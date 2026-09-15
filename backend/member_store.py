"""Private member credentials and revocable sessions, separate from analytics."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping


MEMBER_DB_FILENAME = "goita-members.sqlite3"
SESSION_SECONDS = 30 * 24 * 60 * 60
TEMP_PASSWORD_SECONDS = 24 * 60 * 60
TEMP_SESSION_SECONDS = 30 * 60
PASSWORD_ITERATIONS = 600_000
JST = timezone(timedelta(hours=9))


class MemberError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def resolve_member_path(environ: Mapping[str, str], fallback: Path) -> Path:
    explicit = str(environ.get("GOITA_MEMBER_DB_PATH", "") or "").strip()
    directory = str(environ.get("GOITA_PERSISTENT_DATA_DIR", "") or "").strip()
    return Path(explicit) if explicit else Path(directory) / MEMBER_DB_FILENAME if directory else fallback


def normalize_member_id(value: str) -> str:
    value = value.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{3,31}", value):
        raise MemberError(400, "会員IDは半角英数字・ハイフン・下線の4〜32文字で入力してください。")
    return value


def validate_password(value: str) -> None:
    if not 8 <= len(value) <= 128 or len(value.encode("utf-8")) > 512:
        raise MemberError(400, "パスワードは8〜128文字で入力してください。")
    if not value.strip():
        raise MemberError(400, "空白だけのパスワードは使えません。")


def hash_password(value: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", value.encode("utf-8"), bytes.fromhex(salt), PASSWORD_ITERATIONS)
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt}${digest.hex()}"


def verify_password(value: str, encoded: str) -> bool:
    try:
        algorithm, iterations, salt, digest = encoded.split("$")
        if algorithm != "pbkdf2_sha256" or int(iterations) != PASSWORD_ITERATIONS:
            return False
        actual = hashlib.pbkdf2_hmac("sha256", value.encode("utf-8"), bytes.fromhex(salt), int(iterations))
        return hmac.compare_digest(actual.hex(), digest)
    except (ValueError, TypeError):
        return False


def normalize_expiry(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value or not 2000 <= parsed.year <= 9998:
            raise ValueError
        return value
    except ValueError:
        raise MemberError(400, "有効期限は年月日で指定してください。") from None


class MemberStore:
    PAID_KIFU_LIMIT = 1000
    FREE_KIFU_LIMIT = 20
    SCORE_REWARD_BONUSES = {1: 10, 2: 5, 3: 3}
    SCORE_REWARD_BONUS_CAP = 100
    def __init__(self, path: Path, clock=time.time):
        self.path = Path(path)
        self.clock = clock
        self._lock = threading.Lock()
        self._ready = False

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        return db

    def _schema(self):
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            db = self._connect()
            try:
                db.executescript("""
                    CREATE TABLE IF NOT EXISTS members (
                        member_id TEXT PRIMARY KEY,
                        password_hash TEXT NOT NULL,
                        must_change_password INTEGER NOT NULL,
                        temporary_expires_at REAL,
                        enabled INTEGER NOT NULL DEFAULT 1,
                        paid_enabled INTEGER NOT NULL DEFAULT 1,
                        paid_until TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS member_sessions (
                        token_hash TEXT PRIMARY KEY,
                        member_id TEXT NOT NULL REFERENCES members(member_id),
                        created_at REAL NOT NULL,
                        expires_at REAL NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS member_sessions_owner ON member_sessions(member_id);
                    CREATE TABLE IF NOT EXISTS member_attempts (
                        key TEXT PRIMARY KEY, count INTEGER NOT NULL, expires_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS member_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS member_kifu (
                        id TEXT PRIMARY KEY,
                        member_id TEXT NOT NULL REFERENCES members(member_id) ON DELETE CASCADE,
                        created_at TEXT NOT NULL,
                        title TEXT NOT NULL,
                        memo TEXT NOT NULL,
                        tags_json TEXT NOT NULL,
                        payload_json TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS member_kifu_owner ON member_kifu(member_id, created_at DESC);
                    CREATE TABLE IF NOT EXISTS member_kifu_settings (
                        member_id TEXT PRIMARY KEY REFERENCES members(member_id) ON DELETE CASCADE,
                        auto_save INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE IF NOT EXISTS member_kifu_auto_saves (
                        member_id TEXT NOT NULL REFERENCES members(member_id) ON DELETE CASCADE,
                        round_id TEXT NOT NULL,
                        PRIMARY KEY(member_id, round_id)
                    );
                    CREATE TABLE IF NOT EXISTS member_reward_settings (
                        id INTEGER PRIMARY KEY CHECK(id = 1),
                        free_base_limit INTEGER NOT NULL,
                        paid_base_limit INTEGER NOT NULL,
                        rank1_bonus INTEGER NOT NULL,
                        rank2_bonus INTEGER NOT NULL,
                        rank3_bonus INTEGER NOT NULL,
                        reward_bonus_cap INTEGER NOT NULL,
                        updated_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS member_score_awards (
                        week_start TEXT NOT NULL,
                        member_id TEXT NOT NULL REFERENCES members(member_id) ON DELETE CASCADE,
                        rank INTEGER NOT NULL,
                        kifu_bonus INTEGER NOT NULL,
                        awarded_at REAL NOT NULL,
                        PRIMARY KEY(week_start, member_id)
                    );
                    CREATE TABLE IF NOT EXISTS member_kifu_quota_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        member_id TEXT NOT NULL REFERENCES members(member_id) ON DELETE CASCADE,
                        old_bonus INTEGER NOT NULL,
                        new_bonus INTEGER NOT NULL,
                        note TEXT NOT NULL,
                        changed_at REAL NOT NULL
                    );
                """)
                db.execute("BEGIN IMMEDIATE")
                columns = {row[1] for row in db.execute("PRAGMA table_info(members)")}
                if "is_operator" not in columns:
                    db.execute("ALTER TABLE members ADD COLUMN is_operator INTEGER NOT NULL DEFAULT 0")
                if "research_enabled" not in columns:
                    db.execute("ALTER TABLE members ADD COLUMN research_enabled INTEGER NOT NULL DEFAULT 0")
                if "managed_room_id" not in columns:
                    db.execute("ALTER TABLE members ADD COLUMN managed_room_id TEXT NOT NULL DEFAULT ''")
                if "registration_source" not in columns:
                    db.execute("ALTER TABLE members ADD COLUMN registration_source TEXT NOT NULL DEFAULT 'admin'")
                if "last_login_at" not in columns:
                    db.execute("ALTER TABLE members ADD COLUMN last_login_at REAL")
                if "reward_kifu_bonus" not in columns:
                    db.execute("ALTER TABLE members ADD COLUMN reward_kifu_bonus INTEGER NOT NULL DEFAULT 0")
                if "admin_kifu_bonus" not in columns:
                    db.execute("ALTER TABLE members ADD COLUMN admin_kifu_bonus INTEGER NOT NULL DEFAULT 0")
                db.execute("""INSERT OR IGNORE INTO member_reward_settings VALUES
                              (1, ?, ?, ?, ?, ?, ?, ?)""",
                           (self.FREE_KIFU_LIMIT, self.PAID_KIFU_LIMIT,
                            self.SCORE_REWARD_BONUSES[1], self.SCORE_REWARD_BONUSES[2],
                            self.SCORE_REWARD_BONUSES[3], self.SCORE_REWARD_BONUS_CAP, self.clock()))
                db.execute("CREATE UNIQUE INDEX IF NOT EXISTS member_managed_room ON members(managed_room_id) WHERE managed_room_id <> ''")
                db.execute("INSERT OR IGNORE INTO member_meta VALUES ('throttle_secret', ?)", (secrets.token_hex(32),))
                db.commit()
                self._ready = True
            finally:
                db.close()

    @contextmanager
    def _db(self, write=False):
        self._schema()
        db = self._connect()
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _public(self, row):
        today = datetime.fromtimestamp(self.clock(), JST).date().isoformat()
        paid_active = bool(row["enabled"] and row["paid_enabled"] and
                           (not row["paid_until"] or row["paid_until"] >= today))
        keys = set(row.keys())
        return {
            "member_id": row["member_id"], "enabled": bool(row["enabled"]),
            "paid_enabled": bool(row["paid_enabled"]), "paid_until": row["paid_until"],
            "paid_active": paid_active, "must_change_password": bool(row["must_change_password"]),
            "created_at": row["created_at"],
            "is_operator": bool(row["is_operator"]),
            "research_enabled": bool(row["research_enabled"]),
            "managed_room_id": row["managed_room_id"],
            "registration_source": row["registration_source"],
            "last_login_at": row["last_login_at"],
            "reward_kifu_bonus": int(row["reward_kifu_bonus"]) if "reward_kifu_bonus" in keys else 0,
            "admin_kifu_bonus": int(row["admin_kifu_bonus"]) if "admin_kifu_bonus" in keys else 0,
        }

    @staticmethod
    def _reward_settings_from_db(db):
        row = db.execute("SELECT * FROM member_reward_settings WHERE id = 1").fetchone()
        return {key: int(row[key]) for key in (
            "free_base_limit", "paid_base_limit", "rank1_bonus", "rank2_bonus",
            "rank3_bonus", "reward_bonus_cap",
        )}

    def reward_settings(self):
        with self._db() as db:
            return self._reward_settings_from_db(db)

    def update_reward_settings(self, **values):
        limits = {
            "free_base_limit": (1, 10000), "paid_base_limit": (1, 10000),
            "rank1_bonus": (0, 1000), "rank2_bonus": (0, 1000),
            "rank3_bonus": (0, 1000), "reward_bonus_cap": (0, 10000),
        }
        for key, (minimum, maximum) in limits.items():
            value = values.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
                raise MemberError(400, "棋譜保存枠の設定値を確認してください。")
        if not values["rank1_bonus"] >= values["rank2_bonus"] >= values["rank3_bonus"]:
            raise MemberError(400, "順位報酬は1位から順に同じか小さい値にしてください。")
        with self._db(write=True) as db:
            db.execute("""UPDATE member_reward_settings SET free_base_limit = ?, paid_base_limit = ?,
                          rank1_bonus = ?, rank2_bonus = ?, rank3_bonus = ?, reward_bonus_cap = ?,
                          updated_at = ? WHERE id = 1""",
                       tuple(values[key] for key in limits) + (self.clock(),))
            return self._reward_settings_from_db(db)

    def list_members(self):
        with self._db() as db:
            rows = db.execute("""
                SELECT m.*, COUNT(k.id) AS kifu_count
                FROM members m
                LEFT JOIN member_kifu k ON k.member_id = m.member_id
                GROUP BY m.member_id
                ORDER BY m.created_at DESC, m.member_id
            """).fetchall()
            settings = self._reward_settings_from_db(db)
            return [self._with_usage_db(db, self._public(row), int(row["kifu_count"]), settings)
                    for row in rows]

    def kifu_base_limit(self, member, settings=None):
        settings = settings or self.reward_settings()
        if member.get("paid_active"):
            return settings["paid_base_limit"]
        if not member.get("paid_enabled"):
            return settings["free_base_limit"]
        return 0

    def kifu_limit(self, member, settings=None):
        base = self.kifu_base_limit(member, settings)
        if not base:
            return 0
        return base + max(0, int(member.get("reward_kifu_bonus", 0))) + max(0, int(member.get("admin_kifu_bonus", 0)))

    def can_save_kifu(self, member):
        return bool(member.get("paid_active") or not member.get("paid_enabled"))

    def with_usage(self, member, count=None):
        with self._db() as db:
            if count is None:
                count = db.execute(
                    "SELECT COUNT(*) FROM member_kifu WHERE member_id = ?",
                    (member["member_id"],),
                ).fetchone()[0]
            return self._with_usage_db(db, member, count, self._reward_settings_from_db(db))

    def _with_usage_db(self, db, member, count, settings):
        result = dict(member)
        current_week = (datetime.fromtimestamp(self.clock(), JST).date() -
                        timedelta(days=datetime.fromtimestamp(self.clock(), JST).date().weekday()))
        active_week = (current_week - timedelta(days=7)).isoformat()
        awards = db.execute("""SELECT
              SUM(CASE WHEN rank = 1 THEN 1 ELSE 0 END) AS gold,
              SUM(CASE WHEN rank = 2 THEN 1 ELSE 0 END) AS silver,
              SUM(CASE WHEN rank = 3 THEN 1 ELSE 0 END) AS bronze
              FROM member_score_awards WHERE member_id = ?""", (result["member_id"],)).fetchone()
        active = db.execute("SELECT rank FROM member_score_awards WHERE member_id = ? AND week_start = ?",
                            (result["member_id"], active_week)).fetchone()
        quota_rows = db.execute("""SELECT old_bonus, new_bonus, note, changed_at
            FROM member_kifu_quota_history WHERE member_id = ? ORDER BY changed_at DESC, id DESC LIMIT 5""",
            (result["member_id"],)).fetchall()
        result["kifu_count"] = int(count)
        result["kifu_base_limit"] = self.kifu_base_limit(result, settings)
        result["kifu_limit"] = self.kifu_limit(result, settings)
        result["can_save_kifu"] = self.can_save_kifu(result)
        history = {"gold": int(awards["gold"] or 0), "silver": int(awards["silver"] or 0),
                   "bronze": int(awards["bronze"] or 0)}
        result["score_award_history"] = history
        result["score_champion_stamp"] = history["gold"] > 0
        result["score_title"] = self._score_title(active["rank"]) if active else None
        result["kifu_quota_history"] = [
            {"old_bonus": int(row["old_bonus"]), "new_bonus": int(row["new_bonus"]),
             "note": row["note"], "changed_at": float(row["changed_at"])} for row in quota_rows
        ]
        return result

    @staticmethod
    def _score_title(rank):
        names = {1: ("gold", "金の称号", "🥇"), 2: ("silver", "銀の称号", "🥈"), 3: ("bronze", "銅の称号", "🥉")}
        key, label, medal = names[int(rank)]
        return {"rank": int(rank), "key": key, "label": label, "medal": medal}

    def grant_weekly_score_awards(self, week_start, candidates):
        settings = self.reward_settings()
        granted = []
        with self._db(write=True) as db:
            for candidate in candidates:
                member_id, rank = str(candidate.get("member_id", "")), int(candidate.get("rank", 0))
                if rank not in (1, 2, 3) or not member_id:
                    continue
                member = db.execute("SELECT reward_kifu_bonus FROM members WHERE member_id = ?", (member_id,)).fetchone()
                if not member or db.execute("SELECT 1 FROM member_score_awards WHERE week_start = ? AND member_id = ?",
                                            (week_start, member_id)).fetchone():
                    continue
                requested = settings[f"rank{rank}_bonus"]
                bonus = min(requested, max(0, settings["reward_bonus_cap"] - int(member["reward_kifu_bonus"])))
                db.execute("INSERT INTO member_score_awards VALUES (?, ?, ?, ?, ?)",
                           (week_start, member_id, rank, bonus, self.clock()))
                db.execute("UPDATE members SET reward_kifu_bonus = reward_kifu_bonus + ?, updated_at = ? WHERE member_id = ?",
                           (bonus, self.clock(), member_id))
                granted.append({"member_id": member_id, "rank": rank, "kifu_bonus": bonus})
        return granted

    def score_awards_for_members(self, member_ids, week_start=None):
        member_ids = sorted({value for value in member_ids if value})
        if not member_ids:
            return {}
        placeholders = ",".join("?" for _ in member_ids)
        with self._db() as db:
            if week_start:
                rows = db.execute(f"SELECT member_id, rank, kifu_bonus FROM member_score_awards WHERE week_start = ? AND member_id IN ({placeholders})",
                                  (week_start, *member_ids)).fetchall()
            else:
                current = datetime.fromtimestamp(self.clock(), JST).date()
                active_week = (current - timedelta(days=current.weekday() + 7)).isoformat()
                rows = db.execute(f"SELECT member_id, rank, kifu_bonus FROM member_score_awards WHERE week_start = ? AND member_id IN ({placeholders})",
                                  (active_week, *member_ids)).fetchall()
            return {row["member_id"]: {**self._score_title(row["rank"]), "kifu_bonus": int(row["kifu_bonus"])} for row in rows}

    def active_score_title_for_token(self, token):
        member = self.authenticate(token)
        return self.score_awards_for_members([member["member_id"]]).get(member["member_id"])

    def update_admin_kifu_bonus(self, member_id, bonus, note=""):
        if not isinstance(bonus, int) or isinstance(bonus, bool) or not 0 <= bonus <= 10000:
            raise MemberError(400, "管理者追加枠は0〜10000局で入力してください。")
        note = str(note or "").strip()
        if len(note) > 500:
            raise MemberError(400, "変更メモは500文字以内で入力してください。")
        with self._db(write=True) as db:
            row = db.execute("SELECT admin_kifu_bonus FROM members WHERE member_id = ?", (member_id,)).fetchone()
            if not row:
                raise MemberError(404, "会員が見つかりません。")
            old = int(row["admin_kifu_bonus"])
            if old != bonus:
                now = self.clock()
                db.execute("UPDATE members SET admin_kifu_bonus = ?, updated_at = ? WHERE member_id = ?", (bonus, now, member_id))
                db.execute("INSERT INTO member_kifu_quota_history(member_id, old_bonus, new_bonus, note, changed_at) VALUES (?, ?, ?, ?, ?)",
                           (member_id, old, bonus, note, now))
            updated = db.execute("SELECT * FROM members WHERE member_id = ?", (member_id,)).fetchone()
            return self._with_usage_db(db, self._public(updated), db.execute("SELECT COUNT(*) FROM member_kifu WHERE member_id = ?", (member_id,)).fetchone()[0], self._reward_settings_from_db(db))

    def kifu_auto_save(self, token, enabled=None):
        with self._db(write=enabled is not None) as db:
            member = self._public(self._session_row(db, token))
            if member["must_change_password"]:
                raise MemberError(403, "先にパスワードを変更してください。")
            if enabled is True and not self.can_save_kifu(member):
                raise MemberError(403, "新規保存には有効な会員権限が必要です。")
            if enabled is not None:
                db.execute("INSERT INTO member_kifu_settings VALUES (?, ?) ON CONFLICT(member_id) DO UPDATE SET auto_save = excluded.auto_save",
                           (member["member_id"], int(enabled)))
            row = db.execute("SELECT auto_save FROM member_kifu_settings WHERE member_id = ?", (member["member_id"],)).fetchone()
            return bool(row and row["auto_save"])

    def delete(self, member_id):
        with self._db(write=True) as db:
            if not db.execute("SELECT 1 FROM members WHERE member_id = ?", (member_id,)).fetchone():
                raise MemberError(404, "会員が見つかりません。")
            db.execute("DELETE FROM member_sessions WHERE member_id = ?", (member_id,))
            db.execute("DELETE FROM members WHERE member_id = ?", (member_id,))

    def create(self, member_id, paid_enabled=True, paid_until=None, is_operator=False,
               research_enabled=False, managed_room_id=""):
        member_id = normalize_member_id(member_id)
        paid_until = normalize_expiry(paid_until)
        temporary = secrets.token_urlsafe(18)
        encoded = hash_password(temporary)
        now = self.clock()
        with self._db(write=True) as db:
            try:
                db.execute("""INSERT INTO members
                           (member_id, password_hash, must_change_password, temporary_expires_at,
                            enabled, paid_enabled, paid_until, created_at, updated_at, is_operator,
                            registration_source, last_login_at)
                           VALUES (?, ?, 1, ?, 1, ?, ?, ?, ?, ?, 'admin', NULL)""",
                           (member_id, encoded, now + TEMP_PASSWORD_SECONDS, int(paid_enabled), paid_until, now, now, int(is_operator)))
            except sqlite3.IntegrityError:
                raise MemberError(409, "この会員IDは登録済みです。") from None
            self._assign_room(db, member_id, research_enabled, managed_room_id)
            row = db.execute("SELECT * FROM members WHERE member_id = ?", (member_id,)).fetchone()
        return {"member": self._public(row), "temporary_password": temporary,
                "temporary_expires_at": now + TEMP_PASSWORD_SECONDS}

    def _registration_key(self, db, source_key):
        secret = db.execute("SELECT value FROM member_meta WHERE key = 'throttle_secret'").fetchone()[0]
        value = str(source_key or "unknown")[:256]
        digest = hmac.new(secret.encode(), value.encode(), hashlib.sha256).hexdigest()
        return "register:" + digest

    def _registration_attempt(self, source_key):
        now = self.clock()
        with self._db(write=True) as db:
            db.execute("DELETE FROM member_attempts WHERE expires_at <= ?", (now,))
            limits = [("register:global", 20, 60), (self._registration_key(db, source_key), 5, 900)]
            for key, limit, _seconds in limits:
                row = db.execute("SELECT count FROM member_attempts WHERE key = ?", (key,)).fetchone()
                if row and row[0] >= limit:
                    raise MemberError(429, "登録回数が多いため、時間をおいてお試しください。")
            for key, _limit, seconds in limits:
                db.execute("""INSERT INTO member_attempts VALUES (?, 1, ?)
                              ON CONFLICT(key) DO UPDATE SET count = count + 1""", (key, now + seconds))

    def register(self, member_id, password, source_key=""):
        member_id = normalize_member_id(member_id)
        if len(member_id) < 5:
            raise MemberError(400, "会員IDは5文字以上で入力してください。")
        validate_password(password)
        self._registration_attempt(source_key)
        encoded = hash_password(password)
        now = self.clock()
        with self._db(write=True) as db:
            try:
                db.execute("""INSERT INTO members
                           (member_id, password_hash, must_change_password, temporary_expires_at,
                            enabled, paid_enabled, paid_until, created_at, updated_at, is_operator,
                            registration_source, last_login_at)
                           VALUES (?, ?, 0, NULL, 1, 0, NULL, ?, ?, 0, 'self', ?)""",
                           (member_id, encoded, now, now, now))
            except sqlite3.IntegrityError:
                raise MemberError(409, "この会員IDは登録済みです。") from None
            row = db.execute("SELECT * FROM members WHERE member_id = ?", (member_id,)).fetchone()
            token, seconds = self._issue_session(db, row)
        return self.with_usage(self._public(row)), token, seconds

    def _attempt_key(self, db, member_id):
        secret = db.execute("SELECT value FROM member_meta WHERE key = 'throttle_secret'").fetchone()[0]
        return hmac.new(secret.encode(), member_id.encode(), hashlib.sha256).hexdigest()

    def _attempt(self, member_id):
        # Persist short-lived counters, not IDs, IPs, or submitted passwords.
        now = self.clock()
        with self._db(write=True) as db:
            db.execute("DELETE FROM member_attempts WHERE expires_at <= ?", (now,))
            limits = [("global", 10, 60), (self._attempt_key(db, member_id), 5, 900)]
            for key, limit, _seconds in limits:
                row = db.execute("SELECT count FROM member_attempts WHERE key = ?", (key,)).fetchone()
                if row and row[0] >= limit:
                    raise MemberError(429, "試行回数が多いため、時間をおいてお試しください。")
            for key, _limit, seconds in limits:
                db.execute("""INSERT INTO member_attempts VALUES (?, 1, ?)
                              ON CONFLICT(key) DO UPDATE SET count = count + 1""", (key, now + seconds))

    def _clear_attempt(self, db, member_id):
        db.execute("DELETE FROM member_attempts WHERE key = ?", (self._attempt_key(db, member_id),))

    def _issue_session(self, db, row):
        now = self.clock()
        seconds = TEMP_SESSION_SECONDS if row["must_change_password"] else SESSION_SECONDS
        token = secrets.token_urlsafe(32)
        db.execute("DELETE FROM member_sessions WHERE expires_at <= ?", (now,))
        # Bound concurrent remembered devices per member.
        db.execute("""DELETE FROM member_sessions WHERE member_id = ? AND token_hash NOT IN
                      (SELECT token_hash FROM member_sessions WHERE member_id = ? ORDER BY created_at DESC LIMIT 9)""",
                   (row["member_id"], row["member_id"]))
        db.execute("INSERT INTO member_sessions VALUES (?, ?, ?, ?)",
                   (hashlib.sha256(token.encode()).hexdigest(), row["member_id"], now, now + seconds))
        return token, seconds

    def login(self, member_id, password):
        normalized = member_id.strip().lower()
        self._attempt(normalized)
        member_public = None
        with self._db(write=True) as db:
            row = db.execute("SELECT * FROM members WHERE member_id = ?", (normalized,)).fetchone()
            # An unknown account still pays the same password-verification cost.
            dummy = f"pbkdf2_sha256${PASSWORD_ITERATIONS}${'00' * 16}${'00' * 32}"
            valid = verify_password(password, row["password_hash"] if row else dummy)
            if not valid or not row or not row["enabled"] or (
                row["must_change_password"] and (row["temporary_expires_at"] or 0) <= self.clock()
            ):
                raise MemberError(401, "会員IDまたはパスワードを確認してください。仮パスワードの期限切れは運営へお問い合わせください。")
            self._clear_attempt(db, normalized)
            now = self.clock()
            db.execute("UPDATE members SET last_login_at = ?, updated_at = ? WHERE member_id = ?",
                       (now, now, normalized))
            row = db.execute("SELECT * FROM members WHERE member_id = ?", (normalized,)).fetchone()
            token, seconds = self._issue_session(db, row)
            member_public = self._public(row)
        return self.with_usage(member_public), token, seconds

    def _session_row(self, db, token):
        if not token or len(token) > 128:
            raise MemberError(401, "ログインしてください。")
        row = db.execute("""SELECT m.* FROM members m JOIN member_sessions s USING(member_id)
                            WHERE s.token_hash = ? AND s.expires_at > ? AND m.enabled = 1""",
                         (hashlib.sha256(token.encode()).hexdigest(), self.clock())).fetchone()
        if not row or (row["must_change_password"] and (row["temporary_expires_at"] or 0) <= self.clock()):
            raise MemberError(401, "ログインしてください。")
        return row

    def authenticate(self, token, *, allow_temporary=False, require_paid=False):
        with self._db() as db:
            row = self._session_row(db, token)
        member = self._public(row)
        if member["must_change_password"] and not allow_temporary:
            raise MemberError(403, "先にパスワードを変更してください。")
        if require_paid and (member["must_change_password"] or not member["paid_active"]):
            raise MemberError(403, "有効な有料権限が必要です。")
        return member

    def logout(self, token):
        if not token:
            return
        with self._db(write=True) as db:
            db.execute("DELETE FROM member_sessions WHERE token_hash = ?", (hashlib.sha256(token.encode()).hexdigest(),))

    def change_password(self, token, current_password, new_password):
        validate_password(new_password)
        member = self.authenticate(token, allow_temporary=True)
        self._attempt(member["member_id"])
        with self._db(write=True) as db:
            row = self._session_row(db, token)
            if not verify_password(current_password, row["password_hash"]):
                raise MemberError(400, "現在のパスワードが違います。")
            if hmac.compare_digest(current_password.encode(), new_password.encode()):
                raise MemberError(400, "現在と異なるパスワードを設定してください。")
            db.execute("""UPDATE members SET password_hash = ?, must_change_password = 0,
                          temporary_expires_at = NULL, updated_at = ? WHERE member_id = ?""",
                       (hash_password(new_password), self.clock(), row["member_id"]))
            db.execute("DELETE FROM member_sessions WHERE member_id = ?", (row["member_id"],))
            self._clear_attempt(db, row["member_id"])
            updated = db.execute("SELECT * FROM members WHERE member_id = ?", (row["member_id"],)).fetchone()
            new_token, seconds = self._issue_session(db, updated)
            return self._public(updated), new_token, seconds

    def reset_password(self, member_id):
        temporary = secrets.token_urlsafe(18)
        now = self.clock()
        with self._db(write=True) as db:
            row = db.execute("SELECT * FROM members WHERE member_id = ?", (member_id,)).fetchone()
            if not row:
                raise MemberError(404, "会員が見つかりません。")
            db.execute("""UPDATE members SET password_hash = ?, must_change_password = 1,
                          temporary_expires_at = ?, updated_at = ? WHERE member_id = ?""",
                       (hash_password(temporary), now + TEMP_PASSWORD_SECONDS, now, member_id))
            db.execute("DELETE FROM member_sessions WHERE member_id = ?", (member_id,))
            self._clear_attempt(db, member_id)
        return {"member_id": member_id, "temporary_password": temporary,
                "temporary_expires_at": now + TEMP_PASSWORD_SECONDS}

    def update(self, member_id, *, enabled, paid_enabled, paid_until, is_operator=None,
               research_enabled=None, managed_room_id=None):
        paid_until = normalize_expiry(paid_until)
        with self._db(write=True) as db:
            result = db.execute("""UPDATE members SET enabled = ?, paid_enabled = ?, paid_until = ?,
                                   updated_at = ? WHERE member_id = ?""",
                                (int(enabled), int(paid_enabled), paid_until, self.clock(), member_id))
            if not result.rowcount:
                raise MemberError(404, "会員が見つかりません。")
            if is_operator is not None:
                db.execute("UPDATE members SET is_operator = ? WHERE member_id = ?", (int(is_operator), member_id))
            if research_enabled is not None or managed_room_id is not None:
                previous = db.execute("SELECT * FROM members WHERE member_id = ?", (member_id,)).fetchone()
                research = bool(previous["research_enabled"]) if research_enabled is None else research_enabled
                room_id = previous["managed_room_id"] if managed_room_id is None else managed_room_id
                self._assign_room(db, member_id, research, room_id if research else "")
            if not enabled:
                db.execute("DELETE FROM member_sessions WHERE member_id = ?", (member_id,))
            row = db.execute("SELECT * FROM members WHERE member_id = ?", (member_id,)).fetchone()
        return self._public(row)

    @staticmethod
    def _assign_room(db, member_id, research_enabled, room_id):
        if room_id and not research_enabled:
            raise MemberError(400, "部屋の割り当てには研究用プランが必要です。")
        try:
            db.execute("UPDATE members SET research_enabled = ?, managed_room_id = ? WHERE member_id = ?",
                       (int(research_enabled), room_id, member_id))
        except sqlite3.IntegrityError:
            raise MemberError(409, "この部屋は別の会員に割り当てられています。先に割り当てを解除してください。") from None

    def is_operator_session(self, token):
        if not token:
            return False
        try:
            # Temporary sessions also exclude the initial password-change visit.
            return self.authenticate(token, allow_temporary=True)["is_operator"]
        except MemberError:
            return False

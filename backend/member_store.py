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
                """)
                db.execute("BEGIN IMMEDIATE")
                columns = {row[1] for row in db.execute("PRAGMA table_info(members)")}
                if "is_operator" not in columns:
                    db.execute("ALTER TABLE members ADD COLUMN is_operator INTEGER NOT NULL DEFAULT 0")
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
        return {
            "member_id": row["member_id"], "enabled": bool(row["enabled"]),
            "paid_enabled": bool(row["paid_enabled"]), "paid_until": row["paid_until"],
            "paid_active": paid_active, "must_change_password": bool(row["must_change_password"]),
            "created_at": row["created_at"],
            "is_operator": bool(row["is_operator"]),
        }

    def list_members(self):
        with self._db() as db:
            return [self._public(row) for row in db.execute("SELECT * FROM members ORDER BY created_at DESC, member_id")]

    def kifu_auto_save(self, token, enabled=None):
        with self._db(write=enabled is not None) as db:
            member = self._public(self._session_row(db, token))
            if member["must_change_password"]:
                raise MemberError(403, "先にパスワードを変更してください。")
            if enabled is True and not member["paid_active"]:
                raise MemberError(403, "新規保存には有効な有料権限が必要です。")
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

    def create(self, member_id, paid_enabled=True, paid_until=None, is_operator=False):
        member_id = normalize_member_id(member_id)
        paid_until = normalize_expiry(paid_until)
        temporary = secrets.token_urlsafe(18)
        encoded = hash_password(temporary)
        now = self.clock()
        with self._db(write=True) as db:
            try:
                db.execute("""INSERT INTO members
                           (member_id, password_hash, must_change_password, temporary_expires_at,
                            enabled, paid_enabled, paid_until, created_at, updated_at, is_operator)
                           VALUES (?, ?, 1, ?, 1, ?, ?, ?, ?, ?)""",
                           (member_id, encoded, now + TEMP_PASSWORD_SECONDS, int(paid_enabled), paid_until, now, now, int(is_operator)))
            except sqlite3.IntegrityError:
                raise MemberError(409, "この会員IDは登録済みです。") from None
            row = db.execute("SELECT * FROM members WHERE member_id = ?", (member_id,)).fetchone()
        return {"member": self._public(row), "temporary_password": temporary,
                "temporary_expires_at": now + TEMP_PASSWORD_SECONDS}

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
            token, seconds = self._issue_session(db, row)
            return self._public(row), token, seconds

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

    def update(self, member_id, *, enabled, paid_enabled, paid_until, is_operator=None):
        paid_until = normalize_expiry(paid_until)
        with self._db(write=True) as db:
            result = db.execute("""UPDATE members SET enabled = ?, paid_enabled = ?, paid_until = ?,
                                   updated_at = ? WHERE member_id = ?""",
                                (int(enabled), int(paid_enabled), paid_until, self.clock(), member_id))
            if not result.rowcount:
                raise MemberError(404, "会員が見つかりません。")
            if is_operator is not None:
                db.execute("UPDATE members SET is_operator = ? WHERE member_id = ?", (int(is_operator), member_id))
            if not enabled:
                db.execute("DELETE FROM member_sessions WHERE member_id = ?", (member_id,))
            row = db.execute("SELECT * FROM members WHERE member_id = ?", (member_id,)).fetchone()
        return self._public(row)

    def is_operator_session(self, token):
        if not token:
            return False
        try:
            # Temporary sessions also exclude the initial password-change visit.
            return self.authenticate(token, allow_temporary=True)["is_operator"]
        except MemberError:
            return False

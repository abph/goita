"""Persistent, server-scored trace attempts. Guest records expire after 30 days."""
import asyncio
import hashlib
import json
import logging
import os
import secrets
import sqlite3
import time
from contextlib import asynccontextmanager, contextmanager, suppress
from pathlib import Path

from fastapi import HTTPException

from backend.member_api import MEMBER_COOKIE
from backend.member_store import MemberError
from backend.private_kifu_archive import archive_path

GUEST_COOKIE = "goita_trace_guest"
RETENTION = 30 * 86400


class TraceStore:
    def __init__(self, path, clock=time.time):
        self.path, self.clock = Path(path), clock

    @contextmanager
    def db(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA secure_delete=ON")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS trace_people (
                    owner TEXT PRIMARY KEY, name TEXT NOT NULL, guest INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS trace_challenges (
                    id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS trace_attempts (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL REFERENCES trace_people(owner),
                    challenge TEXT NOT NULL REFERENCES trace_challenges(id),
                    started REAL NOT NULL, finished REAL, expires REAL,
                    ranked INTEGER NOT NULL, actual_ac INTEGER, actual_bd INTEGER,
                    improvement INTEGER);
                CREATE INDEX IF NOT EXISTS trace_owner_challenge ON trace_attempts(owner, challenge);
                CREATE INDEX IF NOT EXISTS trace_ranking ON trace_attempts(challenge, ranked, improvement DESC);
                CREATE INDEX IF NOT EXISTS trace_expiry ON trace_attempts(expires);
                CREATE TABLE IF NOT EXISTS trace_sessions (
                    owner TEXT PRIMARY KEY, expires REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS trace_challenge_labels (
                    number INTEGER PRIMARY KEY AUTOINCREMENT,
                    challenge TEXT UNIQUE NOT NULL REFERENCES trace_challenges(id) ON DELETE CASCADE);
            """)
            db.execute("BEGIN IMMEDIATE")
            columns = {row[1] for row in db.execute("PRAGMA table_info(trace_attempts)")}
            if "attempt_no" not in columns:
                db.execute("ALTER TABLE trace_attempts ADD COLUMN attempt_no INTEGER NOT NULL DEFAULT 0")
                db.execute("""WITH numbered AS (
                    SELECT id, ROW_NUMBER() OVER (PARTITION BY owner,challenge ORDER BY started,id) AS n
                    FROM trace_attempts)
                    UPDATE trace_attempts SET attempt_no=(SELECT n FROM numbered WHERE numbered.id=trace_attempts.id)""")
            db.execute("DELETE FROM trace_attempts WHERE expires <= ?", (self.clock(),))
            db.execute("DELETE FROM trace_sessions WHERE expires <= ?", (self.clock(),))
            db.execute("DELETE FROM trace_people WHERE guest=1 AND owner NOT IN (SELECT owner FROM trace_attempts)")
            db.execute("DELETE FROM trace_challenges WHERE id NOT IN (SELECT challenge FROM trace_attempts)")
            db.execute("INSERT INTO trace_challenge_labels(challenge) SELECT id FROM trace_challenges WHERE id NOT IN (SELECT challenge FROM trace_challenge_labels) ORDER BY rowid")
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def cleanup(self):
        with self.db():
            pass

    def identity(self, request, response, members, *, create=False):
        token = request.cookies.get(MEMBER_COOKIE, "")
        if token:
            try:
                member = members.authenticate(token)
                return "member:" + member["member_id"], False
            except MemberError as error:
                if error.status != 401:
                    raise HTTPException(error.status, str(error)) from error
        guest = request.cookies.get(GUEST_COOKIE, "")
        owner = "guest:" + hashlib.sha256(guest.encode()).hexdigest()
        with self.db() as db:
            known = bool(guest and db.execute("SELECT 1 FROM trace_people WHERE owner=? UNION ALL SELECT 1 FROM trace_sessions WHERE owner=?", (owner, owner)).fetchone())
        if not known:
            if not create:
                raise HTTPException(401, "挑戦したブラウザ、または会員IDで開いてください。")
            guest = secrets.token_urlsafe(32)
            owner = "guest:" + hashlib.sha256(guest.encode()).hexdigest()
        if create:
            with self.db() as db:
                db.execute("INSERT INTO trace_sessions VALUES (?,?) ON CONFLICT(owner) DO UPDATE SET expires=excluded.expires", (owner, self.clock() + RETENTION))
            response.set_cookie(GUEST_COOKIE, guest, max_age=RETENTION, httponly=True,
                                secure=bool(os.environ.get("RENDER")) or request.url.scheme == "https",
                                samesite="strict", path="/")
        return owner, True

    @staticmethod
    def challenge_id(payload):
        # Labels, archive ID and file format do not distinguish the same challenge.
        canonical = {key: payload[key] for key in ("hands", "dealer", "moves", "score_before", "score_after")}
        canonical["rules"] = "trace-current-v1"
        return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def start(self, owner, guest, name, payload, *, practice=False):
        challenge = self.challenge_id(payload)
        attempt_id = secrets.token_urlsafe(24)
        now = self.clock()
        with self.db() as db:
            attempt_no = db.execute("SELECT COALESCE(MAX(attempt_no),0)+1 FROM trace_attempts WHERE owner=? AND challenge=?", (owner, challenge)).fetchone()[0]
            first = attempt_no == 1
            db.execute("INSERT INTO trace_people VALUES (?,?,?) ON CONFLICT(owner) DO UPDATE SET name=excluded.name",
                       (owner, name.strip()[:24] or ("ゲスト" if guest else "プレイヤー"), int(guest)))
            db.execute("INSERT OR IGNORE INTO trace_challenges VALUES (?,?)", (challenge, json.dumps(payload)))
            db.execute("INSERT INTO trace_challenge_labels(challenge) SELECT ? WHERE NOT EXISTS (SELECT 1 FROM trace_challenge_labels WHERE challenge=?)", (challenge, challenge))
            db.execute("INSERT INTO trace_attempts(id,owner,challenge,started,expires,ranked,attempt_no) VALUES (?,?,?,?,?,?,?)",
                       (attempt_id, owner, challenge, now, now + RETENTION if guest else None, int(first and not practice), attempt_no))
        return attempt_id

    def finish(self, attempt_id, actual):
        with self.db() as db:
            row = db.execute("SELECT a.*, c.payload FROM trace_attempts a JOIN trace_challenges c ON c.id=a.challenge WHERE a.id=?", (attempt_id,)).fetchone()
            if row is None or row["finished"] is not None:
                return
            payload = json.loads(row["payload"])
            before, after = payload["score_before"], payload["score_after"]
            ac, bd = int(actual["AC"]) - before["AC"], int(actual["BD"]) - before["BD"]
            original_margin = (after["AC"] - before["AC"]) - (after["BD"] - before["BD"])
            now = self.clock()
            db.execute("UPDATE trace_attempts SET finished=?, expires=?, actual_ac=?, actual_bd=?, improvement=? WHERE id=? AND finished IS NULL",
                       (now, now + RETENTION if row["expires"] is not None else None, ac, bd, ac - bd - original_margin, attempt_id))

    def read(self, owner, attempt_id, *, original=False, mode="best"):
        if mode not in ("all", "best", "first"):
            raise HTTPException(400, "ランキングの種類を確認してください。")
        with self.db() as db:
            row = db.execute("SELECT a.*, c.payload, p.guest FROM trace_attempts a JOIN trace_challenges c ON c.id=a.challenge JOIN trace_people p ON p.owner=a.owner WHERE a.id=? AND a.owner=?", (attempt_id, owner)).fetchone()
            if row is None:
                raise HTTPException(404, "記録が見つかりません。ゲストの保存期間は30日間です。")
            if row["finished"] is None:
                raise HTTPException(409, "結果と元の棋譜は終局後に表示できます。")
            payload = json.loads(row["payload"])
            if original:
                return payload
            ranking = db.execute("""WITH candidates AS (
                SELECT a.*, name,guest,
                    ROW_NUMBER() OVER (PARTITION BY owner ORDER BY improvement DESC,attempt_no,finished,a.id) AS best
                FROM trace_attempts a JOIN trace_people p USING(owner)
                WHERE challenge=? AND finished IS NOT NULL AND (?!='first' OR ranked=1))
                SELECT *, RANK() OVER (ORDER BY improvement DESC) AS position
                FROM candidates WHERE ?='all' OR best=1
                ORDER BY improvement DESC, finished ASC, id ASC""", (row["challenge"], mode, mode)).fetchall()
            own_rank = next((item["position"] for item in ranking
                             if (item["id"] == attempt_id if mode == "all" else item["owner"] == owner)), None)
            best_score = db.execute("SELECT MAX(improvement) FROM trace_attempts WHERE owner=? AND challenge=? AND finished IS NOT NULL", (owner, row["challenge"])).fetchone()[0]
            number = db.execute("SELECT number FROM trace_challenge_labels WHERE challenge=?", (row["challenge"],)).fetchone()[0]
            before, after = payload["score_before"], payload["score_after"]
            return {"attempt_id": attempt_id, "ranked": bool(row["ranked"]),
                    "mode": mode, "attempt_no": row["attempt_no"], "challenge_label": f"課題{number:03d}",
                    "is_best": row["improvement"] == best_score,
                    "guest": bool(row["guest"]), "expires_at": row["expires"],
                    "original": {team: after[team] - before[team] for team in ("AC", "BD")},
                    "actual": {"AC": row["actual_ac"], "BD": row["actual_bd"]},
                    "improvement": row["improvement"], "own_rank": own_rank, "total": len(ranking),
                    "ranking": [{"rank": item["position"], "name": item["name"], "guest": bool(item["guest"]),
                                 "attempt_no": item["attempt_no"],
                                 "improvement": item["improvement"], "finished_at": item["finished"],
                                 "self": item["owner"] == owner} for item in ranking[:100]]}

    def history(self, owner, *, offset=0, limit=30):
        if offset < 0 or not 1 <= limit <= 100:
            raise HTTPException(400, "履歴の表示範囲を確認してください。")
        with self.db() as db:
            total = db.execute("SELECT COUNT(*) FROM trace_attempts WHERE owner=? AND finished IS NOT NULL", (owner,)).fetchone()[0]
            offset = min(offset, ((max(total, 1) - 1) // limit) * limit)
            rows = db.execute("""SELECT a.*, l.number,
                MAX(improvement) OVER (PARTITION BY a.challenge) AS best_score
                FROM trace_attempts a JOIN trace_challenge_labels l ON l.challenge=a.challenge
                WHERE owner=? AND finished IS NOT NULL
                ORDER BY finished DESC,started DESC,attempt_no DESC,a.id DESC LIMIT ? OFFSET ?""", (owner, limit, offset)).fetchall()
            return {"total": total, "offset": offset, "limit": limit, "records": [
                {"attempt_id": row["id"], "challenge_label": f"課題{row['number']:03d}",
                 "finished_at": row["finished"], "improvement": row["improvement"],
                 "attempt_no": row["attempt_no"], "is_best": row["improvement"] == row["best_score"]}
                for row in rows]}

    def latest(self, owner):
        with self.db() as db:
            row = db.execute("SELECT id FROM trace_attempts WHERE owner=? AND finished IS NOT NULL ORDER BY finished DESC LIMIT 1", (owner,)).fetchone()
            return row["id"] if row else None


def get_trace_store():
    base = Path(__file__).resolve().parents[1]
    return TraceStore(archive_path(base).parent / "trace-results.sqlite3")


@asynccontextmanager
async def trace_lifespan(app):
    async def purge():
        while True:
            try:
                await asyncio.to_thread(get_trace_store().cleanup)
            except (OSError, ValueError, sqlite3.Error):
                logging.getLogger(__name__).warning("Trace result cleanup unavailable; check persistent storage.")
            await asyncio.sleep(3600)
    task = asyncio.create_task(purge())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

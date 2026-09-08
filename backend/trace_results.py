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
            """)
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM trace_attempts WHERE expires <= ?", (self.clock(),))
            db.execute("DELETE FROM trace_people WHERE guest=1 AND owner NOT IN (SELECT owner FROM trace_attempts)")
            db.execute("DELETE FROM trace_challenges WHERE id NOT IN (SELECT challenge FROM trace_attempts)")
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
            known = bool(guest and db.execute("SELECT 1 FROM trace_people WHERE owner=?", (owner,)).fetchone())
        if not known:
            if not create:
                raise HTTPException(401, "挑戦したブラウザ、または会員IDで開いてください。")
            guest = secrets.token_urlsafe(32)
            owner = "guest:" + hashlib.sha256(guest.encode()).hexdigest()
        if create:
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
            first = not db.execute("SELECT 1 FROM trace_attempts WHERE owner=? AND challenge=?", (owner, challenge)).fetchone()
            db.execute("INSERT INTO trace_people VALUES (?,?,?) ON CONFLICT(owner) DO UPDATE SET name=excluded.name",
                       (owner, name.strip()[:24] or ("ゲスト" if guest else "プレイヤー"), int(guest)))
            db.execute("INSERT OR IGNORE INTO trace_challenges VALUES (?,?)", (challenge, json.dumps(payload)))
            db.execute("INSERT INTO trace_attempts(id,owner,challenge,started,expires,ranked) VALUES (?,?,?,?,?,?)",
                       (attempt_id, owner, challenge, now, now + RETENTION if guest else None, int(first and not practice)))
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

    def read(self, owner, attempt_id, *, original=False):
        with self.db() as db:
            row = db.execute("SELECT a.*, c.payload, p.guest FROM trace_attempts a JOIN trace_challenges c ON c.id=a.challenge JOIN trace_people p ON p.owner=a.owner WHERE a.id=? AND a.owner=?", (attempt_id, owner)).fetchone()
            if row is None:
                raise HTTPException(404, "記録が見つかりません。ゲストの保存期間は30日間です。")
            if row["finished"] is None:
                raise HTTPException(409, "結果と元の棋譜は終局後に表示できます。")
            payload = json.loads(row["payload"])
            if original:
                return payload
            ranking = db.execute("""SELECT name,guest,improvement,finished,owner,
                RANK() OVER (ORDER BY improvement DESC) AS position
                FROM trace_attempts a JOIN trace_people p USING(owner)
                WHERE challenge=? AND ranked=1 AND finished IS NOT NULL
                ORDER BY improvement DESC, finished ASC, a.id ASC""", (row["challenge"],)).fetchall()
            own_rank = next((item["position"] for item in ranking if item["owner"] == owner), None)
            before, after = payload["score_before"], payload["score_after"]
            return {"attempt_id": attempt_id, "ranked": bool(row["ranked"]),
                    "guest": bool(row["guest"]), "expires_at": row["expires"],
                    "original": {team: after[team] - before[team] for team in ("AC", "BD")},
                    "actual": {"AC": row["actual_ac"], "BD": row["actual_bd"]},
                    "improvement": row["improvement"], "own_rank": own_rank, "total": len(ranking),
                    "ranking": [{"rank": item["position"], "name": item["name"], "guest": bool(item["guest"]),
                                 "improvement": item["improvement"], "finished_at": item["finished"],
                                 "self": item["owner"] == owner} for item in ranking[:100]]}

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

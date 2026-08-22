"""Store private-room research records in a small SQLite library.

Each private room owns an isolated library. The application authenticates room
administrators before calling this module; this layer handles durable, atomic
storage and never exposes records belonging to another room.
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


RESEARCH_KIFU_FILENAME = "goita-research-kifu.sqlite3"
RESEARCH_KIFU_VERSION = 1
_ID_ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"


def resolve_research_kifu_path(
    environ: Mapping[str, str],
    *,
    local_fallback: Optional[Path] = None,
) -> Path:
    """Resolve an explicit path, Render disk path, or ignored local fallback."""

    explicit = str(environ.get("GOITA_KIFU_DB_PATH", "") or "").strip()
    if explicit:
        return Path(explicit)
    persistent = str(
        environ.get("GOITA_PERSISTENT_DATA_DIR", "") or ""
    ).strip()
    if persistent:
        return Path(persistent) / RESEARCH_KIFU_FILENAME
    if local_fallback is not None:
        return Path(local_fallback)
    return Path(RESEARCH_KIFU_FILENAME)


class ResearchKifuStore:
    """Provide room-scoped CRUD operations for research game records."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._schema_lock = threading.Lock()
        self._schema_ready = False

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            with closing(self._connect()) as connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS research_kifu (
                        id TEXT PRIMARY KEY,
                        room_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        title TEXT NOT NULL,
                        memo TEXT NOT NULL,
                        round_index INTEGER NOT NULL,
                        dealer TEXT NOT NULL,
                        winner TEXT,
                        gained_score INTEGER NOT NULL,
                        payload_json TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_research_kifu_room_created
                    ON research_kifu(room_id, created_at DESC, id DESC);
                    """
                )
                connection.execute(
                    f"PRAGMA user_version = {RESEARCH_KIFU_VERSION}"
                )
                connection.commit()
            self._schema_ready = True

    @staticmethod
    def _new_id() -> str:
        return "K-" + "".join(secrets.choice(_ID_ALPHABET) for _ in range(10))

    @staticmethod
    def _summary(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": str(row["id"]),
            "created_at": str(row["created_at"]),
            "title": str(row["title"]),
            "memo": str(row["memo"]),
            "round_index": int(row["round_index"]),
            "dealer": str(row["dealer"]),
            "winner": str(row["winner"] or ""),
            "gained_score": int(row["gained_score"]),
        }

    def save(
        self,
        room_id: str,
        *,
        title: str,
        memo: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._ensure_schema()
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        payload_json = json.dumps(
            dict(payload), ensure_ascii=False, separators=(",", ":")
        )
        for _attempt in range(8):
            record_id = self._new_id()
            try:
                with closing(self._connect()) as connection:
                    connection.execute(
                        """
                        INSERT INTO research_kifu (
                            id, room_id, created_at, title, memo, round_index,
                            dealer, winner, gained_score, payload_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record_id,
                            room_id,
                            created_at,
                            title,
                            memo,
                            int(payload.get("round_index", 1)),
                            str(payload.get("dealer", "A")),
                            str(payload.get("winner", "")) or None,
                            int(payload.get("gained_score", 0)),
                            payload_json,
                        ),
                    )
                    connection.commit()
                return self.get(room_id, record_id) or {}
            except sqlite3.IntegrityError:
                continue
        raise RuntimeError("Unable to allocate a research kifu ID")

    def list(self, room_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        self._ensure_schema()
        safe_limit = max(1, min(int(limit), 500))
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT id, created_at, title, memo, round_index, dealer,
                       winner, gained_score
                FROM research_kifu
                WHERE room_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (room_id, safe_limit),
            ).fetchall()
        return [self._summary(row) for row in rows]

    def get(self, room_id: str, record_id: str) -> Optional[dict[str, Any]]:
        self._ensure_schema()
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT id, created_at, title, memo, round_index, dealer,
                       winner, gained_score, payload_json
                FROM research_kifu
                WHERE room_id = ? AND id = ?
                """,
                (room_id, record_id),
            ).fetchone()
        if row is None:
            return None
        record = self._summary(row)
        record["payload"] = json.loads(str(row["payload_json"]))
        return record

    def update_memo(
        self,
        room_id: str,
        record_id: str,
        memo: str,
    ) -> Optional[dict[str, Any]]:
        self._ensure_schema()
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "UPDATE research_kifu SET memo = ? WHERE room_id = ? AND id = ?",
                (memo, room_id, record_id),
            )
            connection.commit()
        if cursor.rowcount <= 0:
            return None
        return self.get(room_id, record_id)

    def delete(self, room_id: str, record_id: str) -> bool:
        self._ensure_schema()
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "DELETE FROM research_kifu WHERE room_id = ? AND id = ?",
                (room_id, record_id),
            )
            connection.commit()
        return cursor.rowcount > 0


def is_persistent_research_kifu_configured(environ: Mapping[str, str]) -> bool:
    return bool(
        str(environ.get("GOITA_KIFU_DB_PATH", "") or "").strip()
        or str(environ.get("GOITA_PERSISTENT_DATA_DIR", "") or "").strip()
    )

"""Persist privacy-limited product analytics for Solo Goita.

The store accepts only a fixed event/property vocabulary. It deliberately has
no columns for names, seats, game IDs, chat, hands, kifu, partners, opponents,
or raw IP addresses, so gameplay and social relationships cannot enter it.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from backend.analytics_geo import normalize_prefecture


ANALYTICS_FILENAME = "goita-analytics.sqlite3"
ANALYTICS_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,80}$")
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,80}$")
ALLOWED_EVENTS = frozenset({
    "site_visit",
    "room_enter",
    "room_leave",
    "seat_taken",
    "seat_left",
    "game_started",
    "game_completed",
    "beginner_support_enabled",
    "beginner_support_disabled",
    "ai_question_used",
    "chat_opened",
    "kifu_library_opened",
    "kifu_saved",
    "kifu_loaded",
    "heartbeat",
})
ALLOWED_ROOM_TYPES = frozenset({
    "lobby",
    "public_human",
    "public_ai",
    "private",
    "none",
})
ALLOWED_DEVICES = frozenset({"mobile", "desktop", "tablet", "unknown"})
ALLOWED_LANGUAGES = frozenset({"ja", "zh", "en", "other"})
ALLOWED_ROLES = frozenset({"host", "joined", "spectator", "none"})
ALLOWED_PROPERTIES = {
    "role",
    "human_count",
    "ai_count",
    "pair_practice",
    "completed",
}


def resolve_analytics_path(
    environ: Mapping[str, str],
    *,
    local_fallback: Optional[Path] = None,
) -> Path:
    """Resolve an explicit path, persistent Render path, or local fallback."""

    explicit = str(environ.get("GOITA_ANALYTICS_DB_PATH", "") or "").strip()
    if explicit:
        return Path(explicit)
    persistent = str(environ.get("GOITA_PERSISTENT_DATA_DIR", "") or "").strip()
    if persistent:
        return Path(persistent) / ANALYTICS_FILENAME
    if local_fallback is not None:
        return Path(local_fallback)
    return Path(ANALYTICS_FILENAME)


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean_text(value: Any, max_length: int) -> str:
    return str(value or "").strip().replace("\r", " ").replace("\n", " ")[:max_length]


def _safe_properties(properties: Any) -> dict[str, Any]:
    if not isinstance(properties, dict):
        return {}
    cleaned: dict[str, Any] = {}
    role = str(properties.get("role", "none") or "none")
    if role in ALLOWED_ROLES:
        cleaned["role"] = role
    for key in ("human_count", "ai_count"):
        try:
            cleaned[key] = max(0, min(4, int(properties.get(key, 0))))
        except (TypeError, ValueError):
            pass
    for key in ("pair_practice", "completed"):
        if key in properties:
            cleaned[key] = bool(properties[key])
    return {key: value for key, value in cleaned.items() if key in ALLOWED_PROPERTIES}


class AnalyticsStore:
    """Store compact, indefinitely retained session journeys and summaries."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._schema_lock = threading.Lock()
        self._schema_ready = False

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        connection.execute("PRAGMA journal_mode = WAL")
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
                    CREATE TABLE IF NOT EXISTS analytics_visitors (
                        analytics_id TEXT PRIMARY KEY,
                        first_seen TEXT NOT NULL,
                        last_seen TEXT NOT NULL,
                        first_source TEXT NOT NULL DEFAULT '',
                        first_campaign TEXT NOT NULL DEFAULT '',
                        device TEXT NOT NULL DEFAULT 'unknown',
                        language TEXT NOT NULL DEFAULT 'other'
                    );
                    CREATE TABLE IF NOT EXISTS analytics_sessions (
                        session_id TEXT PRIMARY KEY,
                        analytics_id TEXT NOT NULL,
                        started_at TEXT NOT NULL,
                        last_seen TEXT NOT NULL,
                        ended_at TEXT,
                        source TEXT NOT NULL DEFAULT '',
                        medium TEXT NOT NULL DEFAULT '',
                        campaign TEXT NOT NULL DEFAULT '',
                        device TEXT NOT NULL DEFAULT 'unknown',
                        language TEXT NOT NULL DEFAULT 'other',
                        prefecture TEXT NOT NULL DEFAULT '不明',
                        event_count INTEGER NOT NULL DEFAULT 0,
                        FOREIGN KEY (analytics_id) REFERENCES analytics_visitors(analytics_id)
                    );
                    CREATE TABLE IF NOT EXISTS analytics_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        analytics_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        occurred_at TEXT NOT NULL,
                        event_name TEXT NOT NULL,
                        room_type TEXT NOT NULL DEFAULT 'none',
                        properties_json TEXT NOT NULL DEFAULT '{}',
                        FOREIGN KEY (analytics_id) REFERENCES analytics_visitors(analytics_id),
                        FOREIGN KEY (session_id) REFERENCES analytics_sessions(session_id)
                    );
                    CREATE INDEX IF NOT EXISTS idx_analytics_events_time
                    ON analytics_events(occurred_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_analytics_events_name_time
                    ON analytics_events(event_name, occurred_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_analytics_events_visitor_time
                    ON analytics_events(analytics_id, occurred_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_analytics_sessions_time
                    ON analytics_sessions(started_at DESC);
                    """
                )
                session_columns = {
                    str(row[1])
                    for row in connection.execute("PRAGMA table_info(analytics_sessions)")
                }
                if "prefecture" not in session_columns:
                    connection.execute(
                        "ALTER TABLE analytics_sessions "
                        "ADD COLUMN prefecture TEXT NOT NULL DEFAULT '不明'"
                    )
                connection.commit()
            self._schema_ready = True

    def record_event(self, payload: Mapping[str, Any]) -> bool:
        """Validate and store one allow-listed event without accepting identity data."""

        analytics_id = _clean_text(payload.get("analytics_id"), 80)
        session_id = _clean_text(payload.get("session_id"), 80)
        event_name = _clean_text(payload.get("event"), 50)
        if (
            not ANALYTICS_ID_RE.fullmatch(analytics_id)
            or not SESSION_ID_RE.fullmatch(session_id)
            or event_name not in ALLOWED_EVENTS
        ):
            return False

        room_type = _clean_text(payload.get("room_type"), 24)
        if room_type not in ALLOWED_ROOM_TYPES:
            room_type = "none"
        device = _clean_text(payload.get("device"), 16)
        if device not in ALLOWED_DEVICES:
            device = "unknown"
        language = _clean_text(payload.get("language"), 8)
        if language not in ALLOWED_LANGUAGES:
            language = "other"
        source = _clean_text(payload.get("source"), 80)
        medium = _clean_text(payload.get("medium"), 80)
        campaign = _clean_text(payload.get("campaign"), 80)
        prefecture = normalize_prefecture(payload.get("prefecture"))
        properties = _safe_properties(payload.get("properties"))
        now = _utc_now_text()

        self._ensure_schema()
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(
                """
                INSERT INTO analytics_visitors (
                    analytics_id, first_seen, last_seen, first_source,
                    first_campaign, device, language
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(analytics_id) DO UPDATE SET
                    last_seen = excluded.last_seen,
                    device = excluded.device,
                    language = excluded.language
                """,
                (analytics_id, now, now, source, campaign, device, language),
            )
            connection.execute(
                """
                INSERT INTO analytics_sessions (
                    session_id, analytics_id, started_at, last_seen,
                    source, medium, campaign, device, language, prefecture,
                    event_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(session_id) DO UPDATE SET
                    last_seen = excluded.last_seen
                """,
                (
                    session_id,
                    analytics_id,
                    now,
                    now,
                    source,
                    medium,
                    campaign,
                    device,
                    language,
                    prefecture,
                ),
            )
            if event_name != "heartbeat":
                connection.execute(
                    """
                    INSERT INTO analytics_events (
                        analytics_id, session_id, occurred_at, event_name,
                        room_type, properties_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        analytics_id,
                        session_id,
                        now,
                        event_name,
                        room_type,
                        json.dumps(properties, ensure_ascii=True, separators=(",", ":")),
                    ),
                )
                connection.execute(
                    """
                    UPDATE analytics_sessions
                    SET event_count = event_count + 1,
                        ended_at = CASE WHEN ? = 'room_leave' THEN ? ELSE ended_at END
                    WHERE session_id = ?
                    """,
                    (event_name, now, session_id),
                )
            connection.commit()
        return True

    def delete_visitor(self, analytics_id: str) -> bool:
        """Delete one browser's analysis history when that browser opts out."""

        analytics_id = _clean_text(analytics_id, 80)
        if not ANALYTICS_ID_RE.fullmatch(analytics_id):
            return False
        self._ensure_schema()
        with closing(self._connect()) as connection:
            session_ids = [
                str(row["session_id"])
                for row in connection.execute(
                    "SELECT session_id FROM analytics_sessions WHERE analytics_id = ?",
                    (analytics_id,),
                ).fetchall()
            ]
            connection.execute(
                "DELETE FROM analytics_events WHERE analytics_id = ?",
                (analytics_id,),
            )
            if session_ids:
                connection.execute(
                    "DELETE FROM analytics_sessions WHERE analytics_id = ?",
                    (analytics_id,),
                )
            connection.execute(
                "DELETE FROM analytics_visitors WHERE analytics_id = ?",
                (analytics_id,),
            )
            connection.commit()
        return True

    def snapshot(self, days: int = 30, recent_limit: int = 80) -> dict[str, Any]:
        """Return an administrator-safe aggregate and recent anonymous journeys."""

        self._ensure_schema()
        days = max(1, min(3650, int(days)))
        recent_limit = max(1, min(250, int(recent_limit)))
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
        with closing(self._connect()) as connection:
            visitors = int(connection.execute(
                "SELECT COUNT(DISTINCT analytics_id) FROM analytics_events WHERE occurred_at >= ?",
                (since,),
            ).fetchone()[0])
            sessions = int(connection.execute(
                "SELECT COUNT(*) FROM analytics_sessions WHERE started_at >= ?",
                (since,),
            ).fetchone()[0])
            event_rows = connection.execute(
                """
                SELECT event_name, COUNT(*) AS total
                FROM analytics_events
                WHERE occurred_at >= ?
                GROUP BY event_name
                """,
                (since,),
            ).fetchall()
            event_counts = {str(row["event_name"]): int(row["total"]) for row in event_rows}
            room_rows = connection.execute(
                """
                SELECT room_type, COUNT(*) AS total
                FROM analytics_events
                WHERE occurred_at >= ? AND event_name = 'room_enter'
                GROUP BY room_type
                """,
                (since,),
            ).fetchall()
            room_entries = {
                str(row["room_type"]): int(row["total"])
                for row in room_rows
            }
            game_start_rows = connection.execute(
                """
                SELECT properties_json
                FROM analytics_events
                WHERE occurred_at >= ? AND event_name = 'game_started'
                """,
                (since,),
            ).fetchall()
            host_starts = 0
            joined_starts = 0
            pair_practice_games = 0
            for row in game_start_rows:
                try:
                    game_properties = json.loads(str(row["properties_json"] or "{}"))
                except json.JSONDecodeError:
                    game_properties = {}
                host_starts += int(game_properties.get("role") == "host")
                joined_starts += int(game_properties.get("role") == "joined")
                pair_practice_games += int(game_properties.get("pair_practice") is True)
            average_duration = float(connection.execute(
                """
                SELECT COALESCE(AVG((julianday(last_seen) - julianday(started_at)) * 86400.0), 0)
                FROM analytics_sessions WHERE started_at >= ?
                """,
                (since,),
            ).fetchone()[0] or 0.0)
            source_rows = connection.execute(
                """
                SELECT CASE WHEN source = '' THEN 'direct' ELSE source END AS source_name,
                       COUNT(DISTINCT analytics_id) AS visitors,
                       COUNT(*) AS sessions
                FROM analytics_sessions
                WHERE started_at >= ?
                GROUP BY source_name
                ORDER BY visitors DESC, sessions DESC
                LIMIT 20
                """,
                (since,),
            ).fetchall()
            region_rows = connection.execute(
                """
                SELECT CASE WHEN prefecture = '' THEN '不明' ELSE prefecture END AS prefecture,
                       COUNT(DISTINCT analytics_id) AS visitors,
                       COUNT(*) AS sessions
                FROM analytics_sessions
                WHERE started_at >= ?
                GROUP BY CASE WHEN prefecture = '' THEN '不明' ELSE prefecture END
                ORDER BY visitors DESC, sessions DESC
                """,
                (since,),
            ).fetchall()
            recent_sessions = connection.execute(
                """
                SELECT session_id, analytics_id, started_at, last_seen, ended_at,
                       source, campaign, device, language, event_count
                FROM analytics_sessions
                ORDER BY last_seen DESC
                LIMIT ?
                """,
                (recent_limit,),
            ).fetchall()
            recent: list[dict[str, Any]] = []
            for row in recent_sessions:
                event_list = connection.execute(
                    """
                    SELECT occurred_at, event_name, room_type, properties_json
                    FROM analytics_events
                    WHERE session_id = ?
                    ORDER BY occurred_at, id
                    """,
                    (row["session_id"],),
                ).fetchall()
                events = []
                for event in event_list:
                    try:
                        properties = json.loads(str(event["properties_json"] or "{}"))
                    except json.JSONDecodeError:
                        properties = {}
                    events.append({
                        "occurred_at": str(event["occurred_at"]),
                        "event": str(event["event_name"]),
                        "room_type": str(event["room_type"]),
                        "properties": properties,
                    })
                recent.append({
                    "session_id": str(row["session_id"]),
                    "analytics_id": str(row["analytics_id"]),
                    "started_at": str(row["started_at"]),
                    "last_seen": str(row["last_seen"]),
                    "ended_at": str(row["ended_at"] or ""),
                    "source": str(row["source"] or "direct"),
                    "campaign": str(row["campaign"] or ""),
                    "device": str(row["device"]),
                    "language": str(row["language"]),
                    "event_count": int(row["event_count"]),
                    "events": events,
                })

        completed = event_counts.get("game_completed", 0)
        started = event_counts.get("game_started", 0)
        return {
            "days": days,
            "visitors": visitors,
            "sessions": sessions,
            "average_duration_seconds": round(average_duration, 1),
            "game_started": started,
            "game_completed": completed,
            "completion_rate": round((completed / started * 100.0) if started else 0.0, 1),
            "host_game_starts": host_starts,
            "joined_game_starts": joined_starts,
            "pair_practice_games": pair_practice_games,
            "room_entries": room_entries,
            "event_counts": event_counts,
            "sources": [dict(row) for row in source_rows],
            "regions": [dict(row) for row in region_rows],
            "recent_sessions": recent,
            "database_bytes": self.path.stat().st_size if self.path.exists() else 0,
        }

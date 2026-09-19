"""Anonymous survey responses for the room notice survey."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


SURVEY_FILENAME = "goita-survey.sqlite3"
RESPONSE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]{16,80}$")
CAMPAIGN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def resolve_survey_path(environ: Mapping[str, str], *, local_fallback: Optional[Path] = None) -> Path:
    explicit = str(environ.get("GOITA_SURVEY_DB_PATH", "") or "").strip()
    if explicit:
        return Path(explicit)
    persistent = str(environ.get("GOITA_PERSISTENT_DATA_DIR", "") or "").strip()
    if persistent:
        return Path(persistent) / SURVEY_FILENAME
    return Path(local_fallback or SURVEY_FILENAME)


class SurveyStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._schema_lock = threading.Lock()
        self._schema_ready = False

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _ensure_schema(self):
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            with closing(self._connect()) as db:
                columns = {
                    str(row["name"])
                    for row in db.execute("PRAGMA table_info(survey_responses)").fetchall()
                }
                if columns and "campaign_id" not in columns:
                    db.execute("ALTER TABLE survey_responses RENAME TO survey_responses_legacy")
                db.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS survey_responses (
                        campaign_id TEXT NOT NULL,
                        campaign_title TEXT NOT NULL,
                        response_key TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        member_type TEXT NOT NULL,
                        device TEXT NOT NULL,
                        language TEXT NOT NULL,
                        answers_json TEXT NOT NULL,
                        PRIMARY KEY (campaign_id, response_key)
                    );
                    CREATE INDEX IF NOT EXISTS idx_survey_responses_updated
                    ON survey_responses(updated_at DESC);
                    """
                )
                if columns and "campaign_id" not in columns:
                    db.execute(
                        """
                        INSERT INTO survey_responses (
                            campaign_id, campaign_title, response_key, created_at,
                            updated_at, kind, member_type, device, language, answers_json
                        )
                        SELECT 'survey-1', '第1回 そろうごいた改善アンケート',
                               response_key, created_at, updated_at, kind, member_type,
                               device, language, answers_json
                        FROM survey_responses_legacy
                        """
                    )
                    db.execute("DROP TABLE survey_responses_legacy")
                    db.execute(
                        "CREATE INDEX IF NOT EXISTS idx_survey_responses_updated "
                        "ON survey_responses(updated_at DESC)"
                    )
                db.commit()
            self._schema_ready = True

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "campaign_id": str(row["campaign_id"]),
            "campaign_title": str(row["campaign_title"]),
            "response_key": str(row["response_key"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "kind": str(row["kind"]),
            "member_type": str(row["member_type"]),
            "device": str(row["device"]),
            "language": str(row["language"]),
            "answers": json.loads(str(row["answers_json"])),
        }

    def save(self, *, campaign_id: str, campaign_title: str, response_key: str,
             kind: str, member_type: str, device: str,
             language: str, answers: Mapping[str, Any]) -> dict[str, Any]:
        if not CAMPAIGN_ID_RE.fullmatch(campaign_id):
            raise ValueError("invalid campaign id")
        if not RESPONSE_KEY_RE.fullmatch(response_key):
            raise ValueError("invalid response key")
        self._ensure_schema()
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        encoded = json.dumps(dict(answers), ensure_ascii=False, separators=(",", ":"))
        with closing(self._connect()) as db:
            existing = db.execute(
                "SELECT kind, created_at FROM survey_responses WHERE campaign_id = ? AND response_key = ?",
                (campaign_id, response_key),
            ).fetchone()
            # A later quick response must not replace a completed detailed response.
            if existing is not None and existing["kind"] == "detailed" and kind == "quick":
                row = db.execute(
                    "SELECT * FROM survey_responses WHERE campaign_id = ? AND response_key = ?",
                    (campaign_id, response_key),
                ).fetchone()
                return self._decode(row)
            created_at = str(existing["created_at"]) if existing else now
            db.execute(
                """
                INSERT INTO survey_responses (
                    campaign_id, campaign_title, response_key, created_at, updated_at, kind, member_type,
                    device, language, answers_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(campaign_id, response_key) DO UPDATE SET
                    campaign_title = excluded.campaign_title,
                    updated_at = excluded.updated_at,
                    kind = excluded.kind,
                    member_type = excluded.member_type,
                    device = excluded.device,
                    language = excluded.language,
                    answers_json = excluded.answers_json
                """,
                (campaign_id, campaign_title[:80], response_key, created_at, now,
                 kind, member_type, device, language, encoded),
            )
            db.commit()
            row = db.execute(
                "SELECT * FROM survey_responses WHERE campaign_id = ? AND response_key = ?",
                (campaign_id, response_key),
            ).fetchone()
        return self._decode(row)

    def snapshot(self, *, campaign_id: str = "", limit: int = 100, offset: int = 0) -> dict[str, Any]:
        self._ensure_schema()
        safe_limit = max(1, min(int(limit), 200))
        safe_offset = max(0, int(offset))
        with closing(self._connect()) as db:
            campaigns = db.execute(
                """SELECT campaign_id, MAX(campaign_title) AS campaign_title,
                          COUNT(*) AS response_count, MAX(updated_at) AS last_response_at
                   FROM survey_responses GROUP BY campaign_id ORDER BY last_response_at DESC"""
            ).fetchall()
            selected = campaign_id or (str(campaigns[0]["campaign_id"]) if campaigns else "")
            if selected:
                rows = db.execute(
                    "SELECT * FROM survey_responses WHERE campaign_id = ? ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                    (selected, safe_limit, safe_offset),
                ).fetchall()
                all_rows = db.execute(
                    "SELECT * FROM survey_responses WHERE campaign_id = ?", (selected,)
                ).fetchall()
            else:
                rows = []
                all_rows = []
        decoded = [self._decode(row) for row in all_rows]
        kind_counts = Counter(item["kind"] for item in decoded)
        member_counts = Counter(item["member_type"] for item in decoded)
        answer_counts: dict[str, Counter] = {}
        for item in decoded:
            for key, value in item["answers"].items():
                if key in {"free_text", "ai_other", "improvement_other"}:
                    continue
                if isinstance(value, list):
                    values = value
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        answer_counts.setdefault(f"{key}.{subkey}", Counter())[str(subvalue)] += 1
                    continue
                elif value in (None, ""):
                    continue
                else:
                    values = [value]
                for selected in values:
                    answer_counts.setdefault(key, Counter())[str(selected)] += 1
        return {
            "summary": {
                "total": len(decoded),
                "quick": kind_counts["quick"],
                "detailed": kind_counts["detailed"],
                "member_types": dict(member_counts),
                "answer_counts": {key: dict(counts) for key, counts in answer_counts.items()},
            },
            "responses": [
                {key: value for key, value in self._decode(row).items() if key != "response_key"}
                for row in rows
            ],
            "limit": safe_limit,
            "offset": safe_offset,
            "total": len(decoded),
            "campaign_id": selected,
            "campaigns": [dict(row) for row in campaigns],
        }

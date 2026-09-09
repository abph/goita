"""Persistent per-round audit decisions for the private score-attack archive."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


AUDIT_STATUSES = frozenset({"eligible", "review", "excluded", "invalid", "out_of_range"})
MANUAL_STATUSES = frozenset({"eligible", "review", "excluded"})


def audit_path(base_dir: Path) -> Path:
    persistent = os.environ.get("GOITA_PERSISTENT_DATA_DIR", "").strip()
    if os.environ.get("RENDER") and not persistent:
        raise ValueError("棋譜監査データの保存にはGOITA_PERSISTENT_DATA_DIRの永続保存先を設定してください。")
    directory = Path(persistent) / "private-kifu" if persistent else base_dir / "private_data"
    path = (directory / "score-attack-audit.sqlite3").resolve()
    if path.is_relative_to((base_dir / "frontend").resolve()):
        raise ValueError("棋譜監査データの保存先に公開ディレクトリは指定できません。")
    return path


def candidate_id(match: Dict[str, Any], round_obj: Dict[str, Any], position: int) -> str:
    """Return a stable id for one round, changing when its source content changes."""
    value = {
        "match_id": str(match.get("id") or ""),
        "round_position": int(position),
        "round": round_obj,
    }
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ScoreAttackAuditStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS score_attack_audit (
                    candidate_id TEXT PRIMARY KEY,
                    source_revision TEXT NOT NULL DEFAULT '',
                    default_status TEXT NOT NULL,
                    status TEXT NOT NULL,
                    match_id TEXT NOT NULL,
                    round_index INTEGER,
                    score_ac INTEGER,
                    score_bd INTEGER,
                    winner TEXT,
                    gained_score INTEGER,
                    reasons_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    updated_by TEXT NOT NULL DEFAULT 'system',
                    decision_note TEXT NOT NULL DEFAULT ''
                )
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(score_attack_audit)")}
            if "source_revision" not in columns:
                connection.execute(
                    "ALTER TABLE score_attack_audit ADD COLUMN source_revision TEXT NOT NULL DEFAULT ''"
                )
            if "decision_note" not in columns:
                connection.execute(
                    "ALTER TABLE score_attack_audit ADD COLUMN decision_note TEXT NOT NULL DEFAULT ''"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS score_attack_audit_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    changed_at TEXT NOT NULL,
                    changed_by TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS score_attack_audit_scan (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    revision TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'idle',
                    processed INTEGER NOT NULL DEFAULT 0,
                    total INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _decode(row: sqlite3.Row) -> Dict[str, Any]:
        value = dict(row)
        for field in ("reasons_json", "metadata_json"):
            try:
                value[field[:-5]] = json.loads(value.pop(field))
            except (TypeError, ValueError, json.JSONDecodeError):
                value[field[:-5]] = [] if field == "reasons_json" else {}
        return value

    def sync(self, record: Dict[str, Any], source_revision: str = "") -> Dict[str, Any]:
        """Insert/update an automatic scan while preserving a manual decision."""
        candidate = str(record["candidate_id"])
        reasons = list(record.get("reasons") or [])
        metadata = dict(record.get("metadata") or {})
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT status, updated_by FROM score_attack_audit WHERE candidate_id = ?",
                (candidate,),
            ).fetchone()
            if existing is None:
                status = str(record.get("default_status") or "invalid")
                updated_by = "system"
            else:
                status = str(existing["status"])
                updated_by = str(existing["updated_by"] or "system")
            connection.execute(
                """
                INSERT INTO score_attack_audit
                    (candidate_id, source_revision, default_status, status, match_id, round_index,
                     score_ac, score_bd, winner, gained_score, reasons_json,
                     metadata_json, updated_at, updated_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                    source_revision = excluded.source_revision,
                    default_status = excluded.default_status,
                    match_id = excluded.match_id,
                    round_index = excluded.round_index,
                    score_ac = excluded.score_ac,
                    score_bd = excluded.score_bd,
                    winner = excluded.winner,
                    gained_score = excluded.gained_score,
                    reasons_json = excluded.reasons_json,
                    metadata_json = excluded.metadata_json
                """,
                (
                    candidate,
                    str(source_revision or ""),
                    str(record.get("default_status") or "invalid"),
                    status,
                    str(record.get("match_id") or ""),
                    record.get("round_index"),
                    record.get("score_ac"),
                    record.get("score_bd"),
                    record.get("winner"),
                    record.get("gained_score"),
                    json.dumps(reasons, ensure_ascii=False),
                    json.dumps(metadata, ensure_ascii=False),
                    _now(),
                    updated_by,
                ),
            )
            row = connection.execute(
                "SELECT * FROM score_attack_audit WHERE candidate_id = ?", (candidate,)
            ).fetchone()
        return self._decode(row)

    def list(
        self, status: Optional[str] = None, source_revision: Optional[str] = None
    ) -> list[Dict[str, Any]]:
        query = "SELECT * FROM score_attack_audit"
        clauses = []
        args: tuple[Any, ...] = ()
        if source_revision is not None:
            clauses.append("source_revision = ?")
            args += (str(source_revision),)
        if status and status != "all":
            clauses.append("status = ?")
            args += (status,)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY CASE status WHEN 'review' THEN 0 WHEN 'invalid' THEN 1 WHEN 'out_of_range' THEN 2 WHEN 'eligible' THEN 3 ELSE 4 END, match_id, round_index, candidate_id"
        with self._connect() as connection:
            rows = connection.execute(query, args).fetchall()
        return [self._decode(row) for row in rows]

    def summary(self, source_revision: Optional[str] = None) -> Dict[str, int]:
        clauses = []
        args: tuple[Any, ...] = ()
        if source_revision is not None:
            clauses.append("source_revision = ?")
            args += (str(source_revision),)
        query = "SELECT status, COUNT(*) AS count FROM score_attack_audit"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " GROUP BY status"
        result = {status: 0 for status in AUDIT_STATUSES}
        with self._connect() as connection:
            for row in connection.execute(query, args):
                if row["status"] in result:
                    result[row["status"]] = int(row["count"])
        return result

    def touch_source_revision(self, candidate: str, source_revision: str) -> Optional[Dict[str, Any]]:
        """Mark an unchanged round as present in the latest archive revision."""
        with self._connect() as connection:
            connection.execute(
                "UPDATE score_attack_audit SET source_revision = ? WHERE candidate_id = ?",
                (str(source_revision or ""), str(candidate)),
            )
            row = connection.execute(
                "SELECT * FROM score_attack_audit WHERE candidate_id = ?", (str(candidate),)
            ).fetchone()
        return self._decode(row) if row is not None else None

    def scan_state(self) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT revision, status, processed, total, error, updated_at FROM score_attack_audit_scan WHERE id = 1"
            ).fetchone()
        if row is None:
            return {"revision": "", "status": "idle", "processed": 0, "total": 0, "error": "", "updated_at": ""}
        return dict(row)

    def begin_scan(self, revision: str, total: int) -> Dict[str, Any]:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO score_attack_audit_scan
                    (id, revision, status, processed, total, error, updated_at)
                VALUES (1, ?, 'running', 0, ?, '', ?)
                ON CONFLICT(id) DO UPDATE SET
                    revision = excluded.revision,
                    status = excluded.status,
                    processed = excluded.processed,
                    total = excluded.total,
                    error = excluded.error,
                    updated_at = excluded.updated_at
                """,
                (str(revision), max(0, int(total)), _now()),
            )
        return self.scan_state()

    def update_scan(self, revision: str, processed: int, total: int) -> Dict[str, Any]:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE score_attack_audit_scan
                   SET processed = ?, total = ?, updated_at = ?
                 WHERE id = 1 AND revision = ?
                """,
                (max(0, int(processed)), max(0, int(total)), _now(), str(revision)),
            )
        return self.scan_state()

    def finish_scan(self, revision: str, total: int, error: str = "") -> Dict[str, Any]:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE score_attack_audit_scan
                   SET status = ?, processed = ?, total = ?, error = ?, updated_at = ?
                 WHERE id = 1 AND revision = ?
                """,
                (
                    "error" if error else "complete",
                    max(0, int(total)),
                    max(0, int(total)),
                    str(error or "")[:500],
                    _now(),
                    str(revision),
                ),
            )
        return self.scan_state()

    def get(self, candidate: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM score_attack_audit WHERE candidate_id = ?", (candidate,)
            ).fetchone()
        return self._decode(row) if row is not None else None

    def set_status(
        self, candidate: str, status: str, updated_by: str = "admin", note: str = ""
    ) -> Optional[Dict[str, Any]]:
        if status not in MANUAL_STATUSES:
            raise ValueError("監査状態が正しくありません。")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT default_status FROM score_attack_audit WHERE candidate_id = ?", (candidate,)
            ).fetchone()
            if row is None:
                return None
            if row["default_status"] in {"invalid", "out_of_range"} and status == "eligible":
                raise ValueError("形式不正または開始点数の対象外の棋譜は採用できません。")
            connection.execute(
                "UPDATE score_attack_audit SET status = ?, updated_at = ?, updated_by = ?, decision_note = ? WHERE candidate_id = ?",
                (status, _now(), str(updated_by or "admin")[:80], str(note or "")[:500], candidate),
            )
            connection.execute(
                "INSERT INTO score_attack_audit_history (candidate_id, status, note, changed_at, changed_by) VALUES (?, ?, ?, ?, ?)",
                (candidate, status, str(note or "")[:500], _now(), str(updated_by or "admin")[:80]),
            )
            updated = connection.execute(
                "SELECT * FROM score_attack_audit WHERE candidate_id = ?", (candidate,)
            ).fetchone()
        return self._decode(updated)

    def history(self, candidate: str) -> list[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, note, changed_at, changed_by FROM score_attack_audit_history WHERE candidate_id = ? ORDER BY id DESC",
                (candidate,),
            ).fetchall()
        return [dict(row) for row in rows]


def audit_store_for(base_dir: Path) -> ScoreAttackAuditStore:
    return ScoreAttackAuditStore(audit_path(base_dir))

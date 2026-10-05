"""Persistent administrator storage for human-reviewed AI decision cases."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


MAX_REPORT_BYTES = 25 * 1024 * 1024
MAX_CASES = 5000
FILTERS = frozenset({
    "all",
    "unreviewed",
    "different",
    "benchmark",
    "needs_recheck",
    "avoid",
})
ACTION_RATINGS = frozenset({"best", "acceptable", "avoid", "hold"})
THINKING_STYLES = frozenset({"", "safe", "risk", "common", "undecided"})
FINAL_RATINGS = frozenset({
    "",
    "理解できる",
    "少し理解できる",
    "なんともいえない",
    "あまり理解できない",
    "理解できない",
})
MOVE_QUALITIES = frozenset({"", "良い手", "迷う手", "ミスだった", "判断できない"})
PURPOSES = frozenset({
    "",
    "自分の上がり",
    "敵方に王・玉を使わせる",
    "相方の上がりを助ける",
    "し攻め・連続攻め",
    "受け駒・攻め駒の温存",
    "相方への情報",
    "その他",
    "覚えていない",
})
EXPLANATION_RATINGS = frozenset({"", "近い", "一部近い", "違う", "判断できない"})
REASON_CATEGORIES = frozenset({
    "受ける・パスする",
    "攻め順・伏せ駒",
    "しの枚数推定",
    "し攻め・しの差し込み",
    "相方との連携",
    "王か玉の使い方",
    "確定上がり",
    "高得点を狙う",
    "相方への情報",
    "その他",
})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ai_benchmark_path(base_dir: Path) -> Path:
    persistent = os.environ.get("GOITA_PERSISTENT_DATA_DIR", "").strip()
    if os.environ.get("RENDER") and not persistent:
        raise ValueError(
            "AI判断ベンチマークの保存には"
            "GOITA_PERSISTENT_DATA_DIRの永続保存先を設定してください。"
        )
    directory = Path(persistent) / "private-kifu" if persistent else base_dir / "private_data"
    path = (directory / "ai-decision-benchmark.sqlite3").resolve()
    if path.is_relative_to((base_dir / "frontend").resolve()):
        raise ValueError("AI判断ベンチマークは公開ディレクトリに保存できません。")
    return path


def parse_report(raw: bytes) -> Dict[str, Any]:
    if len(raw) > MAX_REPORT_BYTES:
        raise ValueError("確認結果のJSONは25MB以下にしてください。")
    try:
        report = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ValueError("確認結果のJSON形式が正しくありません。") from error
    if not isinstance(report, dict) or not isinstance(report.get("cases"), list):
        raise ValueError("棋譜・AI理解度確認のJSONを選択してください。")
    cases = report["cases"]
    if not cases or len(cases) > MAX_CASES:
        raise ValueError(f"局面は1〜{MAX_CASES}件で登録してください。")
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("局面データの形式が正しくありません。")
        case_id = str(case.get("id", "")).strip()
        if not case_id or len(case_id) > 200 or case_id in seen:
            raise ValueError("局面IDが空、重複、または長すぎます。")
        if not isinstance(case.get("position"), dict):
            raise ValueError(f"盤面情報がありません: {case_id}")
        if not isinstance(case.get("recorded_route"), list) or not isinstance(
            case.get("ai_route"), list
        ):
            raise ValueError(f"判断経路の形式が正しくありません: {case_id}")
        seen.add(case_id)
    return report


def _route_key(route: object) -> str:
    return json.dumps(route, ensure_ascii=False, separators=(",", ":"))


def _candidate_route_keys(case: Mapping[str, object]) -> set[str]:
    result = set()
    for item in case.get("candidate_routes", []) or []:
        route = item.get("route") if isinstance(item, Mapping) else item
        if isinstance(route, list) and route:
            result.add(_route_key(route))
    for field in ("recorded_route", "ai_route"):
        route = case.get(field)
        if isinstance(route, list) and route:
            result.add(_route_key(route))
    for item in case.get("candidate_scores", []) or []:
        action = item.get("action") if isinstance(item, Mapping) else None
        if isinstance(action, list) and len(action) == 3:
            result.add(_route_key([action]))
    return result


def _text(value: object, limit: int) -> str:
    return str(value or "")[:limit]


def normalize_review(
    value: object,
    *,
    case: Optional[Mapping[str, object]] = None,
    validate_completion: bool = False,
) -> Dict[str, Any]:
    source = dict(value or {}) if isinstance(value, Mapping) else {}
    review: Dict[str, Any] = {
        "final_rating": _text(source.get("final_rating"), 40),
        "recorded_move_quality": _text(source.get("recorded_move_quality"), 40),
        "purpose": _text(source.get("purpose"), 80),
        "ai_explanation_close": _text(source.get("ai_explanation_close"), 40),
        "note": _text(source.get("note"), 4000),
        "benchmark_reason": _text(source.get("benchmark_reason"), 4000),
        "benchmark_learning_point": _text(
            source.get("benchmark_learning_point"), 4000
        ),
        "benchmark_thinking_style": _text(
            source.get("benchmark_thinking_style"), 20
        ),
        "benchmark_registered_at": _text(
            source.get("benchmark_registered_at"), 80
        ),
        "benchmark_action_ratings": {},
        "benchmark_reason_categories": [],
        "benchmark_completed": bool(source.get("benchmark_completed", False)),
        "review_completed_after_update": bool(
            source.get("review_completed_after_update", False)
        ),
    }
    allowed_values = (
        ("final_rating", FINAL_RATINGS),
        ("recorded_move_quality", MOVE_QUALITIES),
        ("purpose", PURPOSES),
        ("ai_explanation_close", EXPLANATION_RATINGS),
        ("benchmark_thinking_style", THINKING_STYLES),
    )
    for field, allowed in allowed_values:
        if review[field] not in allowed:
            raise ValueError(f"{field}の値が正しくありません。")

    allowed_routes = _candidate_route_keys(case or {}) if case is not None else None
    raw_ratings = source.get("benchmark_action_ratings", {})
    if not isinstance(raw_ratings, Mapping):
        raise ValueError("候補評価の形式が正しくありません。")
    for raw_key, raw_rating in raw_ratings.items():
        key = str(raw_key)
        rating = str(raw_rating)
        if len(key) > 1000 or rating not in ACTION_RATINGS:
            raise ValueError("候補評価の値が正しくありません。")
        if allowed_routes is not None and key not in allowed_routes:
            raise ValueError("現在の局面にない候補が含まれています。")
        review["benchmark_action_ratings"][key] = rating

    categories = source.get("benchmark_reason_categories", [])
    if not isinstance(categories, Sequence) or isinstance(categories, (str, bytes)):
        raise ValueError("判断理由の分類が正しくありません。")
    for raw_category in categories:
        category = str(raw_category)
        if category not in REASON_CATEGORIES:
            raise ValueError("判断理由の分類が正しくありません。")
        if category not in review["benchmark_reason_categories"]:
            review["benchmark_reason_categories"].append(category)

    if review["benchmark_completed"]:
        if validate_completion and "best" not in review["benchmark_action_ratings"].values():
            raise ValueError("最善手を1つ以上選んでください。")
        if validate_completion and not review["benchmark_reason_categories"]:
            raise ValueError("判断理由の分類を1つ以上選んでください。")
        if validate_completion and not review["benchmark_thinking_style"]:
            raise ValueError("判断の考え方を選んでください。")
        if validate_completion and not review["benchmark_reason"].strip():
            raise ValueError("判断理由を入力してください。")
        if not review["benchmark_registered_at"]:
            review["benchmark_registered_at"] = _now()
    return review


def _review_has_content(review: Mapping[str, object]) -> bool:
    return any(
        bool(value)
        for key, value in review.items()
        if key not in {"benchmark_completed", "review_completed_after_update"}
    ) or bool(review.get("benchmark_completed")) or bool(
        review.get("review_completed_after_update")
    )


class AIBenchmarkStore:
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
                CREATE TABLE IF NOT EXISTS ai_benchmark_dataset (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    metadata_json TEXT NOT NULL,
                    source_name TEXT NOT NULL DEFAULT '',
                    imported_at TEXT NOT NULL,
                    case_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_benchmark_case (
                    case_id TEXT PRIMARY KEY,
                    position_index INTEGER NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    same_route INTEGER NOT NULL DEFAULT 0,
                    needs_recheck INTEGER NOT NULL DEFAULT 0,
                    case_json TEXT NOT NULL,
                    review_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_ai_benchmark_active_position "
                "ON ai_benchmark_case(active, position_index)"
            )

    @staticmethod
    def _loads(value: object, fallback: object) -> Any:
        try:
            return json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            return fallback

    def import_report(self, report: Mapping[str, object], source_name: str = "") -> Dict[str, Any]:
        cases = list(report.get("cases", []) or [])
        metadata = dict(report)
        metadata.pop("cases", None)
        now = _now()
        imported_reviews = 0
        preserved_reviews = 0
        with self._connect() as connection:
            connection.execute("UPDATE ai_benchmark_case SET active = 0")
            for position, raw_case in enumerate(cases):
                case = dict(raw_case)
                case_id = str(case["id"])
                imported = normalize_review(case.pop("review", {}), case=case)
                existing_row = connection.execute(
                    "SELECT review_json FROM ai_benchmark_case WHERE case_id = ?",
                    (case_id,),
                ).fetchone()
                existing = normalize_review(
                    self._loads(existing_row["review_json"], {})
                    if existing_row is not None else {},
                    case=case,
                )
                if _review_has_content(imported):
                    review = imported
                    imported_reviews += 1
                else:
                    review = existing
                    preserved_reviews += int(_review_has_content(existing))
                comparison = dict(case.get("comparison", {}) or {})
                same_route = case.get("recorded_route") == case.get("ai_route")
                completed = bool(review.get("final_rating")) or bool(
                    review.get("review_completed_after_update")
                )
                needs_recheck = bool(comparison.get("needs_recheck")) and not completed
                connection.execute(
                    """
                    INSERT INTO ai_benchmark_case
                        (case_id, position_index, active, same_route,
                         needs_recheck, case_json, review_json, updated_at)
                    VALUES (?, ?, 1, ?, ?, ?, ?, ?)
                    ON CONFLICT(case_id) DO UPDATE SET
                        position_index = excluded.position_index,
                        active = 1,
                        same_route = excluded.same_route,
                        needs_recheck = excluded.needs_recheck,
                        case_json = excluded.case_json,
                        review_json = excluded.review_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        case_id,
                        position,
                        int(same_route),
                        int(needs_recheck),
                        json.dumps(case, ensure_ascii=False),
                        json.dumps(review, ensure_ascii=False),
                        now,
                    ),
                )
            connection.execute(
                """
                INSERT INTO ai_benchmark_dataset
                    (id, metadata_json, source_name, imported_at, case_count)
                VALUES (1, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    metadata_json = excluded.metadata_json,
                    source_name = excluded.source_name,
                    imported_at = excluded.imported_at,
                    case_count = excluded.case_count
                """,
                (
                    json.dumps(metadata, ensure_ascii=False),
                    str(source_name or "")[:260],
                    now,
                    len(cases),
                ),
            )
        return {
            "ok": True,
            "case_count": len(cases),
            "imported_reviews": imported_reviews,
            "preserved_reviews": preserved_reviews,
            **self.status(),
        }

    def status(self) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT source_name, imported_at, case_count FROM ai_benchmark_dataset WHERE id = 1"
            ).fetchone()
        if row is None:
            return {"registered": False, "source_name": "", "imported_at": "", "case_count": 0}
        return {"registered": True, **dict(row)}

    @staticmethod
    def _list_item(row: sqlite3.Row) -> Dict[str, Any]:
        case = AIBenchmarkStore._loads(row["case_json"], {})
        review = AIBenchmarkStore._loads(row["review_json"], {})
        comparison = dict(case.get("comparison", {}) or {})
        ratings = dict(review.get("benchmark_action_ratings", {}) or {})
        return {
            "id": row["case_id"],
            "position_index": int(row["position_index"]),
            "stratum": str(case.get("stratum", "")),
            "seat": str(case.get("seat", "")),
            "recorded_route": case.get("recorded_route", []),
            "ai_route": case.get("ai_route", []),
            "provisional_rating": str(case.get("provisional_rating", "")),
            "final_rating": str(review.get("final_rating", "")),
            "benchmark_completed": bool(review.get("benchmark_completed")),
            "review_completed_after_update": bool(
                review.get("review_completed_after_update")
            ),
            "benchmark_thinking_style": str(
                review.get("benchmark_thinking_style", "")
            ),
            "has_avoid": "avoid" in ratings.values(),
            "same_route": bool(row["same_route"]),
            "needs_recheck": bool(row["needs_recheck"]),
            "decision_changed": bool(comparison.get("decision_changed")),
            "updated_at": str(row["updated_at"]),
        }

    def _where(self, filter_name: str) -> tuple[str, tuple[Any, ...]]:
        if filter_name not in FILTERS:
            raise ValueError("表示条件が正しくありません。")
        clause = "active = 1"
        if filter_name == "unreviewed":
            clause += (
                " AND COALESCE(json_extract(review_json, '$.final_rating'), '') = ''"
                " AND COALESCE(json_extract(review_json, "
                "'$.review_completed_after_update'), 0) = 0"
            )
        elif filter_name == "different":
            clause += " AND same_route = 0"
        elif filter_name == "benchmark":
            clause += " AND json_extract(review_json, '$.benchmark_completed') = 1"
        elif filter_name == "needs_recheck":
            clause += " AND needs_recheck = 1"
        elif filter_name == "avoid":
            clause += " AND review_json LIKE '%\"avoid\"%'"
        return clause, ()

    def list(self, filter_name: str, limit: int, offset: int) -> Dict[str, Any]:
        clause, args = self._where(filter_name)
        with self._connect() as connection:
            total = int(connection.execute(
                f"SELECT COUNT(*) FROM ai_benchmark_case WHERE {clause}", args
            ).fetchone()[0])
            rows = connection.execute(
                f"SELECT * FROM ai_benchmark_case WHERE {clause} "
                "ORDER BY position_index LIMIT ? OFFSET ?",
                (*args, int(limit), int(offset)),
            ).fetchall()
        return {
            "items": [self._list_item(row) for row in rows],
            "total": total,
            "shown": len(rows),
            "offset": int(offset),
            "limit": int(limit),
            "summary": self.summary(),
            **self.status(),
        }

    def summary(self) -> Dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT same_route, needs_recheck, review_json "
                "FROM ai_benchmark_case WHERE active = 1"
            ).fetchall()
        result = {
            "total": len(rows),
            "unreviewed": 0,
            "different": 0,
            "benchmark": 0,
            "needs_recheck": 0,
            "avoid": 0,
        }
        for row in rows:
            review = self._loads(row["review_json"], {})
            ratings = dict(review.get("benchmark_action_ratings", {}) or {})
            result["unreviewed"] += int(
                not review.get("final_rating")
                and not review.get("review_completed_after_update")
            )
            result["different"] += int(not row["same_route"])
            result["benchmark"] += int(bool(review.get("benchmark_completed")))
            result["needs_recheck"] += int(bool(row["needs_recheck"]))
            result["avoid"] += int("avoid" in ratings.values())
        return result

    def get(self, case_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM ai_benchmark_case WHERE case_id = ? AND active = 1",
                (str(case_id),),
            ).fetchone()
        if row is None:
            return None
        case = self._loads(row["case_json"], {})
        case["review"] = self._loads(row["review_json"], {})
        return case

    def update_review(self, case_id: str, value: object) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT case_json FROM ai_benchmark_case WHERE case_id = ? AND active = 1",
                (str(case_id),),
            ).fetchone()
            if row is None:
                return None
            case = self._loads(row["case_json"], {})
            review = normalize_review(
                value,
                case=case,
                validate_completion=True,
            )
            comparison = dict(case.get("comparison", {}) or {})
            completed = bool(review.get("final_rating")) or bool(
                review.get("review_completed_after_update")
            )
            needs_recheck = bool(comparison.get("needs_recheck")) and not completed
            connection.execute(
                "UPDATE ai_benchmark_case SET review_json = ?, needs_recheck = ?, "
                "updated_at = ? WHERE case_id = ?",
                (
                    json.dumps(review, ensure_ascii=False),
                    int(needs_recheck),
                    _now(),
                    str(case_id),
                ),
            )
        case["review"] = review
        return case

    def export_report(self) -> Dict[str, Any]:
        with self._connect() as connection:
            dataset = connection.execute(
                "SELECT metadata_json FROM ai_benchmark_dataset WHERE id = 1"
            ).fetchone()
            rows = connection.execute(
                "SELECT case_json, review_json FROM ai_benchmark_case "
                "WHERE active = 1 ORDER BY position_index"
            ).fetchall()
        if dataset is None:
            raise ValueError("AI判断ベンチマークが登録されていません。")
        report = self._loads(dataset["metadata_json"], {})
        report["cases"] = []
        action_counts = {rating: 0 for rating in ACTION_RATINGS}
        registered = 0
        for row in rows:
            case = self._loads(row["case_json"], {})
            review = self._loads(row["review_json"], {})
            case["review"] = review
            report["cases"].append(case)
            if review.get("benchmark_completed"):
                registered += 1
                for rating in dict(
                    review.get("benchmark_action_ratings", {}) or {}
                ).values():
                    if rating in action_counts:
                        action_counts[rating] += 1
        report["benchmark_summary"] = {
            "schema_version": 1,
            "registered_cases": registered,
            "action_ratings": action_counts,
        }
        return report


__all__ = [
    "AIBenchmarkStore",
    "FILTERS",
    "MAX_REPORT_BYTES",
    "REASON_CATEGORIES",
    "ai_benchmark_path",
    "normalize_review",
    "parse_report",
]

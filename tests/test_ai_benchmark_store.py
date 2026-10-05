import json
from pathlib import Path

import pytest

from backend.ai_benchmark_store import AIBenchmarkStore, ai_benchmark_path, parse_report


def _case(case_id: str = "case-1", *, same: bool = False) -> dict:
    recorded = [["pass", None, None]]
    ai_route = recorded if same else [["receive", "1", None], ["attack", None, "4"]]
    return {
        "id": case_id,
        "stratum": "opening_response",
        "seat": "A",
        "ai_label": "強化中AI2",
        "position": {
            "phase": "response",
            "hand": ["1", "1", "4"],
            "current_attack": "1",
            "public_history": [],
        },
        "recorded_route": recorded,
        "ai_route": ai_route,
        "candidate_routes": [
            {"route": recorded, "root_score": 10},
            {"route": ai_route, "root_score": 20},
        ],
        "candidate_scores": [],
        "comparison": {"needs_recheck": not same, "decision_changed": not same},
        "review": {},
    }


def _report() -> dict:
    return {
        "schema_version": 3,
        "purpose": "private_player_ai_understanding_review",
        "player": "1222",
        "cases": [_case(), _case("case-2", same=True)],
    }


def _completed_review(case: dict) -> dict:
    return {
        "final_rating": "少し理解できる",
        "recorded_move_quality": "良い手",
        "purpose": "相方への情報",
        "ai_explanation_close": "一部近い",
        "note": "確認メモ",
        "review_completed_after_update": True,
        "benchmark_action_ratings": {
            json.dumps(case["recorded_route"], ensure_ascii=False, separators=(",", ":")): "best",
            json.dumps(case["ai_route"], ensure_ascii=False, separators=(",", ":")): "avoid",
        },
        "benchmark_reason_categories": ["相方への情報"],
        "benchmark_thinking_style": "safe",
        "benchmark_reason": "相方に誤った情報を伝えないため。",
        "benchmark_learning_point": "単独の局面に限定しない。",
        "benchmark_completed": True,
    }


def test_store_import_review_filter_export_and_preserve(tmp_path: Path) -> None:
    store = AIBenchmarkStore(tmp_path / "benchmark.sqlite3")
    report = _report()
    imported = store.import_report(report, "review.json")
    assert imported["case_count"] == 2
    assert store.list("unreviewed", 50, 0)["total"] == 2
    assert store.list("different", 50, 0)["total"] == 1

    review = _completed_review(report["cases"][0])
    updated = store.update_review("case-1", review)
    assert updated is not None
    assert updated["review"]["benchmark_completed"] is True
    assert store.list("benchmark", 50, 0)["total"] == 1
    assert store.list("avoid", 50, 0)["total"] == 1
    assert store.list("needs_recheck", 50, 0)["total"] == 0
    assert store.list("unreviewed", 50, 0)["total"] == 1

    exported = store.export_report()
    assert exported["cases"][0]["review"]["final_rating"] == "少し理解できる"
    assert exported["benchmark_summary"]["registered_cases"] == 1

    # A newly generated report has no answers. Re-importing it must keep the
    # review that was saved from the administrator page.
    preserved = store.import_report(_report(), "regenerated.json")
    assert preserved["preserved_reviews"] == 1
    assert store.get("case-1")["review"]["note"] == "確認メモ"


def test_completed_benchmark_requires_a_reason_and_best_move(tmp_path: Path) -> None:
    store = AIBenchmarkStore(tmp_path / "benchmark.sqlite3")
    report = _report()
    store.import_report(report)
    review = _completed_review(report["cases"][0])
    review["benchmark_reason"] = ""
    with pytest.raises(ValueError, match="判断理由"):
        store.update_review("case-1", review)
    review = _completed_review(report["cases"][0])
    review["benchmark_action_ratings"] = {}
    with pytest.raises(ValueError, match="最善手"):
        store.update_review("case-1", review)


def test_report_validation_and_persistent_path(monkeypatch, tmp_path: Path) -> None:
    assert parse_report(json.dumps(_report()).encode("utf-8"))["player"] == "1222"
    with pytest.raises(ValueError):
        parse_report(b"{}")
    monkeypatch.setenv("GOITA_PERSISTENT_DATA_DIR", str(tmp_path / "persistent"))
    path = ai_benchmark_path(tmp_path)
    assert path == (tmp_path / "persistent" / "private-kifu" / "ai-decision-benchmark.sqlite3").resolve()

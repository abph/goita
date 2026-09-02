from __future__ import annotations

from goita_ai2.human_kifu_targeted_recheck import (
    _comparison_summary,
    _direct_pattern_candidate,
    _normalized_pattern_ids,
    _round_robin_candidates,
    _summarize_direct_comparisons,
    prioritized_pattern_ids,
)


def test_normalizes_explicit_pattern_ids() -> None:
    assert _normalized_pattern_ids([
        "14667166127236F4",
        "1466716612",
        " 62A0A4B9B1 ",
        "",
    ]) == ["1466716612", "62a0a4b9b1"]


def test_prioritizes_positive_existing_root_patterns_before_mismatches() -> None:
    metrics = {
        "generic_patterns": {
            "human_root_pattern_details": [
                {
                    "pattern_id": "promising",
                    "candidate_scope_eligible": True,
                    "comparisons": 2,
                    "human_better": 2,
                    "ai_better": 0,
                    "average_value_delta": 100.0,
                },
                {
                    "pattern_id": "unsafe",
                    "candidate_scope_eligible": True,
                    "comparisons": 5,
                    "human_better": 1,
                    "ai_better": 4,
                    "average_value_delta": -50.0,
                },
            ],
            "human_mismatch_details": [
                {
                    "pattern_id": "mismatch",
                    "recommended_action": "receive_same",
                    "actual_action": "pass",
                    "count": 8,
                    "observations": 20,
                },
                {
                    "pattern_id": "wrong_direction",
                    "recommended_action": "pass",
                    "actual_action": "receive_same",
                    "count": 20,
                    "observations": 50,
                },
            ],
        }
    }

    assert prioritized_pattern_ids(metrics) == ["promising", "mismatch"]


def test_candidate_selection_limits_each_pattern() -> None:
    buckets = {
        "a": [
            {"id": "receive-a", "actual_action": ["receive", "2", None]},
            {"id": "pass-a", "actual_action": ["pass", None, None]},
            {"id": "receive-b", "actual_action": ["receive", "2", None]},
            {"id": "pass-b", "actual_action": ["pass", None, None]},
        ],
        "b": [
            {"id": "receive-c", "actual_action": ["receive", "3", None]},
            {"id": "pass-c", "actual_action": ["pass", None, None]},
        ],
    }
    selected = _round_robin_candidates(
        buckets,
        ["a", "b"],
        per_pattern=3,
        max_evaluations=5,
    )
    assert [item["id"] for item in selected] == [
        "pass-a",
        "pass-b",
        "receive-a",
        "pass-c",
        "receive-c",
    ]


def test_comparison_summary_does_not_copy_position_data() -> None:
    summary = _comparison_summary({
        "human_root_comparisons": 3,
        "human_root_completed": 2,
        "hands": {"A": ["1"]},
    })
    assert summary["human_root_comparisons"] == 3
    assert summary["human_root_completed"] == 2
    assert "hands" not in summary


def test_direct_comparison_summary_counts_both_sides() -> None:
    summary = _summarize_direct_comparisons([
        {
            "completed": True,
            "selected_side": "human",
            "common_depth": 7,
            "value_delta": 120.0,
            "human_terminal_loss_rate": 0.1,
            "ai_terminal_loss_rate": 0.2,
        },
        {
            "completed": True,
            "selected_side": "ai",
            "common_depth": 5,
            "value_delta": -20.0,
            "human_terminal_loss_rate": 0.3,
            "ai_terminal_loss_rate": 0.2,
        },
        {"completed": False, "reason": "common_depth_below_five"},
    ])
    assert summary == {
        "attempted": 3,
        "completed": 2,
        "incomplete": 1,
        "human_better": 1,
        "ai_better": 1,
        "tied": 0,
        "average_common_depth": 6.0,
        "average_value_delta": 50.0,
        "average_human_terminal_loss_rate": 0.2,
        "average_ai_terminal_loss_rate": 0.2,
        "terminal_loss_rate_delta": 0.0,
    }


def test_candidate_requires_terminal_loss_safety() -> None:
    summary = {
        "completed": 8,
        "human_better": 7,
        "ai_better": 1,
        "average_value_delta": 200.0,
        "average_human_terminal_loss_rate": 0.25,
        "average_ai_terminal_loss_rate": 0.10,
    }
    assert _direct_pattern_candidate(summary) == (
        False,
        "terminal_loss_rate_increased",
    )

    summary["average_human_terminal_loss_rate"] = 0.05
    assert _direct_pattern_candidate(summary) == (True, "qualified")

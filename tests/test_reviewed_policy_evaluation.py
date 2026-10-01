"""Regression tests for the fixed human-reviewed disagreement set."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from goita_ai2.neural_policy import NeuralPolicyModel
from goita_ai2.reviewed_policy_evaluation import DEFAULT_CASES, DEFAULT_MODEL
from goita_ai2.reviewed_policy_evaluation import evaluate_reviewed_cases
from goita_ai2.reviewed_policy_evaluation import load_reviewed_cases
from goita_ai2.reviewed_policy_evaluation import write_evaluation_report


def test_reviewed_set_keeps_both_kinds_of_policy_win() -> None:
    cases = load_reviewed_cases(DEFAULT_CASES)

    report = evaluate_reviewed_cases(cases)

    assert report["records"] == 4
    assert report["composition"] == {
        "review_reference_not_correction": 1,
        "training_correction": 3,
    }
    assert report["captured_disagreement_outcomes"] == {
        "current_only": 3,
        "neural_only": 1,
        "both_match": 0,
        "neither": 0,
    }
    assert report["captured_at_review"]["current"] == {
        "matches": 3,
        "records": 4,
        "match_rate": 0.75,
    }
    assert report["captured_at_review"]["neural"] == {
        "matches": 1,
        "records": 4,
        "match_rate": 0.25,
    }
    assert report["captured_at_review"]["hybrid"]["matches"] == 0


def test_exported_neural_model_matches_all_reviewed_preferences() -> None:
    cases = load_reviewed_cases(DEFAULT_CASES)
    model = NeuralPolicyModel.load(DEFAULT_MODEL)

    report = evaluate_reviewed_cases(cases, neural_model=model)

    assert report["live_neural"] == {
        "matches": 4,
        "records": 4,
        "match_rate": 1.0,
    }
    assert all(
        item["live_neural_matches"] is True for item in report["details"]
    )


def test_writes_machine_readable_report() -> None:
    with TemporaryDirectory() as directory:
        output = Path(directory) / "reviewed.json"
        report = write_evaluation_report(
            DEFAULT_CASES,
            DEFAULT_MODEL,
            output,
        )
        saved = json.loads(output.read_text(encoding="utf-8"))

    assert saved == report
    assert saved["scope"] == "human_reviewed_disagreement_regression_only"
    assert "overall playing strength" in saved["warning"]

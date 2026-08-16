from __future__ import annotations

from goita_ai2.probabilistic_hand_validation import (
    ProbabilisticHandValidationConfig,
    validate_probabilistic_hand_cases,
)


def _case():
    return {
        "id": "synthetic-opening-A",
        "player": "A",
        "dealer": "A",
        "initial_score": {"AC": 0, "BD": 0},
        "initial_hands": {
            "A": list("12334567"),
            "B": list("13123815"),
            "C": list("12111459"),
            "D": list("17512644"),
        },
        "history": [],
    }


def test_validation_config_rejects_invalid_limits() -> None:
    try:
        ProbabilisticHandValidationConfig(cases=0).validate()
    except ValueError as exc:
        assert "cases" in str(exc)
    else:
        raise AssertionError("cases=0 must be rejected")


def test_private_case_validation_reports_calibration_and_public_pool_checks() -> None:
    config = ProbabilisticHandValidationConfig(
        cases=1,
        sample_count=64,
        max_seconds_per_case=0.08,
        max_retained_candidates=24,
    )

    report = validate_probabilistic_hand_cases([_case()], config)

    assert report["status"] == "passed"
    assert report["summary"]["cases"] == 1
    assert report["summary"]["piece_observations"] == 27
    assert 0.0 <= report["summary"]["mean_brier"] <= 1.0
    assert report["summary"]["timing_ms"]["maximum"] >= 0.0
    assert report["calibration"]
    assert report["cases"][0]["candidate_count"] <= 24
    assert report["cases"][0]["all_candidates_match_public_pool"]
    assert all(item["passed"] for item in report["summary"]["checks"])


if __name__ == "__main__":
    test_validation_config_rejects_invalid_limits()
    test_private_case_validation_reports_calibration_and_public_pool_checks()
    print("PROBABILISTIC_HAND_VALIDATION_TEST_OK")

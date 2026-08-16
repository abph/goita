from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from goita_ai2.branched_attack_validation import (
    BranchedAttackValidationConfig,
    run_branched_attack_validation,
    write_validation_report,
)


def test_validation_config_rejects_invalid_limits() -> None:
    try:
        BranchedAttackValidationConfig(pairs=0).validate()
    except ValueError as exc:
        assert "pairs" in str(exc)
    else:
        raise AssertionError("pairs=0 must be rejected")


def test_one_mirrored_pair_passes_production_invariants() -> None:
    config = BranchedAttackValidationConfig(
        pairs=1,
        seed=1_480_002,
        max_steps=160,
        planner_seconds=0.08,
        latency_guardrail_ms=1000.0,
    )
    report = run_branched_attack_validation(config)

    assert report["status"] == "passed"
    assert report["summary"]["games"] == 2
    assert report["summary"]["finished"] == 2
    assert report["summary"]["errors"] == 0
    assert report["summary"]["planner_calls"] > 0
    assert report["summary"]["maximum_generated_candidates"] <= 10
    assert report["summary"]["maximum_evaluated_candidates"] <= 8
    assert report["summary"]["missed_immediate_wins"] == {
        "planner_enabled": 0,
        "planner_disabled": 0,
    }
    assert report["summary"]["maximum_probability_cache_entries"] <= 128
    assert all(check["passed"] for check in report["summary"]["checks"])
    first, second = report["games"]
    assert first["initial_hands"] == second["initial_hands"]
    assert {first["planner_team"], second["planner_team"]} == {"AC", "BD"}

    with TemporaryDirectory() as directory:
        output = Path(directory) / "branched-validation.json"
        write_validation_report(output, report)
        stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored["config"]["pairs"] == 1
    assert stored["summary"]["passed"] is True


if __name__ == "__main__":
    test_validation_config_rejects_invalid_limits()
    test_one_mirrored_pair_passes_production_invariants()
    print("BRANCHED_ATTACK_VALIDATION_TEST_OK")

"""Exercises the phase-6 information-set validation report.

The small sentinel scenario must prove invariants, prevent strategy fusion,
stay bounded, and produce an explicit rollout decision against legacy search.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from goita_ai2.information_set_benchmark import (
    run_information_set_validation,
    write_information_set_validation,
)


def test_phase_6_information_set_validation_passes_all_four_gates() -> None:
    report = run_information_set_validation(
        search_seconds=1.0,
        runs=2,
        max_depth=3,
        max_nodes=100_000,
    )

    assert report["passed"] is True
    assert all(report[label]["passed"] for label in ("6-1", "6-2", "6-3", "6-4"))
    assert report["6-1"]["candidate_details"]["candidate_count"] == 2
    assert report["6-2"]["checks"]["sample_order_does_not_change_shared_result"]
    assert report["6-2"]["checks"]["live_hidden_deal_does_not_change_shared_result"]
    assert report["6-3"]["checks"]["node_limit_respected"]
    assert report["6-4"]["recommendation"] == "keep_information_set_search_enabled"


def test_phase_6_report_can_be_saved_as_json() -> None:
    report = run_information_set_validation(
        search_seconds=1.0,
        runs=1,
        max_depth=3,
        max_nodes=100_000,
    )
    with TemporaryDirectory() as directory:
        output = Path(directory) / "information-set-validation.json"
        write_information_set_validation(output, report)
        stored = json.loads(output.read_text(encoding="utf-8"))

    assert stored["schema_version"] == 1
    assert stored["scenario"] == "strategy_fusion_sentinel_opening"
    assert stored["6-4"]["comparison"]["legacy_action"]


if __name__ == "__main__":
    test_phase_6_information_set_validation_passes_all_four_gates()
    test_phase_6_report_can_be_saved_as_json()
    print("INFORMATION_SET_BENCHMARK_TEST_OK")

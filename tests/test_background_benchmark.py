from __future__ import annotations

from goita_ai2.background_benchmark import run_background_load_benchmark


def test_background_load_benchmark_finishes_without_exceeding_limits() -> None:
    result = run_background_load_benchmark(
        rooms=6,
        search_seconds=0.1,
        sample_count=4,
        max_depth=3,
    )

    assert result["accepted"] + result["throttled"] == 6
    assert result["wait_completed"] == 6
    assert result["errors"] == 0
    assert result["runtime"]["pending"] == 0
    assert result["runtime"]["active"] == 0
    assert result["runtime"]["max_active_seen"] <= result["runtime"]["max_workers"]
    assert result["runtime"]["max_pending_seen"] <= result["runtime"]["max_pending"]


if __name__ == "__main__":
    test_background_load_benchmark_finishes_without_exceeding_limits()
    print("BACKGROUND_BENCHMARK_TEST_OK")

from __future__ import annotations

from goita_ai2.prefetch_benchmark import (
    PrefetchBenchmarkConfig,
    run_prefetch_benchmark,
)


def test_prefetch_benchmark_reports_accuracy_and_latency() -> None:
    report = run_prefetch_benchmark(
        PrefetchBenchmarkConfig(
            games=1,
            seed=920_000,
            max_steps=80,
            opponent_think_seconds=0.001,
            search_seconds=0.03,
            search_samples=2,
            search_depth=1,
            search_nodes=300,
            background_branches=1,
        )
    )

    summary = report["summary"]
    projection = report["projection"]
    telemetry = report["telemetry"]
    assert summary["games"] == 1
    assert summary["target_decisions"] > 0
    assert 0.0 <= summary["background_hit_rate"] <= 1.0
    assert 0.0 <= summary["path_action_match_rate"] <= 1.0
    assert 0.0 <= summary["path_completion_rate"] <= 1.0
    assert summary["average_target_decision_ms"] >= 0.0
    assert summary["estimated_saved_ms"] >= 0.0
    assert sum(summary["background_cache_hits_by_kind"].values()) == summary[
        "background_cache_hits"
    ]
    assert sum(summary["background_cache_hits_by_context"].values()) == summary[
        "background_cache_hits"
    ]
    assert projection["prefetch_calls"] >= 1
    assert projection["errors"] == 0
    assert report["projection_by_kind"]
    for metrics in report["projection_by_kind"].values():
        assert 0.0 <= metrics["action_match_rate"] <= 1.0
        assert 0.0 <= metrics["completion_rate"] <= 1.0
    assert telemetry["background_prefetch_calls"] == projection["prefetch_calls"]
    assert telemetry["background_branches_scheduled"] == projection["scheduled"]
    assert telemetry["background_errors"] == 0
    assert isinstance(report["adaptive_value"], dict)
    assert sum(
        int(values["cache_hits"])
        for values in report["adaptive_value"]["by_kind"].values()
    ) == summary["background_cache_hits"]


if __name__ == "__main__":
    test_prefetch_benchmark_reports_accuracy_and_latency()
    print("PREFETCH_BENCHMARK_TEST_OK")

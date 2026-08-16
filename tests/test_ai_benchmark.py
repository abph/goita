from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from goita_ai2.benchmark import (
    BenchmarkConfig,
    load_checkpoint,
    percentile,
    render_markdown_summary,
    run_benchmark,
    summarize_decisions,
)


def test_percentile_and_summary_include_stage_breakdown() -> None:
    assert percentile([10.0, 20.0, 30.0], 50) == 20.0
    decisions = [
        {
            "position_phase": "opening",
            "timing": {
                "total_ms": 10.0,
                "rule_based_ms": 4.0,
                "inference_ms": 1.0,
                "cache_ms": 0.0,
                "sample_generation_ms": 2.0,
                "search_ms": 2.0,
                "other_ms": 1.0,
            },
            "search": {"depth": 3, "nodes": 40},
        },
        {
            "position_phase": "endgame",
            "timing": {
                "total_ms": 30.0,
                "rule_based_ms": 10.0,
                "inference_ms": 5.0,
                "cache_ms": 0.0,
                "sample_generation_ms": 0.0,
                "search_ms": 0.0,
                "other_ms": 15.0,
            },
            "search": None,
        },
    ]
    summary = summarize_decisions(decisions)
    assert summary["total"]["average_ms"] == 20.0
    assert summary["total"]["p95_ms"] == 29.0
    assert summary["stages"]["rule_based"]["total_ms"] == 14.0
    assert summary["search"]["decisions"] == 1
    assert summary["search"]["nodes"]["total"] == 40
    assert summary["groups"]["position_phase"]["opening"]["decisions"] == 1
    assert summary["groups"]["hand_size"]["unknown"]["decisions"] == 2
    assert summary["groups"]["legal_action_band"]["1"]["decisions"] == 2
    assert summary["search_breakdown"]["by_depth"]["3"]["nodes"]["total"] == 40.0


def test_small_benchmark_writes_replayable_json_and_can_resume() -> None:
    config = BenchmarkConfig(
        games=1,
        seed=991_000,
        max_steps=80,
        search_seconds=0.01,
        search_samples=2,
        search_depth=1,
        search_nodes=200,
        checkpoint_every=1,
        slow_limit=3,
    )
    with TemporaryDirectory() as directory:
        output = Path(directory) / "benchmark.json"
        report = run_benchmark(config, output=output)
        stored = json.loads(output.read_text(encoding="utf-8"))

        assert report["status"] == "complete"
        assert stored["progress"] == {"completed_games": 1, "target_games": 1}
        assert stored["profile"] == "current"
        assert len(stored["games"]) == 1
        assert stored["games"][0]["initial_hands"]
        assert stored["games"][0]["history"]
        assert stored["decisions"]
        assert stored["slow_decisions"]
        assert stored["slow_decisions"][0]["replay"]["initial_hands"]
        assert "games_per_hour" in stored["game_summary"]
        assert "agent_cumulative_metrics" in stored
        assert "prediction_cache_summary" in stored
        assert "partial_hits" in stored["prediction_cache_summary"]
        markdown = render_markdown_summary(stored)
        assert "## Stage breakdown" in markdown
        assert "## Search breakdown" in markdown
        assert "Prediction samples:" in markdown
        assert "## Slowest decisions" in markdown

        started_at, games, decisions = load_checkpoint(output, config)
        assert started_at == stored["started_at"]
        assert len(games) == 1
        assert len(decisions) == len(stored["decisions"])

        resumed = run_benchmark(config, output=output, resume=True)
        assert resumed["prediction_cache_summary"] == stored[
            "prediction_cache_summary"
        ]


if __name__ == "__main__":
    test_percentile_and_summary_include_stage_breakdown()
    test_small_benchmark_writes_replayable_json_and_can_resume()
    print("AI_BENCHMARK_TEST_OK")

from __future__ import annotations

from goita_ai2.long_comparison import (
    LongComparisonConfig,
    build_comparison_report,
    compare_reports,
    render_comparison_markdown,
)


def _report(average: float, p95: float, elapsed: float, action: str) -> dict:
    return {
        "status": "complete",
        "game_summary": {
            "games": 1,
            "finished": 1,
            "games_per_hour": 100.0,
            "average_steps": 1.0,
        },
        "decision_summary": {
            "decisions": 1,
            "total": {
                "average_ms": average,
                "median_ms": average,
                "p95_ms": p95,
                "p99_ms": p95,
                "max_ms": p95,
            },
            "search": {
                "decision_rate": 1.0,
                "depth": {"average": 3.0},
                "nodes": {"average": 100.0},
            },
            "stages": {
                "rule_based": {"average_ms": 10.0},
                "sample_generation": {"average_ms": 2.0},
                "search": {"average_ms": average - 12.0},
            },
        },
        "prediction_cache_summary": {
            "hits": 1,
            "partial_hits": 2,
            "reuse_rate": 0.5,
            "reused_samples": 4,
            "generated_samples": 5,
        },
        "games": [
            {
                "game_index": 0,
                "elapsed_ms": elapsed,
                "history": [
                    {"player": "A", "action": [action, None, None]},
                ],
            }
        ],
        "decisions": [
            {
                "game_index": 0,
                "timing": {"total_ms": average},
            }
        ],
    }


def test_compare_reports_calculates_paired_deltas_and_divergence() -> None:
    before = _report(100.0, 180.0, 1_000.0, "pass")
    after = _report(80.0, 140.0, 800.0, "receive")
    comparison = compare_reports(before, after)

    assert comparison["decision_average_delta_ms"] == -20.0
    assert comparison["decision_average_delta_percent"] == -20.0
    assert comparison["decision_p95_delta_ms"] == -40.0
    assert comparison["paired_game_decision_average_ms"]["mean_delta"] == -20.0
    assert comparison["paired_game_elapsed_ms"]["mean_delta"] == -200.0
    assert comparison["stage_average_delta_ms"]["search"] == -20.0
    assert comparison["behavior"]["divergent_games"] == 1


def test_comparison_report_and_markdown_cover_all_variants() -> None:
    reports = {
        "baseline": _report(100.0, 180.0, 1_000.0, "pass"),
        "adaptive": _report(90.0, 160.0, 900.0, "pass"),
        "optimized": _report(80.0, 140.0, 800.0, "receive"),
    }
    report = build_comparison_report(LongComparisonConfig(games=1), reports)
    markdown = render_comparison_markdown(report)

    assert set(report["variants"]) == {"baseline", "adaptive", "optimized"}
    assert set(report["comparisons"]) == {
        "baseline_to_adaptive",
        "adaptive_to_optimized",
        "baseline_to_optimized",
    }
    assert "| optimized |" in markdown
    assert "## Stage averages" in markdown
    assert "Paired game average 95% CI" in markdown
    assert "Whole-game elapsed 95% CI" in markdown


if __name__ == "__main__":
    test_compare_reports_calculates_paired_deltas_and_divergence()
    test_comparison_report_and_markdown_cover_all_variants()
    print("LONG_COMPARISON_TEST_OK")

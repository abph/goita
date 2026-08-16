from __future__ import annotations

from backend import app as app_module
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.rule_based_intermediate_middle import (
    RuleBasedAgent as IntermediateMiddleRuleBasedAgent,
)
from goita_ai2.state import GoitaState


HANDS = {
    "A": list("11123457"),
    "B": list("11122334"),
    "C": list("11244556"),
    "D": list("11356789"),
}


def test_current_ai_records_stage_timings_without_changing_action() -> None:
    state = GoitaState(HANDS, dealer="A")
    current = RuleBasedAgent()
    frozen = IntermediateMiddleRuleBasedAgent()
    current.bind_player("A")
    frozen.bind_player("A")
    current.TIME_SEARCH_ENABLED = False
    frozen.TIME_SEARCH_ENABLED = False

    legal = state.legal_actions("A")
    current_action = current.select_action(state, "A", legal)
    frozen_action = frozen.select_action(state, "A", legal)

    assert current_action == frozen_action
    metrics = current.last_performance_metrics
    assert set(metrics) == {
        "total_ms",
        "rule_based_ms",
        "inference_ms",
        "cache_ms",
        "sample_generation_ms",
        "search_ms",
        "other_ms",
    }
    assert metrics["total_ms"] >= metrics["rule_based_ms"]
    assert metrics["inference_ms"] >= 0.0
    assert metrics["cache_ms"] == 0.0
    assert metrics["sample_generation_ms"] == 0.0
    assert metrics["search_ms"] == 0.0
    measured_total = sum(value for key, value in metrics.items() if key != "total_ms")
    assert abs(metrics["total_ms"] - measured_total) < 0.01
    assert not hasattr(frozen, "last_performance_metrics")


def test_timed_search_records_sampling_and_search_separately() -> None:
    state = GoitaState(HANDS, dealer="A")
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent.TIME_SEARCH_SAMPLE_COUNT = 4
    agent.TIME_SEARCH_MAX_SECONDS = 0.03
    agent.TIME_SEARCH_MAX_DEPTH = 3
    agent.TIME_SEARCH_MAX_NODES = 2_000

    agent.select_action(state, "A", state.legal_actions("A"))

    metrics = agent.last_performance_metrics
    assert metrics["sample_generation_ms"] > 0.0
    assert metrics["search_ms"] >= 0.0
    snapshot = agent.performance_metrics_snapshot()
    assert snapshot["totals"]["sample_generation"]["calls"] == 1
    assert snapshot["totals"]["total"]["calls"] == 1


def test_performance_log_format_is_current_ai_only() -> None:
    agent = RuleBasedAgent()
    agent.last_performance_metrics = {
        "total_ms": 12.34,
        "rule_based_ms": 2.0,
        "inference_ms": 1.0,
        "cache_ms": 0.5,
        "sample_generation_ms": 3.0,
        "search_ms": 5.0,
        "other_ms": 1.34,
    }
    formatted = app_module._format_ai_performance(agent)
    assert formatted == (
        " [PERF(ms):total=12.3,rule=2.0,infer=1.0,"
        "cache=0.5,sample=3.0,search=5.0,other=1.3]"
    )
    assert app_module._format_ai_performance(IntermediateMiddleRuleBasedAgent()) == ""


if __name__ == "__main__":
    test_current_ai_records_stage_timings_without_changing_action()
    test_timed_search_records_sampling_and_search_separately()
    test_performance_log_format_is_current_ai_only()
    print("AI_PERFORMANCE_TEST_OK")

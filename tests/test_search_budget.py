from __future__ import annotations

from goita_ai2.current_ai.search_budget import AdaptiveSearchBudgetController
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


HANDS = {
    "A": list("11123457"),
    "B": list("11122334"),
    "C": list("11244556"),
    "D": list("11356789"),
}


def _plan(controller, state, actions, *, warmup=4):
    return controller.plan(
        state,
        "A",
        actions,
        enabled=True,
        warmup_observations=warmup,
        minimum_seconds=0.15,
        minimum_samples=8,
        configured_seconds=1.0,
        configured_samples=80,
        configured_depth=11,
        configured_nodes=250_000,
    )


def test_search_budget_keeps_configured_limits_during_warmup() -> None:
    state = GoitaState(HANDS, dealer="A")
    actions = state.legal_actions("A")
    controller = AdaptiveSearchBudgetController()

    plan = _plan(controller, state, actions)

    assert plan.reason == "warmup"
    assert plan.effective_seconds == 1.0
    assert plan.effective_samples == 80
    assert plan.effective_depth == 11
    assert plan.effective_nodes == 250_000


def test_search_budget_reduces_work_after_repeated_overruns() -> None:
    state = GoitaState(HANDS, dealer="A")
    actions = state.legal_actions("A")
    controller = AdaptiveSearchBudgetController()
    plan = _plan(controller, state, actions)
    for _ in range(4):
        controller.observe(
            plan,
            sample_ms=240.0,
            search_ms=1_100.0,
            samples=80,
            nodes=100_000,
            cache_hit=False,
            cancelled=False,
            alpha=0.25,
        )

    adjusted = _plan(controller, state, actions)

    assert adjusted.reason == "measured_complexity"
    assert 0.15 <= adjusted.effective_seconds < 1.0
    assert 8 <= adjusted.effective_samples < 80
    assert adjusted.effective_depth < 11
    assert 500 <= adjusted.effective_nodes < 250_000
    snapshot = controller.snapshot()
    assert snapshot["totals"]["observations"] == 4
    assert snapshot["totals"]["overruns"] == 4


def test_search_budget_allocates_more_to_harder_positions() -> None:
    state = GoitaState(HANDS, dealer="A")
    controller = AdaptiveSearchBudgetController()
    easy = _plan(
        controller,
        state,
        [("attack", None, "1"), ("attack", None, "2")],
        warmup=0,
    )
    hard = _plan(
        controller,
        state,
        [
            ("attack", None, piece)
            for piece in ("1", "2", "3", "4", "5", "6")
        ],
        warmup=0,
    )

    assert hard.effective_seconds > easy.effective_seconds
    assert hard.effective_samples > easy.effective_samples


def test_agent_effective_budget_does_not_mutate_configured_limits() -> None:
    state = GoitaState(HANDS, dealer="A")
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent.TIME_SEARCH_ADAPTIVE_BUDGET_WARMUP = 0
    configured = (
        agent.TIME_SEARCH_MAX_SECONDS,
        agent.TIME_SEARCH_SAMPLE_COUNT,
        agent.TIME_SEARCH_MAX_DEPTH,
        agent.TIME_SEARCH_MAX_NODES,
    )

    plan = agent._prepare_time_search_budget(
        state,
        "A",
        state.legal_actions("A"),
    )

    assert plan.reason == "measured_complexity"
    assert agent.last_time_search_budget["effective_seconds"] <= configured[0]
    assert (
        agent.TIME_SEARCH_MAX_SECONDS,
        agent.TIME_SEARCH_SAMPLE_COUNT,
        agent.TIME_SEARCH_MAX_DEPTH,
        agent.TIME_SEARCH_MAX_NODES,
    ) == configured


if __name__ == "__main__":
    test_search_budget_keeps_configured_limits_during_warmup()
    test_search_budget_reduces_work_after_repeated_overruns()
    test_search_budget_allocates_more_to_harder_positions()
    test_agent_effective_budget_does_not_mutate_configured_limits()
    print("SEARCH_BUDGET_TEST_OK")

from __future__ import annotations

import time

from goita_ai2.constants import PIECE_TOTALS
from goita_ai2.current_ai.branched_attack_runtime import (
    BranchedAttackPlanCache,
    BranchedAttackRuntimeMixin,
)
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


def _state_with_a_hand(hand: str, *, reverse_hidden: bool = False) -> GoitaState:
    remaining = [
        piece
        for piece, total in PIECE_TOTALS.items()
        for _ in range(total)
    ]
    for piece in hand:
        remaining.remove(piece)
    groups = [remaining[:8], remaining[8:16], remaining[16:24]]
    if reverse_hidden:
        groups.reverse()
    return GoitaState(
        hands={
            "A": list(hand),
            "B": groups[0],
            "C": groups[1],
            "D": groups[2],
        },
        dealer="A",
    )


def _apply(agent: RuleBasedAgent, state: GoitaState, player: str, action) -> None:
    action_type, block, attack = action
    if action_type == "pass":
        state.apply_pass(player)
    elif action_type == "receive":
        assert block is not None
        state.apply_receive(player, block)
    elif action_type == "attack":
        assert attack is not None
        state.apply_attack(player, attack)
    else:
        assert block is not None and attack is not None
        state.apply_attack_after_block(player, block, attack)
    agent.on_public_action(state, player, action)


def _agent() -> RuleBasedAgent:
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent.TIME_SEARCH_ENABLED = False
    return agent


def test_rule_based_agent_uses_branched_attack_runtime_mixin() -> None:
    assert issubclass(RuleBasedAgent, BranchedAttackRuntimeMixin)


def test_production_uses_and_continues_a_representative_attack_plan() -> None:
    state = _state_with_a_hand("11122347")
    agent = _agent()

    first = agent.select_action(state, "A", state.legal_actions("A"))

    assert first == ("attack_after_block", "1", "7")
    assert agent.last_decision_reason == "score_fallback"
    assert agent.last_score_fallback_detail == "attack_sequence_two_kyosha_single_big"
    first_metrics = dict(agent.last_branched_attack_metrics)
    assert first_metrics["continued"] is False
    assert first_metrics["selected_source"] == "representative:two_kyosha_single_big"
    assert "support_counts" in first_metrics["inference_summary"]
    assert sum(first_metrics["inference_summary"]["support_counts"].values()) > 0

    _apply(agent, state, "A", first)
    for passer in ("B", "C", "D"):
        _apply(agent, state, passer, ("pass", None, None))

    second = agent.select_action(state, "A", state.legal_actions("A"))

    assert second == ("attack_after_block", "1", "2")
    assert agent.last_branched_attack_metrics["continued"] is True
    assert agent.last_branched_attack_metrics["revalidated"] is True
    assert agent.last_branched_attack_metrics["selected_plan_id"] == first_metrics["selected_plan_id"]
    assert agent._active_branched_attack_plan(state).reason == "revalidated_with_current_hand_inference"


def test_replan_reuses_cached_evaluation_for_the_same_public_position() -> None:
    state = _state_with_a_hand("11122347")
    agent = _agent()
    agent._ensure_trackers(state)
    actions = state.legal_actions("A")

    first = agent._production_branched_attack_action(state, "A", actions)
    assert first is not None
    assert first.cache_hit is False
    agent._invalidate_branched_attack_plan(state, "test_replan")

    replacement = agent._production_branched_attack_action(state, "A", actions)

    assert replacement is not None
    assert replacement.continued is False
    assert replacement.cache_hit is True
    assert replacement.action == first.action
    snapshot = agent.branched_attack_cache_snapshot()
    assert snapshot["hits"] >= 1
    assert snapshot["hit_rate"] > 0.0


def test_cache_key_does_not_depend_on_opponents_hidden_deal() -> None:
    first_state = _state_with_a_hand("11122347")
    second_state = _state_with_a_hand("11122347", reverse_hidden=True)
    first_agent = _agent()
    second_agent = _agent()
    first_agent._ensure_trackers(first_state)
    second_agent._ensure_trackers(second_state)

    first_key = first_agent._branched_attack_cache_key(
        first_state,
        "A",
        first_state.legal_actions("A"),
    )
    second_key = second_agent._branched_attack_cache_key(
        second_state,
        "A",
        second_state.legal_actions("A"),
    )

    assert first_key == second_key


def test_candidate_and_evaluation_limits_are_enforced() -> None:
    state = _state_with_a_hand("11122347")
    agent = _agent()
    agent.BRANCHED_ATTACK_MAX_TEMPLATE_PLANS = 2
    agent.BRANCHED_ATTACK_MAX_GENERIC_ROOTS = 2
    agent.BRANCHED_ATTACK_MAX_TOTAL_PLANS = 3
    agent.BRANCHED_ATTACK_MAX_EVALUATED_PLANS = 2
    agent.clear_branched_attack_cache()
    agent._ensure_trackers(state)

    choice = agent._production_branched_attack_action(
        state,
        "A",
        state.legal_actions("A"),
    )

    assert choice is not None
    metrics = agent.last_branched_attack_metrics
    assert metrics["generated"] <= 3
    assert metrics["evaluated"] <= 2
    assert isinstance(metrics["timed_out"], bool)
    assert metrics["truncated"] is True
    assert metrics["generation_ms"] >= 0.0
    assert metrics["evaluation_ms"] >= 0.0
    assert metrics["cache_ms"] >= 0.0
    assert metrics["elapsed_ms"] >= 0.0


def test_time_limit_stops_evaluation_between_candidates() -> None:
    state = _state_with_a_hand("11122347")
    agent = _agent()
    agent.BRANCHED_ATTACK_MAX_SECONDS = 0.001
    agent.BRANCHED_ATTACK_MAX_TOTAL_PLANS = 3
    agent.BRANCHED_ATTACK_MAX_EVALUATED_PLANS = 3
    agent.clear_branched_attack_cache()
    original = agent._evaluate_branched_attack_plan

    def slow_evaluation(*args, **kwargs):
        time.sleep(0.005)
        return original(*args, **kwargs)

    agent._evaluate_branched_attack_plan = slow_evaluation
    agent._ensure_trackers(state)

    choice = agent._production_branched_attack_action(
        state,
        "A",
        state.legal_actions("A"),
    )

    assert choice is None or choice.action in state.legal_actions("A")
    assert agent.last_branched_attack_metrics["evaluated"] <= 1
    assert agent.last_branched_attack_metrics["timed_out"] is True


def test_proven_tactic_stays_above_branched_planner() -> None:
    state = _state_with_a_hand("11122347")
    agent = _agent()
    expected = state.legal_actions("A")[0]
    agent._high_score_tsume_action = (
        lambda _state, _player, _actions, **_kwargs: (expected, 50.0, False)
    )

    selected = agent.select_action(state, "A", state.legal_actions("A"))

    assert selected == expected
    assert agent.last_decision_reason == "tsume"
    assert agent.last_score_fallback_detail == "high_score_50"
    assert agent.last_branched_attack_metrics == {}
    assert agent._active_branched_attack_plan(state) is None


def test_runtime_can_be_disabled_without_changing_the_fallback_path() -> None:
    state = _state_with_a_hand("11122347")
    agent = _agent()
    agent.BRANCHED_ATTACK_ENABLED = False
    agent._ensure_trackers(state)

    choice = agent._production_branched_attack_action(
        state,
        "A",
        state.legal_actions("A"),
    )

    assert choice is None
    assert agent._active_branched_attack_plan(state) is None


def test_lru_cache_is_bounded_and_reports_hits_and_evictions() -> None:
    cache = BranchedAttackPlanCache(max_entries=2, ttl_seconds=60.0)
    cache.put("a", ())
    cache.put("b", ())
    assert cache.get("a") == ()
    cache.put("c", ())

    assert cache.get("b") is None
    assert cache.get("a") == ()
    assert cache.get("c") == ()
    snapshot = cache.snapshot()
    assert snapshot["size"] == 2
    assert snapshot["evictions"] == 1
    assert snapshot["hits"] == 3
    assert snapshot["misses"] == 1


if __name__ == "__main__":
    test_rule_based_agent_uses_branched_attack_runtime_mixin()
    test_production_uses_and_continues_a_representative_attack_plan()
    test_replan_reuses_cached_evaluation_for_the_same_public_position()
    test_cache_key_does_not_depend_on_opponents_hidden_deal()
    test_candidate_and_evaluation_limits_are_enforced()
    test_time_limit_stops_evaluation_between_candidates()
    test_proven_tactic_stays_above_branched_planner()
    test_runtime_can_be_disabled_without_changing_the_fallback_path()
    test_lru_cache_is_bounded_and_reports_hits_and_evictions()
    print("BRANCHED_ATTACK_RUNTIME_TEST_OK")

from __future__ import annotations

import copy

from goita_ai2.current_ai.conditional_response import (
    ConditionalResponseMixin,
    ConditionalResponsePlan,
    conditional_response_runtime_snapshot,
    merge_conditional_response_snapshots,
    reset_conditional_response_runtime,
)
from goita_ai2.current_ai.timed_search import TimedSearchResult
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


def _receive_state(*, permuted: bool = False) -> GoitaState:
    other_hands = {
        "B": list("11133457"),
        "C": list("11244556"),
        "D": list("11236789"),
    }
    if permuted:
        other_hands = {
            "B": list("11236789"),
            "C": list("11133457"),
            "D": list("11244556"),
        }
    state = GoitaState(
        hands={"A": list("11234567"), **other_hands},
        dealer="B",
    )
    state.phase = "receive"
    state.turn = "A"
    state.attacker = "B"
    state.current_attack = "2"
    return state


def _agent_for(state: GoitaState) -> RuleBasedAgent:
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent._ensure_trackers(state)
    return agent


def test_rule_based_agent_uses_conditional_response_mixin() -> None:
    assert issubclass(RuleBasedAgent, ConditionalResponseMixin)


def test_conditional_response_key_does_not_read_opponent_hands() -> None:
    first = _receive_state()
    second = _receive_state(permuted=True)
    first_agent = _agent_for(first)
    second_agent = _agent_for(second)
    baseline = ("pass", None, None)

    first_key = first_agent._conditional_response_key(
        first,
        "A",
        first.legal_actions("A"),
        baseline,
    )
    second_key = second_agent._conditional_response_key(
        second,
        "A",
        second.legal_actions("A"),
        baseline,
    )

    assert first_key == second_key


def test_searched_receive_and_followup_are_reused_from_dictionary() -> None:
    reset_conditional_response_runtime()
    state = _receive_state()
    agent = _agent_for(state)
    agent._track[id(state)]["public_seen_counts"]["4"] = 3
    actions = state.legal_actions("A")
    baseline = ("pass", None, None)
    selected = ("receive", "2", None)
    result = TimedSearchResult(
        action=selected,
        depth=7,
        samples=32,
        nodes=100,
        elapsed_seconds=1.0,
        value=500.0,
        margin=100.0,
        agreement=0.75,
        decisive=True,
        information_set=True,
        candidate_count=8,
        information_confidence=0.65,
    )

    stored = agent._remember_conditional_response_plan(
        state,
        "A",
        actions,
        baseline,
        selected,
        result,
        source="kyosha_pass_compare",
    )
    clone = copy.deepcopy(agent)
    reused = clone._lookup_conditional_response_plan(
        state,
        "A",
        actions,
        baseline,
    )

    assert stored is not None
    assert reused == stored
    assert reused.action == selected
    assert reused.followup_attack_piece == "4"
    assert clone.last_conditional_response_hit is True
    clone._record_conditional_response_followup(used=True)
    snapshot = clone.conditional_response_dictionary_snapshot()
    assert snapshot["hits"] == 1
    assert snapshot["receive_hits"] == 1
    assert snapshot["foreground_hits"] == 1
    assert snapshot["followup_hits"] == 1
    assert snapshot["estimated_saved_ms"] == 1000.0

    merged = merge_conditional_response_snapshots([snapshot, snapshot])
    assert merged["hits"] == 2
    assert merged["estimated_saved_seconds"] == 2.0
    assert merged["dictionary_instances"] == 2

    runtime = conditional_response_runtime_snapshot([snapshot])
    assert runtime["hits"] == 1
    assert runtime["stores"] == 1
    assert runtime["estimated_saved_seconds"] == 1.0


def test_runtime_totals_survive_agent_replacement() -> None:
    reset_conditional_response_runtime()
    state = _receive_state()
    first_agent = _agent_for(state)
    first_agent._conditional_response_dictionary.put(
        "first-round-plan",
        ConditionalResponsePlan(
            action=("pass", None, None),
            followup_attack_piece=None,
            baseline_action=("pass", None, None),
            source="test",
            depth=3,
            agreement=1.0,
            information_confidence=1.0,
            margin=10.0,
            cache_source="foreground",
            cache_branch_kind=None,
            cache_branch_context=None,
            cached_compute_ms=100.0,
        ),
    )

    next_round_agent = _agent_for(_receive_state())
    runtime = conditional_response_runtime_snapshot(
        [next_round_agent.conditional_response_dictionary_snapshot()]
    )

    assert runtime["stores"] == 1
    assert runtime["size"] == 0


def test_illegal_cached_response_is_discarded() -> None:
    state = _receive_state()
    agent = _agent_for(state)
    actions = state.legal_actions("A")
    baseline = ("pass", None, None)
    key = agent._conditional_response_key(state, "A", actions, baseline)
    agent._conditional_response_dictionary.put(
        key,
        ConditionalResponsePlan(
            action=("receive", "5", None),
            followup_attack_piece="7",
            baseline_action=baseline,
            source="test",
            depth=7,
            agreement=1.0,
            information_confidence=1.0,
            margin=100.0,
            cache_source="foreground",
            cache_branch_kind=None,
            cache_branch_context=None,
            cached_compute_ms=0.0,
        ),
    )

    assert (
        agent._lookup_conditional_response_plan(
            state,
            "A",
            actions,
            baseline,
        )
        is None
    )
    snapshot = agent.conditional_response_dictionary_snapshot()
    assert snapshot["invalid"] == 1
    assert snapshot["hits"] == 0
    assert snapshot["lookups"] == 1
    assert snapshot["size"] == 0


if __name__ == "__main__":
    test_rule_based_agent_uses_conditional_response_mixin()
    test_conditional_response_key_does_not_read_opponent_hands()
    test_searched_receive_and_followup_are_reused_from_dictionary()
    test_runtime_totals_survive_agent_replacement()
    test_illegal_cached_response_is_discarded()
    print("CONDITIONAL_RESPONSE_TEST_OK")

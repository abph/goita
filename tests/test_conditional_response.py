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


def test_searched_receive_is_reused_without_inventing_a_followup() -> None:
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
    assert reused.followup_attack_piece is None
    assert clone.last_conditional_response_hit is True
    snapshot = clone.conditional_response_dictionary_snapshot()
    assert snapshot["hits"] == 1
    assert snapshot["receive_hits"] == 1
    assert snapshot["foreground_hits"] == 1
    assert snapshot["followup_hits"] == 0
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


def test_review_b_turn6_cached_receive_keeps_three_gold_attack() -> None:
    """2026-09-09 round 3: receiving partner's lance must not force lone rook.

    Repeat with the cache enabled/disabled and every seat rotation. The fixture
    uses real public actions; the original server cache is not in the report,
    so a root-only depth-three result exercises its storage/reuse boundary.
    """
    for rotation in range(4):
        for cached in (False, True):
            seats = "ABCD"
            def seat(p):
                return seats[(seats.index(p) + rotation) % 4]
            state = GoitaState(
                hands={seat(p): list(h) for p, h in {
                    "A": "73134621", "B": "15275514",
                    "C": "85161349", "D": "41213112",
                }.items()},
                dealer=seat("A"),
            )
            player = seat("B")
            agent = RuleBasedAgent()
            agent.bind_player(player)
            agent._ensure_trackers(state)
            agent.TIME_SEARCH_BACKGROUND_ENABLED = False
            agent.TIME_SEARCH_CACHE_ENABLED = False
            agent.TIME_SEARCH_PREDICTION_CACHE_ENABLED = False
            agent.TIME_SEARCH_ADAPTIVE_BUDGET_ENABLED = False
            prefix = (
                ("A", ("attack_after_block", "1", "3")),
                ("B", ("pass", None, None)),
                ("C", ("pass", None, None)),
                ("D", ("receive", "3", None)),
                ("D", ("attack", None, "2")),
                ("A", ("pass", None, None)),
            )
            for p, action in prefix:
                p = seat(p)
                assert action in state.legal_actions(p)
                kind, block, attack = action
                if kind == "pass":
                    state.apply_pass(p)
                elif kind == "receive":
                    state.apply_receive(p, block)
                elif kind == "attack":
                    state.apply_attack(p, attack)
                else:
                    state.apply_attack_after_block(p, block, attack)
                agent.on_public_action(state, p, action)
            receive = ("receive", "2", None)
            if cached:
                actions = state.legal_actions(player)
                baseline = copy.deepcopy(agent)._select_rule_based_action(state, player, actions)
                result = TimedSearchResult(
                    action=receive, depth=3, samples=32, nodes=100,
                    elapsed_seconds=0.1, value=500.0, margin=100.0,
                    agreement=0.75, decisive=True,
                )
                plan = agent._remember_conditional_response_plan(
                    state, player, actions, baseline, receive, result,
                    source="default",
                )
                assert plan is not None and plan.followup_attack_piece is None
                assert agent.select_action(state, player, actions) == receive
                assert agent.last_decision_reason == "response_dictionary"
                assert agent.last_conditional_response_hit
            state.apply_receive(player, "2")
            agent.on_public_action(state, player, receive)
            assert sorted(state.hands[player]) == list("1145557")
            assert agent._track[id(state)]["pending_conditional_response_attack_piece"] is None
            chosen = agent.select_action(state, player, state.legal_actions(player))
            assert chosen == ("attack", None, "5"), (rotation, cached, chosen)
            assert agent.last_decision_reason != "response_dictionary"
            assert not agent.last_score_fallback_detail.startswith("conditional_response_followup_")


if __name__ == "__main__":
    test_rule_based_agent_uses_conditional_response_mixin()
    test_conditional_response_key_does_not_read_opponent_hands()
    test_searched_receive_is_reused_without_inventing_a_followup()
    test_runtime_totals_survive_agent_replacement()
    test_illegal_cached_response_is_discarded()
    test_review_b_turn6_cached_receive_keeps_three_gold_attack()
    print("CONDITIONAL_RESPONSE_TEST_OK")

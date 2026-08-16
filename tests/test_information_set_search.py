"""Locks the public-information search behavior before information-set work.

These tests provide a deliberate baseline for detecting strategy-fusion fixes:
future changes may update the snapshot, but must never read real hidden hands.
"""

from __future__ import annotations

import copy
from collections import Counter

from goita_ai2.constants import PIECE_TOTALS
from goita_ai2.current_ai.information_set_search import InformationSetSearchWorld
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


MY_HAND = list("11123457")
OTHER_HANDS = {
    "B": list("11122334"),
    "C": list("11244556"),
    "D": list("11356789"),
}

# Both deals are legal completions of the same public opening observation.
# The current determinized search can see which completion it is evaluating,
# even though player A could not distinguish them during an actual round.
FUSION_HAND = list("12334567")
FUSION_DEALS = (
    {
        "B": list("13123815"),
        "C": list("12111459"),
        "D": list("17512644"),
    },
    {
        "B": list("24593851"),
        "C": list("11511111"),
        "D": list("62341724"),
    },
)


def _state(*, permuted: bool = False) -> GoitaState:
    others = OTHER_HANDS
    if permuted:
        others = {
            "B": OTHER_HANDS["D"],
            "C": OTHER_HANDS["B"],
            "D": OTHER_HANDS["C"],
        }
    return GoitaState(
        hands={"A": list(MY_HAND), **copy.deepcopy(others)},
        dealer="A",
    )


def _public_state_snapshot(state: GoitaState) -> dict:
    return {
        "hands": copy.deepcopy(state.hands),
        "hidden": copy.deepcopy(state.face_down_hidden),
        "phase": state.phase,
        "turn": state.turn,
        "attack": state.current_attack,
        "attacker": state.attacker,
        "last_block": state.last_block,
        "last_block_player": state.last_block_player,
        "king_block_used": state.king_block_used,
        "finished": state.finished,
        "winner": state.winner,
        "score": dict(state.team_score),
    }


def _opening_information_set_key(state: GoitaState, player: str) -> tuple:
    """Return only information visible to the acting player at the opening."""
    return (
        player,
        tuple(sorted(state.hands[player])),
        state.dealer,
        state.phase,
        state.turn,
        state.current_attack,
        state.attacker,
        tuple((seat, len(state.hands[seat])) for seat in "ABCD"),
        tuple(
            (seat, len(state.face_down_hidden[seat]))
            for seat in "ABCD"
        ),
        state.king_block_used,
        tuple(sorted(state.team_score.items())),
    )


def _search_baseline(state: GoitaState) -> tuple[dict, tuple]:
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent.TIME_SEARCH_CACHE_ENABLED = False
    agent.TIME_SEARCH_PREDICTION_CACHE_ENABLED = False
    agent.TIME_SEARCH_INFORMATION_SET_ENABLED = False
    agent.TIME_SEARCH_ADAPTIVE_BUDGET_ENABLED = False
    agent.TIME_SEARCH_MAX_SECONDS = 2.0
    agent.TIME_SEARCH_SAMPLE_COUNT = 4
    agent.TIME_SEARCH_MAX_DEPTH = 3
    agent.TIME_SEARCH_MAX_NODES = 100_000
    agent._ensure_trackers(state)

    actions = state.legal_actions("A")
    baseline = agent._select_rule_based_action(state, "A", actions)
    samples = agent._timed_search_sample_states(
        state,
        "A",
        agent._track[id(state)],
        4,
    )
    sample_keys = tuple(
        agent._timed_search_prediction_state_key(sample, "A")
        for sample in samples
    )
    result = agent._time_limited_search_from_samples(
        state,
        "A",
        actions,
        baseline,
        samples,
    )
    assert result is not None
    snapshot = result.as_dict()
    snapshot.pop("elapsed_seconds")
    snapshot["baseline_action"] = baseline
    return snapshot, sample_keys


def test_current_public_search_baseline_is_recorded() -> None:
    snapshot, sample_keys = _search_baseline(_state())

    assert len(sample_keys) == 4
    assert snapshot == {
        "action": ("attack_after_block", "3", "1"),
        "depth": 3,
        "samples": 4,
        "nodes": 456,
        "value": 277.82,
        "margin": 183.87,
        "agreement": 1.0,
        "decisive": False,
        "baseline_action": ("attack_after_block", "3", "1"),
    }


def test_public_search_ignores_the_real_opponent_hand_assignment() -> None:
    first_snapshot, first_samples = _search_baseline(_state())
    second_snapshot, second_samples = _search_baseline(_state(permuted=True))

    assert second_samples == first_samples
    assert second_snapshot == first_snapshot


def test_public_search_does_not_mutate_the_live_position() -> None:
    state = _state()
    before = _public_state_snapshot(state)

    _search_baseline(state)

    assert _public_state_snapshot(state) == before


def test_current_determinized_order_reproduces_strategy_fusion() -> None:
    states = [
        GoitaState(
            hands={"A": list(FUSION_HAND), **copy.deepcopy(deal)},
            dealer="A",
        )
        for deal in FUSION_DEALS
    ]
    agent = RuleBasedAgent()
    agent.bind_player("A")

    for state in states:
        assert Counter(
            piece
            for hand in state.hands.values()
            for piece in hand
        ) == Counter(PIECE_TOTALS)
    assert _opening_information_set_key(
        states[0],
        "A",
    ) == _opening_information_set_key(states[1], "A")

    first_candidates = agent._timed_search_ordered_actions(states[0], 3)
    second_candidates = agent._timed_search_ordered_actions(states[1], 3)

    # This intentionally records the current defect. The future information-set
    # policy will replace both lists with one shared choice for this public key.
    assert first_candidates[0][2] == "3"
    assert second_candidates[0][2] == "1"
    assert first_candidates != second_candidates


def _information_set_result(root_index: int, sample_order: tuple[int, ...]):
    states = [
        GoitaState(
            hands={"A": list(FUSION_HAND), **copy.deepcopy(deal)},
            dealer="A",
        )
        for deal in FUSION_DEALS
    ]
    state = states[root_index]
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent.TIME_SEARCH_CACHE_ENABLED = False
    agent.TIME_SEARCH_PREDICTION_CACHE_ENABLED = False
    agent.TIME_SEARCH_ADAPTIVE_BUDGET_ENABLED = False
    agent.TIME_SEARCH_INFORMATION_SET_ENABLED = True
    agent.TIME_SEARCH_MAX_SECONDS = 2.0
    agent.TIME_SEARCH_MAX_DEPTH = 3
    agent.TIME_SEARCH_MAX_NODES = 100_000
    agent._ensure_trackers(state)
    actions = state.legal_actions("A")
    baseline = agent._select_rule_based_action(state, "A", actions)
    result = agent._time_limited_search_from_samples(
        state,
        "A",
        actions,
        baseline,
        [states[index] for index in sample_order],
    )
    assert result is not None
    return agent, result


def test_production_search_builds_and_uses_an_information_set() -> None:
    agent, result = _information_set_result(0, (0, 1))

    assert result.information_set is True
    assert result.candidate_count == 2
    assert result.samples == 2
    assert result.policy_decisions > 0
    assert 0.0 <= result.information_confidence <= 1.0
    assert agent.last_information_set_search["candidates"] == 2
    assert agent.last_information_set_search["observations"] == 2


def test_shared_policy_removes_sample_order_and_live_deal_dependence() -> None:
    _first_agent, first = _information_set_result(0, (0, 1))
    _reversed_agent, reversed_result = _information_set_result(0, (1, 0))
    _other_live_agent, other_live = _information_set_result(1, (0, 1))

    assert first.action == reversed_result.action == other_live.action
    assert first.depth == reversed_result.depth == other_live.depth
    assert first.nodes == reversed_result.nodes == other_live.nodes
    assert round(first.value, 6) == round(reversed_result.value, 6)
    assert round(first.value, 6) == round(other_live.value, 6)
    assert first.policy_decisions == reversed_result.policy_decisions


def test_probability_and_confidence_control_world_value_aggregation() -> None:
    agent = RuleBasedAgent()
    high_positive = (
        InformationSetSearchWorld(0, None, 0.8, 1.0),
        InformationSetSearchWorld(1, None, 0.2, 1.0),
    )
    high_negative = (
        InformationSetSearchWorld(0, None, 0.2, 1.0),
        InformationSetSearchWorld(1, None, 0.8, 1.0),
    )
    uncertain = (
        InformationSetSearchWorld(0, None, 0.8, 0.0),
        InformationSetSearchWorld(1, None, 0.2, 0.0),
    )
    values = {0: 100.0, 1: -100.0}

    positive_value = agent._information_set_weighted_value(high_positive, values)
    negative_value = agent._information_set_weighted_value(high_negative, values)
    uncertain_value = agent._information_set_weighted_value(uncertain, values)

    assert positive_value > negative_value
    assert uncertain_value < positive_value


if __name__ == "__main__":
    test_current_public_search_baseline_is_recorded()
    test_public_search_ignores_the_real_opponent_hand_assignment()
    test_public_search_does_not_mutate_the_live_position()
    test_current_determinized_order_reproduces_strategy_fusion()
    test_production_search_builds_and_uses_an_information_set()
    test_shared_policy_removes_sample_order_and_live_deal_dependence()
    test_probability_and_confidence_control_world_value_aggregation()
    print("INFORMATION_SET_SEARCH_BASELINE_TEST_OK")

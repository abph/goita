from __future__ import annotations

import json

from goita_ai2.current_ai.generic_response_pattern import (
    GenericResponsePatternMixin,
)
from goita_ai2.current_ai.generic_response_store import (
    generic_response_pattern_store,
    generic_response_pattern_snapshot,
    reset_generic_response_patterns,
)
from goita_ai2.current_ai.search_cache import _digest_payload
from goita_ai2.current_ai.timed_search import TimedSearchResult
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


def _receive_state(
    my_hand: str = "11234678",
    *,
    permuted_opponents: bool = False,
) -> GoitaState:
    opponents = {
        "B": list("11122345"),
        "C": list("11123459"),
        "D": list("11234567"),
    }
    if permuted_opponents:
        opponents = {
            "B": list("11234567"),
            "C": list("11122345"),
            "D": list("11123459"),
        }
    state = GoitaState(
        hands={"A": list(my_hand), **opponents},
        dealer="B",
    )
    state.phase = "receive"
    state.turn = "A"
    state.attacker = "B"
    state.current_attack = "2"
    return state


def _pattern(state: GoitaState, player: str = "A"):
    agent = RuleBasedAgent()
    agent.bind_player(player)
    agent._ensure_trackers(state)
    actions = state.legal_actions(player)
    baseline = ("pass", None, None)
    payload = agent._generic_response_pattern_payload(
        state,
        player,
        actions,
        baseline,
    )
    key = agent._generic_response_pattern_key(
        state,
        player,
        actions,
        baseline,
    )
    return payload, key


def test_rule_based_agent_uses_generic_response_pattern_mixin() -> None:
    assert issubclass(RuleBasedAgent, GenericResponsePatternMixin)


def test_pattern_does_not_read_or_encode_opponent_hands() -> None:
    first_payload, first_key = _pattern(_receive_state())
    second_payload, second_key = _pattern(
        _receive_state(permuted_opponents=True)
    )

    assert first_key == second_key
    assert first_payload == second_payload
    encoded = json.dumps(first_payload, ensure_ascii=False)
    assert "my_hand" not in encoded
    assert "estimated_current_hands" not in encoded
    assert "joint_hand_inference" not in encoded


def test_equivalent_middle_singletons_share_one_pattern() -> None:
    silver_payload, silver_key = _pattern(_receive_state("11234678"))
    gold_payload, gold_key = _pattern(_receive_state("11235678"))

    assert silver_key == gold_key
    assert silver_payload["hand"] == gold_payload["hand"]


def test_rotated_seats_share_one_pattern() -> None:
    first_payload, first_key = _pattern(_receive_state())
    rotated = GoitaState(
        hands={
            "A": list("11123459"),
            "B": list("11234567"),
            "C": list("11234678"),
            "D": list("11122345"),
        },
        dealer="D",
    )
    rotated.phase = "receive"
    rotated.turn = "C"
    rotated.attacker = "D"
    rotated.current_attack = "2"
    second_payload, second_key = _pattern(rotated, "C")

    assert first_key == second_key
    assert first_payload == second_payload


def test_royal_and_shi_structure_changes_the_pattern() -> None:
    _royal_payload, royal_key = _pattern(_receive_state("11234678"))
    _no_royal_payload, no_royal_key = _pattern(_receive_state("11123467"))

    assert royal_key != no_royal_key


def _search_result(
    action=("receive", "2", None),
    *,
    depth: int = 7,
    agreement: float = 0.75,
    confidence: float = 0.65,
) -> TimedSearchResult:
    return TimedSearchResult(
        action=action,
        depth=depth,
        samples=8,
        nodes=100,
        elapsed_seconds=1.0,
        value=120.0,
        margin=80.0,
        agreement=agreement,
        decisive=True,
        information_set=True,
        candidate_count=8,
        information_confidence=confidence,
    )


def test_adopted_deep_search_is_aggregated() -> None:
    reset_generic_response_patterns()
    state = _receive_state()
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent._ensure_trackers(state)
    actions = state.legal_actions("A")
    baseline = ("pass", None, None)
    result = _search_result()

    assert agent._record_generic_response_search_result(
        state,
        "A",
        actions,
        baseline,
        result.action,
        result,
        source="default",
    ) is True
    snapshot = generic_response_pattern_snapshot()
    assert snapshot["recorded"] == 1
    assert snapshot["pattern_count"] == 1
    assert snapshot["action_counts"]["receive_same"] == 1


def test_shallow_uncertain_or_unadopted_search_is_rejected() -> None:
    reset_generic_response_patterns()
    state = _receive_state()
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent._ensure_trackers(state)
    actions = state.legal_actions("A")
    baseline = ("pass", None, None)

    for result, selected in (
        (_search_result(depth=4), ("receive", "2", None)),
        (_search_result(agreement=0.59), ("receive", "2", None)),
        (_search_result(confidence=0.44), ("receive", "2", None)),
        (_search_result(), baseline),
    ):
        assert agent._record_generic_response_search_result(
            state,
            "A",
            actions,
            baseline,
            selected,
            result,
            source="default",
        ) is False

    snapshot = generic_response_pattern_snapshot()
    assert snapshot["recorded"] == 0
    assert snapshot["considered"] == 4
    assert snapshot["rejection_counts"] == {
        "depth": 1,
        "agreement": 1,
        "confidence": 1,
        "not_adopted": 1,
    }


def test_shadow_comparison_observes_but_does_not_change_the_action() -> None:
    reset_generic_response_patterns()
    state = _receive_state()
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent._ensure_trackers(state)
    actions = state.legal_actions("A")
    baseline = ("pass", None, None)
    selected = ("receive", "2", None)
    result = _search_result(action=selected)

    for _index in range(5):
        assert agent._record_generic_response_search_result(
            state,
            "A",
            actions,
            baseline,
            selected,
            result,
            source="default",
        ) is True

    hand_before = list(state.hands["A"])
    shadow = agent._compare_generic_response_shadow(
        state,
        "A",
        actions,
        baseline,
        selected,
    )
    priority = agent._generic_response_priority_action(
        state,
        "A",
        actions,
        baseline,
    )

    assert shadow["status"] == "match"
    assert shadow["recommended_action"] == "receive_same"
    assert priority == selected
    assert list(state.hands["A"]) == hand_before
    assert state.phase == "receive"
    assert selected == ("receive", "2", None)
    snapshot = generic_response_pattern_snapshot()
    assert snapshot["priority_queries"] == 1
    assert snapshot["priority_hits"] == 1


def test_medium_pattern_can_prioritize_search_without_detailed_match() -> None:
    reset_generic_response_patterns()
    state = _receive_state()
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent._ensure_trackers(state)
    actions = state.legal_actions("A")
    baseline = ("pass", None, None)
    detailed = agent._generic_response_pattern_payload(
        state,
        "A",
        actions,
        baseline,
    )
    store = generic_response_pattern_store()
    for index in range(10):
        variant = {**detailed, "test_detailed_variant": index}
        store.record(
            pattern_key=_digest_payload(variant),
            features=variant,
            action_label="receive_same",
            followup_label="kyosha_pair",
            source="default",
            depth=7,
            agreement=0.80,
            confidence=0.70,
            margin=100.0,
        )

    priority = agent._generic_response_priority_action(
        state,
        "A",
        actions,
        baseline,
    )
    snapshot = generic_response_pattern_snapshot()
    assert priority == ("receive", "2", None)
    assert agent.last_generic_response_priority["granularity"] == "medium"
    assert snapshot["priority_granularity_counts"] == {"medium": 1}


if __name__ == "__main__":
    test_rule_based_agent_uses_generic_response_pattern_mixin()
    test_pattern_does_not_read_or_encode_opponent_hands()
    test_equivalent_middle_singletons_share_one_pattern()
    test_rotated_seats_share_one_pattern()
    test_royal_and_shi_structure_changes_the_pattern()
    test_adopted_deep_search_is_aggregated()
    test_shallow_uncertain_or_unadopted_search_is_rejected()
    test_shadow_comparison_observes_but_does_not_change_the_action()
    test_medium_pattern_can_prioritize_search_without_detailed_match()
    print("GENERIC_RESPONSE_PATTERN_TEST_OK")

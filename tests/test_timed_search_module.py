from __future__ import annotations

from goita_ai2.current_ai.timed_search import TimedSearchMixin
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


def _initial_state(*, permuted: bool = False) -> GoitaState:
    other_hands = {
        "B": list("11122334"),
        "C": list("11244556"),
        "D": list("11356789"),
    }
    if permuted:
        other_hands = {
            "B": list("11356789"),
            "C": list("11122334"),
            "D": list("11244556"),
        }
    return GoitaState(
        hands={"A": list("11123457"), **other_hands},
        dealer="A",
    )


def _sample_key(states) -> tuple:
    return tuple(
        tuple(
            (seat, tuple(sorted(state.hands[seat])))
            for seat in "BCD"
        )
        for state in states
    )


def test_rule_based_agent_uses_timed_search_mixin() -> None:
    assert issubclass(RuleBasedAgent, TimedSearchMixin)


def test_hidden_hand_sampling_does_not_read_actual_opponent_hands() -> None:
    first_state = _initial_state()
    second_state = _initial_state(permuted=True)
    first_agent = RuleBasedAgent()
    second_agent = RuleBasedAgent()
    first_agent.bind_player("A")
    second_agent.bind_player("A")
    first_agent._ensure_trackers(first_state)
    second_agent._ensure_trackers(second_state)

    first_samples = first_agent._timed_search_sample_states(
        first_state,
        "A",
        first_agent._track[id(first_state)],
        6,
    )
    second_samples = second_agent._timed_search_sample_states(
        second_state,
        "A",
        second_agent._track[id(second_state)],
        6,
    )

    assert len(first_samples) == 6
    assert len(second_samples) == 6
    assert _sample_key(first_samples) == _sample_key(second_samples)


def test_time_limited_search_returns_a_completed_legal_result() -> None:
    state = _initial_state()
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent.TIME_SEARCH_MAX_SECONDS = 0.4
    agent.TIME_SEARCH_SAMPLE_COUNT = 4
    agent.TIME_SEARCH_MAX_DEPTH = 5
    agent.TIME_SEARCH_MAX_NODES = 20_000
    agent._ensure_trackers(state)
    actions = state.legal_actions("A")
    baseline = agent._select_rule_based_action(state, "A", actions)

    result = agent._time_limited_search_action(
        state,
        "A",
        actions,
        baseline,
    )

    assert result is not None
    assert result.action in actions
    assert result.depth in (1, 3, 5)
    assert result.samples == 4
    assert result.nodes > 0
    assert result.elapsed_seconds <= 0.8


def test_select_action_records_search_without_changing_public_state() -> None:
    state = _initial_state()
    hands_before = {seat: list(hand) for seat, hand in state.hands.items()}
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent.TIME_SEARCH_MAX_SECONDS = 0.4
    agent.TIME_SEARCH_SAMPLE_COUNT = 4
    agent.TIME_SEARCH_MAX_DEPTH = 5
    agent.TIME_SEARCH_MAX_NODES = 20_000

    action = agent.select_action(state, "A", state.legal_actions("A"))
    search = agent._track[id(state)].get("last_time_limited_search")

    assert action in state.legal_actions("A")
    assert search is not None
    assert search["depth"] in (1, 3, 5)
    assert state.hands == hands_before


if __name__ == "__main__":
    test_rule_based_agent_uses_timed_search_mixin()
    test_hidden_hand_sampling_does_not_read_actual_opponent_hands()
    test_time_limited_search_returns_a_completed_legal_result()
    test_select_action_records_search_without_changing_public_state()
    print("TIMED_SEARCH_MODULE_TEST_OK")

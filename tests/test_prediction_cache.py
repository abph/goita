from __future__ import annotations

import copy

from goita_ai2.current_ai.prediction_cache import (
    PredictionSample,
    PredictionSampleCache,
    clear_prediction_sample_cache,
    prediction_sample_cache_snapshot,
)
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


HANDS_ONE = {
    "A": list("11123457"),
    "B": list("11122334"),
    "C": list("11244556"),
    "D": list("11356789"),
}
HANDS_TWO = {
    "A": list("11123457"),
    "B": list("11356789"),
    "C": list("11122334"),
    "D": list("11244556"),
}


def _sample(piece: str) -> PredictionSample:
    return PredictionSample(
        opponent_hands=(("B", (piece,)),),
        opponent_hidden=(("B", tuple()),),
        opponent_had_both_kings=(("B", False),),
        last_block=None,
    )


def _sample_key(states) -> tuple:
    return tuple(
        tuple(
            (seat, tuple(sorted(state.hands[seat])))
            for seat in "BCD"
        )
        for state in states
    )


def test_prediction_cache_requires_enough_samples_and_keeps_larger_entry() -> None:
    cache = PredictionSampleCache(max_entries=2, ttl_seconds=30.0)
    owner, _event = cache.claim("position")
    assert owner is True
    cache.finish("position", (_sample("1"), _sample("2")))

    assert cache.get("position", 2) is not None
    assert cache.get("position", 3) is None

    owner, _event = cache.claim("position")
    assert owner is True
    cache.finish(
        "position",
        tuple(_sample(piece) for piece in ("1", "2", "3", "4")),
    )
    assert len(cache.get("position", 4) or ()) == 4
    snapshot = cache.snapshot()
    assert snapshot["partial_misses"] == 1
    assert snapshot["replacements"] == 1
    assert snapshot["sample_counts"] == [4]


def test_prediction_cache_key_never_reads_real_opponent_hands() -> None:
    first_state = GoitaState(HANDS_ONE, dealer="A")
    second_state = GoitaState(HANDS_TWO, dealer="A")
    first = RuleBasedAgent()
    second = RuleBasedAgent()
    first.bind_player("A")
    second.bind_player("A")
    first._ensure_trackers(first_state)
    second._ensure_trackers(second_state)

    first_key = first._timed_search_prediction_cache_key(
        first_state,
        "A",
        first._track[id(first_state)],
    )
    second_key = second._timed_search_prediction_cache_key(
        second_state,
        "A",
        second._track[id(second_state)],
    )

    assert first_key == second_key


def test_prediction_samples_are_reused_and_materialized_independently() -> None:
    clear_prediction_sample_cache()
    first_state = GoitaState(HANDS_ONE, dealer="A")
    second_state = GoitaState(HANDS_TWO, dealer="A")
    first = RuleBasedAgent()
    second = RuleBasedAgent()
    first.bind_player("A")
    second.bind_player("A")
    first._ensure_trackers(first_state)
    second._ensure_trackers(second_state)

    generated = first._timed_search_sample_states(
        first_state,
        "A",
        first._track[id(first_state)],
        6,
    )
    reused = second._timed_search_sample_states(
        second_state,
        "A",
        second._track[id(second_state)],
        6,
    )

    assert first.last_prediction_cache_hit is False
    assert second.last_prediction_cache_hit is True
    assert _sample_key(generated) == _sample_key(reused)
    reused[0].hands["B"].clear()
    reused_again = second._timed_search_sample_states(
        second_state,
        "A",
        second._track[id(second_state)],
        6,
    )
    assert reused_again[0].hands["B"]
    snapshot = prediction_sample_cache_snapshot()
    assert snapshot["hits"] >= 2
    assert snapshot["reused_samples"] >= 12


def test_prediction_cache_extends_after_a_partial_miss() -> None:
    clear_prediction_sample_cache()
    state = GoitaState(HANDS_ONE, dealer="A")
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent._ensure_trackers(state)
    tracker = agent._track[id(state)]

    assert len(agent._timed_search_sample_states(state, "A", tracker, 2)) == 2
    assert len(agent._timed_search_sample_states(state, "A", tracker, 6)) == 6
    assert agent.last_prediction_cache_hit is True
    assert agent.last_prediction_cache_samples == 2
    assert len(agent._timed_search_sample_states(state, "A", tracker, 6)) == 6
    assert agent.last_prediction_cache_hit is True
    snapshot = prediction_sample_cache_snapshot()
    assert snapshot["partial_misses"] == 1
    assert snapshot["partial_hits"] == 1
    assert snapshot["replacements"] == 1
    assert snapshot["generated_samples"] == 6
    assert snapshot["reused_samples"] == 8
    assert snapshot["reuse_rate"] > 0.0
    assert snapshot["sample_counts"] == [6]


def test_prediction_samples_roll_forward_across_public_actions() -> None:
    clear_prediction_sample_cache()
    state = GoitaState(HANDS_ONE, dealer="A")
    state.hands["D"].remove("1")
    state.phase = "receive"
    state.turn = "A"
    state.attacker = "D"
    state.current_attack = "1"
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent._ensure_trackers(state)
    tracker = agent._track[id(state)]
    # The synthetic state starts after D's visible attack, so remove that
    # already-played piece from the unknown pool normally built by tracking.
    tracker["unknown_piece_pool"]["1"] -= 1

    initial = agent._timed_search_sample_states(state, "A", tracker, 8)
    assert len(initial) == 8
    received = ("receive", "1", None)
    assert received in state.legal_actions("A")
    state.apply_receive("A", "1")
    agent.on_public_action(state, "A", received)

    reused = agent._timed_search_sample_states(
        state,
        "A",
        agent._track[id(state)],
        8,
    )
    assert len(reused) == 8
    assert agent.last_prediction_cache_hit is True
    assert agent.last_prediction_cache_samples == 8
    snapshot = prediction_sample_cache_snapshot()
    assert snapshot["rollforward_stores"] >= 1
    assert snapshot["rollforward_samples"] >= 8


def test_rollforward_ignores_an_opponents_actual_hidden_block() -> None:
    clear_prediction_sample_cache()
    first_state = GoitaState(HANDS_ONE, dealer="B")
    second_state = GoitaState(HANDS_ONE, dealer="B")
    first = RuleBasedAgent()
    second = RuleBasedAgent()
    first.bind_player("A")
    second.bind_player("A")
    first._ensure_trackers(first_state)
    second._ensure_trackers(second_state)

    inferred_root = GoitaState(HANDS_ONE, dealer="B")
    first._prediction_rollforward_states = [copy.deepcopy(inferred_root)]
    second._prediction_rollforward_states = [copy.deepcopy(inferred_root)]
    first_action = ("attack_after_block", "2", "1")
    second_action = ("attack_after_block", "3", "1")
    assert first_action in first_state.legal_actions("B")
    assert second_action in second_state.legal_actions("B")

    first_state.apply_attack_after_block("B", "2", "1")
    second_state.apply_attack_after_block("B", "3", "1")
    first.on_public_action(first_state, "B", first_action)
    second.on_public_action(second_state, "B", second_action)

    assert first._prediction_rollforward_key == second._prediction_rollforward_key
    assert _sample_key(first._prediction_rollforward_states) == _sample_key(
        second._prediction_rollforward_states
    )


def test_background_workers_cannot_roll_predictions_into_shared_cache() -> None:
    state = GoitaState(HANDS_ONE, dealer="A")
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent._ensure_trackers(state)
    agent._prediction_rollforward_states = [copy.deepcopy(state)]

    worker, future_state = agent._background_search_clone(state)
    branch_worker, _branch_state = agent._background_clone_branch(
        worker,
        future_state,
    )
    assert worker._prediction_cache_rollforward_enabled is False
    assert worker._prediction_rollforward_states == []
    assert branch_worker._prediction_cache_rollforward_enabled is False
    assert branch_worker._prediction_rollforward_states == []


if __name__ == "__main__":
    test_prediction_cache_requires_enough_samples_and_keeps_larger_entry()
    test_prediction_cache_key_never_reads_real_opponent_hands()
    test_prediction_samples_are_reused_and_materialized_independently()
    test_prediction_cache_extends_after_a_partial_miss()
    test_prediction_samples_roll_forward_across_public_actions()
    test_rollforward_ignores_an_opponents_actual_hidden_block()
    test_background_workers_cannot_roll_predictions_into_shared_cache()
    print("PREDICTION_CACHE_TEST_OK")

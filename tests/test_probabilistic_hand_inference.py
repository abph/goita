"""Tests the stage-one schema and stage-two public legal-deal prior."""

from __future__ import annotations

import copy
from collections import Counter

from goita_ai2.constants import PIECE_TOTALS
from goita_ai2.current_ai.probabilistic_hand_inference import (
    PosteriorProbabilisticHandInference,
    ProbabilisticHandInferenceCache,
    ProbabilisticHandInferenceMixin,
)
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


MY_HAND = list("12334567")
DEALS = (
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


def _state(deal_index: int = 0) -> GoitaState:
    state = GoitaState(
        hands={"A": list(MY_HAND), **copy.deepcopy(DEALS[deal_index])},
        dealer="A",
    )
    assert Counter(
        piece
        for hand in state.hands.values()
        for piece in hand
    ) == Counter(PIECE_TOTALS)
    return state


def _model(deal_index: int = 0, samples: int = 192):
    state = _state(deal_index)
    agent = RuleBasedAgent()
    agent.bind_player("A")
    model = agent._initial_probabilistic_hand_inference(
        state,
        "A",
        sample_count=samples,
    )
    return agent, state, model


def _probability_sum(items) -> float:
    return sum(item.probability for item in items)


def _apply_public_action(agent, state, player, action) -> None:
    action_type, block, attack = action
    if action_type == "pass":
        state.apply_pass(player)
    elif action_type == "receive":
        state.apply_receive(player, block)
    elif action_type == "attack":
        state.apply_attack(player, attack)
    elif action_type == "attack_after_block":
        state.apply_attack_after_block(player, block, attack)
    else:
        raise AssertionError(f"unexpected action: {action}")
    agent.on_public_action(state, player, action)


def _public_action_state(deal_index: int = 0):
    state = _state(deal_index)
    state.dealer = "B"
    state.turn = "B"
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent._ensure_trackers(state)
    _apply_public_action(
        agent,
        state,
        "B",
        ("attack_after_block", "1" if deal_index == 0 else "2", "3"),
    )
    return agent, state


def _probability_at_least(player_model, piece: str, minimum: int) -> float:
    return sum(
        item.probability
        for item in player_model.piece(piece).original_count_distribution
        if item.count >= minimum
    )


def test_rule_based_agent_exposes_probabilistic_hand_inference() -> None:
    assert issubclass(RuleBasedAgent, ProbabilisticHandInferenceMixin)


def test_initial_prior_has_all_stage_one_outputs() -> None:
    _agent, _state_value, model = _model(samples=128)

    assert model.observer == "A"
    assert model.requested_samples == 128
    assert model.accepted_samples == 128
    assert model.rejected_samples == 0
    assert tuple(item.seat for item in model.players) == ("B", "C", "D")
    assert model.prior_sources == (
        "uniform_public_unknown_pool_deals",
        "natural_multiset_structure_frequency",
        "relative_hand_rank_table_classification",
    )
    for player in model.players:
        assert len(player.pieces) == 9
        assert player.top_hands
        assert player.absolute_rank_distribution
        assert player.relative_rank_distribution
        assert player.structure_distribution
        assert abs(_probability_sum(player.absolute_rank_distribution) - 1.0) < 1e-9
        assert abs(_probability_sum(player.relative_rank_distribution) - 1.0) < 1e-9
        assert abs(_probability_sum(player.structure_distribution) - 1.0) < 1e-9
        for piece in player.pieces:
            assert abs(_probability_sum(piece.current_count_distribution) - 1.0) < 1e-9
            assert abs(_probability_sum(piece.original_count_distribution) - 1.0) < 1e-9
            assert 0.0 <= piece.current_holding_probability <= 1.0
            assert 0.0 <= piece.original_holding_probability <= 1.0


def test_every_candidate_obeys_slots_and_the_public_unknown_pool() -> None:
    agent, state, model = _model(samples=160)
    tracker = agent._track[id(state)]
    expected_pool = Counter(tracker["unknown_piece_pool"])
    hidden_counts = tracker["hidden_block_counts"]

    assert sum(candidate.observations for candidate in model.candidates) == 160
    assert abs(sum(
        candidate.prior_probability for candidate in model.candidates
    ) - 1.0) < 1e-9
    for candidate in model.candidates:
        hands = dict(candidate.prediction.opponent_hands)
        hidden = dict(candidate.prediction.opponent_hidden)
        combined = Counter()
        for seat in ("B", "C", "D"):
            assert len(hands[seat]) == len(state.hands[seat])
            assert len(hidden[seat]) == int(hidden_counts.get(seat, 0))
            combined.update(hands[seat])
            combined.update(hidden[seat])
        assert combined == expected_pool


def test_piece_expectations_conserve_every_copy_in_the_deck() -> None:
    _agent, state, model = _model(samples=256)
    for piece, total in PIECE_TOTALS.items():
        expected_other_count = 0.0
        for seat in ("B", "C", "D"):
            distribution = model.player(seat).piece(piece).current_count_distribution
            expected_other_count += sum(
                item.count * item.probability
                for item in distribution
            )
        assert abs(
            expected_other_count + state.hands["A"].count(piece) - total
        ) < 1e-9


def test_initial_prior_does_not_read_real_opponent_hands() -> None:
    _first_agent, _first_state, first = _model(0, samples=128)
    _second_agent, _second_state, second = _model(1, samples=128)

    assert [candidate.as_dict() for candidate in first.candidates] == [
        candidate.as_dict() for candidate in second.candidates
    ]
    assert [player.as_dict() for player in first.players] == [
        player.as_dict() for player in second.players
    ]


def test_soft_action_estimates_do_not_change_the_stage_two_prior() -> None:
    state = _state(0)
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent._ensure_trackers(state)
    tracker = agent._track[id(state)]
    first = agent._initial_probabilistic_hand_inference(
        state,
        "A",
        sample_count=96,
    )

    tracker["estimated_current_hands"]["B"]["1"].update({
        "min": 0,
        "max": 8,
        "expected": 7.5,
        "confidence": 1.0,
        "map_count": 8,
    })
    second = agent._initial_probabilistic_hand_inference(
        state,
        "A",
        sample_count=96,
    )

    assert [candidate.as_dict() for candidate in first.candidates] == [
        candidate.as_dict() for candidate in second.candidates
    ]


def test_requested_sample_count_is_bounded_by_configuration() -> None:
    state = _state(0)
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent.PROBABILISTIC_HAND_INITIAL_MAX_SAMPLES = 24

    model = agent._initial_probabilistic_hand_inference(
        state,
        "A",
        sample_count=1000,
    )

    assert model.requested_samples == 24
    assert model.accepted_samples == 24


def test_first_attack_weights_multiple_piece_hands_above_the_prior() -> None:
    agent, state = _public_action_state(0)
    prior = agent._initial_probabilistic_hand_inference(
        state,
        "A",
        sample_count=512,
    )
    posterior = agent._posterior_probabilistic_hand_inference(
        state,
        "A",
        sample_count=512,
    )

    assert isinstance(posterior, PosteriorProbabilisticHandInference)
    assert _probability_at_least(
        posterior.player("B"), "3", 2
    ) > _probability_at_least(prior.player("B"), "3", 2)
    assert any(
        "B:first_attack_middle_repeat" in candidate.evidence
        for candidate in posterior.candidates
    )


def test_pass_lowers_probability_of_currently_holding_the_attack_piece() -> None:
    agent, state = _public_action_state(0)
    _apply_public_action(agent, state, "C", ("pass", None, None))
    prior = agent._initial_probabilistic_hand_inference(
        state,
        "A",
        sample_count=512,
    )
    posterior = agent._posterior_probabilistic_hand_inference(
        state,
        "A",
        sample_count=512,
    )

    assert (
        posterior.player("C").piece("3").current_holding_probability
        < prior.player("C").piece("3").current_holding_probability
    )
    assert any(
        "C:pass_3_1" in candidate.evidence
        for candidate in posterior.candidates
    )


def test_posterior_is_one_joint_three_player_deck_distribution() -> None:
    agent, state = _public_action_state(0)
    _apply_public_action(agent, state, "C", ("pass", None, None))
    posterior = agent._posterior_probabilistic_hand_inference(
        state,
        "A",
        sample_count=384,
    )
    tracker = agent._track[id(state)]
    expected_pool = Counter(tracker["unknown_piece_pool"])

    assert abs(sum(
        candidate.probability for candidate in posterior.candidates
    ) - 1.0) < 1e-9
    for candidate in posterior.candidates:
        combined = Counter()
        for _seat, hand in candidate.prediction.opponent_hands:
            combined.update(hand)
        for _seat, hidden in candidate.prediction.opponent_hidden:
            combined.update(hidden)
        assert combined == expected_pool

    for piece, pool_count in expected_pool.items():
        expected_count = sum(
            sum(
                count.probability * count.count
                for count in posterior.player(seat).piece(piece).current_count_distribution
            )
            + sum(
                count.probability * count.count
                for count in posterior.player(seat).piece(piece).original_count_distribution
            )
            - sum(
                count.probability * count.count
                for count in posterior.player(seat).piece(piece).current_count_distribution
            )
            - agent._observed_piece_count_for_player(tracker, seat, piece)
            for seat in ("B", "C", "D")
        )
        assert abs(expected_count - pool_count) < 1e-9


def test_posterior_outputs_top_hand_rank_counts_and_confidence() -> None:
    agent, state = _public_action_state(0)
    _apply_public_action(agent, state, "C", ("pass", None, None))
    posterior = agent._posterior_probabilistic_hand_inference(
        state,
        "A",
        sample_count=192,
    )
    exported = posterior.as_dict()

    assert exported["stage"] == "action_weighted_joint_posterior"
    assert exported["action_evidence_count"] > 0
    assert exported["evidence_sources"]
    assert 0.0 <= exported["confidence"] <= 1.0
    assert 0.0 <= exported["normalized_entropy"] <= 1.0
    for player in posterior.players:
        assert player.most_likely_hand is not None
        assert player.top_hands
        assert player.absolute_rank_distribution
        assert player.relative_rank_distribution
        assert 0.0 <= player.confidence <= 1.0


def test_activated_posterior_refreshes_after_each_public_action() -> None:
    agent, state = _public_action_state(0)
    first = agent._posterior_probabilistic_hand_inference(
        state,
        "A",
        sample_count=96,
    )
    first_revision = first.evidence_revision
    _apply_public_action(agent, state, "C", ("pass", None, None))
    tracker = agent._track[id(state)]
    refreshed = tracker["probabilistic_hand_inference"]

    assert isinstance(refreshed, PosteriorProbabilisticHandInference)
    assert refreshed.evidence_revision > first_revision
    assert refreshed.key.digest != first.key.digest
    assert tracker["probabilistic_hand_inference_error"] is None
    assert any("C:pass_3_1" in item for item in refreshed.evidence_sources)


def test_action_weighted_posterior_does_not_read_real_opponent_hands() -> None:
    first_agent, first_state = _public_action_state(0)
    second_agent, second_state = _public_action_state(1)
    _apply_public_action(first_agent, first_state, "C", ("pass", None, None))
    _apply_public_action(second_agent, second_state, "C", ("pass", None, None))
    first = first_agent._posterior_probabilistic_hand_inference(
        first_state,
        "A",
        sample_count=160,
    )
    second = second_agent._posterior_probabilistic_hand_inference(
        second_state,
        "A",
        sample_count=160,
    )

    assert [candidate.as_dict() for candidate in first.candidates] == [
        candidate.as_dict() for candidate in second.candidates
    ]
    assert [player.as_dict() for player in first.players] == [
        player.as_dict() for player in second.players
    ]


def test_posterior_cache_reuses_the_same_public_information_set() -> None:
    agent, state = _public_action_state(0)
    agent.clear_probabilistic_hand_inference_cache()

    first = agent._posterior_probabilistic_hand_inference(
        state,
        "A",
        sample_count=64,
    )
    second = agent._posterior_probabilistic_hand_inference(
        state,
        "A",
        sample_count=64,
    )

    snapshot = agent.probabilistic_hand_inference_cache_snapshot()
    assert first is second
    assert snapshot["hits"] == 1
    assert snapshot["misses"] == 1


def test_posterior_candidate_retention_is_bounded_and_renormalized() -> None:
    agent, state = _public_action_state(0)
    posterior = agent._posterior_probabilistic_hand_inference(
        state,
        "A",
        sample_count=256,
        max_candidates=9,
        minimum_probability=0.0,
        use_cache=False,
    )

    assert len(posterior.candidates) <= 9
    assert abs(sum(item.probability for item in posterior.candidates) - 1.0) < 1e-9
    assert 0.0 < posterior.retained_probability_mass <= 1.0


def test_posterior_generation_honors_a_short_time_budget() -> None:
    agent, state = _public_action_state(0)
    posterior = agent._posterior_probabilistic_hand_inference(
        state,
        "A",
        sample_count=2048,
        max_seconds=0.001,
        max_candidates=8,
        use_cache=False,
    )

    assert posterior.timed_out
    assert 1 <= posterior.accepted_samples <= 2048
    assert len(posterior.candidates) <= 8


def test_probabilistic_cache_is_bounded() -> None:
    cache = ProbabilisticHandInferenceCache(1)
    agent, state = _public_action_state(0)
    first = agent._posterior_probabilistic_hand_inference(
        state,
        "A",
        sample_count=16,
        use_cache=False,
    )
    cache.put(("first",), first)
    cache.put(("second",), first)

    assert cache.snapshot()["size"] == 1
    assert cache.snapshot()["evictions"] == 1


if __name__ == "__main__":
    test_rule_based_agent_exposes_probabilistic_hand_inference()
    test_initial_prior_has_all_stage_one_outputs()
    test_every_candidate_obeys_slots_and_the_public_unknown_pool()
    test_piece_expectations_conserve_every_copy_in_the_deck()
    test_initial_prior_does_not_read_real_opponent_hands()
    test_soft_action_estimates_do_not_change_the_stage_two_prior()
    test_requested_sample_count_is_bounded_by_configuration()
    test_first_attack_weights_multiple_piece_hands_above_the_prior()
    test_pass_lowers_probability_of_currently_holding_the_attack_piece()
    test_posterior_is_one_joint_three_player_deck_distribution()
    test_posterior_outputs_top_hand_rank_counts_and_confidence()
    test_activated_posterior_refreshes_after_each_public_action()
    test_action_weighted_posterior_does_not_read_real_opponent_hands()
    test_posterior_cache_reuses_the_same_public_information_set()
    test_posterior_candidate_retention_is_bounded_and_renormalized()
    test_posterior_generation_honors_a_short_time_budget()
    test_probabilistic_cache_is_bounded()
    print("PROBABILISTIC_HAND_INFERENCE_TEST_OK")

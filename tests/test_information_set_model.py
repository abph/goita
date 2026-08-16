"""Tests public keys, grouped deals, and candidate belief weights."""

from __future__ import annotations

import copy
from collections import Counter

from goita_ai2.constants import PIECE_TOTALS
from goita_ai2.current_ai.information_set import InformationSetMixin
from goita_ai2.current_ai.information_set_action_model import (
    InformationSetActionModelMixin,
)
from goita_ai2.current_ai.information_set_policy import InformationSetPolicy
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


def _agent_and_tracker(state: GoitaState):
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent._ensure_trackers(state)
    return agent, agent._track[id(state)]


def test_rule_based_agent_exposes_information_set_model() -> None:
    assert issubclass(RuleBasedAgent, InformationSetMixin)
    assert issubclass(RuleBasedAgent, InformationSetActionModelMixin)


def test_public_key_excludes_real_opponent_hands() -> None:
    first = _state(0)
    second = _state(1)
    first_agent, first_tracker = _agent_and_tracker(first)
    second_agent, second_tracker = _agent_and_tracker(second)

    first_key = first_agent._information_set_key(first, "A", first_tracker)
    second_key = second_agent._information_set_key(second, "A", second_tracker)

    assert first_key.digest == second_key.digest
    assert first_key.hidden_candidate_slots == 24
    second.team_score["AC"] = 10
    assert first_key.digest != second_agent._information_set_key(
        second,
        "A",
        second_tracker,
    ).digest


def test_candidate_deals_are_grouped_under_one_public_key() -> None:
    root = _state(0)
    other = _state(1)
    agent, tracker = _agent_and_tracker(root)

    information_set = agent._build_information_set(
        root,
        "A",
        tracker,
        [copy.deepcopy(root), copy.deepcopy(root), other],
    )

    assert len(information_set.candidates) == 2
    assert information_set.total_observations == 3
    assert sorted(
        candidate.observations for candidate in information_set.candidates
    ) == [1, 2]
    assert abs(sum(
        candidate.probability for candidate in information_set.candidates
    ) - 1.0) < 1e-9
    assert 1.0 <= information_set.effective_candidate_count <= 2.0


def test_candidate_probability_and_confidence_follow_public_estimates() -> None:
    root = _state(0)
    other = _state(1)
    agent, tracker = _agent_and_tracker(root)
    estimate = tracker["estimated_current_hands"]["B"]["1"]
    estimate.update({
        "min": 0,
        "max": 8,
        "expected": 3.0,
        "confidence": 1.0,
    })
    tracker["joint_hand_inference"] = {"feasible": False}

    information_set = agent._build_information_set(
        root,
        "A",
        tracker,
        [root, other],
    )
    by_b_shi = {
        dict(candidate.prediction.opponent_hands)["B"].count("1"): candidate
        for candidate in information_set.candidates
    }

    assert by_b_shi[3].probability > by_b_shi[1].probability
    assert all(
        0.0 <= candidate.confidence <= 1.0
        for candidate in information_set.candidates
    )
    assert 0.0 <= information_set.confidence <= 1.0
    assert 0.0 <= information_set.normalized_entropy <= 1.0


def test_generated_prediction_samples_build_one_information_set() -> None:
    root = GoitaState(
        hands={
            "A": list("11123457"),
            "B": list("11122334"),
            "C": list("11244556"),
            "D": list("11356789"),
        },
        dealer="A",
    )
    agent, tracker = _agent_and_tracker(root)
    agent.TIME_SEARCH_PREDICTION_CACHE_ENABLED = False
    samples = agent._timed_search_sample_states(root, "A", tracker, 8)

    information_set = agent._build_information_set(
        root,
        "A",
        tracker,
        samples,
    )

    assert len(samples) == 8
    assert information_set.total_observations == 8
    assert information_set.key == agent._information_set_key(root, "A", tracker)
    assert abs(sum(
        candidate.probability for candidate in information_set.candidates
    ) - 1.0) < 1e-9


def test_self_future_move_is_shared_across_indistinguishable_deals() -> None:
    root = _state(0)
    other = _state(1)
    agent, tracker = _agent_and_tracker(root)
    information_set = agent._build_information_set(
        root,
        "A",
        tracker,
        [root, other],
    )
    policy = InformationSetPolicy()

    decisions = agent._information_set_self_future_decisions(
        root,
        "A",
        information_set,
        policy=policy,
    )
    repeated = agent._information_set_self_future_decisions(
        root,
        "A",
        information_set,
        policy=policy,
    )

    assert len(decisions) == 1
    assert decisions[0].candidate_count == 2
    assert decisions[0].action == repeated[0].action
    assert repeated[0].reused is True
    assert len(policy) == 1


def test_ally_future_move_uses_ally_information_not_enemy_hands() -> None:
    ally_hand = list("12334567")
    first = GoitaState(
        hands={
            "A": list("11123457"),
            "B": list("11962135"),
            "C": ally_hand,
            "D": list("44511218"),
        },
        dealer="C",
    )
    second = GoitaState(
        hands={
            "A": list("11123457"),
            "B": list("44511218"),
            "C": ally_hand,
            "D": list("11962135"),
        },
        dealer="C",
    )
    agent, tracker = _agent_and_tracker(first)
    information_set = agent._build_information_set(
        first,
        "A",
        tracker,
        [first, second],
    )

    decisions = agent._information_set_ally_future_decisions(
        first,
        "A",
        information_set,
    )

    assert len(decisions) == 1
    assert decisions[0].actor == "C"
    assert decisions[0].role == "ally"
    assert decisions[0].candidate_count == 2
    assert decisions[0].action in first.legal_actions("C")
    assert decisions[0].action in second.legal_actions("C")


def test_contradictory_candidate_is_removed_before_weighting() -> None:
    root = _state(0)
    contradictory = copy.deepcopy(root)
    contradictory.hands["B"][0] = "9"
    agent, tracker = _agent_and_tracker(root)

    information_set = agent._build_information_set(
        root,
        "A",
        tracker,
        [root, contradictory],
    )

    assert len(information_set.candidates) == 1
    assert information_set.total_observations == 1
    assert information_set.rejected_observations == 1
    assert information_set.candidates[0].probability == 1.0


def test_enemy_receive_prediction_uses_public_rank_and_known_copy_count() -> None:
    state = _state(0)
    state.apply_attack_after_block("A", "4", "3")
    agent, tracker = _agent_and_tracker(state)
    rank_model = tracker["public_hand_models"]["B"]

    rank_model.update({"estimated_rank": "S", "rank_confidence": 1.0})
    strong_receive = agent._information_set_action_evaluation(
        state,
        state,
        "A",
        "B",
        ("receive", "3", None),
        tracker,
    )
    strong_pass = agent._information_set_action_evaluation(
        state,
        state,
        "A",
        "B",
        ("pass", None, None),
        tracker,
    )

    assert strong_receive.role == "enemy"
    assert strong_receive.total_score > strong_pass.total_score
    assert "enemy_rank_same_piece_receive" in dict(strong_receive.adjustments)

    one_copy_state = copy.deepcopy(state)
    one_copy_state.hands["B"].remove("3")
    one_copy_state.hands["B"].append("4")
    rank_model.update({"estimated_rank": "F", "rank_confidence": 1.0})
    weak_receive = agent._information_set_action_evaluation(
        state,
        one_copy_state,
        "A",
        "B",
        ("receive", "3", None),
        tracker,
    )
    weak_pass = agent._information_set_action_evaluation(
        state,
        one_copy_state,
        "A",
        "B",
        ("pass", None, None),
        tracker,
    )

    assert weak_pass.total_score > weak_receive.total_score
    assert "enemy_rank_same_piece_pass" in dict(weak_pass.adjustments)


def test_enemy_shared_decision_records_role_and_public_reasons() -> None:
    state = _state(0)
    state.dealer = "B"
    state.turn = "B"
    agent, tracker = _agent_and_tracker(state)
    information_set = agent._build_information_set(
        state,
        "A",
        tracker,
        [state],
    )

    decisions = agent._information_set_shared_future_decisions(
        state,
        "A",
        "B",
        information_set,
    )

    assert len(decisions) == 1
    assert decisions[0].role == "enemy"
    assert decisions[0].action in state.legal_actions("B")
    assert any(
        "natural_multi_shi_opening" in reason
        for reason, _score in decisions[0].reason_scores
    )


def test_ally_action_prediction_values_shi_signal_and_kakarigotae() -> None:
    state = GoitaState(
        hands={
            "A": list("12345678"),
            "B": list("11122345"),
            "C": list("11123457"),
            "D": list("11234569"),
        },
        dealer="A",
    )
    state.turn = "C"
    state.phase = "attack"
    state.attacker = "C"
    state.current_attack = None
    agent, tracker = _agent_and_tracker(state)
    tracker.update({
        "shi_attack_mode": True,
        "ally_shi_signal": "returned_shi",
        "ally_pending_response_piece": "1",
        "my_past_attacks": {"1"},
    })

    shi = agent._information_set_action_evaluation(
        state,
        state,
        "A",
        "C",
        ("attack", None, "1"),
        tracker,
    )
    other = agent._information_set_action_evaluation(
        state,
        state,
        "A",
        "C",
        ("attack", None, "3"),
        tracker,
    )
    reasons = dict(shi.adjustments)

    assert shi.role == "ally"
    assert shi.total_score > other.total_score
    assert "ally_shi_mode" in reasons
    assert "ally_returned_shi_signal" in reasons
    assert "ally_pending_kakarigotae" in reasons
    assert "ally_answers_my_attack" in reasons


def test_ally_action_prediction_uses_remaining_hand_attack_plan() -> None:
    state = GoitaState(
        hands={
            "A": list("11145678"),
            "B": list("11122345"),
            "C": list("11223345"),
            "D": list("11234569"),
        },
        dealer="A",
    )
    state.turn = "C"
    state.phase = "attack"
    state.attacker = "C"
    state.current_attack = None
    agent, tracker = _agent_and_tracker(state)

    planned = agent._information_set_action_evaluation(
        state,
        state,
        "A",
        "C",
        ("attack", None, "3"),
        tracker,
    )
    later = agent._information_set_action_evaluation(
        state,
        state,
        "A",
        "C",
        ("attack", None, "2"),
        tracker,
    )

    assert dict(planned.adjustments)["hand_attack_plan_next"] == 46.0
    assert dict(later.adjustments)["hand_attack_plan_later"] == 10.0
    assert planned.total_score > later.total_score


if __name__ == "__main__":
    test_rule_based_agent_exposes_information_set_model()
    test_public_key_excludes_real_opponent_hands()
    test_candidate_deals_are_grouped_under_one_public_key()
    test_candidate_probability_and_confidence_follow_public_estimates()
    test_generated_prediction_samples_build_one_information_set()
    test_self_future_move_is_shared_across_indistinguishable_deals()
    test_ally_future_move_uses_ally_information_not_enemy_hands()
    test_contradictory_candidate_is_removed_before_weighting()
    test_enemy_receive_prediction_uses_public_rank_and_known_copy_count()
    test_enemy_shared_decision_records_role_and_public_reasons()
    test_ally_action_prediction_values_shi_signal_and_kakarigotae()
    test_ally_action_prediction_uses_remaining_hand_attack_plan()
    print("INFORMATION_SET_MODEL_TEST_OK")

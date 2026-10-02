"""Tests the exported policy and neural tie-break 強化中AI2 wrapper."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from goita_ai2.current_ai.agent import RuleBasedAgent as CurrentRuleBasedAgent
from goita_ai2.experimental_ai2 import RuleBasedAgent as ExperimentalAI2RuleBasedAgent
from goita_ai2.instruction_case_audit import apply_action
from goita_ai2.neural_policy import NeuralPolicyModel, encode_state, live_state_payload
from goita_ai2.state import GoitaState
from goita_ai2.train_neural_policy import _expand_acceptable_actions


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "goita_ai2" / "experimental_ai2" / "data" / "neural_policy.json"
CORRECTIONS_PATH = ROOT / "goita_ai2" / "experimental_ai2" / "data" / "review_corrections.jsonl"


def _hands() -> dict[str, list[str]]:
    return {
        "A": list("11123557"),
        "B": list("11144478"),
        "C": list("12334456"),
        "D": list("11122369"),
    }


def _three_kyosha_continuation():
    hands = {
        "A": list("31251163"),
        "B": list("71242652"),
        "C": list("44411987"),
        "D": list("31151153"),
    }
    history = [
        ("D", ("attack_after_block", "1", "5")),
        ("A", ("receive", "5", None)),
        ("A", ("attack", None, "3")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("receive", "3", None)),
        ("D", ("attack", None, "5")),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("receive", "9", None)),
        ("C", ("attack", None, "4")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("receive", "4", None)),
        ("B", ("attack", None, "2")),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
    ]
    state = GoitaState(copy.deepcopy(hands), dealer="D")
    agent = ExperimentalAI2RuleBasedAgent()
    agent.bind_player("B")
    for actor, action in history:
        assert action in state.legal_actions(actor)
        apply_action(state, actor, action)
        agent.on_public_action(state, actor, action)
    return state, agent


def _last_kyosha_finish_route():
    hands = {
        "A": list("43716223"),
        "B": list("11755434"),
        "C": list("93511218"),
        "D": list("26511114"),
    }
    history = [
        ("D", ("attack_after_block", "1", "1")),
        ("A", ("receive", "1", None)),
        ("A", ("attack", None, "3")),
        ("B", ("receive", "3", None)),
        ("B", ("attack", None, "4")),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("receive", "4", None)),
        ("A", ("attack", None, "2")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("receive", "2", None)),
        ("D", ("attack", None, "4")),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("receive", "9", None)),
        ("C", ("attack", None, "2")),
        ("D", ("pass", None, None)),
    ]
    state = GoitaState(copy.deepcopy(hands), dealer="D")
    agent = ExperimentalAI2RuleBasedAgent()
    agent.bind_player("A")
    for actor, action in history:
        assert action in state.legal_actions(actor)
        apply_action(state, actor, action)
        agent.on_public_action(state, actor, action)
    return state, agent


def _enemy_immediate_finish_receive():
    hands = {
        "A": list("11224457"),
        "B": list("11112259"),
        "C": list("11333458"),
        "D": list("11345667"),
    }
    history = [
        ("A", ("attack_after_block", "1", "4")),
        ("B", ("pass", None, None)),
        ("C", ("receive", "4", None)),
        ("C", ("attack", None, "3")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("attack_after_block", "1", "3")),
        ("D", ("receive", "3", None)),
        ("D", ("attack", None, "6")),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("attack_after_block", "1", "6")),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("receive", "8", None)),
        ("C", ("attack", None, "3")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
    ]
    state = GoitaState(copy.deepcopy(hands), dealer="A")
    agent = ExperimentalAI2RuleBasedAgent()
    agent.bind_player("B")
    for actor, action in history:
        assert action in state.legal_actions(actor)
        apply_action(state, actor, action)
        agent.on_public_action(state, actor, action)
    return state, agent


def test_exported_policy_ranks_only_the_supplied_legal_actions() -> None:
    model = NeuralPolicyModel.load(MODEL_PATH)
    state = GoitaState(_hands(), dealer="A")
    legal = state.legal_actions("A")
    payload = live_state_payload(
        state,
        "A",
        initial_hand=_hands()["A"],
        history=[],
    )
    ranked = model.rank_actions(payload, legal)

    assert {action for action, _score in ranked} == set(legal)
    assert ranked == sorted(ranked, key=lambda item: item[1], reverse=True)


def test_features_keep_older_per_seat_attack_and_receive_context() -> None:
    history = [
        {"actor": "self", "type": "attack", "block": None, "block_known": False, "attack": "1"},
        {"actor": "next", "type": "receive", "block": "1", "block_known": True, "attack": None},
        {"actor": "self", "type": "attack", "block": None, "block_known": False, "attack": "6"},
        {"actor": "next", "type": "receive", "block": "8", "block_known": True, "attack": None},
        *[
            {"actor": "previous", "type": "pass", "block": None, "block_known": False, "attack": None}
            for _ in range(7)
        ],
    ]
    names, values = encode_state({"history": history}, legal_action_count=4)
    features = dict(zip(names, values))

    assert features["relation_attack_piece_counts.self.1"] == 1.0
    assert features["relation_receive_piece_counts.next.1"] == 1.0
    assert features["self_attack_received_by.next.1"] == 1.0
    assert features["relation_receive_piece_counts.next.8"] == 1.0
    assert features["self_attack_received_by.next.6"] == 1.0
    assert features["history.0.present"] == 1.0


def test_features_identify_owned_publicly_unstoppable_current_attack() -> None:
    state, agent = _last_kyosha_finish_route()
    payload = live_state_payload(
        state,
        "A",
        initial_hand=agent._get_my_initial_hand(state),
        history=agent._neural_public_history_by_state_id[id(state)],
    )
    names, values = encode_state(payload, legal_action_count=2)
    features = dict(zip(names, values))

    assert features["own_current_attack_count"] == 1.0
    assert features["current_attack_all_copies_accounted"] == 1.0
    assert features["own_current_attack_is_publicly_unstoppable"] == 1.0


def test_experimental_profile_uses_neural_only_for_current_ai_close_tie(monkeypatch) -> None:
    state = GoitaState(_hands(), dealer="A")
    agent = ExperimentalAI2RuleBasedAgent()
    agent.bind_player("A")
    legal = state.legal_actions("A")
    rule_choice = legal[-1]
    neural_choice = legal[0]

    class FakeModel:
        def rank_actions(self, _payload, actions):
            return [(neural_choice, 5.0)] + [
                (action, 1.0) for action in actions if action != neural_choice
            ]

    monkeypatch.setattr(ExperimentalAI2RuleBasedAgent, "_shared_neural_model", FakeModel())
    monkeypatch.setattr(ExperimentalAI2RuleBasedAgent, "_shared_neural_error", None)
    def close_rule(self, current_state, player, actions):
        self.last_rule_search_authority = "ordinary"
        self.last_decision_reason = "score_fallback"
        self.last_score_fallback_detail = "attack_piece_value"
        self.last_attack_candidate_scores = [
            {"action": rule_choice, "score": 100.0},
            {"action": neural_choice, "score": 98.0},
        ]
        return rule_choice

    monkeypatch.setattr(CurrentRuleBasedAgent, "select_action", close_rule)

    selected = agent.select_action(state, "A", legal)

    assert selected == neural_choice
    assert agent.last_neural_shadow["available"] is True
    assert agent.last_neural_shadow["mode"] == "tiebreak"
    assert agent.last_neural_shadow["applied"] is True
    assert agent.last_neural_shadow["current_choice_status"] == "close"
    assert agent.last_neural_shadow["current_score_gap"] == 2.0
    assert tuple(agent.last_neural_shadow["rule_action"]) == rule_choice
    assert tuple(agent.last_neural_shadow["recommended_action"]) == neural_choice
    assert agent.last_decision_reason == "neural_tiebreak"
    assert tuple(agent.last_attack_candidate_scores[0]["action"]) == rule_choice
    assert agent.last_attack_candidate_scores[0]["score"] == 100.0


def test_experimental_profile_keeps_current_ai_when_score_gap_is_clear(monkeypatch) -> None:
    state = GoitaState(_hands(), dealer="A")
    agent = ExperimentalAI2RuleBasedAgent()
    agent.bind_player("A")
    legal = state.legal_actions("A")
    rule_choice = legal[-1]
    neural_choice = legal[0]

    class FakeModel:
        def rank_actions(self, _payload, actions):
            return [(neural_choice, 10.0), (rule_choice, 1.0)] + [
                (action, 0.0)
                for action in actions
                if action not in {neural_choice, rule_choice}
            ]

    def clear_rule(self, current_state, player, actions):
        self.last_rule_search_authority = "ordinary"
        self.last_decision_reason = "score_fallback"
        self.last_score_fallback_detail = "attack_piece_value"
        self.last_attack_candidate_scores = [
            {"action": rule_choice, "score": 100.0},
            {"action": neural_choice, "score": 50.0},
        ]
        return rule_choice

    monkeypatch.setattr(ExperimentalAI2RuleBasedAgent, "_shared_neural_model", FakeModel())
    monkeypatch.setattr(ExperimentalAI2RuleBasedAgent, "_shared_neural_error", None)
    monkeypatch.setattr(CurrentRuleBasedAgent, "select_action", clear_rule)

    selected = agent.select_action(state, "A", legal)

    assert selected == rule_choice
    assert agent.last_neural_shadow["applied"] is False
    assert agent.last_neural_shadow["current_choice_status"] == "clear"


def test_experimental_profile_keeps_proven_rule_as_safety_guard(monkeypatch) -> None:
    state = GoitaState(_hands(), dealer="A")
    agent = ExperimentalAI2RuleBasedAgent()
    agent.bind_player("A")
    legal = state.legal_actions("A")
    rule_choice = legal[-1]
    neural_choice = legal[0]

    class FakeModel:
        def rank_actions(self, _payload, actions):
            return [(neural_choice, 5.0)] + [
                (action, 1.0) for action in actions if action != neural_choice
            ]

    def proven_rule(self, current_state, player, actions):
        self.last_rule_search_authority = "proven"
        self.last_decision_reason = "win_now"
        self.last_score_fallback_detail = "high_score_20"
        return rule_choice

    monkeypatch.setattr(ExperimentalAI2RuleBasedAgent, "_shared_neural_model", FakeModel())
    monkeypatch.setattr(ExperimentalAI2RuleBasedAgent, "_shared_neural_error", None)
    monkeypatch.setattr(CurrentRuleBasedAgent, "select_action", proven_rule)

    selected = agent.select_action(state, "A", legal)

    assert selected == rule_choice
    assert agent.last_neural_shadow["mode"] == "tiebreak"
    assert agent.last_neural_shadow["applied"] is False
    assert agent.last_neural_shadow["safety_locked"] is True
    assert agent.last_decision_reason == "win_now"


def test_experimental_profile_requires_more_confidence_to_override_strong_shi_plan(monkeypatch) -> None:
    state = GoitaState(_hands(), dealer="A")
    agent = ExperimentalAI2RuleBasedAgent()
    agent.bind_player("A")
    legal = state.legal_actions("A")
    rule_choice = legal[-1]
    neural_choice = legal[0]

    class FakeModel:
        def rank_actions(self, _payload, actions):
            return [(neural_choice, 5.0), (rule_choice, 4.0)] + [
                (action, 1.0)
                for action in actions
                if action not in {neural_choice, rule_choice}
            ]

    def strong_shi_rule(self, current_state, player, actions):
        self.last_rule_search_authority = "ordinary"
        self.last_decision_reason = "score_fallback"
        self.last_score_fallback_detail = "attack_enemy_team_shi_remaining_2"
        return rule_choice

    monkeypatch.setattr(ExperimentalAI2RuleBasedAgent, "_shared_neural_model", FakeModel())
    monkeypatch.setattr(ExperimentalAI2RuleBasedAgent, "_shared_neural_error", None)
    monkeypatch.setattr(CurrentRuleBasedAgent, "select_action", strong_shi_rule)

    selected = agent.select_action(state, "A", legal)

    assert selected == rule_choice
    assert agent.last_neural_shadow["applied"] is False
    assert agent.last_neural_shadow["current_choice_status"] == "rule_locked"
    assert agent.last_neural_shadow["rule_authority"] == "strong"


def test_experimental_profile_preserves_repeated_attack_reserve(monkeypatch) -> None:
    state, agent = _three_kyosha_continuation()
    legal = state.legal_actions("B")
    rule_choice = ("attack_after_block", "1", "2")
    neural_choice = ("attack_after_block", "2", "2")
    second_choice = ("attack_after_block", "6", "2")

    class FakeModel:
        def rank_actions(self, _payload, actions):
            ranked = [(neural_choice, 5.0), (second_choice, 4.8)]
            ranked.extend(
                (action, 1.0)
                for action in actions
                if action not in {neural_choice, second_choice}
            )
            return ranked

    monkeypatch.setattr(ExperimentalAI2RuleBasedAgent, "_shared_neural_model", FakeModel())
    monkeypatch.setattr(ExperimentalAI2RuleBasedAgent, "_shared_neural_error", None)
    monkeypatch.setattr(
        CurrentRuleBasedAgent,
        "select_action",
        lambda self, current_state, player, actions: rule_choice,
    )

    selected = agent.select_action(state, "B", legal)

    assert selected == rule_choice
    assert agent.last_neural_shadow["block_only_disagreement"] is True
    assert agent.last_neural_shadow["protect_attack_reserve"] is True
    assert agent.last_neural_shadow["current_choice_status"] == "rule_locked"
    assert agent.last_neural_shadow["rule_authority"] == "strong"


def test_reported_three_kyosha_position_keeps_a_kyosha_for_continuation() -> None:
    state, agent = _three_kyosha_continuation()

    selected = agent.select_action(state, "B", state.legal_actions("B"))

    assert selected[0] == "attack_after_block"
    assert selected[2] == "2"
    assert selected[1] != "2"


def test_experimental_profile_protects_publicly_unstoppable_finish_piece(monkeypatch) -> None:
    state, agent = _last_kyosha_finish_route()
    neural_choice = ("receive", "2", None)
    pass_choice = ("pass", None, None)

    class FakeModel:
        def rank_actions(self, _payload, actions):
            assert set(actions) == {pass_choice, neural_choice}
            return [(neural_choice, 10.0), (pass_choice, 4.0)]

    monkeypatch.setattr(ExperimentalAI2RuleBasedAgent, "_shared_neural_model", FakeModel())
    monkeypatch.setattr(ExperimentalAI2RuleBasedAgent, "_shared_neural_error", None)

    selected = agent.select_action(state, "A", state.legal_actions("A"))

    assert selected == pass_choice
    assert agent.last_score_fallback_detail == "pass_preserve_public_unstoppable_finish_piece_2"
    assert agent.last_neural_shadow["protect_unstoppable_finish"] is True
    assert agent.last_neural_shadow["current_choice_status"] == "rule_locked"


def test_reported_last_kyosha_position_passes_instead_of_receiving() -> None:
    state, agent = _last_kyosha_finish_route()

    selected = agent.select_action(state, "A", state.legal_actions("A"))

    assert selected == ("pass", None, None)


def test_reported_enemy_immediate_finish_forces_receive_over_neural_pass() -> None:
    state, agent = _enemy_immediate_finish_receive()

    selected = agent.select_action(state, "B", state.legal_actions("B"))

    assert selected == ("receive", "9", None)
    assert agent.last_decision_reason == "score_fallback"
    assert (
        agent.last_score_fallback_detail
        == "receive_prevent_enemy_immediate_finish"
    )
    assert agent.last_neural_shadow["safety_locked"] is True
    assert agent.last_neural_shadow["prevent_enemy_immediate_finish"] is True
    assert agent.last_neural_shadow["current_choice_status"] == "rule_locked"


def test_enemy_immediate_finish_guard_corrects_an_accidental_current_ai_pass(
    monkeypatch,
) -> None:
    state, agent = _enemy_immediate_finish_receive()

    def accidental_pass(self, current_state, player, actions):
        self.last_rule_search_authority = "ordinary"
        self.last_decision_reason = "score_fallback"
        self.last_score_fallback_detail = "enemy_later_pass"
        self.last_attack_candidate_scores = [
            {"action": ("pass", None, None), "score": 1.0},
            {"action": ("receive", "9", None), "score": 0.0},
        ]
        return ("pass", None, None)

    monkeypatch.setattr(CurrentRuleBasedAgent, "select_action", accidental_pass)

    selected = agent.select_action(state, "B", state.legal_actions("B"))

    assert selected == ("receive", "9", None)
    assert agent.last_neural_shadow["safety_locked"] is True


def test_review_correction_teaches_target_shi_continuation() -> None:
    records = [
        json.loads(line)
        for line in CORRECTIONS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record = next(
        item for item in records
        if item["decision_id"] == "review-20261001063211-round6-B-turn7-attack"
    )
    legal = [
        (item["type"], item.get("block"), item.get("attack"))
        for item in record["legal_actions"]
    ]
    corrected = legal[int(record["selected_action_index"])]
    model = NeuralPolicyModel.load(MODEL_PATH)

    ranked = model.rank_actions(record["state"], legal)

    assert corrected == ("attack", None, "1")
    assert ranked[0][0] == corrected


def test_review_correction_accepts_several_hidden_blocks_without_kyosha() -> None:
    records = [
        json.loads(line)
        for line in CORRECTIONS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record = next(
        item for item in records
        if item["decision_id"] == "review-20261001082211-round1-B-turn15-hidden-block"
    )
    legal = [
        (item["type"], item.get("block"), item.get("attack"))
        for item in record["legal_actions"]
    ]
    acceptable = {
        legal[index] for index in record["acceptable_action_indices"]
    }
    expanded = _expand_acceptable_actions([record])
    model = NeuralPolicyModel.load(MODEL_PATH)

    ranked = model.rank_actions(record["state"], legal)

    assert len(expanded) == 3
    assert {_selected["selected_action_index"] for _selected in expanded} == set(
        record["acceptable_action_indices"]
    )
    assert ranked[0][0] in acceptable
    assert ("attack_after_block", "2", "2") not in acceptable


def test_review_correction_targets_receive_decision_before_reported_attack() -> None:
    records = [
        json.loads(line)
        for line in CORRECTIONS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record = next(
        item for item in records
        if item["decision_id"] == "review-20261001085804-round1-A-turn14-receive"
    )
    legal = [
        (item["type"], item.get("block"), item.get("attack"))
        for item in record["legal_actions"]
    ]
    corrected = legal[int(record["selected_action_index"])]
    model = NeuralPolicyModel.load(MODEL_PATH)

    ranked = model.rank_actions(record["state"], legal)

    assert corrected == ("pass", None, None)
    assert record["review"]["corrected_decision_log_index"] == 19
    assert record["review"]["reported_target_decision_log_index"] == 20
    assert ranked[0][0] == corrected


def test_review_correction_teaches_immediate_enemy_finish_defense() -> None:
    records = [
        json.loads(line)
        for line in CORRECTIONS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record = next(
        item for item in records
        if item["decision_id"]
        == "review-20261001095745-round2-B-turn18-receive"
    )
    legal = [
        (item["type"], item.get("block"), item.get("attack"))
        for item in record["legal_actions"]
    ]
    corrected = legal[int(record["selected_action_index"])]
    model = NeuralPolicyModel.load(MODEL_PATH)

    ranked = model.rank_actions(record["state"], legal)

    assert corrected == ("receive", "9", None)
    assert ranked[0][0] == corrected


def test_experimental_profile_owns_new_ally_and_shi_guards() -> None:
    current = CurrentRuleBasedAgent()
    experimental = ExperimentalAI2RuleBasedAgent()

    assert current.PRESERVE_SHI_FOR_THIRD_ATTACK_ENABLED is False
    assert current.ALLY_GUARANTEED_WIN_NO_SELF_FINISH_ENABLED is False
    assert current.PRESERVE_PUBLIC_UNSTOPPABLE_FINISH_ENABLED is False
    assert experimental.PRESERVE_SHI_FOR_THIRD_ATTACK_ENABLED is True
    assert experimental.ALLY_GUARANTEED_WIN_NO_SELF_FINISH_ENABLED is True
    assert experimental.PRESERVE_PUBLIC_UNSTOPPABLE_FINISH_ENABLED is True
    assert experimental.NEURAL_PRIMARY_ENABLED is False
    assert experimental.NEURAL_TIEBREAK_ENABLED is True


def test_experimental_history_never_keeps_an_opponent_hidden_piece() -> None:
    state = GoitaState(_hands(), dealer="A")
    agent = ExperimentalAI2RuleBasedAgent()
    agent.bind_player("B")
    action = ("attack_after_block", "1", "5")
    state.apply_attack_after_block("A", "1", "5")
    agent.on_public_action(state, "A", action)

    saved = agent._neural_public_history_by_state_id[id(state)][0]
    assert saved == {
        "player": "A",
        "action": ["attack_after_block", None, "5"],
    }


def test_false_big_kyosha_signal_is_locked_against_neural_override() -> None:
    """Regression for 2026-10-02 B turn 7 review in 強化中AI2."""
    state = GoitaState(
        {
            "A": list("11345567"),
            "B": list("41193265"),
            "C": list("11122234"),
            "D": list("11134578"),
        },
        dealer="D",
    )
    agent = ExperimentalAI2RuleBasedAgent()
    agent.bind_player("B")
    agent.TIME_LIMITED_SEARCH_ENABLED = False
    agent.TIME_SEARCH_ENABLED = False
    history = (
        ("D", ("attack_after_block", "3", "1")),
        ("A", ("receive", "1", None)),
        ("A", ("attack", None, "5")),
        ("B", ("receive", "5", None)),
        ("B", ("attack", None, "6")),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("receive", "6", None)),
        ("A", ("attack", None, "7")),
        ("B", ("receive", "9", None)),
    )
    for actor, action in history:
        apply_action(state, actor, action)
        agent.on_public_action(state, actor, action)

    selected = agent.select_action(state, "B", state.legal_actions("B"))

    assert selected == ("attack", None, "4")
    assert agent.last_score_fallback_detail == "attack_avoid_false_sequence_signal"
    assert agent.last_neural_shadow["recommended_action"] == ["attack", None, "2"]
    assert agent.last_neural_shadow["applied"] is False
    assert agent.last_neural_shadow["current_choice_status"] == "rule_locked"


def test_truthful_first_pair_signal_agrees_with_global_neural_choice() -> None:
    """Regression for 2026-10-02 round 4, A turn 10 in 強化中AI2."""
    state = GoitaState(
        {
            "A": list("12128716"),
            "B": list("15141443"),
            "C": list("23615147"),
            "D": list("35915132"),
        },
        dealer="D",
    )
    agent = ExperimentalAI2RuleBasedAgent()
    agent.bind_player("A")
    history = (
        ("D", ("attack_after_block", "1", "5")),
        ("A", ("pass", None, None)),
        ("B", ("receive", "5", None)),
        ("B", ("attack", None, "4")),
        ("C", ("receive", "4", None)),
        ("C", ("attack", None, "6")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("attack_after_block", "1", "7")),
        ("D", ("receive", "9", None)),
        ("D", ("attack", None, "3")),
        ("A", ("receive", "8", None)),
    )
    for actor, action in history:
        apply_action(state, actor, action)
        agent.on_public_action(state, actor, action)

    selected = agent.select_action(state, "A", state.legal_actions("A"))

    assert selected == ("attack", None, "2")
    assert agent.last_score_fallback_detail == "attack_truthful_first_pair_signal_2"
    assert agent.last_neural_shadow["recommended_action"] == ["attack", None, "2"]
    assert agent.last_neural_shadow["global_recommended_action"] == [
        "attack",
        None,
        "2",
    ]
    assert agent.last_neural_shadow["applied"] is False


if __name__ == "__main__":
    test_exported_policy_ranks_only_the_supplied_legal_actions()
    print("NEURAL_POLICY_TEST_OK")

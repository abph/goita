"""Tests the exported policy and neural-first 強化中AI2 wrapper."""

from __future__ import annotations

import json
from pathlib import Path

from goita_ai2.current_ai.agent import RuleBasedAgent as CurrentRuleBasedAgent
from goita_ai2.experimental_ai2 import RuleBasedAgent as ExperimentalAI2RuleBasedAgent
from goita_ai2.neural_policy import NeuralPolicyModel, encode_state, live_state_payload
from goita_ai2.state import GoitaState


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


def test_experimental_profile_uses_neural_action_outside_proven_safety(monkeypatch) -> None:
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
    monkeypatch.setattr(
        CurrentRuleBasedAgent,
        "select_action",
        lambda self, current_state, player, actions: rule_choice,
    )

    selected = agent.select_action(state, "A", legal)

    assert selected == neural_choice
    assert agent.last_neural_shadow["available"] is True
    assert agent.last_neural_shadow["mode"] == "primary"
    assert agent.last_neural_shadow["applied"] is True
    assert tuple(agent.last_neural_shadow["rule_action"]) == rule_choice
    assert tuple(agent.last_neural_shadow["recommended_action"]) == neural_choice
    assert agent.last_decision_reason == "neural_primary"
    assert tuple(agent.last_attack_candidate_scores[0]["action"]) == neural_choice
    assert agent.last_attack_candidate_scores[0]["score"] == 5.0


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
    assert agent.last_neural_shadow["mode"] == "primary"
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
    assert agent.last_neural_shadow["confidence_deferred"] is True
    assert agent.last_neural_shadow["required_margin"] == 3.0


def test_review_correction_teaches_target_shi_continuation() -> None:
    record = json.loads(CORRECTIONS_PATH.read_text(encoding="utf-8").strip())
    legal = [
        (item["type"], item.get("block"), item.get("attack"))
        for item in record["legal_actions"]
    ]
    corrected = legal[int(record["selected_action_index"])]
    model = NeuralPolicyModel.load(MODEL_PATH)

    ranked = model.rank_actions(record["state"], legal)

    assert corrected == ("attack", None, "1")
    assert ranked[0][0] == corrected


def test_experimental_profile_owns_new_ally_and_shi_guards() -> None:
    current = CurrentRuleBasedAgent()
    experimental = ExperimentalAI2RuleBasedAgent()

    assert current.PRESERVE_SHI_FOR_THIRD_ATTACK_ENABLED is False
    assert current.ALLY_GUARANTEED_WIN_NO_SELF_FINISH_ENABLED is False
    assert experimental.PRESERVE_SHI_FOR_THIRD_ATTACK_ENABLED is True
    assert experimental.ALLY_GUARANTEED_WIN_NO_SELF_FINISH_ENABLED is True
    assert experimental.NEURAL_PRIMARY_ENABLED is True


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


if __name__ == "__main__":
    test_exported_policy_ranks_only_the_supplied_legal_actions()
    print("NEURAL_POLICY_TEST_OK")

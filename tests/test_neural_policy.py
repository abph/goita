"""Tests the exported policy and the non-authoritative 強化中AI2 wrapper."""

from __future__ import annotations

from pathlib import Path

from goita_ai2.current_ai.agent import RuleBasedAgent as CurrentRuleBasedAgent
from goita_ai2.experimental_ai2 import RuleBasedAgent as ExperimentalAI2RuleBasedAgent
from goita_ai2.neural_policy import NeuralPolicyModel, live_state_payload
from goita_ai2.state import GoitaState


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "goita_ai2" / "experimental_ai2" / "data" / "neural_policy.json"
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


def test_experimental_profile_keeps_rule_action_and_records_shadow(monkeypatch) -> None:
    state = GoitaState(_hands(), dealer="A")
    agent = ExperimentalAI2RuleBasedAgent()
    agent.bind_player("A")
    legal = state.legal_actions("A")
    rule_choice = legal[-1]
    monkeypatch.setattr(
        CurrentRuleBasedAgent,
        "select_action",
        lambda self, current_state, player, actions: rule_choice,
    )

    selected = agent.select_action(state, "A", legal)

    assert selected == rule_choice
    assert agent.last_neural_shadow["available"] is True
    assert agent.last_neural_shadow["mode"] == "shadow"
    assert tuple(agent.last_neural_shadow["rule_action"]) == rule_choice
    assert tuple(agent.last_neural_shadow["recommended_action"]) in legal


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

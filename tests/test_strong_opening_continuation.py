from __future__ import annotations

from goita_ai2.current_ai.agent import RuleBasedAgent as CurrentRuleBasedAgent
from goita_ai2.experimental_ai2 import RuleBasedAgent as ExperimentalAI2RuleBasedAgent
from goita_ai2.instruction_case_audit import apply_action
from goita_ai2.state import GoitaState


REPORTED_HANDS = {
    "A": list("61143443"),
    "B": list("15132614"),
    "C": list("25819722"),
    "D": list("11537151"),
}


def _agent(agent_class):
    state = GoitaState(REPORTED_HANDS, dealer="C")
    agent = agent_class()
    agent.bind_player("C")
    agent.TIME_LIMITED_SEARCH_ENABLED = False
    agent.TIME_SEARCH_ENABLED = False
    if hasattr(agent, "NEURAL_TIEBREAK_ENABLED"):
        agent.NEURAL_TIEBREAK_ENABLED = False
    return state, agent


def _public(agent, state, player, action) -> None:
    assert action in state.legal_actions(player)
    apply_action(state, player, action)
    agent.on_public_action(state, player, action)


def test_absolute_a_opening_protects_guaranteed_score_across_receive_branches() -> None:
    """Regression for the 2026-10-06 round 2, C turn 1 report."""
    state, agent = _agent(ExperimentalAI2RuleBasedAgent)

    selected = agent.select_action(state, "C", state.legal_actions("C"))
    plan = agent._track[id(state)]["shallow_eight_card_plan"]
    proof = plan["opening_continuation_proof"]

    assert agent._initial_hand_axes_for_state(state, "C")["absolute_rank"] == "A"
    assert selected[2] == "2"
    assert selected[1] in {"5", "7"}
    assert selected[1] != "1"
    assert proof["absolute_rank"] == "A"
    assert proof["minimum_score"] == 50.0
    assert proof["maximum_score"] == 100.0
    assert agent.last_decision_reason == "tsume"
    assert agent.last_score_fallback_detail.startswith(
        "high_score_opening_continuation_absA_safe_50_"
    )


def test_saved_current_ai_keeps_its_existing_opening_choice() -> None:
    state, agent = _agent(CurrentRuleBasedAgent)

    selected = agent.select_action(state, "C", state.legal_actions("C"))

    assert selected == ("attack_after_block", "1", "2")
    assert "opening_continuation_proof" not in agent._track[id(state)][
        "shallow_eight_card_plan"
    ]


def test_opening_continuation_proof_is_not_used_below_absolute_a(monkeypatch) -> None:
    state, agent = _agent(ExperimentalAI2RuleBasedAgent)
    monkeypatch.setattr(
        agent,
        "_initial_hand_axes_for_state",
        lambda _state, _player: {"rank": "B", "absolute_rank": "B"},
    )

    selected = agent.select_action(state, "C", state.legal_actions("C"))

    assert selected == ("attack_after_block", "1", "2")
    assert "opening_continuation_proof" not in agent._track[id(state)][
        "shallow_eight_card_plan"
    ]


def test_strong_opening_keeps_the_second_kyosha_after_all_pass() -> None:
    state, agent = _agent(ExperimentalAI2RuleBasedAgent)
    first = agent.select_action(state, "C", state.legal_actions("C"))
    _public(agent, state, "C", first)
    _public(agent, state, "D", ("pass", None, None))
    _public(agent, state, "A", ("pass", None, None))
    _public(agent, state, "B", ("pass", None, None))

    second = agent.select_action(state, "C", state.legal_actions("C"))

    assert second[0] == "attack_after_block"
    assert second[2] == "2"
    assert agent.last_decision_reason == "tsume"
    assert agent.last_score_fallback_detail == "high_score_50"


def test_strong_opening_can_receive_shi_after_kyosha_is_received() -> None:
    state, agent = _agent(ExperimentalAI2RuleBasedAgent)
    first = agent.select_action(state, "C", state.legal_actions("C"))
    _public(agent, state, "C", first)
    _public(agent, state, "D", ("pass", None, None))
    _public(agent, state, "A", ("pass", None, None))
    _public(agent, state, "B", ("receive", "2", None))
    _public(agent, state, "B", ("attack", None, "1"))

    response = agent.select_action(state, "C", state.legal_actions("C"))

    assert response == ("receive", "1", None)


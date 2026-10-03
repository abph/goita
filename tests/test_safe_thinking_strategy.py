from __future__ import annotations

from goita_ai2.current_ai.agent import RuleBasedAgent as CurrentRuleBasedAgent
from goita_ai2.experimental_ai2 import RuleBasedAgent as ExperimentalAI2RuleBasedAgent
from goita_ai2.instruction_case_audit import apply_action
from goita_ai2.state import GoitaState


HANDS = {
    "A": list("11331525"),
    "B": list("21371244"),
    "C": list("21671864"),
    "D": list("91314515"),
}

HISTORY = (
    ("D", ("attack_after_block", "1", "5")),
    ("A", ("receive", "5", None)),
    ("A", ("attack", None, "1")),
    ("B", ("receive", "1", None)),
    ("B", ("attack", None, "4")),
    ("C", ("receive", "4", None)),
    ("C", ("attack", None, "6")),
    ("D", ("pass", None, None)),
    ("A", ("pass", None, None)),
    ("B", ("pass", None, None)),
)


def _reported_position(agent_class):
    state = GoitaState(HANDS, dealer="D")
    agent = agent_class()
    agent.bind_player("C")
    agent.TIME_LIMITED_SEARCH_ENABLED = False
    agent.TIME_SEARCH_ENABLED = False
    if hasattr(agent, "NEURAL_TIEBREAK_ENABLED"):
        agent.NEURAL_TIEBREAK_ENABLED = False
    for player, action in HISTORY:
        assert action in state.legal_actions(player)
        apply_action(state, player, action)
        agent.on_public_action(state, player, action)
    return state, agent


def test_safe_thinking_spends_spare_shi_and_keeps_rook_for_third_attack() -> None:
    state, agent = _reported_position(ExperimentalAI2RuleBasedAgent)

    selected = agent.select_action(state, "C", state.legal_actions("C"))

    assert selected == ("attack_after_block", "1", "6")
    assert agent.last_decision_reason == "score_fallback"
    assert agent.last_score_fallback_detail == "block_spare_shi_keep_third_big_attack"


def test_spare_shi_rule_stays_isolated_from_saved_current_ai() -> None:
    state, agent = _reported_position(CurrentRuleBasedAgent)

    selected = agent.select_action(state, "C", state.legal_actions("C"))

    assert selected == ("attack_after_block", "7", "6")
    assert agent.ALLY_SHI_SPARE_THIRD_BIG_ATTACK_ENABLED is False


def test_spare_shi_rule_requires_no_enemy_shi_attack_pressure() -> None:
    state, agent = _reported_position(ExperimentalAI2RuleBasedAgent)
    agent._ensure_trackers(state)
    tracker = agent._track[id(state)]
    tracker["enemy_past_attacks"].add("1")

    adjustment = agent._ally_shi_spare_third_big_attack_adjustment(
        state,
        "C",
        "attack_after_block",
        "1",
        "6",
    )

    assert adjustment == 0.0

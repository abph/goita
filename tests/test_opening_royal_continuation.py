from __future__ import annotations

from goita_ai2.current_ai import RuleBasedAgent as CurrentRuleBasedAgent
from goita_ai2.experimental_ai2 import RuleBasedAgent as ExperimentalAI2RuleBasedAgent
from goita_ai2.instruction_case_audit import apply_action
from goita_ai2.state import GoitaState


REPORTED_HANDS = {
    "A": list("11134555"),
    "B": list("12334677"),
    "C": list("11112246"),
    "D": list("11234589"),
}

HISTORY_BEFORE_D_FIRST_ATTACK = (
    ("A", ("attack_after_block", "1", "5")),
    ("B", ("pass", None, None)),
    ("C", ("pass", None, None)),
    ("D", ("pass", None, None)),
    ("A", ("attack_after_block", "1", "5")),
    ("B", ("pass", None, None)),
    ("C", ("pass", None, None)),
    ("D", ("receive", "5", None)),
)


def _reported_position(agent_class):
    state = GoitaState(REPORTED_HANDS, dealer="A")
    agent = agent_class()
    agent.bind_player("D")
    agent.TIME_LIMITED_SEARCH_ENABLED = False
    agent.TIME_SEARCH_ENABLED = False
    agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    for actor, action in HISTORY_BEFORE_D_FIRST_ATTACK:
        assert action in state.legal_actions(actor)
        apply_action(state, actor, action)
        agent.on_public_action(state, actor, action)
    return state, agent


def test_reported_first_attack_keeps_royals_and_uses_kyosha_probe() -> None:
    current_state, current = _reported_position(CurrentRuleBasedAgent)
    assert current.select_action(
        current_state,
        "D",
        current_state.legal_actions("D"),
    ) in (("attack", None, "8"), ("attack", None, "9"))

    state, agent = _reported_position(ExperimentalAI2RuleBasedAgent)
    chosen = agent.select_action(state, "D", state.legal_actions("D"))

    assert chosen == ("attack", None, "2")
    assert agent.last_decision_reason == "score_fallback"
    assert agent.last_score_fallback_detail == (
        "attack_preserve_royals_without_continuation_2"
    )
    assert agent.last_rule_search_authority == "strong"
    assert agent.last_neural_shadow["applied"] is False
    assert agent.last_neural_shadow["rule_authority"] == "strong"
    assert agent.last_attack_candidate_snapshot["chosen"]["attack"] == "2"
    comparison = agent.last_attack_candidate_snapshot["attack_intent"]
    assert comparison["kind"] == "preserve_royals_without_continuation"
    assert comparison["planned_attack"] in ("8", "9")
    assert comparison["selected_attack"] == "2"


def test_proven_opening_royal_route_is_not_replaced() -> None:
    state, agent = _reported_position(ExperimentalAI2RuleBasedAgent)
    actions = state.legal_actions("D")
    royal = ("attack", None, "8")

    assert agent._opening_royal_without_continuation_fallback(
        state,
        "D",
        actions,
        royal,
        rule_reason="tsume",
        rule_authority="proven",
        attack_count_before=0,
    ) is None

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


def _lance_signal_opening(agent_class, *, silver_pair=False, risk=False):
    hands = {"A": "12233677", "B": "11234589", "C": "11114456", "D": "11123455"}
    if silver_pair:
        hands.update(B="11234489", D="11123555")
    state = GoitaState({seat: list(hand) for seat, hand in hands.items()}, dealer="B")
    agent = agent_class()
    agent.bind_player("B")
    agent.TIME_SEARCH_ENABLED = False
    agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    agent.final_shi_risk = risk
    return state, agent


def test_reported_dealer_uses_lance_to_signal_royals_over_singleton_silver() -> None:
    state, agent = _lance_signal_opening(ExperimentalAI2RuleBasedAgent)

    assert agent.select_action(state, "B", state.legal_actions("B")) == (
        "attack_after_block", "1", "2",
    )
    assert agent.last_score_fallback_detail == "attack_opening_lance_royal_signal"
    assert agent.last_rule_search_authority == "strong"
    signal = agent.last_attack_candidate_snapshot["opening_lance_signal"]
    assert signal["compared_action"] == ["attack_after_block", "1", "4"]
    assert signal["return_coverage"]["uncovered_returns"] == {"2": 2}
    assert signal["selected_value"] > signal["compared_value"]
    plan = agent._track[id(state)]["shallow_eight_card_plan"]
    assert plan["steps"][0] == ("1", "2")
    assert set(plan["final_pair"]) == {"8", "9"}


def test_lance_signal_does_not_replace_a_real_silver_pair() -> None:
    state, agent = _lance_signal_opening(ExperimentalAI2RuleBasedAgent, silver_pair=True)

    assert agent.select_action(state, "B", state.legal_actions("B"))[2] == "4"
    assert "opening_lance_signal" not in agent.last_attack_candidate_snapshot


def test_lance_signal_is_scoped_to_the_ai2_safe_profile() -> None:
    for agent_class, risk in (
        (CurrentRuleBasedAgent, False),
        (ExperimentalAI2RuleBasedAgent, True),
    ):
        state, agent = _lance_signal_opening(agent_class, risk=risk)
        assert agent.select_action(state, "B", state.legal_actions("B"))[2] == "4"

from __future__ import annotations

from goita_ai2.current_ai.upside_finish import UpsideFinishMixin
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


HANDS = {
    "A": list("54269335"),
    "B": list("71271116"),
    "C": list("34124114"),
    "D": list("85211513"),
}


def _upside_position(*, ac_score: int = 20, bd_score: int = 100):
    state = GoitaState(hands=HANDS, dealer="B")
    state.team_score = {"AC": ac_score, "BD": bd_score}
    agents = {player: RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)
        agent._ensure_trackers(state)

    def apply_public(player: str, action) -> None:
        action_type, block, attack = action
        if action_type == "pass":
            state.apply_pass(player)
        elif action_type == "receive":
            state.apply_receive(player, block)
        elif action_type == "attack":
            state.apply_attack(player, attack)
        else:
            state.apply_attack_after_block(player, block, attack)
        for agent in agents.values():
            agent.on_public_action(state, player, action)

    opening = (
        ("B", ("attack_after_block", "1", "1")),
        ("C", ("receive", "1", None)),
        ("C", ("attack", None, "4")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("attack_after_block", "3", "4")),
        ("D", ("receive", "8", None)),
        ("D", ("attack", None, "5")),
        ("A", ("receive", "5", None)),
        ("A", ("attack", None, "4")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("attack_after_block", "6", "3")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("receive", "3", None)),
        ("D", ("attack", None, "5")),
        ("A", ("receive", "5", None)),
    )
    for player, action in opening:
        apply_public(player, action)
    return state, agents


def test_rule_based_agent_uses_upside_finish_mixin() -> None:
    assert issubclass(RuleBasedAgent, UpsideFinishMixin)


def test_trailing_team_risks_the_horse_for_a_royal_fifty_finish() -> None:
    state, agents = _upside_position()
    agent = agents["A"]

    chosen = agent.select_action(state, "A", state.legal_actions("A"))
    metrics = agent.last_upside_finish_metrics

    assert chosen == ("attack", None, "3")
    assert agent.last_decision_reason == "upside_finish"
    assert agent.last_score_fallback_detail.startswith("safe_20_target_50_chance_")
    assert metrics["safe_action"] == ("attack", None, "9")
    assert metrics["safe_score"] == 20.0
    assert metrics["maximum_score"] == 50.0
    assert metrics["high_score_probability"] >= 0.35
    assert metrics["safe_retention_probability"] >= 0.75
    assert metrics["adjusted_failure_risk"] <= metrics["allowed_failure_risk"]


def test_upside_finish_never_risks_a_safe_match_win() -> None:
    state, agents = _upside_position(ac_score=130, bd_score=100)
    agent = agents["A"]

    chosen = agent.select_action(state, "A", state.legal_actions("A"))

    assert chosen == ("attack", None, "9")
    assert agent.last_decision_reason == "tsume"
    assert agent.last_score_fallback_detail == "high_score_20"
    assert agent.last_upside_finish_metrics == {}


def test_even_score_accepts_a_strong_positive_value_upside() -> None:
    state, agents = _upside_position(ac_score=20, bd_score=20)
    agent = agents["A"]

    chosen = agent.select_action(state, "A", state.legal_actions("A"))

    assert chosen == ("attack", None, "3")
    assert agent.last_decision_reason == "upside_finish"
    assert agent.last_upside_finish_metrics["allowed_failure_risk"] == 0.2


def test_large_lead_keeps_the_guaranteed_finish() -> None:
    state, agents = _upside_position(ac_score=100, bd_score=20)
    agent = agents["A"]

    chosen = agent.select_action(state, "A", state.legal_actions("A"))

    assert chosen == ("attack", None, "9")
    assert agent.last_decision_reason == "tsume"
    assert agent.last_score_fallback_detail == "high_score_20"
    assert agent.last_upside_finish_metrics == {}

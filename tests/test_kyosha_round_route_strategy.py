from __future__ import annotations

from goita_ai2.experimental_ai2 import RuleBasedAgent
from goita_ai2.state import GoitaState


def test_kyosha_route_keeps_truthful_pair_plan_over_two_shi_opening() -> None:
    """Regression for round 1, D turn 2 of the 2026-10-02 review."""
    state = GoitaState(
        hands={
            "A": list("81411713"),
            "B": list("31515615"),
            "C": list("14259274"),
            "D": list("32123164"),
        },
        dealer="C",
    )
    agent = RuleBasedAgent()
    agent.bind_player("D")
    agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    agent._ensure_trackers(state)

    opening = ("attack_after_block", "4", "2")
    state.apply_attack_after_block("C", "4", "2")
    agent.on_public_action(state, "C", opening)

    receive = agent.select_action(state, "D", state.legal_actions("D"))
    comparison = agent._track[id(state)]["last_round_route_comparison"]

    assert receive == ("receive", "2", None)
    assert comparison is not None
    assert comparison["attack_values"]["1"] > comparison["attack_values"]["3"]
    assert comparison["search_best_attack"] == "1"
    assert comparison["followup_rule_action"] == ["attack", None, "3"]
    assert comparison["followup_rule_detail"] == (
        "attack_sequence_two_kyosha_middle_pair"
    )
    assert comparison["followup_rule_authority"] == "strong"
    assert comparison["followup_two_shi_signal_risk"] is True
    assert comparison["followup_override_blocked"] is True
    assert comparison["chosen"] == (
        ("receive", "2", None),
        ("attack", None, "3"),
    )
    assert "receive_attack_3_followup_rule_kept_3_search_1" in (
        agent.last_score_fallback_detail
    )
    assert agent.last_score_fallback_detail.endswith("two_shi_signal")

    state.apply_receive("D", "2")
    agent.on_public_action(state, "D", receive)

    chosen = agent.select_action(
        state,
        "D",
        state.legal_actions("D"),
    )

    assert chosen == ("attack", None, "3")
    assert agent.last_score_fallback_detail == "kyosha_round_route_followup_3"

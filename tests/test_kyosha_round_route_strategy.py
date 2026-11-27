from __future__ import annotations

from goita_ai2.decision_check import apply_action, replay
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


def test_kyosha_route_preserves_unproven_early_royals_and_uses_hisha() -> None:
    """Regression for round 1, C turn 4 of the 2026-10-07 review."""
    state = GoitaState(
        hands={
            "A": list("11123455"),
            "B": list("11345667"),
            "C": list("12345789"),
            "D": list("11114232"),
        },
        dealer="D",
    )
    agent = RuleBasedAgent()
    agent.bind_player("C")
    agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    agent._ensure_trackers(state)

    opening = ("attack_after_block", "1", "2")
    state.apply_attack_after_block("D", "1", "2")
    agent.on_public_action(state, "D", opening)
    for player in ("A", "B"):
        passed = ("pass", None, None)
        state.apply_pass(player)
        agent.on_public_action(state, player, passed)

    comparison = agent._compare_kyosha_round_routes(state, "C")
    assert comparison is not None
    assert comparison["raw_search_best_attack"] == "8"
    assert comparison["early_royal_proof_statuses"] == {
        "8": "counterexample",
        "9": "counterexample",
    }
    assert comparison["blocked_early_royal_attacks"] == ["8", "9"]
    assert comparison["search_best_attack"] == "4"
    assert comparison["followup_rule_action"] == ["attack", None, "7"]
    assert comparison["royal_preservation_followup_kept"] is True
    assert comparison["best_receive_attack"] == "7"
    # C is the later receiver of D's opening kyosha.  The existing later-seat
    # policy therefore keeps pass as the final response, which the review also
    # allows.  If the route does receive, its saved follow-up must be hisha.
    assert comparison["chosen"] == (("pass", None, None),)

    state.apply_receive("C", "2")
    agent.on_public_action(state, "C", ("receive", "2", None))
    tracker = agent._track[id(state)]
    tracker["pending_kyosha_receive_attack_piece"] = "7"
    tracker["pending_kyosha_receive_source"] = (
        "round_route_royal_preservation"
    )
    chosen = agent.select_action(state, "C", state.legal_actions("C"))

    assert chosen == ("attack", None, "7")
    assert agent.last_score_fallback_detail == (
        "kyosha_round_route_followup_7_royal_preserved_unproven"
    )


def test_rejected_early_royal_cannot_return_as_lance_followup() -> None:
    state, agent = _royal_attack_review_position()

    assert agent.select_action(state, "B", state.legal_actions("B")) == (
        "pass", None, None,
    )

    tracker = agent._track[id(state)]
    comparison = tracker["last_round_route_comparison"]
    assert comparison["early_royal_proof_statuses"] == {
        "8": "counterexample", "9": "counterexample",
    }
    assert comparison["rejected_royal_followup"] == "8"
    assert comparison["best_receive_attack"] == "1"
    assert comparison["pass_value"] > comparison["receive_value"]
    assert tracker["pending_kyosha_receive_attack_piece"] is None
    assert agent.last_score_fallback_detail == "pass_kyosha_unproven_royal_followup"


def test_second_attack_keeps_last_royal_when_third_cannot_secure_finish() -> None:
    state, agent = _royal_attack_review_position(second_attack=True)
    agent.TIME_SEARCH_ENABLED = False

    chosen = agent.select_action(state, "B", state.legal_actions("B"))

    assert chosen[0] == "attack_after_block"
    assert "9" not in chosen[1:]
    assert agent.last_score_fallback_detail == (
        "attack_preserve_last_royal_without_continuation_9"
    )
    assert agent._track[id(state)]["my_attack_count"] == 2


def test_second_royal_attack_remains_available_with_unreceivable_third_attack() -> None:
    state, agent = _royal_attack_review_position(second_attack=True, two_kaku=True)
    actions = state.legal_actions("B")
    viable = ("attack_after_block", "1", "9")

    eligible, reserved = agent._second_attack_preserving_royal(
        state, "B", actions, agent._track[id(state)],
    )

    assert viable in eligible
    assert reserved is None


def _royal_attack_review_position(*, second_attack=False, two_kaku=False):
    # Round 1, B turn 2 of the 2026-11-27 review. Swap three middle
    # pieces for both kaku and a hisha in the positive proof control.
    hands = {"A": "36127273", "B": "49853112", "C": "41114156", "D": "21115543"}
    if two_kaku:
        hands.update(A="35123273", B="69867112", C="41114154")
    position = {
        "initial_hands": {seat: list(hand) for seat, hand in hands.items()},
        "dealer": "A", "player": "B",
        "history": [{"player": "A", "action": ["attack_after_block", "3", "2"]}],
    }
    agent = RuleBasedAgent()
    agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    state = replay(position, agent)
    if second_attack:
        apply_action(state, "B", ("receive", "2", None), agent)
        apply_action(state, "B", ("attack", None, "8"), agent)
        for seat in "CDA":
            apply_action(state, seat, ("pass", None, None), agent)
    return state, agent

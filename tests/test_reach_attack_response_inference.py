"""A late attack uses publicly missed chances to infer remaining receivers."""

import pytest

from goita_ai2.decision_check import replay
from goita_ai2.experimental_ai2 import RuleBasedAgent


HANDS = {"A": "14241751", "B": "12111524", "C": "61543932", "D": "38561371"}
HISTORY = (
    ("D", ("attack_after_block", "1", "3")),
    ("A", ("pass", None, None)),
    ("B", ("pass", None, None)),
    ("C", ("receive", "3", None)),
    ("C", ("attack", None, "6")),
    ("D", ("receive", "6", None)),
    ("D", ("attack", None, "3")),
    ("A", ("pass", None, None)),
    ("B", ("pass", None, None)),
    ("C", ("receive", "3", None)),
    ("C", ("attack", None, "5")),
    ("D", ("receive", "5", None)),
    ("D", ("attack", None, "7")),
    ("A", ("receive", "7", None)),
    ("A", ("attack", None, "5")),
    ("B", ("receive", "5", None)),
    ("B", ("attack", None, "2")),
    ("C", ("pass", None, None)),
    ("D", ("pass", None, None)),
    ("A", ("pass", None, None)),
    ("B", ("attack_after_block", "1", "2")),
    ("C", ("pass", None, None)),
    ("D", ("pass", None, None)),
    ("A", ("pass", None, None)),
    ("B", ("attack_after_block", "1", "1")),
    ("C", ("receive", "1", None)),
)


def position(*, offset=0, hands=HANDS):
    seats = {seat: "ABCD"[(index + offset) % 4] for index, seat in enumerate("ABCD")}
    player = seats["C"]
    agent = RuleBasedAgent()
    agent.TIME_SEARCH_ENABLED = False
    agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    state = replay({
        "initial_hands": {seats[seat]: list(hand) for seat, hand in hands.items()},
        "dealer": seats["D"],
        "player": player,
        "history": [{"player": seats[seat], "action": action} for seat, action in HISTORY],
    }, agent)
    return state, agent, seats


@pytest.mark.parametrize("offset", range(4))
def test_reported_lance_attack_is_selected_without_timed_search(offset):
    state, agent, seats = position(offset=offset)
    player = seats["C"]
    assert agent.select_action(state, player, state.legal_actions(player)) == ("attack", None, "2")
    assert agent.last_score_fallback_detail == "attack_reach_public_receive_risk"
    review = agent.last_attack_candidate_snapshot["reach_attack_response_review"]
    assert review["previous_attack"] == "4"
    lance, silver = review["candidates"]["2"], review["candidates"]["4"]
    assert 0 < lance["holding"][seats["B"]] < 0.20
    assert 0 < lance["holding"][seats["D"]] < 0.15
    assert lance["ready_enemy_receive_mass"] < silver["ready_enemy_receive_mass"]
    assert {item["seat"]: item["reason"] for item in lance["evidence"]} == {
        seats["B"]: "repeat_attack_switch", seats["D"]: "declined_own_finish",
    }


@pytest.mark.parametrize("enemy, spare_piece", [("B", "4"), ("D", "1")])
def test_forecast_does_not_look_at_the_actual_remaining_hands(enemy, spare_piece):
    hands = HANDS.copy()
    hands["A"] = hands["A"].replace("2", spare_piece, 1)
    prefix, suffix = hands[enemy].rsplit(spare_piece, 1)
    hands[enemy] = prefix + "2" + suffix
    expected, first, _ = position()
    alternate, second, _ = position(hands=hands)
    assert "2" not in expected.hands[enemy] and "2" in alternate.hands[enemy]
    assert first._reach_attack_response_review(expected, "C", expected.legal_actions("C")) == (
        second._reach_attack_response_review(alternate, "C", alternate.legal_actions("C"))
    )


def test_repeated_pass_is_not_counted_twice_and_ready_partner_can_be_left_to_finish():
    state, agent, _ = position()
    tracker = agent._track[id(state)]
    events = tracker["piece_pass_evidence"]["D"]["2"]
    assert len(events) == 2
    assert [event.get("response_context") for event in events] == ["declined_own_finish", None]
    events.clear()
    tracker["active_attack_context"] = {"piece": "2", "attacker": "B", "attack_no": 3}
    assert len(state.hands["B"]) == len(state.hands["D"]) == 2
    agent._record_piece_pass_evidence(state, tracker, "D")
    assert "response_context" not in events[0]


def test_generic_enemy_reuse_rule_still_applies_without_missed_opportunities():
    state, agent, _ = position()
    tracker = agent._track[id(state)]
    tracker["public_hand_models"]["B"].pop("repeat_attack_switch")
    for event in tracker["piece_pass_evidence"]["D"]["2"]:
        event.pop("response_context", None)
    assert agent.select_action(state, "C", state.legal_actions("C")) == ("attack", None, "4")
    assert agent.last_score_fallback_detail == "attack_avoid_enemy_used_piece"

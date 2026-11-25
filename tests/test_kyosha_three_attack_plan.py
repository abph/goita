"""Lance receipts must account for the remaining attacks and lost initiative."""
import copy
from collections import Counter

import pytest

from goita_ai2.constants import PIECE_TOTALS
from goita_ai2.current_ai import RuleBasedAgent as CurrentAI
from goita_ai2.decision_check import apply_action as replay_action, replay
from goita_ai2.experimental_ai2 import RuleBasedAgent
from goita_ai2.instruction_case_audit import apply_action
from goita_ai2.rule_based_intermediate_middle3 import RuleBasedAgent as Middle3
from goita_ai2.state import GoitaState


HANDS = {"A": list("15374111"), "B": list("33761513"),
         "C": list("11491525"), "D": list("84126422")}

REACH_HANDS = {"A": "73151224", "B": "51114582", "C": "71431123", "D": "91645163"}
REACH_HISTORY = [
    ("A", ("attack_after_block", "1", "7")),
    ("B", ("pass", None, None)), ("C", ("pass", None, None)), ("D", ("pass", None, None)),
    ("A", ("attack_after_block", "3", "2")),
    ("B", ("pass", None, None)), ("C", ("pass", None, None)), ("D", ("pass", None, None)),
    ("A", ("attack_after_block", "5", "2")),
]


def reach_position(*, offset=0, before_reach=False, move_hidden_lance=False):
    seats = {p: "ABCD"[(n + offset) % 4] for n, p in enumerate("ABCD")}
    hands = REACH_HANDS.copy()
    if move_hidden_lance:
        hands["C"] = hands["C"].replace("2", "3", 1)
        hands["D"] = hands["D"].replace("3", "2", 1)
    history = REACH_HISTORY[:5] if before_reach else REACH_HISTORY
    position = {
        "initial_hands": {seats[p]: list(hand) for p, hand in hands.items()},
        "dealer": seats["A"], "player": seats["B"],
        "history": [{"player": seats[p], "action": action} for p, action in history],
    }
    agent = RuleBasedAgent()
    agent.TIME_SEARCH_ENABLED = False
    agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    return replay(position, agent), agent, seats["B"], seats["D"]


@pytest.mark.parametrize("offset", range(4))
def test_lance_receive_stops_enemy_finish_even_with_ally_still_to_act(offset):
    state, agent, player, ally = reach_position(offset=offset)
    receive = agent.select_action(state, player, state.legal_actions(player))
    assert receive == ("receive", "2", None)
    assert agent.last_score_fallback_detail == "receive_prevent_enemy_finish_without_ally_cover"
    assert agent.last_rule_search_authority == "strong"
    risk = agent.last_attack_candidate_snapshot["enemy_reach_pass_risk"]
    assert risk["ally"] == ally
    assert 0.6 <= risk["no_ally_receive_mass"] < 1.0
    replay_action(state, player, receive, agent)
    assert agent.select_action(state, player, state.legal_actions(player)) == ("attack", None, "5")


def test_lance_before_enemy_reach_still_passes():
    state, agent, player, _ = reach_position(before_reach=True)
    assert agent.select_action(state, player, state.legal_actions(player)) == ("pass", None, None)
    assert "enemy_reach_pass_risk" not in agent.last_attack_candidate_snapshot


def test_enemy_reach_cover_uses_hypothetical_hands_only():
    state, agent, player, ally = reach_position()
    alternate, other_agent, _, _ = reach_position(move_hidden_lance=True)
    assert "2" not in state.hands[ally] and "2" in alternate.hands[ally]
    assert agent._enemy_reach_pass_risk(state, player) == other_agent._enemy_reach_pass_risk(alternate, player)


def test_lance_pass_can_stand_when_ally_is_likely_to_cover(monkeypatch):
    state, agent, player, ally = reach_position()
    monkeypatch.setattr(agent, "_enemy_reach_pass_risk", lambda *_: {
        "ally": ally, "no_ally_receive_mass": 0.2,
    })
    assert agent.select_action(state, player, state.legal_actions(player)) == ("pass", None, None)


def position(hands=None, offset=0):
    seats = {p: "ABCD"[(n + offset) % 4] for n, p in enumerate("ABCD")}
    hands = {seats[p]: list(hand) for p, hand in (hands or HANDS).items()}
    assert Counter(p for hand in hands.values() for p in hand) == Counter(PIECE_TOTALS)
    state = GoitaState(hands, dealer=seats["D"])
    agent = RuleBasedAgent()
    player = seats["C"]
    agent.bind_player(player)
    agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    agent._ensure_trackers(state)
    for who, action in [("D", ("attack_after_block", "4", "2")),
                        ("A", ("pass", None, None)), ("B", ("pass", None, None))]:
        assert action in state.legal_actions(seats[who])
        apply_action(state, seats[who], action)
        agent.on_public_action(state, seats[who], action)
    return state, agent, player


def with_own_hand(hand):
    pool = Counter(PIECE_TOTALS) - Counter(hand) - Counter("42")
    assert sum(pool.values()) == 22
    hands = {"A": [], "B": [], "C": list(hand), "D": list("42")}
    for piece in sorted(pool.elements()):
        seat = min((p for p in "ABD" if len(hands[p]) < 8), key=lambda p: len(hands[p]))
        hands[seat].append(piece)
    return hands


def assert_outline_resources(hand, outline):
    cards = list(hand)
    for step in outline["steps"]:
        if step["block"] is not None:
            cards.remove(step["block"])
        cards.remove(step["attack"])
        assert sorted(cards) == step["remaining_hand"]
    assert cards == []


def test_reported_vrc_lance_receipt_passes_after_three_attack_review():
    state, agent, player = position()
    action = agent.select_action(state, player, state.legal_actions(player))
    assert action == ("pass", None, None)
    assert agent.last_score_fallback_detail == "pass_kyosha_three_attack_plan_554_penalty_160"
    comparison = agent._track[id(state)]["last_round_route_comparison"]
    assert comparison["raw_receive_value"] > comparison["pass_value"] + 200
    assert comparison["receive_value"] <= comparison["pass_value"] + 200
    assert comparison["three_attack_plan_changed_response"] is True
    review = comparison["three_attack_reviews"]["5"]
    assert [step["attack"] for step in review["outline"]["steps"][:3]] == list("554")
    assert all(branch["continuation"] is None for branch in review["interception_branches"])
    assert agent._track[id(state)]["pending_kyosha_receive_attack_piece"] is None
    assert tuple(agent.last_neural_shadow["selected_action"]) == ("pass", None, None)


@pytest.mark.parametrize("offset", range(4))
@pytest.mark.parametrize("middle", ["3", "4", "5"])
def test_weak_pair_is_not_specific_to_one_seat_or_piece(offset, middle):
    mapping = {"5": middle, middle: "5"} if middle != "5" else {}
    hands = {p: [mapping.get(piece, piece) for piece in hand] for p, hand in HANDS.items()}
    # The opening hidden piece is fixed to silver by the fixture.
    if middle == "4":
        hands["D"].remove("5")
        hands["D"].append("4")
        hands["A"].remove("4")
        hands["A"].append("5")
    state, agent, player = position(hands, offset)
    review = agent._kyosha_continuation_review(state, player, middle)
    assert review["uncertainty_penalty"] >= 120
    assert review["proven"] is False
    hand = list(state.hands[player])
    hand.remove("2")
    assert_outline_resources(hand, review["outline"])


def test_repeated_attacks_can_force_royals_and_keep_a_finishing_route():
    state, agent, player = position(with_own_hand("25551194"))
    review = agent._kyosha_continuation_review(state, player, "5")
    purposes = [purpose for step in review["outline"]["steps"] for purpose in step["purposes"]]
    assert "force_royal_or_keep_lead" in purposes
    assert review["strength"] >= 2.5
    assert review["uncertainty_penalty"] < 120
    # This is an outline, not a claim that opponents must receive the golds.
    assert review["proven"] is False


def test_interception_spends_the_receive_and_rebuilds_attacks_two_and_three():
    state, agent, player = position(with_own_hand("11192255"))
    review = agent._kyosha_continuation_review(state, player, "5")
    assert review["interception_branches"]
    assert all(branch["continuation"] for branch in review["interception_branches"])
    for branch in review["interception_branches"]:
        continuation = branch["continuation"]
        assert continuation["receive"] == "2"
        hand = list(state.hands[player])
        hand.remove("2")
        hand.remove("5")
        hand.remove(continuation["receive"])
        assert_outline_resources(hand, continuation)
        assert continuation["steps"][0]["number"] == 2
    assert review["uncertainty_penalty"] < 160


def test_public_ally_signal_has_value_but_does_not_invent_hidden_hands():
    state, agent, player = position()
    before = agent._kyosha_continuation_review(state, player, "5")
    tracker = agent._track[id(state)]
    tracker["ally_first_attack"] = "5"
    tracker["ally_past_attacks"] = {"5"}
    after = agent._kyosha_continuation_review(state, player, "5")
    assert after["strength"] > before["strength"]
    assert any("support_public_ally_attack" in step["purposes"] for step in after["outline"]["steps"])
    tracker["public_hand_models"][tracker["ally"]]["strategy_broken"] = True
    assert agent._kyosha_continuation_review(state, player, "5")["strength"] == before["strength"]


def test_review_does_not_read_other_players_hidden_cards():
    state, agent, player = position()
    before = agent._kyosha_continuation_review(state, player, "5")
    state.hands["A"], state.hands["B"] = state.hands["B"], state.hands["A"]
    state.face_down_hidden["D"] = ["9"]
    assert agent._kyosha_continuation_review(state, player, "5") == before


def test_cached_or_incomplete_receive_cannot_skip_continuation_review(monkeypatch):
    state, agent, player = position()
    tracker = agent._track[id(state)]
    before = copy.deepcopy(tracker)

    def cached_receive(self, state, player, actions):
        self._set_decision_reason("response_dictionary")
        self._set_score_fallback_detail("conditional_response_receive_cached_depth_7")
        self._track[id(state)]["pending_kyosha_receive_attack_piece"] = "5"
        return ("receive", "2", None)

    monkeypatch.setattr(CurrentAI, "_select_action_with_measurement", cached_receive)
    assert agent._select_action_with_measurement(state, player, state.legal_actions(player)) == ("pass", None, None)
    assert tracker["pending_kyosha_receive_attack_piece"] == before["pending_kyosha_receive_attack_piece"]
    assert tracker["my_attack_count"] == before["my_attack_count"]
    assert tracker["last_kyosha_continuation_review"]["uncertainty_penalty"] >= 120


def test_proven_receive_is_never_blocked_by_a_weak_outline(monkeypatch):
    state, agent, player = position()

    def proven_receive(self, state, player, actions):
        self._set_decision_reason("score_fallback")
        self._set_score_fallback_detail("receive_win_after")
        return ("receive", "2", None)

    monkeypatch.setattr(CurrentAI, "_select_action_with_measurement", proven_receive)
    monkeypatch.setattr(agent, "_kyosha_continuation_review",
                        lambda *args: pytest.fail("proven receipt must keep priority"))
    assert agent._select_action_with_measurement(state, player, state.legal_actions(player)) == ("receive", "2", None)


def test_current_and_saved_middle_three_are_unchanged():
    assert not hasattr(CurrentAI(), "_kyosha_continuation_review")
    assert not hasattr(Middle3(), "_kyosha_continuation_review")

"""First late shi approval must communicate support without spending the reserve."""
import copy
from collections import Counter

import pytest

from goita_ai2.constants import PIECE_TOTALS
from goita_ai2.current_ai import RuleBasedAgent as CurrentAI
from goita_ai2.experimental_ai2 import RuleBasedAgent
from goita_ai2.instruction_case_audit import apply_action
from goita_ai2.rule_based_intermediate_middle3 import RuleBasedAgent as Middle3
from goita_ai2.state import GoitaState


HANDS = {"A": list("15531411"), "B": list("34346613"),
         "C": list("21191287"), "D": list("51152724")}
HISTORY = (
    ("A", ("attack_after_block", "1", "1")),
    ("B", ("receive", "1", None)), ("B", ("attack", None, "6")),
    ("C", ("pass", None, None)), ("D", ("pass", None, None)),
    ("A", ("pass", None, None)), ("B", ("attack_after_block", "3", "6")),
    ("C", ("receive", "9", None)), ("C", ("attack", None, "2")),
    ("D", ("receive", "2", None)), ("D", ("attack", None, "5")),
    ("A", ("receive", "5", None)), ("A", ("attack", None, "1")),
    ("B", ("pass", None, None)),
)


def position(offset=0, hands=None, timed=False, agent_class=RuleBasedAgent):
    seats = {p: "ABCD"[(n + offset) % 4] for n, p in enumerate("ABCD")}
    mapped = {seats[p]: list(hand) for p, hand in (hands or HANDS).items()}
    assert Counter(p for hand in mapped.values() for p in hand) == Counter(PIECE_TOTALS)
    state = GoitaState(mapped, dealer=seats["A"])
    player = seats["C"]
    agent = agent_class()
    agent.bind_player(player)
    agent.TIME_SEARCH_ENABLED = timed
    agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    agent._ensure_trackers(state)
    for actor, action in HISTORY:
        play(state, agent, seats[actor], action)
    return state, agent, player


def play(state, agent, actor, action):
    assert action in state.legal_actions(actor)
    apply_action(state, actor, action)
    agent.on_public_action(state, actor, action)


@pytest.mark.parametrize("timed", [False, True])
def test_reported_c_receives_and_returns_shi_once(timed):
    state, agent, player = position(timed=timed)
    tracker = agent._track[id(state)]
    receive = agent.select_action(state, player, state.legal_actions(player))
    assert receive == ("receive", "1", None)
    assert agent.last_score_fallback_detail == "receive_first_three_shi_approval"
    assert tracker["my_shi_approval_sent"] is False
    assert tracker["my_shi_approval_pending"] is True
    assert tracker["my_attack_count"] == 1
    play(state, agent, player, receive)
    assert tracker["my_shi_approval_sent"] is False
    attack = agent.select_action(state, player, state.legal_actions(player))
    assert attack == ("attack", None, "1")
    assert agent.last_score_fallback_detail == "attack_first_three_shi_approval"
    assert tuple(agent.last_neural_shadow["selected_action"]) == attack
    assert tracker["my_shi_approval_sent"] is False
    play(state, agent, player, attack)
    assert Counter(state.hands[player]) == Counter("1287")
    assert tracker["my_attack_count"] == 2
    assert tracker["my_shi_approval_sent"] is True
    assert tracker["my_shi_approval_pending"] is False
    assert tracker["shi_attack_mode_source"] == "ally_signal"
    assert "first_three_shi_approval_plan" not in tracker


@pytest.mark.parametrize("offset", range(4))
def test_first_approval_is_not_specific_to_c_or_a(offset):
    state, agent, player = position(offset)
    assert agent.select_action(state, player, state.legal_actions(player)) == ("receive", "1", None)
    evidence = agent._track[id(state)]["last_first_three_shi_approval"]
    assert evidence["own_shi_after_return"] == 1
    assert evidence["ally_remaining_shi_min"] == 0
    assert evidence["ally_remaining_shi_map"] >= 1
    assert evidence["source"] == "own_hand_and_public_inference"


def test_repeated_preview_does_not_claim_approval_was_actually_sent():
    state, agent, player = position()
    for _ in range(2):
        assert agent.select_action(state, player, state.legal_actions(player)) == ("receive", "1", None)
    tracker = agent._track[id(state)]
    assert tracker["my_attack_count"] == 1
    assert tracker["my_shi_approval_sent"] is False
    assert tracker["first_three_shi_approval_plan"]["received"] is False


@pytest.mark.parametrize("condition", ["already_sent", "self_led", "fewer_shi",
                                       "no_prior_attack", "ally_first_shi", "ally_ready",
                                       "ally_no_reserve", "ally_uncertain", "ally_stopped"])
def test_first_approval_requires_a_new_useful_signal_and_reserve(condition):
    state, agent, player = position()
    tracker = agent._track[id(state)]
    if condition == "already_sent":
        tracker["my_shi_approval_sent"] = True
    elif condition == "self_led":
        tracker["shi_attack_mode"] = True
        tracker["shi_attack_mode_source"] = "self"
    elif condition == "fewer_shi":
        state.hands[player].remove("1")
    elif condition == "no_prior_attack":
        tracker["my_attack_history"] = []
    elif condition == "ally_first_shi":
        tracker["ally_attack_history"] = ["1"]
    elif condition == "ally_ready":
        state.hands["A"] = state.hands["A"][:2]
    elif condition in ("ally_no_reserve", "ally_uncertain"):
        estimate = tracker["estimated_current_hands"]["A"]["1"]
        estimate.update(min=0, max=0 if condition == "ally_no_reserve" else 2,
                        expected=0 if condition == "ally_no_reserve" else 0.4, map_count=0)
    elif condition == "ally_stopped":
        tracker["public_hand_models"]["A"]["strategy_broken"] = True
    assert agent._first_three_shi_approval_evidence(state, player, state.legal_actions(player)) is None
    if condition == "already_sent":
        assert agent.select_action(state, player, state.legal_actions(player)) == ("pass", None, None)
        assert agent.last_score_fallback_detail == "ally_shi_approval_already_sent_pass"


def test_keeps_an_existing_pair_attack_instead_of_intercepting_for_approval():
    hands = copy.deepcopy(HANDS)
    hands["C"].remove("8")
    hands["C"].remove("2")
    hands["C"].extend("55")
    hands["A"].remove("5")
    hands["A"].append("2")
    hands["D"].remove("5")
    hands["D"].append("8")
    state, agent, player = position(hands=hands)
    assert state.hands[player].count("5") == 2
    assert agent._first_three_shi_approval_evidence(state, player, state.legal_actions(player)) is None


def test_hidden_ally_hand_does_not_determine_approval():
    state, agent, player = position()
    before = agent._first_three_shi_approval_evidence(state, player, state.legal_actions(player))
    state.hands["A"].remove("1")
    state.hands["A"].append("4")
    state.hands["B"].remove("4")
    state.hands["B"].append("1")
    assert state.hands["A"].count("1") == 0
    assert agent._first_three_shi_approval_evidence(state, player, state.legal_actions(player)) == before
    assert agent.select_action(state, player, state.legal_actions(player)) == ("receive", "1", None)


def test_another_actual_attack_cancels_the_unsent_approval():
    state, agent, player = position()
    receive = agent.select_action(state, player, state.legal_actions(player))
    play(state, agent, player, receive)
    play(state, agent, player, ("attack", None, "7"))
    tracker = agent._track[id(state)]
    assert tracker["my_shi_approval_sent"] is False
    assert tracker["my_shi_approval_pending"] is False
    assert "first_three_shi_approval_plan" not in tracker


def test_pending_return_preserves_proven_finishes(monkeypatch):
    state, agent, player = position()
    receive = agent.select_action(state, player, state.legal_actions(player))
    play(state, agent, player, receive)

    def proven(self, state, player, actions):
        self._set_decision_reason("tsume")
        self._set_score_fallback_detail("high_score_50")
        return ("attack", None, "8")

    monkeypatch.setattr(CurrentAI, "_select_rule_based_action", proven)
    assert agent._select_rule_based_action(state, player, state.legal_actions(player)) == ("attack", None, "8")
    assert agent.last_score_fallback_detail == "high_score_50"
    assert agent._track[id(state)]["my_shi_approval_pending"] is False


@pytest.mark.parametrize("agent_class", [CurrentAI, Middle3])
def test_current_and_saved_middle_three_keep_their_policy(agent_class):
    state, agent, player = position(agent_class=agent_class)
    assert agent.select_action(state, player, state.legal_actions(player)) == ("pass", None, None)
    assert not hasattr(agent, "_first_three_shi_approval_evidence")

"""Keep a guaranteed floor while waiting for a higher royal finish."""
from __future__ import annotations

import copy
from collections import Counter

import pytest

from goita_ai2.constants import PIECE_TOTALS
from goita_ai2.current_ai import RuleBasedAgent as CurrentAI
from goita_ai2.current_ai.endgame import ForcedWinResult, ForcedWinStatus
from goita_ai2.experimental_ai2 import RuleBasedAgent as ExperimentalAI2
from goita_ai2.instruction_case_audit import apply_action
from goita_ai2.rule_based_intermediate_middle3 import RuleBasedAgent as Middle3
from goita_ai2.state import GoitaState


HANDS = {"A": list("11922412"), "B": list("51115264"),
         "C": list("73431534"), "D": list("31571186")}
HISTORY = (
    ("A", ("attack_after_block", "1", "2")),
    ("B", ("pass", None, None)), ("C", ("pass", None, None)),
    ("D", ("pass", None, None)),
    ("A", ("attack_after_block", "1", "2")),
    ("B", ("receive", "2", None)), ("B", ("attack", None, "5")),
    ("C", ("pass", None, None)), ("D", ("pass", None, None)),
)


def play(state, agent, actor, action):
    assert action in state.legal_actions(actor)
    apply_action(state, actor, action)
    agent.on_public_action(state, actor, action)


def position(agent_class=ExperimentalAI2, *, seat="A", pieces=None, hands=None):
    piece_map = pieces or {}
    offset = "ABCD".index(seat)
    seats = {actor: "ABCD"[(n + offset) % 4] for n, actor in enumerate("ABCD")}
    mapped_hands = {seats[actor]: [piece_map.get(p, p) for p in hand]
                    for actor, hand in (hands or HANDS).items()}
    assert Counter(p for hand in mapped_hands.values() for p in hand) == Counter(PIECE_TOTALS)
    state = GoitaState(mapped_hands, dealer=seat)
    agent = agent_class()
    agent.bind_player(seat)
    agent.TIME_SEARCH_ENABLED = False
    agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    agent._ensure_trackers(state)
    for actor, (kind, block, attack) in HISTORY:
        mapped = (kind, piece_map.get(block, block), piece_map.get(attack, attack))
        play(state, agent, seats[actor], mapped)
    return state, agent


@pytest.mark.parametrize("seat", list("ABCD"))
def test_reported_position_passes_with_proven_floor_and_upside(seat):
    state, agent = position(seat=seat)
    chosen = agent.select_action(state, seat, state.legal_actions(seat))
    assert chosen == ("pass", None, None)
    assert agent.last_score_fallback_detail == "pass_full_receive_cover_public_route_safe_30_target_50"
    assert agent.last_rule_search_authority == "proven"
    assert agent.last_neural_shadow["safety_locked"] is True
    proof = agent._track[id(state)]["last_royal_wait_comparison"]
    assert proof["source"] == "public_information"
    assert proof["wait_minimum_score"] == proof["receive_minimum_score"] == 30
    assert proof["wait_target_score"] == 50
    assert {branch["incoming"] for branch in proof["branches"]} == set("1345678")


@pytest.mark.parametrize("pieces,floor", [
    ({"8": "9", "9": "8"}, 30),
    ({"3": "4", "4": "3"}, 20),
    ({"4": "5", "5": "4"}, 30),
])
def test_wait_is_not_limited_to_one_royal_or_middle_piece(pieces, floor):
    state, agent = position(pieces=pieces)
    assert agent.select_action(state, "A", state.legal_actions("A")) == ("pass", None, None)
    assert agent.last_score_fallback_detail == f"pass_full_receive_cover_public_route_safe_{floor}_target_50"


@pytest.mark.parametrize("incoming,block", [("1", "4"), ("4", "1")])
def test_natural_receive_keeps_control_piece_and_finishes_for_fifty(incoming, block):
    state, agent = position()
    chosen = agent.select_action(state, "A", state.legal_actions("A"))
    play(state, agent, "A", chosen)
    play(state, agent, "B", ("attack_after_block", "1", incoming))
    for actor in "CD":
        play(state, agent, actor, ("pass", None, None))
    receive = agent.select_action(state, "A", state.legal_actions("A"))
    assert receive == ("receive", incoming, None)
    play(state, agent, "A", receive)
    attack = agent.select_action(state, "A", state.legal_actions("A"))
    assert attack == ("attack", None, "2")
    play(state, agent, "A", attack)
    for actor in "BCD":
        play(state, agent, actor, ("pass", None, None))
    finish = agent.select_action(state, "A", state.legal_actions("A"))
    assert finish == ("attack_after_block", block, "9")
    play(state, agent, "A", finish)
    assert state.finished and state.winner == "A" and state.team_score["AC"] == 50


def test_wait_stops_before_enemy_can_finish_and_keeps_thirty_point_fallback():
    state, agent = position()
    for attack in ("5", "6"):
        assert agent.select_action(state, "A", state.legal_actions("A")) == ("pass", None, None)
        play(state, agent, "A", ("pass", None, None))
        play(state, agent, "B", ("attack_after_block", "1", attack))
        for actor in "CD":
            play(state, agent, actor, ("pass", None, None))
    assert len(state.hands["B"]) == 2
    receive = agent.select_action(state, "A", state.legal_actions("A"))
    assert receive == ("receive", "9", None)
    play(state, agent, "A", receive)
    attack = agent.select_action(state, "A", state.legal_actions("A"))
    assert attack == ("attack", None, "2")
    play(state, agent, "A", attack)
    for actor in "BCD":
        play(state, agent, actor, ("pass", None, None))
    finish = agent.select_action(state, "A", state.legal_actions("A"))
    play(state, agent, "A", finish)
    assert state.finished and state.team_score["AC"] == 30


def test_another_two_card_enemy_blocks_waiting():
    state, agent = position()
    # A conservative public hand-size guard, independent of its concealed faces.
    state.hands["D"] = state.hands["D"][:2]
    assert agent._full_receive_cover_royal_wait_pass_action(state, "A", state.legal_actions("A")) is None
    assert agent.select_action(state, "A", state.legal_actions("A")) == ("receive", "9", None)


def test_uncovered_shi_keeps_the_current_guaranteed_receive():
    hands = copy.deepcopy(HANDS)
    hands["A"][6], hands["C"][2] = hands["C"][2], hands["A"][6]
    state, agent = position(hands=hands)
    assert "1" not in state.hands["A"]
    assert agent._full_receive_cover_royal_wait_pass_action(state, "A", state.legal_actions("A")) is None
    assert agent.select_action(state, "A", state.legal_actions("A")) == ("receive", "9", None)


def test_control_piece_must_be_publicly_unreceivable():
    hands = copy.deepcopy(HANDS)
    hands["A"][7], hands["D"][0] = hands["D"][0], hands["A"][7]
    state, agent = position(hands=hands)
    assert agent._public_unknown_piece_pool(state, "A", agent._track[id(state)])["2"] == 1
    assert agent._full_receive_cover_royal_wait_pass_action(state, "A", state.legal_actions("A")) is None


@pytest.mark.parametrize("result", [
    ForcedWinResult(ForcedWinStatus.UNKNOWN),
    ForcedWinResult(ForcedWinStatus.COUNTEREXAMPLE),
    ForcedWinResult(ForcedWinStatus.PROVEN, 20, 20, 20),
    ForcedWinResult(ForcedWinStatus.PROVEN, 30, 30, 30),
])
def test_unknown_lower_score_or_no_upside_does_not_replace_proven_receive(monkeypatch, result):
    state, agent = position()
    monkeypatch.setattr(agent, "_forced_win_resolve_external_return", lambda *a, **k: result)
    assert agent._full_receive_cover_royal_wait_pass_action(state, "A", state.legal_actions("A")) is None
    assert agent.select_action(state, "A", state.legal_actions("A")) == ("receive", "9", None)


def test_hidden_piece_reallocation_does_not_change_public_wait():
    state, agent = position()
    cards = [piece for seat in "BCD" for piece in state.hands[seat]][::-1]
    offset = 0
    for seat in "BCD":
        size = len(state.hands[seat])
        state.hands[seat] = cards[offset:offset + size]
        offset += size
    assert agent.select_action(state, "A", state.legal_actions("A")) == ("pass", None, None)


def test_proven_wait_cannot_be_overturned_by_timed_search_or_neural_receive(monkeypatch):
    state, agent = position()
    agent.TIME_SEARCH_ENABLED = True

    def unexpected_search(*args, **kwargs):
        pytest.fail("A publicly proven wait must skip sampled search")

    class ReceiveModel:
        def rank_actions(self, payload, actions):
            receive = ("receive", "9", None)
            return [(receive, 100)] + [(action, 0) for action in actions if action != receive]

    monkeypatch.setattr(agent, "_time_limited_search_action", unexpected_search)
    monkeypatch.setattr(ExperimentalAI2, "_shared_neural_model", ReceiveModel())
    monkeypatch.setattr(ExperimentalAI2, "_shared_neural_error", None)
    assert agent.select_action(state, "A", state.legal_actions("A")) == ("pass", None, None)
    assert agent.last_search_skip_reason == "proven_rule"
    assert agent.last_neural_shadow["safety_locked"]
    assert not agent.last_neural_shadow["applied"]


def test_no_current_guaranteed_receive_means_no_proven_wait(monkeypatch):
    state, agent = position()
    monkeypatch.setattr(agent, "_forced_win_result_after_receive_action",
                        lambda *a, **k: ForcedWinResult(ForcedWinStatus.UNKNOWN))
    assert agent._full_receive_cover_royal_wait_pass_action(state, "A", state.legal_actions("A")) is None


@pytest.mark.parametrize("agent_class", [CurrentAI, Middle3])
def test_saved_and_existing_ai_profiles_keep_their_decision(agent_class):
    state, agent = position(agent_class)
    assert agent.select_action(state, "A", state.legal_actions("A")) == ("receive", "9", None)

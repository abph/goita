"""Endgame estimates compare public-consistent worlds, never actual hidden hands."""
import copy
import random
from collections import Counter

import pytest

from goita_ai2.constants import PIECE_TOTALS
from goita_ai2.current_ai import RuleBasedAgent as CurrentAI
from goita_ai2.experimental_ai2 import RuleBasedAgent
from goita_ai2.instruction_case_audit import apply_action
from goita_ai2.rule_based_intermediate_middle3 import RuleBasedAgent as Middle3
from goita_ai2.state import GoitaState


HANDS = {"A": list("33512427"), "B": list("18151622"),
         "C": list("11165413"), "D": list("54711493")}
HISTORY = (
    ("A", ("attack_after_block", "1", "3")),
    ("B", ("pass", None, None)), ("C", ("pass", None, None)),
    ("D", ("receive", "3", None)), ("D", ("attack", None, "4")),
    ("A", ("pass", None, None)), ("B", ("pass", None, None)),
    ("C", ("receive", "4", None)), ("C", ("attack", None, "3")),
    ("D", ("pass", None, None)), ("A", ("pass", None, None)),
    ("B", ("pass", None, None)), ("C", ("attack_after_block", "5", "1")),
    ("D", ("receive", "1", None)), ("D", ("attack", None, "7")),
    ("A", ("receive", "7", None)), ("A", ("attack", None, "2")),
    ("B", ("receive", "2", None)), ("B", ("attack", None, "6")),
    ("C", ("pass", None, None)), ("D", ("pass", None, None)),
    ("A", ("pass", None, None)), ("B", ("attack_after_block", "1", "5")),
    ("C", ("pass", None, None)), ("D", ("pass", None, None)),
    ("A", ("receive", "5", None)), ("A", ("attack", None, "2")),
    ("B", ("receive", "2", None)), ("B", ("attack", None, "1")),
    ("C", ("receive", "1", None)), ("C", ("attack", None, "6")),
)


def play(state, agent, actor, action):
    assert action in state.legal_actions(actor)
    apply_action(state, actor, action)
    agent.on_public_action(state, actor, action)


def position(offset=0, timed=False, agent_class=RuleBasedAgent):
    seats = {p: "ABCD"[(i + offset) % 4] for i, p in enumerate("ABCD")}
    hands = {seats[p]: list(hand) for p, hand in HANDS.items()}
    assert Counter(p for hand in hands.values() for p in hand) == Counter(PIECE_TOTALS)
    state = GoitaState(hands, dealer=seats["A"])
    player = seats["D"]
    agent = agent_class()
    agent.bind_player(player)
    agent.TIME_SEARCH_ENABLED = timed
    agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    agent._ensure_trackers(state)
    for actor, action in HISTORY:
        play(state, agent, seats[actor], action)
    return state, agent, player


@pytest.mark.parametrize("timed", [False, True])
def test_report_receives_then_compares_gold_and_silver_again(timed):
    state, agent, player = position(timed=timed)
    receive = agent.select_action(state, player, state.legal_actions(player))
    assert receive == ("receive", "9", None)
    tracker = agent._track[id(state)]
    before = tracker["last_belief_endgame"]
    assert before["followup"] == ("attack", None, "5")
    assert len([row for row in before["candidates"] if row["action"] == receive]) == 3
    play(state, agent, player, receive)
    attack = agent.select_action(state, player, state.legal_actions(player))
    assert attack == ("attack", None, "5")
    after = tracker["last_belief_endgame"]
    assert after is not before
    assert after["source"] == "public_action_weighted_worlds"
    assert after["proven"] is False
    assert after["unique_worlds"] > 1
    assert 8 <= after["worlds"] <= agent.BELIEF_ENDGAME_WORLDS
    assert after["nodes"] <= agent.BELIEF_ENDGAME_NODES
    scores = {row["action"][2]: row for row in after["candidates"]}
    assert scores["5"]["wins"] > scores["4"]["wins"]
    assert scores["5"]["value"] > scores["4"]["value"]
    assert agent.last_score_fallback_detail.startswith("belief_endgame_worlds_")
    snapshot = agent.last_attack_candidate_snapshot
    assert snapshot["belief_endgame"]["action"] == attack
    assert all(row["score"] is not None for row in snapshot["alternatives"])


@pytest.mark.parametrize("offset", range(4))
def test_report_tactic_is_not_tied_to_d_seat(offset):
    state, agent, player = position(offset=offset)
    receive = agent.select_action(state, player, state.legal_actions(player))
    assert receive == ("receive", "9", None)
    play(state, agent, player, receive)
    assert agent.select_action(state, player, state.legal_actions(player)) == ("attack", None, "5")


def test_conventions_do_not_exclude_observed_actual_possibilities():
    state, agent, _ = position()
    tracker = agent._track[id(state)]
    estimates = tracker["other_piece_count_estimates"]
    assert estimates["B"]["8"]["max"] >= 1  # Third shi does not prove no gyoku.
    # The endgame posterior admits A's retained horse and C's singleton return,
    # independently of the conventional ranges used by ordinary opening rules.
    prediction = agent._information_set_prediction(state, "D")
    assert agent._probabilistic_candidate_action_evidence(state, "D", tracker, prediction) is not None
    assert tracker["public_hand_models"]["B"]["soft_third_nonroyal"]
    assert tracker["public_hand_models"]["C"]["soft_opening_response_piece"] == "3"
    assert tracker["public_hand_models"]["B"]["soft_unreturned_opening"]["piece"] == "4"
    # Not returning partner's silver is negative evidence, never an impossibility.
    assert agent._behaviour_piece_likelihood(tracker, "B", "4", 0) == 1
    assert 0 < agent._behaviour_piece_likelihood(tracker, "B", "4", 1) < 1


def test_same_public_information_gives_same_comparison_despite_other_actual_cards():
    state, agent, player = position()
    baseline = agent._belief_endgame_comparison(state, player, state.legal_actions(player))
    other, second, _ = position()
    # This alternate physical world swaps only concealed information; no public event changes.
    other.hands["A"], other.hands["B"] = other.hands["B"], other.hands["A"]
    other.face_down_hidden["A"], other.face_down_hidden["B"] = (
        other.face_down_hidden["B"], other.face_down_hidden["A"])
    other.had_both_kings["A"] = not other.had_both_kings["A"]
    random.seed(8675309)
    result = second._belief_endgame_comparison(other, player, other.legal_actions(player))
    assert result is not None and baseline is not None
    for key in ("action", "followup", "candidates", "worlds", "unique_worlds", "nodes"):
        assert result[key] == baseline[key]


def test_publicly_impossible_candidate_is_rejected():
    state, agent, player = position()
    impossible = copy.deepcopy(state)
    impossible.hands["A"] = ["9", "9"]
    prediction = agent._information_set_prediction(impossible, player)
    assert agent._probabilistic_candidate_action_evidence(
        state, player, agent._track[id(state)], prediction) is None


def test_comparison_reuses_public_position_cache(monkeypatch):
    state, agent, player = position()
    actions = state.legal_actions(player)
    result = agent._belief_endgame_comparison(state, player, actions)
    assert result is not None

    def must_not_resample(*args):
        raise AssertionError("An unchanged public position must reuse its comparison")

    monkeypatch.setattr(agent, "_probabilistic_uniform_prediction", must_not_resample)
    assert agent._belief_endgame_comparison(state, player, actions) is result


def test_partial_completed_worlds_are_not_reported_as_a_finished_comparison(monkeypatch):
    state, agent, player = position()
    calls = 0

    def stop_after_one_world(*args):
        nonlocal calls
        calls += 1
        return (100010, player, 10, ()) if calls <= 4 else None

    monkeypatch.setattr(agent, "_solve_inferred_endgame", stop_after_one_world)
    assert agent._belief_endgame_comparison(state, player, state.legal_actions(player)) is None
    assert calls == 5
    assert not agent._track[id(state)].get("last_belief_endgame")


@pytest.mark.parametrize("limit", ["time", "nodes"])
def test_incomplete_comparison_drops_stale_followup_instead_of_claiming_one_world_win(limit):
    state, agent, player = position()
    if limit == "time":
        agent.BELIEF_ENDGAME_SECONDS = 0
    else:
        agent.BELIEF_ENDGAME_NODES = 1
    tracker = agent._track[id(state)]
    tracker["pending_inferred_endgame_attack"] = ("attack", None, "4")
    assert agent._inferred_endgame_team_result_action(state, player, state.legal_actions(player)) is None
    assert tracker["pending_inferred_endgame_attack"] is None
    assert not tracker.get("last_belief_endgame")


def test_pending_attack_is_not_reused_after_comparison_times_out():
    state, agent, player = position()
    receive = agent.select_action(state, player, state.legal_actions(player))
    play(state, agent, player, receive)
    agent.BELIEF_ENDGAME_SECONDS = 0
    agent._select_rule_based_action(state, player, state.legal_actions(player))
    assert agent._track[id(state)]["pending_inferred_endgame_attack"] is None
    assert not agent.last_score_fallback_detail.startswith("belief_endgame_")


def test_publicly_proven_receive_is_never_replaced_by_sampled_higher_score(monkeypatch):
    state, agent, player = position()
    state.hands[player] = ["9", "7"]
    actions = state.legal_actions(player)
    assert agent._guaranteed_finish_receive_action(state, player, actions) == ("receive", "9", None)

    def must_not_sample(*args):
        raise AssertionError("A sample estimate cannot override a proven finish")

    monkeypatch.setattr(agent, "_belief_endgame_comparison", must_not_sample)
    assert agent._inferred_endgame_team_result_action(state, player, actions) is None
    assert agent.select_action(state, player, actions) == ("receive", "9", None)


@pytest.mark.parametrize("agent_class", [CurrentAI, Middle3])
def test_saved_profiles_keep_their_original_decision(agent_class):
    state, agent, player = position(agent_class=agent_class)
    receive = agent.select_action(state, player, state.legal_actions(player))
    assert receive == ("receive", "9", None)
    play(state, agent, player, receive)
    assert agent.select_action(state, player, state.legal_actions(player)) == ("attack", None, "4")
    assert "last_belief_endgame" not in agent._track[id(state)]

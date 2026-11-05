"""Public response inference and fair, bounded upside searches (experimental AI2)."""
import copy
import json
import math
from collections import Counter
from types import SimpleNamespace

import pytest

from goita_ai2.constants import PIECE_TOTALS
from goita_ai2.current_ai import RuleBasedAgent as CurrentAI
from goita_ai2.current_ai.information_set_search import (
    InformationSetSearchOutcome, InformationSetSearchDeadline, InformationSetSearchCancelled,
)
from goita_ai2.experimental_ai2 import RuleBasedAgent
from goita_ai2.experimental_ai2.upside_search import UpsideSearchMixin
from goita_ai2.instruction_case_audit import apply_action
from goita_ai2.rule_based_intermediate_middle3 import RuleBasedAgent as Middle3
from goita_ai2.state import GoitaState


HANDS = {"A": list("42596414"), "B": list("11733365"),
         "C": list("11755411"), "D": list("21283121")}
HISTORY = (
    ("A", ("attack_after_block", "1", "4")),
    ("B", ("pass", None, None)), ("C", ("pass", None, None)),
    ("D", ("pass", None, None)),
    ("A", ("attack_after_block", "6", "4")),
    ("B", ("pass", None, None)), ("C", ("pass", None, None)),
    ("D", ("receive", "8", None)), ("D", ("attack", None, "2")),
    ("A", ("receive", "2", None)),
)


def position(*, offset=0, timed=False, score=0):
    seats = {p: "ABCD"[(i + offset) % 4] for i, p in enumerate("ABCD")}
    hands = {seats[p]: list(hand) for p, hand in HANDS.items()}
    assert Counter(p for hand in hands.values() for p in hand) == Counter(PIECE_TOTALS)
    state = GoitaState(hands, dealer=seats["A"])
    player = seats["A"]
    state.team_score["AC" if player in "AC" else "BD"] = score
    agent = RuleBasedAgent()
    agent.bind_player(player)
    agent.TIME_SEARCH_ENABLED = timed
    agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    agent.TIME_SEARCH_MAX_SECONDS = 1
    agent.TIME_SEARCH_MAX_DEPTH = 11
    agent.UPSIDE_FINISH_MAX_SECONDS = 2
    # Fixed completed depth makes the probability/seat tests independent of CPU
    # load. The reported runtime settings are exercised separately below.
    agent.UPSIDE_FINISH_MAX_DEPTH = 5
    agent._ensure_trackers(state)
    for actor, action in HISTORY:
        actor = seats[actor]
        assert action in state.legal_actions(actor)
        apply_action(state, actor, action)
        agent.on_public_action(state, actor, action)
    return state, agent, player, seats


@pytest.mark.parametrize("offset", range(4))
def test_report_keeps_royal_and_finishes_for_fifty(offset):
    state, agent, player, seats = position(offset=offset, timed=(offset == 0))
    if offset == 0:
        agent.UPSIDE_FINISH_MAX_SECONDS = 10
        agent.UPSIDE_FINISH_MAX_DEPTH = 15
    action = agent.select_action(state, player, state.legal_actions(player))
    assert action == ("attack", None, "4")
    assert agent.last_decision_reason == "upside_finish"
    assert agent.last_score_fallback_detail.endswith("_public_response_4")
    metrics = agent.last_upside_finish_metrics
    assert metrics["safe_score"] == 30 and metrics["maximum_score"] == 50
    assert metrics["proven"] is False
    assert metrics["depth"] >= 5
    assert metrics["unknown_probability"] <= agent.UPSIDE_FINISH_MAX_UNKNOWN_PROBABILITY
    assert metrics["adjusted_failure_risk"] <= metrics["allowed_failure_risk"]
    assert agent.last_attack_candidate_snapshot["upside_finish"] == metrics
    json.dumps(agent.last_attack_candidate_snapshot)  # Report export remains valid JSON.
    assert len(agent.last_attack_candidate_snapshot["upside_finish_comparison"]["candidates"]) == 2
    rows = agent._track[id(state)]["last_upside_finish_comparison"]["candidates"]
    assert all(row["completed_depth"] is not None for row in rows.values())
    apply_action(state, player, action)
    for p in ("B", "C", "D"):
        actor = seats[p]
        apply_action(state, actor, ("pass", None, None))
    apply_action(state, player, ("attack_after_block", "5", "9"))
    assert state.finished and state.winner == player
    assert state.team_score["AC" if player in "AC" else "BD"] == 50


def test_guaranteed_match_win_is_not_risked_for_fifty():
    state, agent, player, _ = position(score=120)
    assert agent.select_action(state, player, state.legal_actions(player)) == ("attack", None, "9")
    assert agent.last_decision_reason == "tsume"
    assert not agent._track[id(state)].get("last_upside_finish_comparison")


def test_royal_receive_and_near_reach_pass_are_soft_public_evidence():
    state, agent, player, _ = position()
    tr = agent._track[id(state)]
    event, = tr["public_hand_models"]["D"]["royal_instead_of_same_evidence"]
    assert event["piece"] == "4" and event["hand_slots"] == 7
    assert tr["piece_pass_evidence"]["B"]["4"][-1]["response_context"] == "enemy_near_reach"
    assert "response_context" not in tr["piece_pass_evidence"]["C"]["4"][-1]
    actual = agent._information_set_prediction(state, player)
    baseline = agent._information_set_candidate_evidence(actual, player, tr)[0]
    # Same visible history, but the enemy actually holds the fourth silver.
    alternate = copy.deepcopy(state)
    alternate.hands["C"].remove("4"); alternate.hands["C"].append("3")
    alternate.hands["D"].remove("3"); alternate.hands["D"].append("4")
    prediction = agent._information_set_prediction(alternate, player)
    assert agent._information_set_candidate_consistent(state, player, tr, prediction)
    weighted = agent._information_set_candidate_evidence(prediction, player, tr)
    assert weighted is not None and math.isfinite(weighted[0]) and weighted[0] < baseline
    assert agent._probabilistic_candidate_action_evidence(state, player, tr, prediction) is not None
    # The public proof must still admit that counterexample.
    assert agent._forced_win_result_after_attack_action(state, player, ("attack", None, "4")).minimum_score is None


@pytest.mark.parametrize("piece", ["3", "4", "5", "6", "7"])
def test_same_piece_inference_is_not_silver_specific(piece):
    state, agent, _, _ = position()
    tr = copy.deepcopy(agent._track[id(state)])
    tr["active_attack_context"] = {"piece": piece, "attacker": "A"}
    model = tr["public_hand_models"]["D"]
    model.pop("royal_instead_of_same_evidence", None)
    agent._update_public_hand_model(state, tr, "D", "receive", "8", None)
    assert model["royal_instead_of_same_evidence"][0]["piece"] == piece


@pytest.mark.parametrize("actor,size", [("C", 7), ("D", 3), ("D", 1)])
def test_partner_receives_and_finishing_royal_receives_do_not_imply_absence(actor, size):
    state, agent, _, _ = position()
    tr = copy.deepcopy(agent._track[id(state)])
    model = tr["public_hand_models"][actor]
    model.pop("royal_instead_of_same_evidence", None)
    tr["active_attack_context"] = {"piece": "4", "attacker": "A"}
    state.hands[actor] = state.hands[actor][:size]
    agent._update_public_hand_model(state, tr, actor, "receive", "8", None)
    assert not model.get("royal_instead_of_same_evidence")


def test_hidden_copies_do_not_count_as_held_receivers():
    event = {"observed_used": 0, "hidden_slots": 1, "hand_slots": 1}
    assert RuleBasedAgent._response_likelihood(event, 1, .12) == pytest.approx(.56)
    assert RuleBasedAgent._response_likelihood(event, 0, .12) == 1


def test_actual_opponent_cards_cannot_change_decision():
    state, agent, player, _ = position()
    other, second, _, _ = position()
    other.hands["B"], other.hands["C"] = other.hands["C"], other.hands["B"]
    assert agent.select_action(state, player, state.legal_actions(player)) == second.select_action(
        other, player, other.legal_actions(player)) == ("attack", None, "4")
    for key in ("high_score_probability", "unknown_probability", "safe_retention_probability"):
        assert agent.last_upside_finish_metrics[key] == pytest.approx(second.last_upside_finish_metrics[key])


def test_saved_profiles_keep_their_original_planners():
    for cls in (CurrentAI, Middle3):
        assert not issubclass(cls, UpsideSearchMixin)
        assert not hasattr(cls, "_public_response_evidence")


def test_slow_first_candidate_cannot_hide_second_candidate(monkeypatch):
    agent = RuleBasedAgent()
    agent.UPSIDE_FINISH_MAX_DEPTH = 7
    tracker, calls = {}, []
    actions = [(('attack', None, '5'), 50), (('attack', None, '4'), 50)]
    outcome = InformationSetSearchOutcome(125000, ((0, 125000),), 0)
    def search(state, worlds, player, action, information_set, baseline, depth, deadline, stats, tr, cancel):
        calls.append((action[2], depth))
        if action[2] == '5':
            raise InformationSetSearchDeadline()
        return outcome
    monkeypatch.setattr(agent, "_information_set_search_root_action", search)
    monkeypatch.setattr('goita_ai2.experimental_ai2.upside_search.time.perf_counter', lambda: 0)
    completed, _ = agent._search_upside_candidates(
        None, 'A', tracker, actions, [SimpleNamespace(index=0)], None, {}, 10)
    assert ('4', 5) in calls
    assert completed[actions[1][0]] == (5, outcome)


def test_interrupted_deeper_iteration_keeps_completed_result(monkeypatch):
    agent = RuleBasedAgent(); agent.UPSIDE_FINISH_MAX_DEPTH = 7
    action = ('attack', None, '4'); tracker = {}
    outcome = InformationSetSearchOutcome(0, ((0, 0),), 0)
    def search(*args):
        if args[6] == 7:
            raise InformationSetSearchDeadline()
        return outcome
    monkeypatch.setattr(agent, '_information_set_search_root_action', search)
    monkeypatch.setattr('goita_ai2.experimental_ai2.upside_search.time.perf_counter', lambda: 0)
    completed, _ = agent._search_upside_candidates(
        None, 'A', tracker, [(action, 50)], [SimpleNamespace(index=0)], None, {}, 10)
    assert completed[action] == (5, outcome)
    assert tracker['last_upside_finish_comparison']['candidates'][action]['completed_depth'] == 5


def test_cancelled_search_discards_all_previous_results(monkeypatch):
    agent = RuleBasedAgent(); tracker = {}
    actions = [(('attack', None, '5'), 50), (('attack', None, '4'), 50)]
    def cancel(*args):
        if args[3][2] == '5':
            return InformationSetSearchOutcome(0, ((0, 0),), 0)
        raise InformationSetSearchCancelled()
    monkeypatch.setattr(agent, '_information_set_search_root_action', cancel)
    monkeypatch.setattr('goita_ai2.experimental_ai2.upside_search.time.perf_counter', lambda: 0)
    completed, _ = agent._search_upside_candidates(
        None, 'A', tracker, actions, [SimpleNamespace(index=0)], None, {}, 10)
    assert not completed and tracker['last_upside_finish_comparison']['cancelled']


def test_unknown_or_losing_comparisons_keep_proven_thirty(monkeypatch):
    state, agent, player, _ = position()
    # Simulate a completed comparison that finds only losses or unresolved
    # positions. Neither may be promoted to a presumed fifty-point finish.
    original = agent._information_set_search_root_action
    for value in (-125000, 0):
        def outcome(state, worlds, *args):
            return InformationSetSearchOutcome(value, tuple((w.index, value) for w in worlds), 0)
        monkeypatch.setattr(agent, '_information_set_search_root_action', outcome)
        assert agent._upside_finish_action(
            state, player, state.legal_actions(player), ('attack', None, '9'), 30) is None
    monkeypatch.setattr(agent, '_information_set_search_root_action', original)


def test_node_budget_is_shared_with_later_candidates(monkeypatch):
    agent = RuleBasedAgent(); agent.UPSIDE_FINISH_MAX_NODES = 20
    agent.UPSIDE_FINISH_MAX_DEPTH = 5
    actions = [(('attack', None, '5'), 50), (('attack', None, '4'), 50)]
    allocations = []
    def spend(*args):
        stats = args[8]
        allocations.append((args[3][2], stats['max_nodes']))
        stats['nodes'] = stats['max_nodes']
        raise InformationSetSearchDeadline()
    monkeypatch.setattr(agent, '_information_set_search_root_action', spend)
    monkeypatch.setattr('goita_ai2.experimental_ai2.upside_search.time.perf_counter', lambda: 0)
    completed, nodes = agent._search_upside_candidates(
        None, 'A', {}, actions, [], None, {}, 10)
    assert allocations == [('5', 10), ('4', 10)]
    assert nodes == 20 and not completed


def test_promising_completed_candidate_is_deepened_first(monkeypatch):
    agent = RuleBasedAgent(); agent.UPSIDE_FINISH_MAX_DEPTH = 7
    calls = []
    def search(*args):
        attack, depth = args[3][2], args[6]
        calls.append((attack, depth))
        value = 100 if attack == '4' else 10
        return InformationSetSearchOutcome(value, ((0, value),), 0)
    monkeypatch.setattr(agent, '_information_set_search_root_action', search)
    monkeypatch.setattr('goita_ai2.experimental_ai2.upside_search.time.perf_counter', lambda: 0)
    agent._search_upside_candidates(None, 'A', {}, [
        (('attack', None, '5'), 50), (('attack', None, '4'), 50),
    ], [SimpleNamespace(index=0)], None, {}, 10)
    assert calls == [('5', 5), ('4', 5), ('4', 7), ('5', 7)]

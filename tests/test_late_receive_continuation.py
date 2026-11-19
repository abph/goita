"""AI2 compares late receipts and complete continuations without a fixed pass tax."""
import copy
from collections import Counter
from types import SimpleNamespace

import pytest

from goita_ai2.experimental_ai2 import RuleBasedAgent
from goita_ai2.current_ai.agent import RuleBasedAgent as CurrentAgent
from goita_ai2.current_ai.endgame import ForcedWinStatus
from goita_ai2.current_ai import generic_response_store as stores
from goita_ai2.state import GoitaState


@pytest.fixture(autouse=True)
def independent_store(monkeypatch):
    monkeypatch.setattr(stores, "_GENERIC_RESPONSE_PATTERN_STORE", stores.GenericResponsePatternStore())


def position(*, rotate=0, swap_royals=False, agent_class=RuleBasedAgent):
    seats = {seat: "ABCD"[(index + rotate) % 4] for index, seat in enumerate("ABCD")}
    piece = lambda p: {"8": "9", "9": "8"}.get(p, p) if swap_royals else p
    hands = {"A": "31827714", "B": "31443231", "C": "25161115", "D": "46259151"}
    state = GoitaState({seats[s]: [piece(p) for p in h] for s, h in hands.items()}, dealer=seats["A"])
    agent = agent_class()
    agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    agent._relative_hand_rank_table = {}
    agent.bind_player(seats["A"])
    agent._ensure_trackers(state)
    history = [
        ("A", ("attack_after_block", "1", "7")),
        ("B", ("pass", None, None)), ("C", ("pass", None, None)), ("D", ("pass", None, None)),
        ("A", ("attack_after_block", "3", "7")),
        ("B", ("pass", None, None)), ("C", ("pass", None, None)),
        ("D", ("receive", "9", None)), ("D", ("attack", None, "5")),
        ("A", ("pass", None, None)), ("B", ("pass", None, None)),
        ("C", ("receive", "5", None)), ("C", ("attack", None, "6")),
        ("D", ("receive", "6", None)), ("D", ("attack", None, "5")),
    ]
    for seat, action in history:
        actor = seats[seat]
        kind, block, attack = (action[0], piece(action[1]), piece(action[2]))
        if kind == "pass": state.apply_pass(actor)
        elif kind == "receive": state.apply_receive(actor, block)
        elif kind == "attack": state.apply_attack(actor, attack)
        else: state.apply_attack_after_block(actor, block, attack)
        agent.on_public_action(state, actor, (kind, block, attack))
    return state, agent, seats["A"]


def test_only_ai2_removes_late_non_shi_offsets_and_preserves_other_terms():
    state, agent, seat = position()
    for action, offset in [(("pass", None, None), -10000), (("receive", "8", None), 1000)]:
        raw = CurrentAgent._score_receive_phase(agent, state, seat, action[0], action[1])
        assert agent._score_receive_phase(state, seat, action[0], action[1]) == raw - offset
    tracker = agent._track[id(state)]
    for attack, count in [("5", 1), ("1", 2)]:
        state.current_attack = attack
        tracker["enemy_attack_counts"][state.attacker] = count
        for kind, block in [("pass", None), ("receive", "8")]:
            assert agent._score_receive_phase(state, seat, kind, block) == CurrentAgent._score_receive_phase(
                agent, state, seat, kind, block)


def test_proven_receive_is_not_reweighted(monkeypatch):
    state, agent, seat = position()
    monkeypatch.setattr(agent, "_forced_win_result_after_receive_action", lambda *args:
                        SimpleNamespace(status=ForcedWinStatus.PROVEN, minimum_score=30))
    assert agent._score_receive_phase(state, seat, "receive", "8") == 1e8 + 30


def test_same_piece_receive_also_loses_the_fixed_ten_thousand_bonus(monkeypatch):
    state, agent, seat = position()
    state.current_attack = "4"
    monkeypatch.setattr(agent, "_forced_win_result_after_receive_action", lambda *args:
                        SimpleNamespace(status=ForcedWinStatus.COUNTEREXAMPLE, minimum_score=None))
    original = CurrentAgent._score_receive_phase(agent, state, seat, "receive", "4")
    assert agent._score_receive_phase(state, seat, "receive", "4") == original - 10000


@pytest.mark.parametrize("rotate,swap_royals", [(0, False), (1, False), (0, True)])
def test_report_passes_in_rotated_and_other_royal_positions(rotate, swap_royals):
    state, agent, seat = position(rotate=rotate, swap_royals=swap_royals)
    hands_before = copy.deepcopy(state.hands)
    assert agent.select_action(state, seat, state.legal_actions(seat)) == ("pass", None, None)
    result = agent._track[id(state)]["last_late_receive_comparison"]
    assert result["applied"] and result["proven"] is False
    assert (result["pass_value"] > result["receive_value"]
            or result["selection_basis"] == "royal_reserve_within_sample_dispersion")
    if result["selection_basis"] == "royal_reserve_within_sample_dispersion":
        assert abs(result["pass_value"] - result["receive_value"]) <= result["dispersion_band"] + 0.01
    assert result["public_bridge_hits"] > 0
    assert result["public_bridge_scores"] == [20, 30]
    assert all(len(plan["remaining"]) == 2
               for hand in result["attacker_hypotheses"] for plan in hand["plans"])
    assert 0 <= result["forced_round_return_mass"] <= 1
    assert "_pass" in agent.last_score_fallback_detail
    assert agent.last_rule_search_authority == "strong"
    assert agent.last_attack_candidate_snapshot["late_receive_comparison"]["chosen"] == (("pass", None, None),)
    assert state.hands == hands_before


def test_bridge_uses_public_royal_counts_and_requires_a_natural_receive(monkeypatch):
    state, agent, seat = position()
    tracker = agent._track[id(state)]
    for attack, score in [("1", 30), ("2", 30), ("4", 20)]:
        state.current_attack = attack
        assert agent._late_public_royal_bridge_score(state, seat, tracker) == score
    state.current_attack = "5"
    assert agent._late_public_royal_bridge_score(state, seat, tracker) is None
    state.current_attack = "1"
    monkeypatch.setattr(agent, "_public_unknown_piece_pool", lambda *args: Counter({"9": 1}))
    assert agent._late_public_royal_bridge_score(state, seat, tracker) is None


def test_actual_enemy_cards_do_not_change_comparison():
    results = []
    for swap in (False, True):
        state, agent, seat = position()
        if swap:
            state.hands["B"].remove("3"); state.hands["C"].remove("1")
            state.hands["B"].append("1"); state.hands["C"].append("3")
        result = agent._compare_late_receive_routes(state, seat, state.legal_actions(seat))
        assert result is not None
        result.pop("elapsed_ms")
        results.append(result)
    assert results[0] == results[1]


def test_incomplete_comparison_and_immediate_threat_do_not_install_a_route():
    state, agent, seat = position()
    agent.LATE_RECEIVE_ROUTE_SECONDS = 0
    before = copy.deepcopy(state.hands)
    assert agent._compare_late_receive_routes(state, seat, state.legal_actions(seat)) is None
    assert getattr(agent, "_late_receive_route_context", None) is None
    assert "pending_late_receive_route" not in agent._track[id(state)]
    assert state.hands == before
    state.hands["D"] = state.hands["D"][:2]
    assert agent._compare_late_receive_routes(state, seat, state.legal_actions(seat)) is None


def test_receiving_can_win_the_comparison_and_its_followup_is_kept(monkeypatch):
    state, agent, seat = position()
    receive, attack = ("receive", "8", None), ("attack", None, "4")
    comparison = {"chosen": (receive, attack), "margin": 1000, "depth": 8, "samples": 16,
                  "proven": False, "reliable": True, "routes": [{"actions": [("pass", None, None)], "value": 0},
                                              {"actions": [receive, attack], "value": 1000}]}
    monkeypatch.setattr(agent, "_compare_late_receive_routes", lambda *args: copy.deepcopy(comparison))
    assert agent.select_action(state, seat, state.legal_actions(seat)) == receive
    before = agent._track[id(state)]["my_attack_count"]
    state.apply_receive(seat, "8")
    agent.on_public_action(state, seat, receive)
    assert agent.select_action(state, seat, state.legal_actions(seat)) == attack
    assert agent.last_score_fallback_detail == "late_receive_followup_4"
    assert agent._track[id(state)]["my_attack_count"] == before + 1


def test_unchanged_current_profile_still_has_the_original_offsets():
    state, agent, seat = position(agent_class=CurrentAgent)
    assert agent._score_receive_phase(state, seat, "pass", None) < -10000
    assert agent._score_receive_phase(state, seat, "receive", "8") > 1000


def test_reserve_preference_requires_full_receive_cover_and_no_repeat_attack():
    state, agent, seat = position()
    tracker = agent._track[id(state)]
    assert agent._late_royal_reserve_wait_available(state, seat, tracker, state.legal_actions(seat))
    state.hands[seat].remove("4"); state.hands[seat].append("2")
    assert not agent._late_royal_reserve_wait_available(state, seat, tracker, state.legal_actions(seat))
    state.hands[seat].remove("2"); state.hands[seat].remove("1"); state.hands[seat].extend(["4", "3"])
    assert not agent._late_royal_reserve_wait_available(state, seat, tracker, state.legal_actions(seat))


def test_other_players_plan_from_their_own_hand_not_our_missing_piece():
    state, agent, seat = position()
    # A physically consistent worst-case alternative: D has the fourth gold.
    state.hands["D"].remove("4"); state.hands["D"].append("5")
    state.hands["C"].remove("5"); state.hands["C"].append("4")
    for actor in "ABC":
        state.apply_pass(actor)
    tracker = agent._track[id(state)]
    actions = state.legal_actions("D")
    before = [agent._late_other_action_value(state, "D", action, tracker) for action in actions]
    state.hands["A"].remove("4"); state.hands["A"].append("3")
    after = [agent._late_other_action_value(state, "D", action, tracker) for action in actions]
    assert before == after
    assert actions[max(range(len(actions)), key=lambda index: before[index])][2] == "5"


def test_receiving_and_finishing_are_detected_even_at_a_search_leaf():
    state = GoitaState({"A": ["1", "2"], "B": ["3", "5"],
                        "C": ["5", "6"], "D": ["4", "1"]}, dealer="A")
    state.turn, state.phase, state.attacker, state.current_attack = "D", "receive", "C", "4"
    agent = RuleBasedAgent()
    final = agent._late_immediate_finish(state)
    assert final is not None and final.finished and final.winner == "D"
    assert not state.finished and state.hands["D"] == ["4", "1"]


def test_receive_comparison_is_exported_even_without_an_attack_candidate():
    import json
    from urllib.parse import unquote
    from backend.app import _format_ai_attack_candidates
    snapshot = {"late_receive_comparison": {"chosen": [("pass", None, None)], "proven": False}}
    suffix = _format_ai_attack_candidates(SimpleNamespace(last_attack_candidate_snapshot=snapshot))
    assert suffix.startswith(" [AI-CANDIDATES:")
    assert json.loads(unquote(suffix.split(":", 1)[1][:-1]))["late_receive_comparison"]["proven"] is False


def test_low_confidence_comparison_does_not_override_the_baseline(monkeypatch):
    state, agent, seat = position()
    comparison = {"chosen": (("pass", None, None),), "margin": 10000, "depth": 10,
                  "samples": 48, "proven": False, "reliable": False,
                  "selection_basis": "paired_route_value"}
    monkeypatch.setattr(agent, "_compare_late_receive_routes", lambda *args: copy.deepcopy(comparison))
    assert agent.select_action(state, seat, state.legal_actions(seat)) == ("receive", "8", None)
    assert agent._track[id(state)]["last_late_receive_comparison"]["applied"] is False
    assert not agent.last_score_fallback_detail.startswith("late_receive_")


def test_generic_dictionary_records_only_the_final_adopted_response(monkeypatch):
    recorded = []
    def record(self, state, player, actions, baseline, selected, **evidence):
        recorded.append(selected)
        return False
    monkeypatch.setattr(CurrentAgent, "_record_generic_response_evidence", record)
    state, agent, seat = position()
    chosen = agent.select_action(state, seat, state.legal_actions(seat))
    assert chosen == ("pass", None, None)
    assert all(action == chosen for action in recorded)
    assert agent._late_receive_deferred_records is None


def test_extra_route_comparison_does_not_run_in_background(monkeypatch):
    import threading
    state, agent, seat = position()
    agent._time_search_cancel_event = threading.Event()
    monkeypatch.setattr(agent, "_timed_search_sample_states", lambda *args: pytest.fail("extra background search"))
    assert agent._compare_late_receive_routes(state, seat, state.legal_actions(seat)) is None

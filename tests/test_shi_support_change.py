"""Regression and public-information guards for the 2026-11-11 D turn 4 report."""
import copy
from collections import Counter

import pytest

from goita_ai2.constants import PIECE_TOTALS
from goita_ai2.current_ai import RuleBasedAgent as CurrentAI
from goita_ai2.current_ai.conditional_response import ConditionalResponsePlan
from goita_ai2.experimental_ai2 import RuleBasedAgent
from goita_ai2.instruction_case_audit import apply_action
from goita_ai2.rule_based_intermediate_middle3 import RuleBasedAgent as Middle3
from goita_ai2.state import GoitaState


HANDS = {"A": list("15319711"), "B": list("61231151"),
         "C": list("48325174"), "D": list("34262145")}
PREFIX = (
    ("A", ("attack_after_block", "1", "1")),
    ("B", ("receive", "1", None)), ("B", ("attack", None, "1")),
    ("C", ("receive", "1", None)), ("C", ("attack", None, "4")),
)


def position(*, offset=0, timed=False, hands=None, ranked=False, agent_class=RuleBasedAgent, prefix=PREFIX):
    seats = {p: "ABCD"[(i + offset) % 4] for i, p in enumerate("ABCD")}
    mapped = {seats[p]: list(hand) for p, hand in (hands or HANDS).items()}
    assert Counter(piece for hand in mapped.values() for piece in hand) == Counter(PIECE_TOTALS)
    state = GoitaState(mapped, dealer=seats["A"])
    agents = {seat: agent_class() for seat in "ABCD"}
    for seat, agent in agents.items():
        agent.bind_player(seat)
        agent.TIME_SEARCH_ENABLED = timed
        agent.TIME_SEARCH_BACKGROUND_ENABLED = False
        if not ranked:
            # The optional locally generated rank table is not deployed.
            agent._relative_hand_rank_table = {}
    for actor, action in prefix:
        play(state, agents, seats[actor], action)
    return state, agents, seats


def play(state, agents, actor, action):
    assert action in state.legal_actions(actor)
    apply_action(state, actor, action)
    for agent in agents.values():
        agent.on_public_action(state, actor, action)


@pytest.mark.parametrize("timed,ranked", [(False, False), (True, False), (False, True)])
def test_report_receives_silver_and_keeps_a_kyosha_continuation(timed, ranked):
    state, agents, seats = position(timed=timed, ranked=ranked)
    player, partner = seats["D"], seats["B"]
    agent = agents[player]
    tracker = agent._track[id(state)]
    before = copy.deepcopy(agents[partner]._track[id(state)])

    receive = agent.select_action(state, player, state.legal_actions(player))
    assert receive == ("receive", "4", None)
    assert agent.last_score_fallback_detail == "receive_limited_shi_support_1_followup_2"
    assert agent.last_rule_search_authority == "strong"
    assert not agent.last_neural_shadow["applied"]
    assert tracker["my_attack_count"] == 0
    plan = tracker["limited_shi_support_plan"]
    assert plan["future_attacks"][:3] == ["2", "2", "6"]
    assert plan["proven"] is False
    assert plan["received"] is False
    # Thinking about a signal does not send it to another player.
    assert agents[partner]._track[id(state)] == before
    play(state, agents, player, receive)
    assert plan["received"] is True
    assert agents[partner]._track[id(state)]["ally_shi_signal"] == "unknown"

    attack = agent.select_action(state, player, state.legal_actions(player))
    assert attack == ("attack", None, "2")
    assert agent.last_score_fallback_detail == "attack_limited_shi_support_2"
    assert agent.last_attack_candidate_snapshot["shi_support_change"]["followup"] == "2"
    play(state, agents, player, attack)
    assert tracker["my_attack_count"] == 1
    assert tracker["my_limited_shi_support_sent"] is True
    assert "limited_shi_support_plan" not in tracker
    partner_tracker = agents[partner]._track[id(state)]
    signal = partner_tracker["ally_limited_shi_support"]
    assert signal["received_piece"] == "4" and signal["returned_piece"] == "2"
    assert signal["is_hard_count_bound"] is False
    assert partner_tracker["ally_shi_signal"] == "weak"
    assert partner_tracker["shi_attack_mode"] is False
    assert agents[partner]._estimate_remaining_range(partner_tracker, player, "1")[1] >= 1


@pytest.mark.parametrize("offset", range(4))
def test_signal_is_not_tied_to_d_seat_or_b_partner(offset):
    state, agents, seats = position(offset=offset)
    player = seats["D"]
    agent = agents[player]
    assert agent.select_action(state, player, state.legal_actions(player)) == ("receive", "4", None)
    evidence = agent._track[id(state)]["limited_shi_support_plan"]
    assert evidence["source"] == seats["B"]
    assert evidence["interceptor"] == seats["C"]


def test_other_actual_hands_do_not_determine_the_signal():
    changed = copy.deepcopy(HANDS)
    changed["B"].remove("1")
    changed["C"].remove("4")
    changed["B"].append("4")
    changed["C"].append("1")
    plans = []
    for hands in (HANDS, changed):
        state, agents, _ = position(hands=hands)
        agent = agents["D"]
        assert agent.select_action(state, "D", state.legal_actions("D")) == ("receive", "4", None)
        plans.append(agent._track[id(state)]["limited_shi_support_plan"])
    assert plans[0] == plans[1]


def test_zero_shi_also_requires_a_concrete_followup():
    hands = copy.deepcopy(HANDS)
    hands["D"].remove("1")
    hands["C"].remove("3")
    hands["D"].append("3")
    hands["C"].append("1")
    state, agents, _ = position(hands=hands)
    agent = agents["D"]
    assert agent.select_action(state, "D", state.legal_actions("D")) == ("receive", "4", None)
    assert agent.last_score_fallback_detail.startswith("receive_limited_shi_support_0_")


@pytest.mark.parametrize("condition", ["sufficient_shi", "self_led", "already_sent",
                                       "no_current_signal", "ready_enemy", "no_followup"])
def test_signal_is_not_a_blanket_receive_for_low_shi(condition):
    state, agents, _ = position()
    agent = agents["D"]
    tracker = agent._track[id(state)]
    if condition == "sufficient_shi":
        state.hands["D"][0] = "1"
    elif condition == "self_led":
        tracker["shi_attack_mode"] = True
    elif condition == "already_sent":
        tracker["my_limited_shi_support_sent"] = True
    elif condition == "no_current_signal":
        agent._neural_public_history_by_state_id[id(state)][-3]["action"][2] = "5"
    elif condition == "ready_enemy":
        state.hands["C"] = state.hands["C"][:2]
    elif condition == "no_followup":
        # Receiving from this six-card hand leaves only singleton middles.
        state.hands["D"] = list("123445")
    assert agent._limited_shi_support_plan(state, "D", state.legal_actions("D")) is None


def test_a_proven_action_has_priority_over_communication(monkeypatch):
    state, agents, _ = position()
    agent = agents["D"]
    monkeypatch.setattr(agent, "_high_score_tsume_action", lambda *_args, **_kw: (("pass", None, None), 30, False))
    monkeypatch.setattr(agent, "_upside_finish_action", lambda *_args, **_kw: None)
    assert agent.select_action(state, "D", state.legal_actions("D")) == ("pass", None, None)
    assert agent.last_score_fallback_detail == "high_score_30"
    assert "limited_shi_support_plan" not in agent._track[id(state)]


def test_actual_different_followup_cancels_the_plan_and_does_not_reject_shi():
    state, agents, _ = position()
    agent = agents["D"]
    receive = agent.select_action(state, "D", state.legal_actions("D"))
    play(state, agents, "D", receive)
    play(state, agents, "D", ("attack", None, "1"))
    assert "limited_shi_support_plan" not in agent._track[id(state)]
    assert not agent._track[id(state)].get("my_limited_shi_support_sent")
    assert not agents["B"]._track[id(state)].get("ally_limited_shi_support")


def test_partner_observes_the_switch_even_when_passes_precede_the_interception():
    prefix = PREFIX[:3] + (
        ("C", ("pass", None, None)), ("D", ("pass", None, None)),
        ("A", ("receive", "1", None)), ("A", ("attack", None, "5")),
        ("B", ("pass", None, None)), ("C", ("pass", None, None)),
    )
    state, agents, _ = position(prefix=prefix)
    play(state, agents, "D", ("receive", "5", None))
    play(state, agents, "D", ("attack", None, "2"))
    signal = agents["B"]._track[id(state)]["ally_limited_shi_support"]
    assert signal["interceptor"] == "A"
    assert signal["received_piece"] == "5"


def test_later_actual_shi_response_supersedes_the_earlier_weak_signal():
    state, agents, _ = position()
    receive = agents["D"].select_action(state, "D", state.legal_actions("D"))
    play(state, agents, "D", receive)
    play(state, agents, "D", ("attack", None, "2"))
    for seat in "ABC":
        play(state, agents, seat, ("pass", None, None))
    play(state, agents, "D", ("attack_after_block", "3", "1"))
    tracker = agents["B"]._track[id(state)]
    assert "ally_limited_shi_support" not in tracker
    assert tracker["ally_shi_signal"] == "returned_shi"
    assert tracker["shi_attack_mode"] is True


def test_partner_reduces_shi_pressure_without_turning_a_signal_into_card_counts(monkeypatch):
    state, agents, _ = position()
    partner = agents["B"]
    tracker = partner._track[id(state)]
    assert partner._weak_shi_attack_plan_value(state, "B") > 0
    monkeypatch.setattr(partner, "_opponents_piece_count_summary", lambda *_args: {
        "min": 0, "max": 2, "map_count": 1, "expected": 1.2, "confidence": 0.8,
    })
    assert partner._inferred_enemy_team_shi_pressure(state, "B") is not None
    receive = agents["D"].select_action(state, "D", state.legal_actions("D"))
    play(state, agents, "D", receive)
    play(state, agents, "D", ("attack", None, "2"))
    assert partner._weak_shi_attack_plan_value(state, "B") == 0
    assert partner._inferred_enemy_team_shi_pressure(state, "B") is None


def test_cached_ordinary_pass_does_not_discard_the_new_receive_purpose():
    state, agents, _ = position()
    agent = agents["D"]
    actions = state.legal_actions("D")
    baseline = ("receive", "4", None)
    key = agent._conditional_response_key(state, "D", actions, baseline)
    agent._conditional_response_dictionary.put(key, ConditionalResponsePlan(
        action=("pass", None, None), followup_attack_piece=None,
        baseline_action=baseline, source="foreground", depth=7, agreement=0.8,
        information_confidence=0.8, margin=100, cache_source="foreground",
        cache_branch_kind=None, cache_branch_context=None, cached_compute_ms=100,
    ))
    assert agent.select_action(state, "D", actions) == baseline
    assert agent.last_score_fallback_detail.startswith("receive_limited_shi_support_")


def test_publicly_exhausted_enemies_can_still_be_attacked_after_a_weak_signal(monkeypatch):
    state, agents, _ = position()
    partner = agents["B"]
    partner._track[id(state)]["ally_limited_shi_support"] = {"is_hard_count_bound": False}
    monkeypatch.setattr(partner, "_opponents_piece_count_summary", lambda *_args: {
        "min": 0, "max": 0, "map_count": 0, "expected": 0, "confidence": 1.0,
    })
    assert partner._inferred_enemy_team_shi_pressure(state, "B") is not None


@pytest.mark.parametrize("agent_class", [CurrentAI, Middle3])
def test_saved_profiles_keep_their_original_pass(agent_class):
    state, agents, _ = position(agent_class=agent_class)
    agent = agents["D"]
    assert agent.select_action(state, "D", state.legal_actions("D")) == ("pass", None, None)
    assert agent.last_score_fallback_detail == "pass_piece_count_inference"

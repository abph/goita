"""A sufficient shi attack reserve can retain other receivers and attack axes."""
import copy
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
    ("B", ("pass", None, None)),
)


def position(offset=0, mapping=None, timed=False, agent_class=RuleBasedAgent, hands=None):
    seats = {p: "ABCD"[(i + offset) % 4] for i, p in enumerate("ABCD")}
    mapping = mapping or {}
    hands = {seats[p]: [mapping.get(piece, piece) for piece in hand]
             for p, hand in (hands or HANDS).items()}
    assert Counter(p for hand in hands.values() for p in hand) == Counter(PIECE_TOTALS)
    state = GoitaState(hands, dealer=seats["A"])
    player = seats["C"]
    agent = agent_class()
    agent.bind_player(player)
    agent.TIME_SEARCH_ENABLED = timed
    agent.TIME_SEARCH_MAX_SECONDS = 1
    agent.TIME_SEARCH_SAMPLE_COUNT = 80
    agent.TIME_SEARCH_MAX_DEPTH = 11
    agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    agent._ensure_trackers(state)
    for actor, action in HISTORY:
        action = tuple(mapping.get(piece, piece) for piece in action)
        assert action in state.legal_actions(seats[actor])
        apply_action(state, seats[actor], action)
        agent.on_public_action(state, seats[actor], action)
    return state, agent, player


@pytest.mark.parametrize("timed", [False, True])
def test_report_folds_shi_preserving_gold_and_bishop(timed):
    state, agent, player = position(timed=timed)
    assert Counter(state.hands[player]) == Counter("111156")
    chosen = agent.select_action(state, player, state.legal_actions(player))
    assert chosen == ("attack_after_block", "1", "1")
    assert agent.last_score_fallback_detail == "block_shi_keep_receivers_reserve_2_needed_2"
    snapshot = agent.last_attack_candidate_snapshot
    evidence = snapshot["shi_block_reserve"]
    assert evidence["remaining_hand"] == list("1156")
    assert evidence["other_receive_types"] == list("56")
    assert evidence["shi_after_block_and_attack"] == evidence["future_attack_slots"] == 2
    assert evidence["additional_shi_receives_reserved"] == 0
    assert evidence["proven"] is False
    assert evidence["previous_action"] == ("attack_after_block", "5", "1")
    assert all(row["score_gap"] > 0 for row in snapshot["block_alternatives"])
    assert agent.last_neural_shadow["global_recommended_action"] == list(chosen)
    apply_action(state, player, chosen)
    agent.on_public_action(state, player, chosen)
    assert Counter(state.hands[player]) == Counter("1156")
    assert agent._track[id(state)]["my_attack_count"] == 2


@pytest.mark.parametrize("offset", range(4))
@pytest.mark.parametrize("mapping", [{}, {"5": "3", "3": "5"}, {"5": "4", "4": "5"},
                                     {"5": "2", "2": "5"}, {"6": "7", "7": "6"}])
def test_reserve_tactic_generalizes_across_seats_and_receiving_pieces(offset, mapping):
    state, agent, player = position(offset=offset, mapping=mapping)
    assert agent.select_action(state, player, state.legal_actions(player)) == ("attack_after_block", "1", "1")
    evidence = agent.last_attack_candidate_snapshot["shi_block_reserve"]
    assert set(evidence["other_receive_types"]) == {mapping.get("5", "5"), mapping.get("6", "6")}


def test_three_shi_cannot_be_spent_as_though_two_future_shi_attacks_remain():
    hands = copy.deepcopy(HANDS)
    hands["C"].remove("1")
    hands["C"].append("2")
    hands["B"].remove("2")
    hands["B"].append("1")
    state, agent, player = position(hands=hands)
    assert state.hands[player].count("1") == 3
    agent._track[id(state)]["shi_attack_mode"] = True
    agent._track[id(state)]["shi_attack_mode_source"] = "self"
    assert agent._weak_shi_attack_plan_value(state, player) > 0
    assert agent._shi_block_reserve_context(state, player, "attack_after_block", "1") is None
    adjustment = agent._shi_attack_hidden_block_adjustment(state, player, "attack_after_block", "1", "1")
    assert adjustment == CurrentAI._shi_attack_hidden_block_adjustment(
        agent, state, player, "attack_after_block", "1", "1") == -700


def test_dealer_opening_four_shi_signal_keeps_its_existing_bonus():
    state = GoitaState(copy.deepcopy(HANDS), dealer="C")
    agent = RuleBasedAgent()
    agent.bind_player("C")
    agent._ensure_trackers(state)
    assert agent._shi_block_reserve_context(state, "C", "attack_after_block", "1") is None
    assert agent._shi_attack_hidden_block_adjustment(state, "C", "attack_after_block", "1", "1") == (
        CurrentAI._shi_attack_hidden_block_adjustment(agent, state, "C", "attack_after_block", "1", "1"))


def test_last_pair_logic_and_other_attack_pieces_keep_the_existing_scores():
    state, agent, player = position()
    for action in (("attack_after_block", "1", "6"), ("attack", None, "1")):
        assert agent._shi_attack_hidden_block_adjustment(state, player, *action) == (
            CurrentAI._shi_attack_hidden_block_adjustment(agent, state, player, *action))
    state.hands[player] = list("1116")
    assert agent._shi_block_reserve_context(state, player, "attack_after_block", "1") is None
    assert agent._shi_attack_hidden_block_adjustment(state, player, "attack_after_block", "1", "1") == (
        agent.WEAK_SHI_ENDGAME_MIXED_BLOCK_BONUS)


def test_final_immediate_win_is_not_replaced_by_preserving_receivers():
    state, agent, player = position()
    state.hands[player] = list("16")
    chosen = agent.select_action(state, player, state.legal_actions(player))
    assert chosen == ("attack_after_block", "1", "6")
    assert agent.last_decision_reason == "win_now"
    assert "shi_block_reserve" not in agent.last_attack_candidate_snapshot


def test_unplayed_opponents_cards_do_not_supply_the_reserve_evaluation():
    state, agent, player = position()
    state.hands["A"].remove("2")
    state.hands["A"].append("1")
    state.hands["B"].remove("1")
    state.hands["B"].append("2")
    assert agent.select_action(state, player, state.legal_actions(player)) == ("attack_after_block", "1", "1")
    assert agent.last_attack_candidate_snapshot["shi_block_reserve"]["source"] == (
        "own_hand_and_remaining_attack_slots")


def test_special_log_requires_an_actual_change_in_candidate_order():
    state, agent, player = position()
    chosen = ("attack_after_block", "1", "1")
    agent.last_attack_candidate_scores = [{"action": chosen, "score": 1000},
                                         {"action": ("attack_after_block", "5", "1"), "score": 0}]
    assert not agent._classify_score_fallback(
        state, player, chosen, has_non_king_attack_option=True).startswith("block_shi_keep_receivers_")


@pytest.mark.parametrize("agent_class", [CurrentAI, Middle3])
def test_current_and_saved_middle_three_keep_the_original_fold(agent_class):
    state, agent, player = position(agent_class=agent_class)
    assert agent.select_action(state, player, state.legal_actions(player)) == ("attack_after_block", "5", "1")
    assert not hasattr(agent, "_shi_block_reserve_context")

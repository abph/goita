"""Regression for allocating spare attackers to blocks and keeping receivers."""

from collections import Counter

import pytest

from goita_ai2.constants import PIECE_TOTALS
from goita_ai2.experimental_ai2 import RuleBasedAgent
from goita_ai2.instruction_case_audit import apply_action
from goita_ai2.state import GoitaState


PASS = ("pass", None, None)
OPEN = ("attack_after_block", "3", "5")
HANDS = {"A": "57315463", "B": "21168113", "C": "21134275", "D": "91124154"}


def position(rotation=0, swap_middle=False, hands=None):
    seats = {seat: "ABCD"[(i + rotation) % 4] for i, seat in enumerate("ABCD")}
    swap = {"3": "4", "4": "3"} if swap_middle else {}
    mapped = {seats[seat]: [swap.get(piece, piece) for piece in hand]
              for seat, hand in (hands or HANDS).items()}
    assert Counter(piece for hand in mapped.values() for piece in hand) == Counter(PIECE_TOTALS)
    state = GoitaState(mapped, dealer=seats["A"])
    agent = RuleBasedAgent()
    agent.bind_player(seats["A"])
    agent.TIME_SEARCH_ENABLED = agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    agent._relative_hand_rank_table = {}
    agent._ensure_trackers(state)
    return state, agent


def play(state, agent, seat, action):
    assert action in state.legal_actions(seat)
    apply_action(state, seat, action)
    agent.on_public_action(state, seat, action)


@pytest.mark.parametrize("rotation,swap_middle", [(0, False), (2, True)])
def test_opening_assigns_a_spare_middle_piece_to_the_block(rotation, swap_middle):
    state, agent = position(rotation=rotation, swap_middle=swap_middle)
    seat = state.turn
    spare = "4" if swap_middle else "3"
    before = Counter(state.hands[seat])
    chosen = agent.select_action(state, seat, state.legal_actions(seat))
    assert chosen == ("attack_after_block", spare, "5")
    plan = agent.last_attack_candidate_snapshot["attack_and_block_plan"]
    assert plan["receive_reserves"] == ["1", "6", "7"]
    attacks = [attack for _, attack in plan["steps"]]
    assert attacks[0] == attacks[2] == "5" and attacks[1] in ("6", "7")
    assert Counter(piece for step in plan["steps"] for piece in step) == before
    assert agent.last_score_fallback_detail == "attack_and_block_plan_opening"
    assert not agent.last_neural_shadow["safety_locked"]


def test_all_pass_uses_next_attack_and_block_and_keeps_gold_for_later():
    state, agent = position()
    chosen = agent.select_action(state, "A", state.legal_actions("A"))
    play(state, agent, "A", chosen)
    for seat in "BCD":
        play(state, agent, seat, PASS)
    followup = agent.select_action(state, "A", state.legal_actions("A"))
    assert followup[0:2] == ("attack_after_block", "3") and followup[2] in ("6", "7")
    assert agent.last_score_fallback_detail == "attack_and_block_plan_review"
    plan = agent.last_attack_candidate_snapshot["attack_and_block_plan"]
    assert plan["steps"][1][1] == "5" and "1" in plan["receive_reserves"]


def test_ally_gold_changes_the_attack_order_and_reallocates_blocks(monkeypatch):
    state, agent = position()
    play(state, agent, "A", agent.select_action(state, "A", state.legal_actions("A")))
    def unexpected_search(*args, **kwargs):
        pytest.fail("A public move must not launch a new search")
    monkeypatch.setattr(agent, "_search_future_attack_plan", unexpected_search)
    monkeypatch.setattr(agent, "_time_limited_search_action", unexpected_search)
    tracker = agent._track[id(state)]["round_strategy"]
    before = tracker["live_review"]["attack_and_block_plan"]["steps"]
    assert before[0][1] in ("6", "7")
    for seat, action in [
        ("B", ("receive", "8", None)), ("B", ("attack", None, "1")),
        ("C", ("receive", "1", None)), ("C", ("attack", None, "5")),
    ]:
        play(state, agent, seat, action)
    review = tracker["live_review"]
    plan = review["attack_and_block_plan"]
    assert review["status"] == "updated"
    assert plan["steps"][0] == ("3", "5")
    assert plan["steps"][1][1] in ("6", "7")
    assert plan["steps"][1] != before[1]
    assert plan["receive_reserves"] == ["1", "6", "7"]


def test_two_pairs_are_needed_to_set_aside_one_of_them():
    hands = dict(HANDS)
    hands.update(A=HANDS["A"].replace("3", "2", 1), B=HANDS["B"].replace("2", "3", 1))
    state, agent = position(hands=hands)
    agent.select_action(state, "A", state.legal_actions("A"))
    assert "attack_and_block_plan" not in agent.last_attack_candidate_snapshot

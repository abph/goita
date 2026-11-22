"""Strong initial hands can wait for a better proven receive and finish."""

from collections import Counter

import pytest

from goita_ai2.constants import PIECE_TOTALS
from goita_ai2.current_ai.endgame import ForcedWinStatus
from goita_ai2.experimental_ai2 import RuleBasedAgent
from goita_ai2.instruction_case_audit import apply_action
from goita_ai2.state import GoitaState


PASS = ("pass", None, None)
HANDS = {"A": "11112337", "B": "11246689", "C": "11123455", "D": "12344557"}


def position(*, swap_big=False, deal=HANDS):
    swap = {"6": "7", "7": "6"} if swap_big else {}
    hands = {seat: [swap.get(piece, piece) for piece in hand] for seat, hand in deal.items()}
    assert Counter(piece for hand in hands.values() for piece in hand) == Counter(PIECE_TOTALS)
    state = GoitaState(hands, dealer="D")
    agent = RuleBasedAgent()
    agent.bind_player("B")
    agent.TIME_SEARCH_ENABLED = agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    agent._relative_hand_rank_table = {}
    agent._ensure_trackers(state)
    return state, agent


def play(state, agent, seat, action):
    assert seat == state.turn and action in state.legal_actions(seat)
    apply_action(state, seat, action)
    agent.on_public_action(state, seat, action)


def choose(state, agent):
    return agent.select_action(state, "B", state.legal_actions("B"))


def approach_gold(state, agent):
    play(state, agent, "D", ("attack_after_block", "4", "5"))
    play(state, agent, "A", PASS)


@pytest.mark.parametrize("swap_big", [False, True])
def test_initial_priorities_lead_to_shi_silver_then_two_big_attacks(swap_big):
    state, agent = position(swap_big=swap_big)
    big = "7" if swap_big else "6"
    opening = agent._track[id(state)]["round_strategy"]["opening_receive_plans"]
    assert opening["absolute_rank"] == "S"
    main, alternative, fallback = opening["plans"]
    assert (main["minimum_score"], main["maximum_score"]) == (50, 100)
    assert [a for _, a in main["routes"][0]["all_pass_steps"][:2]] == [big, big]
    assert main["routes"][0]["receive"] == "1" and main["routes"][0]["attack"] == "4"
    assert alternative["minimum_score"] == 50 and alternative["routes"][0]["receive"] == "2"
    assert fallback["minimum_score"] == 30
    assert {r["receive"] for r in fallback["routes"]} == {"8", "9"}
    approach_gold(state, agent)
    assert choose(state, agent) == PASS
    comparison = agent.last_attack_candidate_snapshot["guaranteed_receive_comparison"]
    wait = next(row for row in comparison["candidates"] if row["actions"] == (PASS,))
    assert (wait["minimum_score"], wait["maximum_score"]) == (30, 100)
    assert agent.last_rule_search_authority == "proven"
    for seat, action in [
        ("B", PASS), ("C", PASS), ("D", ("attack_after_block", "3", "1")), ("A", PASS),
    ]:
        play(state, agent, seat, action)
    receive = choose(state, agent)
    assert receive == ("receive", "1", None)
    comparison = agent.last_attack_candidate_snapshot["guaranteed_receive_comparison"]
    assert comparison["selected"] == (receive, ("attack", None, "4"))
    play(state, agent, "B", receive)
    for attack in ("4", big, big, None):
        action = choose(state, agent)
        if attack is None:
            assert set(action[1:]) == {"8", "9"}
        else:
            assert action[2] == attack
        play(state, agent, "B", action)
        if not state.finished:
            for seat in "CDA":
                play(state, agent, seat, PASS)
    assert state.winner == "B" and state.team_score["BD"] == 100


def test_intercepted_silver_preserves_fifty():
    state, agent = position()
    approach_gold(state, agent)
    for seat, action in [
        ("B", PASS), ("C", PASS), ("D", ("attack_after_block", "3", "1")), ("A", PASS),
        ("B", ("receive", "1", None)), ("B", ("attack", None, "4")),
        ("C", ("receive", "4", None)), ("C", ("attack", None, "5")), ("D", PASS), ("A", PASS),
    ]:
        if seat == "B":
            assert choose(state, agent) == action
        play(state, agent, seat, action)
    for _ in range(40):
        if state.finished:
            break
        seat = state.turn
        play(state, agent, seat, choose(state, agent) if seat == "B" else PASS)
    assert state.winner == "B" and state.team_score["BD"] == 50


def test_enemy_with_two_cards_stops_wait_and_rechecks_the_risky_attack():
    state, agent = position()
    approach_gold(state, agent)
    for seat, action in [
        ("B", PASS), ("C", ("receive", "5", None)), ("C", ("attack", None, "1")),
        ("D", PASS), ("A", PASS), ("B", PASS), ("C", ("attack_after_block", "1", "2")),
        ("D", PASS), ("A", PASS), ("B", PASS), ("C", ("attack_after_block", "5", "4")),
        ("D", PASS), ("A", PASS),
    ]:
        play(state, agent, seat, action)
    assert len(state.hands["C"]) == 2
    tracker = agent._track[id(state)]
    assert agent._proven_wait_branches(state, "B", tracker) == []
    receive = choose(state, agent)
    assert receive == ("receive", "4", None)
    play(state, agent, "B", receive)
    attack = choose(state, agent)
    assert attack[2] != "1"
    proof = agent._forced_win_result_after_attack_action(state, "B", attack)
    assert proof.status == ForcedWinStatus.PROVEN and proof.minimum_score >= 50


def test_rank_b_has_no_prepared_receive_plans(monkeypatch):
    from goita_ai2.current_ai import RuleBasedAgent as CurrentAI

    def forbidden(*args):
        pytest.fail("rank B must not expand the opening proof")

    monkeypatch.setattr(RuleBasedAgent, "_initial_receive_plan_options", forbidden)
    monkeypatch.setattr(CurrentAI, "_classify_hand_axes", lambda *args, **kwargs: {"rank": "B"})
    state, agent = position()
    assert "opening_receive_plans" not in agent._track[id(state)]["round_strategy"]


def test_rank_a_prepares_provisional_routes_without_claiming_a_score_floor():
    deal = {**HANDS, "A": "11112336", "B": "11236789", "D": "12444557"}
    state, agent = position(deal=deal)
    opening = agent._track[id(state)]["round_strategy"]["opening_receive_plans"]
    assert opening["absolute_rank"] == "A"
    assert [p["priority"] for p in opening["plans"]] == [1, 2, 3]
    assert all(p["minimum_score"] is None and p["maximum_score"] is None for p in opening["plans"])
    for plan in opening["plans"]:
        for route in plan["routes"]:
            cards = [route["receive"], route["attack"]]
            cards += [piece for step in route["all_pass_steps"] for piece in step]
            assert Counter(cards) == Counter(state.hands["B"])

"""Regression for choosing and carrying forward guaranteed scoring routes."""

from collections import Counter

import pytest

from goita_ai2.constants import PIECE_TOTALS
from goita_ai2.experimental_ai2 import RuleBasedAgent
from goita_ai2.experimental_ai2.guaranteed_receive_comparison import GuaranteedRoute
from goita_ai2.instruction_case_audit import apply_action
from goita_ai2.state import GoitaState


PASS = ("pass", None, None)
RECEIVE = ("receive", "2", None)
SILVER = ("attack", None, "4")
SHI = ("attack", None, "1")
HANDS = {"A": "13128264", "B": "11163413", "C": "41457715", "D": "25231591"}
HISTORY = (
    ("A", ("attack_after_block", "1", "6")), ("B", ("receive", "6", None)),
    ("B", ("attack", None, "3")), ("C", PASS), ("D", PASS),
    ("A", ("receive", "3", None)), ("A", ("attack", None, "2")),
    ("B", PASS), ("C", PASS), ("D", ("receive", "2", None)),
    ("D", ("attack", None, "3")), ("A", PASS), ("B", PASS), ("C", PASS),
    ("D", ("attack_after_block", "1", "2")),
)


def position():
    state = GoitaState({seat: list(hand) for seat, hand in HANDS.items()}, dealer="A")
    assert Counter(piece for hand in state.hands.values() for piece in hand) == Counter(PIECE_TOTALS)
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent.TIME_SEARCH_ENABLED = False
    agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    agent._relative_hand_rank_table = {}
    agent._ensure_trackers(state)
    for actor, action in HISTORY:
        play(state, agent, actor, action)
    return state, agent


def play(state, agent, actor, action):
    assert action in state.legal_actions(actor)
    apply_action(state, actor, action)
    agent.on_public_action(state, actor, action)


def test_report_passes_for_a_higher_guaranteed_floor():
    state, agent = position()
    assert agent.select_action(state, "A", state.legal_actions("A")) == PASS
    review = agent.last_attack_candidate_snapshot["guaranteed_receive_comparison"]
    candidates = {row["actions"]: row for row in review["candidates"]}
    assert review["basis"] == "minimum_score"
    assert (candidates[(PASS,)]["minimum_score"], candidates[(RECEIVE, SILVER)]["minimum_score"]) == (30, 10)
    assert all(row["maximum_score"] == 50 for row in candidates.values())
    assert agent.last_neural_shadow["safety_locked"]


@pytest.mark.parametrize("incoming,hidden,receive,finish", [
    ("5", "1", "8", "4"), ("1", "5", "1", "8"),
])
def test_after_wait_original_deal_finishes_for_thirty_or_fifty(incoming, hidden, receive, finish):
    state, agent = position()
    play(state, agent, "A", agent.select_action(state, "A", state.legal_actions("A")))
    for seat in ("B", "C"):
        assert state.legal_actions(seat) == [PASS]
        play(state, agent, seat, PASS)
    play(state, agent, "D", ("attack_after_block", hidden, incoming))
    for expected in (("receive", receive, None), ("attack", None, "2")):
        chosen = agent.select_action(state, "A", state.legal_actions("A"))
        assert chosen == expected
        play(state, agent, "A", chosen)
    for seat in ("B", "C", "D"):
        assert state.legal_actions(seat) == [PASS]
        play(state, agent, seat, PASS)
    chosen = agent.select_action(state, "A", state.legal_actions("A"))
    assert chosen[2] == finish
    play(state, agent, "A", chosen)
    assert state.winner == "A"
    assert state.team_score["AC"] == (30 if finish == "4" else 50)


def test_equal_floor_compares_each_followup_and_keeps_the_chosen_attack(monkeypatch):
    state, agent = position()
    mocked_shi = GuaranteedRoute((RECEIVE, SHI), 10, 50)
    silver = GuaranteedRoute((RECEIVE, SILVER), 10, 50)
    # Replace only the candidate input to exercise the equal-floor comparison.
    # The real position still has a provable silver continuation to revalidate.
    monkeypatch.setattr(agent, "_proven_receive_routes", lambda *args: [mocked_shi, silver])
    monkeypatch.setattr(agent, "_proven_wait_branches", lambda *args: [])
    monkeypatch.setattr(agent, "_paired_guaranteed_receive_scores",
                        lambda *args: ({mocked_shi: 22.0, silver: 40.0}, 24))
    chosen = agent.select_action(state, "A", state.legal_actions("A"))
    assert chosen == RECEIVE
    review = agent.last_attack_candidate_snapshot["guaranteed_receive_comparison"]
    assert review["basis"] == "expected_score"
    assert review["selected"] == (RECEIVE, SILVER)
    play(state, agent, "A", chosen)
    assert agent.select_action(state, "A", state.legal_actions("A")) == SILVER
    assert agent.last_search_skip_reason == "proven_rule"


def test_each_estimate_uses_the_same_complete_deals(monkeypatch):
    from types import SimpleNamespace

    state, agent = position()
    wait = GuaranteedRoute((PASS,), 30, 50)
    receive = GuaranteedRoute((RECEIVE, SILVER), 10, 50)
    calls = []
    worlds = tuple(SimpleNamespace(state=i, probability=0.1) for i in range(10))
    monkeypatch.setattr(agent, "GUARANTEED_RECEIVE_SECONDS", 10)
    monkeypatch.setattr(agent, "_timed_search_generate_sample_states", lambda *args: worlds)
    monkeypatch.setattr(agent, "_information_set_search_worlds", lambda *args: (None, worlds))

    def rollout(world, player, tracker, route, deadline):
        calls.append((world, route))
        if world >= 8 and route == receive:
            return None
        return 50 if world >= 8 else route.minimum

    monkeypatch.setattr(agent, "_guaranteed_receive_rollout", rollout)
    scores, used = agent._paired_guaranteed_receive_scores(state, "A", {}, [wait, receive])
    assert used == 8
    assert len(calls) == 20
    assert scores[wait] == pytest.approx(30)
    assert scores[receive] == pytest.approx(10)

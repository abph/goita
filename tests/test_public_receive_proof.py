"""Eight-card receives can enter bounded exact proofs using only public pieces."""
import copy
from collections import Counter

import pytest

from goita_ai2.constants import PIECE_TOTALS
from goita_ai2.current_ai import RuleBasedAgent as CurrentAI
from goita_ai2.current_ai.endgame import ForcedWinStatus
from goita_ai2.experimental_ai2 import RuleBasedAgent as AI2
from goita_ai2.instruction_case_audit import apply_action
from goita_ai2.rule_based_intermediate_middle3 import RuleBasedAgent as Middle3
from goita_ai2.state import GoitaState


HANDS = {"A": "15121543", "B": "34278162", "C": "54961351", "D": "71243111"}
HISTORY = [
    ("A", ("attack_after_block", "1", "5")),
    ("B", ("pass", None, None)), ("C", ("pass", None, None)), ("D", ("pass", None, None)),
    ("A", ("attack_after_block", "1", "5")),
    ("B", ("receive", "8", None)), ("B", ("attack", None, "7")),
    ("C", ("pass", None, None)), ("D", ("pass", None, None)), ("A", ("pass", None, None)),
    ("B", ("attack_after_block", "1", "2")),
    ("C", ("pass", None, None)), ("D", ("pass", None, None)), ("A", ("pass", None, None)),
    ("B", ("attack_after_block", "3", "2")),
    ("C", ("pass", None, None)), ("D", ("pass", None, None)),
    ("A", ("receive", "2", None)), ("A", ("attack", None, "3")),
    ("B", ("pass", None, None)),
]


def position(agent_class=AI2, *, rotate=0, swap_royals=False, swap_middle=False):
    seats = {s: "ABCD"[(i + rotate) % 4] for i, s in enumerate("ABCD")}
    mapping = {}
    if swap_royals:
        mapping.update({"8": "9", "9": "8"})
    if swap_middle:
        mapping.update({"4": "5", "5": "4"})
    piece = lambda p: mapping.get(p, p)
    state = GoitaState({seats[s]: list(map(piece, hand)) for s, hand in HANDS.items()}, dealer=seats["A"])
    assert Counter(p for hand in state.hands.values() for p in hand) == Counter(PIECE_TOTALS)
    agent = agent_class()
    agent.TIME_SEARCH_ENABLED = False
    agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    agent._relative_hand_rank_table = {}
    actor = seats["C"]
    agent.bind_player(actor)
    agent._ensure_trackers(state)
    for seat, (kind, block, attack) in HISTORY:
        play(state, agent, seats[seat], (kind, piece(block), piece(attack)))
    return state, agent, actor, piece


def play(state, agent, actor, action):
    assert action in state.legal_actions(actor)
    apply_action(state, actor, action)
    agent.on_public_action(state, actor, action)


@pytest.mark.parametrize("rotate", range(4))
@pytest.mark.parametrize("swap_royals", [False, True])
@pytest.mark.parametrize("swap_middle", [False, True])
def test_report_receives_and_keeps_the_proven_royal_middle_middle_finish(rotate, swap_royals, swap_middle):
    state, agent, actor, piece = position(rotate=rotate, swap_royals=swap_royals, swap_middle=swap_middle)
    before = copy.deepcopy(state.hands)
    chosen = agent.select_action(state, actor, state.legal_actions(actor))
    assert chosen == ("receive", "3", None)
    assert state.hands == before
    proof = agent.last_attack_candidate_snapshot["public_receive_proof"]
    assert proof["proven"] and proof["source"] == "public_information"
    assert proof["minimum_score"] == 40
    assert proof["reserved_followup"] == ["attack", None, piece("9")]
    assert agent.last_score_fallback_detail == "receive_tsume_after"
    assert agent.last_neural_shadow["safety_locked"]
    play(state, agent, actor, chosen)
    attacks = []
    while not state.finished:
        if state.turn == actor:
            chosen = agent.select_action(state, actor, state.legal_actions(actor))
            attacks.append(chosen[2])
            assert agent.last_decision_reason in ("tsume", "win_now")
            play(state, agent, actor, chosen)
        else:
            # This is a forced pass, not a cooperative assumption in the replay.
            assert state.legal_actions(state.turn) == [("pass", None, None)]
            play(state, agent, state.turn, ("pass", None, None))
    assert attacks == [piece("9"), piece("5"), piece("5"), "6"]
    assert state.winner == actor
    assert state.team_score["AC" if actor in "AC" else "BD"] == 40
    assert agent.EXACT_FORCED_WIN_MAX_HAND == 6


@pytest.mark.parametrize("agent_class", [CurrentAI, Middle3])
def test_saved_and_current_profiles_keep_the_original_eight_card_limit(agent_class):
    state, agent, actor, _ = position(agent_class)
    proof = agent._forced_win_result_after_receive_action(state, actor, ("receive", "3", None))
    assert proof.status == ForcedWinStatus.UNKNOWN
    assert not hasattr(agent, "LARGE_HAND_RECEIVE_PUBLIC_PROOF_ENABLED")


def test_no_extended_search_when_the_first_attack_can_be_received(monkeypatch):
    state, agent, actor, _ = position()
    tracker = agent._track[id(state)]
    # With an unaccounted royal, every middle/royal attack may be received;
    # the remaining shi are not exhausted either. No exact branch is launched.
    tracker["public_seen_counts"]["8"] = 0
    monkeypatch.setattr(agent, "_forced_win_result_after_attack_action",
                        lambda *args: pytest.fail("receivable attack must not enter extended proof"))
    monkeypatch.setattr(agent, "_apply_action_on_copy",
                        lambda *args: pytest.fail("ordinary positions must not copy the tracker"))
    before = set(agent._track)
    result = agent._forced_win_result_after_receive_action(state, actor, ("receive", "3", None))
    assert result.status == ForcedWinStatus.UNKNOWN
    assert set(agent._track) == before


def test_missing_middle_exhaustion_cannot_be_reported_as_a_proven_win():
    state, agent, actor, _ = position()
    agent._track[id(state)]["public_seen_counts"]["5"] = 1
    result = agent._forced_win_result_after_receive_action(state, actor, ("receive", "3", None))
    assert result.status != ForcedWinStatus.PROVEN


def test_proof_failure_cleans_up_the_speculative_tracker(monkeypatch):
    state, agent, actor, _ = position()
    original = set(agent._track)
    def failed(*args):
        raise RuntimeError("test proof failure")
    monkeypatch.setattr(agent, "_forced_win_result_after_attack_action", failed)
    with pytest.raises(RuntimeError, match="test proof failure"):
        agent._forced_win_result_after_receive_action(state, actor, ("receive", "3", None))
    assert set(agent._track) == original


def test_opposing_hidden_pieces_do_not_change_the_proof_or_action():
    outcomes = []
    for swap in (False, True):
        state, agent, actor, _ = position()
        if swap:
            state.hands["B"].remove("6"); state.hands["B"].append("3")
            state.hands["D"].remove("3"); state.hands["D"].append("6")
        chosen = agent.select_action(state, actor, state.legal_actions(actor))
        outcomes.append((chosen, agent.last_attack_candidate_snapshot["public_receive_proof"]))
    assert outcomes[0] == outcomes[1]


def test_changed_public_state_cancels_the_reserved_first_attack():
    state, agent, actor, _ = position()
    chosen = agent.select_action(state, actor, state.legal_actions(actor))
    play(state, agent, actor, chosen)
    tracker = agent._track[id(state)]
    assert tracker.get("pending_large_hand_receive_proof")
    tracker["public_seen_counts"]["8"] = 0
    agent.select_action(state, actor, state.legal_actions(actor))
    assert tracker.get("pending_large_hand_receive_proof") is None


def test_extension_can_be_disabled():
    state, agent, actor, _ = position()
    agent.LARGE_HAND_RECEIVE_PUBLIC_PROOF_ENABLED = False
    assert agent._forced_win_result_after_receive_action(state, actor, ("receive", "3", None)).status == ForcedWinStatus.UNKNOWN


def test_immediate_defence_does_not_replace_an_already_proven_receive(monkeypatch):
    state, agent, actor, _ = position()
    legal = state.legal_actions(actor)
    assert ("receive", "9", None) in legal
    monkeypatch.setattr(agent, "_enemy_immediate_finish_receive_action",
                        lambda *args: ("receive", "9", None))
    assert agent.select_action(state, actor, legal) == ("receive", "3", None)
    comparison = agent.last_attack_candidate_snapshot["guaranteed_receive_comparison"]
    assert comparison["selected"] == (("receive", "3", None), ("attack", None, "9"))
    assert agent.last_score_fallback_detail == "receive_tsume_after"


def test_disabling_after_receiving_discards_the_pending_order():
    state, agent, actor, _ = position()
    chosen = agent.select_action(state, actor, state.legal_actions(actor))
    play(state, agent, actor, chosen)
    agent.LARGE_HAND_RECEIVE_PUBLIC_PROOF_ENABLED = False
    assert agent.select_action(state, actor, state.legal_actions(actor)) == ("attack", None, "5")
    assert agent._track[id(state)].get("pending_large_hand_receive_proof") is None

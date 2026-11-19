"""Opening pair ownership must be compared without a blanket weak-hand pass bias."""
import copy
from collections import Counter
from types import SimpleNamespace

import pytest

from goita_ai2.constants import PIECE_TOTALS
from goita_ai2.current_ai.agent import RuleBasedAgent as CurrentAI
from goita_ai2.current_ai.endgame import ForcedWinStatus
from goita_ai2.current_ai import generic_response_store as stores
from goita_ai2.experimental_ai2 import RuleBasedAgent
from goita_ai2.rule_based_intermediate_middle3 import RuleBasedAgent as Middle3
from goita_ai2.state import GoitaState


@pytest.fixture(autouse=True)
def independent_store(monkeypatch):
    monkeypatch.setattr(stores, "_GENERIC_RESPONSE_PATTERN_STORE", stores.GenericResponsePatternStore())


def position(piece="4", rotate=0, *, agent_class=RuleBasedAgent, own_count=2, with_royal=False):
    mapping = {"4": piece, piece: "4"}
    seats = {s: "ABCD"[(i + rotate) % 4] for i, s in enumerate("ABCD")}
    hands = {"A": list("52442361"), "B": list("24374152"),
             "C": list("57113161"), "D": list("31181195")}
    if own_count != 2:
        source, target = ("B", "A") if own_count == 1 else ("A", "B")
        hands[source].remove("4"); hands[source].append("3")
        hands[target].remove("3"); hands[target].append("4")
    if with_royal:
        hands["B"].remove("3"); hands["B"].append("9")
        hands["D"].remove("9"); hands["D"].append("3")
    hands = {seats[s]: [mapping.get(p, p) for p in hand] for s, hand in hands.items()}
    assert Counter(p for hand in hands.values() for p in hand) == Counter(PIECE_TOTALS)
    state = GoitaState(hands, dealer=seats["A"])
    agent = agent_class()
    agent.TIME_SEARCH_ENABLED = False
    agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    agent._relative_hand_rank_table = {}
    player = seats["B"]
    agent.bind_player(player)
    agent._ensure_trackers(state)
    action = ("attack_after_block", "1", piece)
    state.apply_attack_after_block(seats["A"], "1", piece)
    agent.on_public_action(state, seats["A"], action)
    return state, agent, player


@pytest.mark.parametrize("piece", "2345")
@pytest.mark.parametrize("rotate", range(4))
def test_all_middle_pieces_and_seats_remove_only_the_initial_rank_bias(piece, rotate):
    state, agent, player = position(piece, rotate)
    old_state, old_agent, old_player = position(piece, rotate, agent_class=CurrentAI)
    context = agent._opening_shared_pair_context(state, player)
    assert context and not context["proven"]
    for kind, block in [("pass", None), ("receive", piece)]:
        old_bonus = old_agent._enemy_first_receive_strength_bonus(old_state, old_player, kind, block)
        old_score = old_agent._score_receive_phase(old_state, old_player, kind, block)
        assert agent._score_receive_phase(state, player, kind, block) == old_score - old_bonus
        assert agent._enemy_first_receive_strength_bonus(state, player, kind, block) == 0
    assert agent._score_receive_phase(state, player, "receive", piece) > agent._score_receive_phase(state, player, "pass", None)
    assert agent.select_action(state, player, state.legal_actions(player)) == ("receive", piece, None)


@pytest.mark.parametrize("rotate", range(4))
def test_report_receives_silver_then_uses_the_existing_attack_choice(rotate):
    state, agent, player = position(rotate=rotate)
    before = copy.deepcopy(state.hands)
    assert agent.select_action(state, player, state.legal_actions(player)) == ("receive", "4", None)
    assert agent.last_score_fallback_detail == "receive_enemy_first_shared_pair_4"
    assert agent.last_attack_candidate_snapshot["opening_shared_pair"]["scores"] == [
        {"action": ("pass", None, None), "score": -26.0},
        {"action": ("receive", "4", None), "score": 67.0}]
    assert state.hands == before
    state.apply_receive(player, "4")
    agent.on_public_action(state, player, ("receive", "4", None))
    action = agent.select_action(state, player, state.legal_actions(player))
    assert action[0] == "attack" and action[2] in ("2", "7")
    assert state.hands[player].count("4") == 1
    assert "opening_shared_pair" not in agent.last_attack_candidate_snapshot


@pytest.mark.parametrize("own_count", [1, 3])
def test_one_or_three_own_copies_retain_the_existing_evaluation(own_count):
    state, agent, player = position(own_count=own_count)
    assert agent._opening_shared_pair_context(state, player) is None
    for kind, block in [("pass", None), ("receive", "4")]:
        assert agent._score_receive_phase(state, player, kind, block) == CurrentAI._score_receive_phase(
            agent, state, player, kind, block)


def test_later_lance_receiver_keeps_the_existing_pass_policy():
    state, agent, player = position("2")
    assert agent._enemy_first_same_piece_rank_policy_action(state, player, state.legal_actions(player)) is None
    # Move an identical decision shape to the later receiver. That role has an
    # existing, deliberately stronger preference for passing the first lance.
    state.hands["B"], state.hands["D"] = state.hands["D"], state.hands["B"]
    state.turn = "D"
    assert agent._opening_shared_pair_context(state, "D") is None
    assert agent._enemy_first_same_piece_rank_policy_action(state, "D", state.legal_actions("D")) == ("pass", None, None)


def test_middle_rank_lance_pair_also_reaches_continuation_comparison():
    state, agent, player = position("2", with_royal=True)
    assert agent._initial_hand_axes_for_state(state, player)["absolute_rank"] == "C"
    assert agent._opening_shared_pair_context(state, player)
    assert agent._enemy_first_same_piece_rank_policy_action(state, player, state.legal_actions(player)) is None
    assert agent._enemy_first_receive_strength_bonus(state, player, "pass", None) == 0


@pytest.mark.parametrize("purpose", ["ally_open_shi_attack_pending", "shi_attack_mode", "inherit_ally_shi_attack"])
def test_existing_shi_purposes_keep_priority(purpose):
    state, agent, player = position()
    agent._track[id(state)][purpose] = True
    assert agent._opening_shared_pair_context(state, player) is None


def test_decisive_search_can_still_choose_pass(monkeypatch):
    from goita_ai2.current_ai.timed_search import TimedSearchResult
    state, agent, player = position()
    agent.TIME_SEARCH_ENABLED = True
    result = TimedSearchResult(action=("pass", None, None), depth=9, samples=80,
                              nodes=100, elapsed_seconds=0.01, value=10000,
                              margin=10000, agreement=1, decisive=True)
    monkeypatch.setattr(agent, "_time_limited_search_action", lambda *args, **kwargs: result)
    chosen = agent.select_action(state, player, state.legal_actions(player))
    assert chosen == ("pass", None, None)
    assert agent.last_decision_reason == "time_search"
    assert "opening_shared_pair" not in agent.last_attack_candidate_snapshot


def test_hidden_enemy_cards_and_hidden_fold_do_not_change_the_decision():
    results = []
    for change in (False, True):
        state, agent, player = position()
        if change:
            # In this equally visible world the attacker actually had one silver.
            state.hands["A"].remove("4"); state.hands["A"].append("1")
            state.hands["C"].remove("1"); state.hands["C"].append("4")
            state.face_down_hidden["A"] = ["7"]
            state.hands["C"].remove("7"); state.hands["C"].append("1")
        choice = agent.select_action(state, player, state.legal_actions(player))
        results.append((choice, agent.last_attack_candidate_scores, agent.last_score_fallback_detail))
    assert results[0] == results[1]


@pytest.mark.parametrize("change", ["second_attack", "broken_signal", "other_public_copy", "immediate_enemy_finish", "own_attack_started"])
def test_different_public_contexts_keep_their_existing_policy(change):
    state, agent, player = position()
    tracker = agent._track[id(state)]
    if change == "second_attack": tracker["enemy_attack_counts"]["A"] = 2
    elif change == "broken_signal": tracker["public_hand_models"]["A"]["strategy_broken"] = True
    elif change == "other_public_copy": tracker["public_seen_counts"]["4"] = 2
    elif change == "immediate_enemy_finish": state.hands["C"] = state.hands["C"][:2]
    else: tracker["my_attack_count"] = 1
    assert agent._opening_shared_pair_context(state, player) is None


def test_proven_receive_keeps_priority_and_is_not_explained_as_a_pair_heuristic(monkeypatch):
    state, agent, player = position()
    monkeypatch.setattr(agent, "_forced_win_result_after_receive_action", lambda *args:
                        SimpleNamespace(status=ForcedWinStatus.PROVEN, minimum_score=30))
    assert agent._score_receive_phase(state, player, "receive", "4") == 1e8 + 30


def test_cached_response_cannot_restore_the_old_weak_hand_pass(monkeypatch):
    state, agent, player = position()
    monkeypatch.setattr(CurrentAI, "_lookup_conditional_response_plan",
                        lambda *args: pytest.fail("old response cache must not override the new comparison"))
    assert agent._lookup_conditional_response_plan(
        state, player, state.legal_actions(player), ("receive", "4", None)) is None


def test_saved_and_current_profiles_keep_their_original_weak_hand_pass():
    for profile in (CurrentAI, Middle3):
        state, agent, player = position(agent_class=profile)
        assert not hasattr(agent, "_opening_shared_pair_context")
        assert agent.select_action(state, player, state.legal_actions(player)) == ("pass", None, None)


def test_pair_scores_are_exported_for_a_receive_without_attack_candidates():
    import json
    from urllib.parse import unquote
    from backend.app import _format_ai_attack_candidates
    state, agent, player = position()
    agent.select_action(state, player, state.legal_actions(player))
    suffix = _format_ai_attack_candidates(agent)
    snapshot = json.loads(unquote(suffix.split(":", 1)[1][:-1]))
    assert snapshot["opening_shared_pair"]["proven"] is False
    assert len(snapshot["opening_shared_pair"]["scores"]) == 2

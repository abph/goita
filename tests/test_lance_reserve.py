"""Do not spend the team's last lance stop to take a weak optional initiative."""
import copy
from collections import Counter
from types import SimpleNamespace

import pytest

from goita_ai2.constants import PIECE_TOTALS
from goita_ai2.current_ai import generic_response_store as stores
from goita_ai2.current_ai.agent import RuleBasedAgent as CurrentAI
from goita_ai2.current_ai.endgame import ForcedWinStatus
from goita_ai2.experimental_ai2 import RuleBasedAgent
from goita_ai2.experimental_ai2.kyosha_continuation import KyoshaContinuationMixin
from goita_ai2.rule_based_intermediate_middle3 import RuleBasedAgent as Middle3
from goita_ai2.state import GoitaState


@pytest.fixture(autouse=True)
def isolated_store(monkeypatch):
    monkeypatch.setattr(stores, "_GENERIC_RESPONSE_PATTERN_STORE", stores.GenericResponsePatternStore())


def position(rotate=0, *, warm=False, agent_class=RuleBasedAgent):
    seats = {s: "ABCD"[(i + rotate) % 4] for i, s in enumerate("ABCD")}
    hands = {"A": "11122345", "B": "12344579", "C": "11124566", "D": "11133578"}
    state = GoitaState({seats[s]: list(h) for s, h in hands.items()}, dealer=seats["A"])
    assert Counter(p for h in state.hands.values() for p in h) == Counter(PIECE_TOTALS)
    agent = agent_class()
    agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    agent._relative_hand_rank_table = {}
    player = seats["B"]
    agent.bind_player(player)
    agent._ensure_trackers(state)
    history = [
        ("A", ("attack_after_block", "1", "2")),
        ("B", ("pass", None, None)), ("C", ("pass", None, None)), ("D", ("pass", None, None)),
        ("A", ("attack_after_block", "3", "1")),
        ("B", ("receive", "1", None)), ("B", ("attack", None, "4")),
        ("C", ("receive", "4", None)), ("C", ("attack", None, "2")),
        ("D", ("pass", None, None)), ("A", ("pass", None, None)),
    ]
    for seat, action in history:
        actor = seats[seat]
        if warm and actor == player:
            before = copy.deepcopy(agent._track[id(state)])
            predicted = agent.select_action(state, player, state.legal_actions(player))
            if predicted != action:
                # The recorded earlier B turns were human moves. Keep that
                # public history and undo the proposed private plan, while
                # retaining the prediction caches to exercise the warm path.
                agent._track[id(state)] = before
        kind, block, attack = action
        assert action in state.legal_actions(actor)
        if kind == "pass": state.apply_pass(actor)
        elif kind == "receive": state.apply_receive(actor, block)
        elif kind == "attack": state.apply_attack(actor, attack)
        else: state.apply_attack_after_block(actor, block, attack)
        agent.on_public_action(state, actor, action)
    return state, agent, player, seats


@pytest.mark.parametrize("rotate", range(4))
def test_report_passes_and_keeps_the_last_lance_in_every_seat(rotate):
    state, agent, player, seats = position(rotate)
    before = copy.deepcopy(state.hands)
    chosen = agent.select_action(state, player, state.legal_actions(player))
    assert chosen == ("pass", None, None)
    if "lance_reserve" not in agent.last_attack_candidate_snapshot:
        # The existing sampled comparison can already prefer pass in a rotated
        # seat. Its actual reason should remain intact in that case.
        assert (agent.last_score_fallback_detail.endswith("_pass")
                or agent.last_score_fallback_detail.startswith("pass_kyosha_three_attack_plan_"))
        context = agent._last_team_lance_context(state, player, state.legal_actions(player))
        review = agent._last_team_lance_review(state, player, context)
        assert review["enemy"] == seats["A"] and not review["proven"]
        assert state.hands == before
        return
    assert agent.last_score_fallback_detail == f"pass_preserve_last_team_lance_enemy_{seats['A']}"
    review = agent.last_attack_candidate_snapshot["lance_reserve"]
    assert review["enemy"] == seats["A"] and review["attacker"] == seats["C"]
    assert not review["proven"] and review["own_lances_before"] == 1
    assert all(row["strength"] == 0 for row in review["followups"])
    assert agent.last_rule_search_authority == "strong"
    assert agent._track[id(state)]["pending_kyosha_receive_attack_piece"] is None
    assert state.hands == before


def test_earlier_thinking_and_cached_candidates_do_not_erase_the_lance_stop():
    state, agent, player, _ = position(warm=True)
    assert agent.select_action(state, player, state.legal_actions(player)) == ("pass", None, None)
    assert agent.last_score_fallback_detail == "pass_preserve_last_team_lance_enemy_A"
    review = agent._track[id(state)]["last_round_route_comparison"]
    assert review["proposed_chosen"][0] == ("receive", "2", None)
    assert review["chosen"] == (("pass", None, None),)
    assert review["last_lance_reserve_changed_response"] is True


@pytest.mark.parametrize("change", ["own_pair", "ally_lance", "current_attacker_lance", "buried_threat", "enemy_two_cards", "ally_ready", "second_attack", "later_enemy_phase"])
def test_different_threats_or_urgent_finishes_keep_their_existing_policy(change):
    state, agent, player, _ = position()
    tracker = agent._track[id(state)]
    if change == "own_pair": state.hands[player].append("2")
    elif change == "ally_lance": tracker["estimated_current_hands"]["D"]["2"]["max"] = 1
    elif change == "current_attacker_lance": tracker["estimated_current_hands"]["C"]["2"]["max"] = 1
    elif change == "buried_threat": tracker["estimated_current_hands"]["A"]["2"]["expected"] = 0.1
    elif change == "enemy_two_cards": state.hands["A"] = state.hands["A"][:2]
    elif change == "ally_ready": state.hands["D"] = state.hands["D"][:2]
    elif change == "second_attack": tracker["enemy_attack_counts"]["C"] = 2
    else: state.hands["C"] = state.hands["C"][:4]
    assert agent._last_team_lance_context(state, player, state.legal_actions(player)) is None


def test_actual_hidden_lance_holder_does_not_change_public_safety_preference():
    contexts = []
    for swapped in (False, True):
        state, agent, player, _ = position()
        if swapped:
            state.hands["A"].remove("2"); state.hands["A"].append("3")
            state.hands["D"].remove("3"); state.hands["D"].append("2")
        context = agent._last_team_lance_context(state, player, state.legal_actions(player))
        contexts.append((context, agent._last_team_lance_review(state, player, context)))
    assert contexts[0] == contexts[1]


def cached_receive(self, state, player, actions):
    self._set_decision_reason("response_dictionary")
    self._set_score_fallback_detail("conditional_response_receive_cached_depth_7")
    self.last_rule_search_authority = "ordinary"
    self._track[id(state)]["pending_kyosha_receive_attack_piece"] = "7"
    self._record_generic_response_evidence(state, player, actions,
                                           ("pass", None, None), ("receive", "2", None),
                                           followup_piece="7", source="conditional_response_cache",
                                           depth=7, agreement=1.0, information_confidence=1.0,
                                           margin=200.0)
    return ("receive", "2", None)


def test_cached_receive_is_rejected_without_learning_it_as_the_final_move(monkeypatch):
    state, agent, player, _ = position()
    recorded = []
    monkeypatch.setattr(CurrentAI, "_record_generic_response_evidence",
                        lambda self, state, actor, legal, baseline, selected, **kw: recorded.append(selected))
    monkeypatch.setattr(KyoshaContinuationMixin, "_select_action_with_measurement", cached_receive)
    assert agent.select_action(state, player, state.legal_actions(player)) == ("pass", None, None)
    assert ("receive", "2", None) not in recorded
    assert agent._track[id(state)]["pending_kyosha_receive_attack_piece"] is None


def test_proven_receiving_win_keeps_priority(monkeypatch):
    state, agent, player, _ = position()
    monkeypatch.setattr(KyoshaContinuationMixin, "_select_action_with_measurement", cached_receive)
    monkeypatch.setattr(agent, "_forced_win_result_after_receive_action", lambda *args:
                        SimpleNamespace(status=ForcedWinStatus.PROVEN))
    assert agent.select_action(state, player, state.legal_actions(player)) == ("receive", "2", None)
    assert "lance_reserve" not in agent.last_attack_candidate_snapshot


def test_strong_continuation_is_still_compared_by_the_existing_strategy(monkeypatch):
    state, agent, player, _ = position()
    monkeypatch.setattr(KyoshaContinuationMixin, "_select_action_with_measurement", cached_receive)
    monkeypatch.setattr(agent, "_kyosha_continuation_review", lambda *args:
                        {"strength": 3, "uncertainty_penalty": 40, "outline": {"conditional_finish_score": 30}})
    assert agent.select_action(state, player, state.legal_actions(player)) == ("receive", "2", None)
    assert "lance_reserve" not in agent.last_attack_candidate_snapshot


def test_passing_preserves_the_lance_and_later_attacks_are_reassessed():
    state, agent, player, _ = position()
    state.apply_pass(player)
    agent.on_public_action(state, player, ("pass", None, None))
    state.apply_attack_after_block("C", "1", "6")
    agent.on_public_action(state, "C", ("attack_after_block", "1", "6"))
    assert state.hands[player].count("2") == 1
    assert agent._last_team_lance_context(state, player, state.legal_actions(player)) is None


def test_reserve_explanation_is_exported_without_attack_candidates():
    import json
    from urllib.parse import unquote
    from backend.app import _format_ai_attack_candidates
    snapshot = {"lance_reserve": {"enemy": "A", "proven": False}}
    payload = _format_ai_attack_candidates(SimpleNamespace(last_attack_candidate_snapshot=snapshot))
    assert json.loads(unquote(payload.split(":", 1)[1][:-1])) == snapshot


def test_current_and_saved_profiles_do_not_gain_the_new_reserve_rule():
    for profile in (CurrentAI, Middle3):
        assert not hasattr(profile(), "_last_team_lance_context")

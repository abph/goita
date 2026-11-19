"""Claims, partner echoes and consumed cards must describe the same four pieces."""
import copy
from collections import Counter

import pytest

from goita_ai2.constants import PIECE_TOTALS
from goita_ai2.current_ai.agent import RuleBasedAgent as CurrentAI
from goita_ai2.current_ai import generic_response_store as stores
from goita_ai2.experimental_ai2 import RuleBasedAgent
from goita_ai2.rule_based_intermediate_middle3 import RuleBasedAgent as Middle3
from goita_ai2.state import GoitaState


@pytest.fixture(autouse=True)
def isolated_dictionary(monkeypatch):
    monkeypatch.setattr(stores, "_GENERIC_RESPONSE_PATTERN_STORE", stores.GenericResponsePatternStore())


def position(piece="2", rotate=0, agent_class=RuleBasedAgent, prefix_length=None, observer="D"):
    seats = {s: "ABCD"[(i + rotate) % 4] for i, s in enumerate("ABCD")}
    pieces = {"2": piece, piece: "2"}
    convert = lambda p: pieces.get(p, p)
    hands = {"A": "22511341", "B": "73414925", "C": "42615161", "D": "33511178"}
    state = GoitaState({seats[s]: [convert(p) for p in h] for s, h in hands.items()}, dealer=seats["A"])
    assert Counter(p for h in state.hands.values() for p in h) == Counter(PIECE_TOTALS)
    agent = agent_class()
    agent._relative_hand_rank_table = {}
    agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    player = seats[observer]
    agent.bind_player(player)
    agent._ensure_trackers(state)
    history = [
        ("A", ("attack_after_block", "1", "2")),
        ("B", ("pass", None, None)), ("C", ("pass", None, None)), ("D", ("pass", None, None)),
        ("A", ("attack_after_block", "3", "1")),
        ("B", ("receive", "1", None)), ("B", ("attack", None, "4")),
        ("C", ("receive", "4", None)), ("C", ("attack", None, "2")),
        ("D", ("pass", None, None)), ("A", ("pass", None, None)),
        ("B", ("receive", "2", None)), ("B", ("attack", None, "7")),
        ("C", ("pass", None, None)), ("D", ("pass", None, None)), ("A", ("pass", None, None)),
        ("B", ("attack_after_block", "5", "4")), ("C", ("pass", None, None)),
    ]
    for who, action in history[:prefix_length]:
        actor = seats[who]
        kind, block, attack = action[0], convert(action[1]), convert(action[2])
        move = kind, block, attack
        assert move in state.legal_actions(actor)
        if kind == "pass": state.apply_pass(actor)
        elif kind == "receive": state.apply_receive(actor, block)
        elif kind == "attack": state.apply_attack(actor, attack)
        else: state.apply_attack_after_block(actor, block, attack)
        agent.on_public_action(state, actor, move)
    return state, agent, player, seats


@pytest.mark.parametrize("piece", "2345")
@pytest.mark.parametrize("rotate", range(4))
def test_observed_receive_resolves_claim_and_echo_for_all_middle_pieces(piece, rotate):
    state, agent, _, seats = position(piece, rotate)
    tracker = agent._track[id(state)]
    assert tracker["middle_first_attack_signals"][seats["A"]]["role"] == "independent_claim"
    assert tracker["middle_first_attack_signals"][seats["C"]]["role"] == "ally_echo"
    assert tracker["public_hand_models"][seats["C"]]["first_attack_role"] == "ally_echo"
    for seat, count in [("A", 2), ("B", 1), ("C", 1)]:
        item = agent._piece_count_estimate(tracker, seats[seat], piece)
        assert (item["min"], item["max"]) == (count, count)
        assert item["bounds_are"] == "strategic_preference"
    assert agent._estimate_remaining_range(tracker, seats["A"], piece) == (0, 1)
    assert agent._estimate_remaining_range(tracker, seats["C"], piece) == (0, 0)
    # A's unobserved fold may have used the remaining copy. The preferred
    # initial distribution never proves a current card or an actual win.
    assert tracker["estimated_current_hands"][seats["A"]][piece]["confidence"] < 1


def test_claim_survives_the_echo_and_then_the_public_receive():
    for length, a_min in [(1, 2), (9, 2), (12, 2)]:
        state, agent, _, _ = position(prefix_length=length)
        tracker = agent._track[id(state)]
        assert agent._piece_count_estimate(tracker, "A", "2")["min"] == a_min
        if length >= 9:
            assert agent._piece_count_estimate(tracker, "C", "2")["min"] == 1
        if length == 12:
            assert agent._piece_count_estimate(tracker, "C", "2")["max"] == 1


@pytest.mark.parametrize("observer", "ABCD")
@pytest.mark.parametrize("piece", "2345")
def test_each_observer_combines_own_known_count_with_the_same_public_distribution(observer, piece):
    state, agent, player, _ = position(piece=piece, observer=observer)
    tracker = agent._track[id(state)]
    assert tracker["middle_first_attack_signals"]["C"]["role"] == "ally_echo"
    for seat, expected in [("A", 2), ("B", 1), ("C", 1), ("D", 0)]:
        if seat == player:
            assert tracker["my_init_count"][piece] == expected
        else:
            estimate = agent._piece_count_estimate(tracker, seat, piece)
            assert (estimate["min"], estimate["max"]) == (expected, expected)


def test_explanation_metadata_does_not_invalidate_unchanged_inference():
    state, agent, _, _ = position()
    tracker = agent._track[id(state)]
    before = copy.deepcopy(tracker["estimated_current_hands"])
    agent._refresh_public_piece_inference_impl(state, tracker, reason="unchanged_verification")
    assert tracker["estimated_current_hands"] == before
    assert tracker["piece_inference_changed_players"] == ()


def test_decision_inference_summary_does_not_change_with_later_actions():
    state, agent, _, _ = position()
    tracker = agent._track[id(state)]
    summary = agent._middle_signal_summary(tracker)
    old = copy.deepcopy(summary)
    tracker["middle_first_attack_signals"]["C"]["role"] = "changed_later"
    tracker["joint_hand_inference"]["map_original_counts"]["A"]["2"] = 1
    assert summary == old


def test_real_deal_and_a_bluff_alternative_remain_possible_sample_worlds():
    state, agent, player, _ = position()
    tracker = agent._track[id(state)]
    actual = agent._information_set_prediction(state, player)
    assert agent._information_set_candidate_consistent(state, player, tracker, actual)
    assert agent._information_set_candidate_evidence(actual, player, tracker) is not None
    alternative = copy.deepcopy(state)
    alternative.hands["A"].remove("2"); alternative.hands["A"].append("1")
    alternative.hands["C"].remove("1"); alternative.hands["C"].append("2")
    prediction = agent._information_set_prediction(alternative, player)
    assert agent._information_set_candidate_consistent(state, player, tracker, prediction)
    assert agent._information_set_candidate_evidence(prediction, player, tracker) is not None
    samples = agent._timed_search_sample_states(state, player, tracker, 128)
    assert any("2" in sample.hands["A"] for sample in samples)
    assert any("2" in sample.hands["C"] for sample in samples)
    assert all(Counter(p for hand in sample.hands.values() for p in hand)
               + Counter(p for h in sample.face_down_hidden.values() for p in h)
               == Counter(p for h in state.hands.values() for p in h)
               + Counter(p for h in state.face_down_hidden.values() for p in h) for sample in samples)


def test_new_public_use_overrides_an_earlier_unconfirmed_claim():
    state, agent, _, _ = position()
    tracker = copy.deepcopy(agent._track[id(state)])
    # A future second public C lance would prove C's original count >= 2.
    tracker["public_hand_models"]["C"]["attacks"]["2"] = 2
    agent._reconcile_piece_count_estimates(state, tracker, "2")
    assert agent._piece_count_estimate(tracker, "C", "2")["min"] == 2
    assert agent._piece_count_estimate(tracker, "A", "2")["min"] == 1
    assert sum(agent._piece_count_estimate(tracker, s, "2")["min"] for s in "ABC") == 4


@pytest.mark.parametrize("rotate", range(4))
def test_report_receives_with_royal_and_offers_the_rook_to_the_ready_ally(rotate):
    state, agent, player, seats = position(rotate=rotate)
    before = copy.deepcopy(state.hands)
    chosen = agent.select_action(state, player, state.legal_actions(player))
    assert chosen == ("receive", "8", None)
    assert agent.last_score_fallback_detail.startswith("ally_reach_rescue_routes_depth_")
    review = agent._track[id(state)]["last_late_receive_comparison"]
    assert review["context"] == "ally_reach_rescue" and review["applied"]
    assert review["middle_signal_inference"]["preferred_initial_counts"][seats["A"]]["2"] == 2
    assert review["ally_reach_interception"]["unchecked_next_attack_mass"] > 0
    assert review["proven"] is False and state.hands == before
    state.apply_receive(player, "8")
    agent.on_public_action(state, player, chosen)
    assert agent.select_action(state, player, state.legal_actions(player)) == ("attack", None, "7")
    assert agent.last_score_fallback_detail == "ally_reach_rescue_followup_7"


def test_actual_hidden_contents_do_not_change_the_inferred_comparison():
    results = []
    for swapped in (False, True):
        state, agent, player, _ = position()
        if swapped:
            state.hands["B"].remove("9"); state.hands["B"].append("4")
            state.hands["A"].remove("4"); state.hands["A"].append("9")
        action = agent.select_action(state, player, state.legal_actions(player))
        review = agent._track[id(state)]["last_late_receive_comparison"]
        results.append((action, review["chosen"], review["routes"]))
    assert results[0] == results[1]


def test_deeper_timeout_keeps_a_complete_rescue_cycle_and_the_paired_rook(monkeypatch):
    from goita_ai2.current_ai.information_set_search import InformationSetSearchDeadline
    state, agent, player, _ = position()
    search = agent._information_set_search_bundle
    def limited(*args, **kwargs):
        if len(args[10]) == 1 and args[5] == 9:
            raise InformationSetSearchDeadline("time_limit")
        return search(*args, **kwargs)
    monkeypatch.setattr(agent, "_information_set_search_bundle", limited)
    chosen = agent.select_action(state, player, state.legal_actions(player))
    assert chosen == ("receive", "8", None)
    assert agent.last_score_fallback_detail == "ally_reach_rescue_routes_depth_8_samples_48_receive_attack_7_budget_limited"
    review = agent._track[id(state)]["last_late_receive_comparison"]
    assert review["depth"] == 8 and review["completed_depths"] == [8]
    state.apply_receive(player, "8")
    agent.on_public_action(state, player, chosen)
    assert agent.select_action(state, player, state.legal_actions(player)) == ("attack", None, "7")


def test_publicly_proven_ally_win_is_not_interrupted(monkeypatch):
    state, agent, player, _ = position()
    def proved_pass(self, state, player, actions):
        self._set_decision_reason("tsume")
        self._set_score_fallback_detail("pass_ally_guaranteed_win_no_self_finish")
        return ("pass", None, None)
    monkeypatch.setattr(CurrentAI, "_select_action_with_measurement", proved_pass)
    monkeypatch.setattr(agent, "_compare_late_receive_routes", lambda *args: pytest.fail("proven ally win"))
    assert agent.select_action(state, player, state.legal_actions(player)) == ("pass", None, None)


def test_a_completed_pass_comparison_can_keep_the_ally_attack(monkeypatch):
    state, agent, player, _ = position()
    comparison = {"context": "ally_reach_rescue", "chosen": (("pass", None, None),),
                  "reliable": True, "margin": 1000, "depth": 6, "samples": 48,
                  "routes": [{"actions": [("pass", None, None)], "value": 2000},
                             {"actions": [("receive", "8", None), ("attack", None, "7")], "value": 1000}]}
    monkeypatch.setattr(agent, "_compare_late_receive_routes", lambda *args: copy.deepcopy(comparison))
    assert agent.select_action(state, player, state.legal_actions(player)) == ("pass", None, None)
    assert agent.last_score_fallback_detail == "ally_reach_rescue_routes_depth_6_samples_48_pass"


def test_old_cached_pass_is_reconsidered_and_not_recorded_as_the_adopted_response(monkeypatch):
    from goita_ai2.current_ai.conditional_response import ConditionalResponsePlan
    state, agent, player, _ = position()
    legal = state.legal_actions(player)
    passed = ("pass", None, None)
    key = agent._conditional_response_key(state, player, legal, passed)
    agent._conditional_response_dictionary.put(key, ConditionalResponsePlan(
        action=passed, followup_attack_piece=None, baseline_action=passed,
        source="foreground", depth=7, agreement=0.8, information_confidence=0.8,
        margin=100, cache_source="foreground", cache_branch_kind=None,
        cache_branch_context=None, cached_compute_ms=100))
    records = []
    monkeypatch.setattr(CurrentAI, "_record_generic_response_evidence",
                        lambda self, state, actor, legal, baseline, selected, **kw: records.append(selected))
    chosen = agent.select_action(state, player, legal)
    assert chosen == ("receive", "8", None)
    assert passed not in records


def test_last_defender_is_not_predicted_to_allow_an_immediate_enemy_finish():
    state, agent, player, _ = position()
    state.apply_pass(player)
    tracker = agent._track[id(state)]
    agent._late_receive_route_context = {"ally_rescue": True}
    choices = agent._late_other_action_distribution(state, "A", tracker)
    assert choices and all(action[0] == "receive" for action, _weight in choices)
    # Only the ready attacker's public hand size is relevant to that urgency.
    state.hands["B"] = ["2", "5"]
    assert agent._late_other_action_distribution(state, "A", tracker) == choices


def test_no_extra_rescue_search_in_background_or_without_budget(monkeypatch):
    import threading
    state, agent, player, _ = position(rotate=1)
    agent._time_search_cancel_event = threading.Event()
    monkeypatch.setattr(agent, "_timed_search_sample_states", lambda *args: pytest.fail("background extra search"))
    assert agent._compare_late_receive_routes(state, player, state.legal_actions(player)) is None
    agent._time_search_cancel_event = None
    assert agent._compare_late_receive_routes(state, player, state.legal_actions(player), 0) is None
    assert agent._late_receive_comparison_attempt["stop_reason"] == "time_limit"


def test_saved_profiles_keep_the_previous_inference_convention():
    for profile in (CurrentAI, Middle3):
        state, agent, _, _ = position(agent_class=profile)
        assert not hasattr(agent, "_middle_signal_sampling_tracker")
        assert agent._estimate_remaining_range(agent._track[id(state)], "A", "2") == (0, 0)

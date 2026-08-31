from __future__ import annotations

import copy
import time

from goita_ai2.current_ai.prediction_cache import clear_prediction_sample_cache
from goita_ai2.current_ai.generic_response_store import (
    generic_response_pattern_snapshot,
    reset_generic_response_patterns,
)
from goita_ai2.current_ai.search_budget import reset_time_search_budget_model
from goita_ai2.current_ai.timed_search import TimedSearchMixin, TimedSearchResult
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


def _initial_state(*, permuted: bool = False) -> GoitaState:
    other_hands = {
        "B": list("11122334"),
        "C": list("11244556"),
        "D": list("11356789"),
    }
    if permuted:
        other_hands = {
            "B": list("11356789"),
            "C": list("11122334"),
            "D": list("11244556"),
        }
    return GoitaState(
        hands={"A": list("11123457"), **other_hands},
        dealer="A",
    )


def _sample_key(states) -> tuple:
    return tuple(
        tuple(
            (seat, tuple(sorted(state.hands[seat])))
            for seat in "BCD"
        )
        for state in states
    )


def test_rule_based_agent_uses_timed_search_mixin() -> None:
    assert issubclass(RuleBasedAgent, TimedSearchMixin)


def test_terminal_summary_separates_leaf_value_components() -> None:
    summary = TimedSearchMixin._timed_search_terminal_summary([
        (110000.0, 0.25),
        (-105000.0, 0.25),
        (200.0, 0.50),
    ])

    assert summary["terminal_win_rate"] == 0.25
    assert summary["terminal_loss_rate"] == 0.25
    assert summary["terminal_point_swing"] == 2.5
    assert summary["mean_value"] == 1350.0
    assert summary["terminal_outcome_component"] == 0.0
    assert summary["terminal_score_component"] == 1250.0
    assert summary["nonterminal_component"] == 100.0


def test_generic_hint_only_reorders_root_actions() -> None:
    actions = [
        ("pass", None, None),
        ("receive", "2", None),
        ("receive", "8", None),
    ]
    priority = ("receive", "2", None)

    ordered = TimedSearchMixin._timed_search_prioritize_root_actions(
        actions,
        priority,
    )

    assert ordered[0] == priority
    assert set(ordered) == set(actions)
    assert len(ordered) == len(actions)
    assert actions[0] == ("pass", None, None)


def test_generic_hint_effect_is_measured_without_changing_legal_actions() -> None:
    reset_generic_response_patterns()
    state = _initial_state()
    state.phase = "receive"
    state.turn = "A"
    state.attacker = "B"
    state.current_attack = "5"
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent._ensure_trackers(state)
    agent.TIME_SEARCH_CACHE_ENABLED = False
    agent.TIME_SEARCH_MAX_SECONDS = 0.3
    agent.TIME_SEARCH_SAMPLE_COUNT = 2
    agent.TIME_SEARCH_MAX_DEPTH = 1
    agent.TIME_SEARCH_MAX_NODES = 10_000
    actions = state.legal_actions("A")
    baseline = ("pass", None, None)
    priority = ("receive", "5", None)
    agent._generic_response_priority_action = (
        lambda _state, _player, _actions, _baseline: priority
    )

    result = agent._time_limited_search_action(
        state,
        "A",
        actions,
        baseline,
    )
    snapshot = generic_response_pattern_snapshot()

    assert result is not None
    assert result.action in actions
    assert set(actions) == {baseline, priority}
    assert snapshot["priority_effect_comparisons"] == 1
    assert snapshot["priority_effect_exact"] == 1
    assert snapshot["priority_effect_incomplete"] == 0
    assert snapshot["priority_effect_reorders"] == 1
    assert (
        snapshot["priority_effect_changed"]
        + snapshot["priority_effect_unchanged"]
        == 1
    )


def test_tactical_hint_takes_priority_and_records_the_final_search_choice() -> None:
    reset_generic_response_patterns()
    state = _initial_state()
    state.phase = "receive"
    state.turn = "A"
    state.attacker = "B"
    state.current_attack = "5"
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent._ensure_trackers(state)
    agent.TIME_SEARCH_CACHE_ENABLED = False
    agent.TIME_SEARCH_INFORMATION_SET_ENABLED = False
    agent.TIME_SEARCH_MAX_SECONDS = 0.3
    agent.TIME_SEARCH_SAMPLE_COUNT = 2
    agent.TIME_SEARCH_MAX_DEPTH = 1
    agent.TIME_SEARCH_MAX_NODES = 10_000
    actions = state.legal_actions("A")
    baseline = ("pass", None, None)
    priority = ("receive", "5", None)
    agent._tactical_response_priority_action = (
        lambda _state, _player, _actions, _baseline: priority
    )
    agent._generic_response_priority_action = (
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("generic priority must not replace tactical priority")
        )
    )

    result = agent._time_limited_search_action(
        state,
        "A",
        actions,
        baseline,
    )
    snapshot = generic_response_pattern_snapshot()

    assert result is not None
    assert result.action in actions
    assert snapshot["tactical_priority_effects"] == 1
    assert snapshot["tactical_priority_reordered"] == 1
    assert snapshot["tactical_priority_baseline_disagreements"] == 1
    assert snapshot["priority_effect_comparisons"] == 0


def test_tactical_hint_runs_an_isolated_paired_comparison() -> None:
    reset_generic_response_patterns()
    state = _initial_state()
    state.phase = "receive"
    state.turn = "A"
    state.attacker = "B"
    state.current_attack = "5"
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent._ensure_trackers(state)
    agent.TIME_SEARCH_CACHE_ENABLED = False
    agent.TIME_SEARCH_INFORMATION_SET_ENABLED = False
    agent.TIME_SEARCH_MAX_SECONDS = 0.3
    agent.TIME_SEARCH_SAMPLE_COUNT = 2
    agent.TIME_SEARCH_MAX_DEPTH = 1
    agent.TIME_SEARCH_MAX_NODES = 10_000
    agent.GENERIC_RESPONSE_TACTICAL_PAIRED_COMPARISON_ENABLED = True
    actions = state.legal_actions("A")
    baseline = ("pass", None, None)
    priority = ("receive", "5", None)
    agent._tactical_response_priority_action = (
        lambda _state, _player, _actions, _baseline: priority
    )
    generic_calls = []

    def no_generic_priority(*_args, **_kwargs):
        generic_calls.append(True)
        return None

    agent._generic_response_priority_action = no_generic_priority

    result = agent._time_limited_search_action(
        state,
        "A",
        actions,
        baseline,
    )
    snapshot = generic_response_pattern_snapshot()

    assert result is not None
    assert result.action in actions
    assert generic_calls == [True]
    assert snapshot["tactical_priority_effects"] == 1
    assert snapshot["tactical_pair_comparisons"] == 1
    assert snapshot["tactical_pair_completed"] == 1
    assert snapshot["tactical_pair_incomplete"] == 0
    assert (
        snapshot["tactical_pair_action_matches"]
        + snapshot["tactical_pair_action_changes"]
        == 1
    )
    assert snapshot["tactical_pair_average_with_depth"] == 1.0
    assert snapshot["tactical_pair_average_without_depth"] == 1.0


def test_human_hint_runs_equal_budget_root_fixed_comparisons() -> None:
    reset_generic_response_patterns()
    state = _initial_state()
    state.phase = "receive"
    state.turn = "A"
    state.attacker = "B"
    state.current_attack = "5"
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent._ensure_trackers(state)
    agent.TIME_SEARCH_CACHE_ENABLED = False
    agent.GENERIC_RESPONSE_HUMAN_PAIRED_COMPARISON_ENABLED = True
    actions = state.legal_actions("A")
    baseline = ("pass", None, None)
    ai_action = ("receive", "5", None)
    calls = []

    agent._timed_search_sample_states = lambda *_args, **_kwargs: [state]
    agent._human_response_comparison_actions = (
        lambda *_args, **_kwargs: (
            {
                "status": "recommended",
                "pattern_key": "human-pattern-root-test",
                "recommended_action": "pass",
            },
            (baseline,),
        )
    )

    def fake_search(
        _state,
        _player,
        _actions,
        _baseline,
        samples,
        *,
        run_context=None,
        forced_priority_action=None,
        **_kwargs,
    ):
        calls.append((
            id(samples),
            tuple(_actions),
            forced_priority_action,
            _kwargs.get("max_seconds_override"),
            _kwargs.get("min_depth_override"),
            _kwargs.get("max_nodes_multiplier_override"),
            _kwargs.get("disable_stable_stop", False),
        ))
        if forced_priority_action is None:
            return TimedSearchResult(
                action=ai_action,
                depth=5,
                samples=1,
                nodes=1200,
                elapsed_seconds=1.25,
                value=100.0,
                margin=20.0,
                agreement=1.0,
                decisive=False,
            )
        if forced_priority_action == ai_action:
            run_context["depth_results"] = {
                5: {
                    "evaluation_value": 90.0,
                    "terminal_win_rate": 0.20,
                    "terminal_loss_rate": 0.10,
                    "terminal_point_swing": 4.0,
                    "mean_value": 80.0,
                    "terminal_outcome_component": 10.0,
                    "terminal_score_component": 5.0,
                    "nonterminal_component": 65.0,
                },
                7: {
                    "evaluation_value": 100.0,
                    "terminal_win_rate": 0.25,
                    "terminal_loss_rate": 0.10,
                    "terminal_point_swing": 5.0,
                    "mean_value": 90.0,
                    "terminal_outcome_component": 15.0,
                    "terminal_score_component": 5.0,
                    "nonterminal_component": 70.0,
                },
            }
            return TimedSearchResult(
                action=ai_action,
                depth=7,
                samples=1,
                nodes=1700,
                elapsed_seconds=1.65,
                value=100.0,
                margin=0.0,
                agreement=1.0,
                decisive=False,
            )
        run_context["depth_results"] = {
            5: {
                "evaluation_value": 110.0,
                "terminal_win_rate": 0.40,
                "terminal_loss_rate": 0.05,
                "terminal_point_swing": 10.0,
                "mean_value": 100.0,
                "terminal_outcome_component": 20.0,
                "terminal_score_component": 10.0,
                "nonterminal_component": 70.0,
            },
            7: {
                "evaluation_value": 125.0,
                "terminal_win_rate": 0.50,
                "terminal_loss_rate": 0.05,
                "terminal_point_swing": 12.5,
                "mean_value": 120.0,
                "terminal_outcome_component": 30.0,
                "terminal_score_component": 15.0,
                "nonterminal_component": 75.0,
            },
            9: {
                "evaluation_value": 999.0,
                "terminal_win_rate": 0.90,
                "terminal_loss_rate": 0.01,
                "terminal_point_swing": 30.0,
                "mean_value": 990.0,
                "terminal_outcome_component": 90.0,
                "terminal_score_component": 30.0,
                "nonterminal_component": 870.0,
            },
        }
        return TimedSearchResult(
            action=baseline,
            depth=9,
            samples=1,
            nodes=1800,
            elapsed_seconds=1.75,
            value=999.0,
            margin=25.0,
            agreement=1.0,
            decisive=False,
        )

    agent._time_limited_search_from_samples = fake_search
    result = agent._time_limited_search_action(
        state,
        "A",
        actions,
        baseline,
    )
    snapshot = generic_response_pattern_snapshot()

    assert result is not None
    assert result.action == ai_action
    assert len(calls) == 3
    assert calls[0][0] == calls[1][0] == calls[2][0]
    assert calls[1][2] == ai_action
    assert calls[2][2] == baseline
    assert calls[1][3] == calls[2][3] == 5.0
    assert calls[1][4] == calls[2][4] == 5
    assert calls[1][5] == calls[2][5] == 1.5
    assert calls[1][6] is True
    assert calls[2][6] is True
    assert snapshot["human_pair_comparisons"] == 0
    assert snapshot["human_root_comparisons"] == 1
    assert snapshot["human_root_completed"] == 1
    assert snapshot["human_root_incomplete"] == 0
    assert snapshot["human_root_human_better"] == 1
    assert snapshot["human_root_ai_better"] == 0
    assert snapshot["human_root_average_ai_depth"] == 7.0
    assert snapshot["human_root_average_human_depth"] == 9.0
    assert snapshot["human_root_average_common_depth"] == 7.0
    assert snapshot["human_root_average_value_delta"] == 25.0
    assert snapshot["human_root_average_ai_terminal_win_rate"] == 0.25
    assert snapshot["human_root_average_human_terminal_win_rate"] == 0.5
    assert snapshot["human_root_average_ai_terminal_point_swing"] == 5.0
    assert snapshot["human_root_average_human_terminal_point_swing"] == 12.5
    assert snapshot["human_root_diag_completed"] == 1
    assert snapshot["human_root_diag_average_mean_value_delta"] == 30.0
    assert snapshot[
        "human_root_diag_average_terminal_outcome_delta"
    ] == 15.0
    assert snapshot[
        "human_root_diag_average_terminal_score_delta"
    ] == 10.0
    assert snapshot["human_root_diag_average_nonterminal_delta"] == 5.0
    assert snapshot["human_root_diag_action_pairs"][0]["human_action"] == "pass"
    assert snapshot["human_root_diag_action_pairs"][0][
        "ai_action"
    ] == "receive_same"
    assert snapshot["human_root_pattern_detail_count"] == 1
    assert snapshot["human_root_pattern_details"][0]["pattern_id"] == (
        "human-pattern-root-test"[:10]
    )
    assert snapshot["human_root_pattern_details"][0]["human_better"] == 1
    assert snapshot["human_root_pattern_details"][0]["ai_better"] == 0
    assert snapshot["human_root_pattern_details"][0][
        "average_value_delta"
    ] == 25.0


def test_generic_hint_narrowing_shadow_compares_after_depth_three() -> None:
    reset_generic_response_patterns()
    state = _initial_state()
    state.hands["A"].remove("7")
    state.hands["A"].append("8")
    state.hands["D"].remove("8")
    state.hands["D"].append("7")
    state.phase = "receive"
    state.turn = "A"
    state.attacker = "B"
    state.current_attack = "5"
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent._ensure_trackers(state)
    agent.TIME_SEARCH_CACHE_ENABLED = False
    agent.TIME_SEARCH_INFORMATION_SET_ENABLED = False
    agent.TIME_SEARCH_MAX_SECONDS = 1.0
    agent.TIME_SEARCH_SAMPLE_COUNT = 2
    agent.TIME_SEARCH_MAX_DEPTH = 3
    agent.TIME_SEARCH_MAX_NODES = 30_000
    actions = state.legal_actions("A")
    baseline = ("pass", None, None)
    priority = ("receive", "5", None)
    agent._generic_response_priority_action = (
        lambda _state, _player, _actions, _baseline: priority
    )

    result = agent._time_limited_search_action(
        state,
        "A",
        actions,
        baseline,
    )
    snapshot = generic_response_pattern_snapshot()

    assert result is not None
    assert result.depth == 3
    assert len(actions) == 3
    assert snapshot["narrowing_shadow_considered"] == 1
    assert snapshot["narrowing_shadow_comparisons"] == 1
    assert snapshot["narrowing_shadow_incomplete"] == 0
    assert snapshot["narrowing_shadow_average_full_candidates"] == 3.0
    assert snapshot["narrowing_shadow_average_kept_candidates"] == 2.0
    assert snapshot["narrowing_shadow_average_removed_candidates"] == 1.0
    assert (
        snapshot["narrowing_shadow_matches"]
        + snapshot["narrowing_shadow_mismatches"]
        == 1
    )


def test_generic_narrowing_requires_a_clear_depth_three_gap() -> None:
    priority = ("receive", "5", None)
    rival = ("receive", "8", None)
    excluded = ("pass", None, None)
    actions = [priority, rival, excluded]

    narrowed, status = RuleBasedAgent._timed_search_safe_generic_narrowing(
        actions,
        {priority: 1800.0, rival: 1400.0, excluded: 800.0},
        priority,
        enabled=True,
        search_profile="default",
        information_enabled=True,
        information_confidence=0.70,
        minimum_confidence=0.45,
        minimum_excluded_margin=450.0,
    )
    assert status == "applied"
    assert narrowed == [priority, rival]

    narrowed, status = RuleBasedAgent._timed_search_safe_generic_narrowing(
        actions,
        {priority: 1800.0, rival: 1400.0, excluded: 1100.0},
        priority,
        enabled=True,
        search_profile="default",
        information_enabled=True,
        information_confidence=0.70,
        minimum_confidence=0.45,
        minimum_excluded_margin=450.0,
    )
    assert narrowed is None
    assert status == "insufficient_margin"


def test_generic_narrowing_rejects_specialized_or_uncertain_searches() -> None:
    priority = ("receive", "5", None)
    rival = ("receive", "8", None)
    excluded = ("pass", None, None)
    actions = [priority, rival, excluded]
    aggregate = {priority: 1800.0, rival: 1400.0, excluded: 800.0}

    for profile, information_enabled, confidence, expected_status in (
        ("kyosha_pass_compare", True, 0.70, "specialized_profile"),
        ("default", False, 0.70, "information_unavailable"),
        ("default", True, 0.30, "low_confidence"),
    ):
        narrowed, status = RuleBasedAgent._timed_search_safe_generic_narrowing(
            actions,
            aggregate,
            priority,
            enabled=True,
            search_profile=profile,
            information_enabled=information_enabled,
            information_confidence=confidence,
            minimum_confidence=0.45,
            minimum_excluded_margin=450.0,
        )
        assert narrowed is None
        assert status == expected_status


def test_generic_narrowing_records_when_dictionary_move_is_outside_top_two() -> None:
    priority = ("receive", "5", None)
    rival = ("receive", "8", None)
    best = ("pass", None, None)
    actions = [priority, rival, best]

    narrowed, status = RuleBasedAgent._timed_search_safe_generic_narrowing(
        actions,
        {priority: 700.0, rival: 1400.0, best: 1800.0},
        priority,
        enabled=True,
        search_profile="default",
        information_enabled=True,
        information_confidence=0.70,
        minimum_confidence=0.45,
        minimum_excluded_margin=450.0,
    )
    assert narrowed is None
    assert status == "priority_not_top_two"


def test_generic_narrowing_reserves_one_second_within_hard_limit() -> None:
    deadline, extension = (
        RuleBasedAgent._timed_search_extend_narrowing_deadline(
            deadline=10.5,
            hard_deadline=30.0,
            now=10.4,
            minimum_continuation_seconds=1.0,
        )
    )
    assert deadline == 11.4
    assert round(extension, 6) == 0.9

    deadline, extension = (
        RuleBasedAgent._timed_search_extend_narrowing_deadline(
            deadline=29.8,
            hard_deadline=30.0,
            now=29.7,
            minimum_continuation_seconds=1.0,
        )
    )
    assert deadline == 30.0
    assert round(extension, 6) == 0.2


def test_generic_narrowing_adds_twenty_five_percent_nodes() -> None:
    maximum_nodes, added_nodes = (
        RuleBasedAgent._timed_search_extend_narrowing_nodes(
            maximum_nodes=10_000,
            extension_ratio=0.25,
        )
    )
    assert maximum_nodes == 12_500
    assert added_nodes == 2_500


def _search_result(
    *,
    depth: int = 7,
    agreement: float = 0.70,
    margin: float = 600.0,
    decisive: bool = True,
) -> TimedSearchResult:
    return TimedSearchResult(
        action=("pass", None, None),
        depth=depth,
        samples=8,
        nodes=100,
        elapsed_seconds=0.01,
        value=1000.0,
        margin=margin,
        agreement=agreement,
        decisive=decisive,
    )


def test_rule_search_authority_separates_proven_and_strategic_rules() -> None:
    agent = RuleBasedAgent()

    assert agent._rule_search_authority("win_now", "") == "proven"
    assert agent._rule_search_authority("tsume", "high_score_50") == "proven"
    assert agent._rule_search_authority("score_fallback", "receive_tsume_after") == "proven"
    assert agent._rule_search_authority("kakari", "") == "strong"
    assert agent._rule_search_authority("shi_signal", "") == "strong"
    assert agent._rule_search_authority("tsume", "") == "strong"
    assert agent._rule_search_authority(
        "score_fallback",
        "attack_sequence_two_kyosha_middle_pair",
    ) == "strong"
    assert agent._rule_search_authority(
        "score_fallback",
        "attack_occupancy",
    ) == "ordinary"


def test_strong_rule_requires_deep_agreed_search_before_override() -> None:
    agent = RuleBasedAgent()

    assert agent._search_may_override_rule("ordinary", _search_result(depth=5))
    assert agent._search_may_override_rule("strong", _search_result())
    assert not agent._search_may_override_rule(
        "strong",
        _search_result(depth=6),
    )
    assert not agent._search_may_override_rule(
        "strong",
        _search_result(agreement=0.69),
    )
    assert not agent._search_may_override_rule(
        "strong",
        _search_result(margin=599.0),
    )
    assert not agent._search_may_override_rule("proven", _search_result())


def test_strong_rule_runs_search_instead_of_stopping_it() -> None:
    class SearchProbeAgent(RuleBasedAgent):
        search_calls = 0

        def _select_rule_based_action(self, state, player, actions):
            self._set_decision_reason("kakari")
            self._set_score_fallback_detail("")
            return actions[0]

        def _time_limited_search_action(
            self,
            state,
            player,
            actions,
            baseline_action,
            *,
            cancel_event=None,
        ):
            type(self).search_calls += 1
            return TimedSearchResult(
                action=baseline_action,
                depth=1,
                samples=1,
                nodes=1,
                elapsed_seconds=0.0,
                value=0.0,
                margin=0.0,
                agreement=1.0,
                decisive=False,
            )

    SearchProbeAgent.search_calls = 0
    state = _initial_state()
    agent = SearchProbeAgent()
    chosen = agent.select_action(state, "A", state.legal_actions("A"))

    assert chosen in state.legal_actions("A")
    assert SearchProbeAgent.search_calls == 1
    assert agent.last_rule_search_authority == "strong"
    assert agent.last_search_skip_reason == ""


def test_proven_rule_still_skips_search() -> None:
    class ProvenProbeAgent(RuleBasedAgent):
        search_calls = 0

        def _select_rule_based_action(self, state, player, actions):
            self._set_decision_reason("win_now")
            self._set_score_fallback_detail("")
            return actions[0]

        def _time_limited_search_action(
            self,
            state,
            player,
            actions,
            baseline_action,
            *,
            cancel_event=None,
        ):
            type(self).search_calls += 1
            return None

    ProvenProbeAgent.search_calls = 0
    state = _initial_state()
    agent = ProvenProbeAgent()
    chosen = agent.select_action(state, "A", state.legal_actions("A"))

    assert chosen in state.legal_actions("A")
    assert ProvenProbeAgent.search_calls == 0
    assert agent.last_rule_search_authority == "proven"
    assert agent.last_search_skip_reason == "proven_rule"


def test_hidden_hand_sampling_does_not_read_actual_opponent_hands() -> None:
    first_state = _initial_state()
    second_state = _initial_state(permuted=True)
    first_agent = RuleBasedAgent()
    second_agent = RuleBasedAgent()
    first_agent.bind_player("A")
    second_agent.bind_player("A")
    first_agent._ensure_trackers(first_state)
    second_agent._ensure_trackers(second_state)

    first_samples = first_agent._timed_search_sample_states(
        first_state,
        "A",
        first_agent._track[id(first_state)],
        6,
    )
    second_samples = second_agent._timed_search_sample_states(
        second_state,
        "A",
        second_agent._track[id(second_state)],
        6,
    )

    assert len(first_samples) == 6
    assert len(second_samples) == 6
    assert _sample_key(first_samples) == _sample_key(second_samples)


def test_time_limited_search_returns_a_completed_legal_result() -> None:
    state = _initial_state()
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent.TIME_SEARCH_MAX_SECONDS = 0.4
    agent.TIME_SEARCH_SAMPLE_COUNT = 4
    agent.TIME_SEARCH_MAX_DEPTH = 5
    agent.TIME_SEARCH_MAX_NODES = 20_000
    agent._ensure_trackers(state)
    actions = state.legal_actions("A")
    baseline = agent._select_rule_based_action(state, "A", actions)

    result = agent._time_limited_search_action(
        state,
        "A",
        actions,
        baseline,
    )

    assert result is not None
    assert result.action in actions
    assert result.depth in (1, 3, 5)
    assert result.samples == 4
    assert result.nodes > 0
    assert result.elapsed_seconds <= 0.8


def test_comparison_search_applies_depth_and_node_overrides() -> None:
    state = _initial_state()
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent._ensure_trackers(state)
    agent.TIME_SEARCH_CACHE_ENABLED = False
    agent.TIME_SEARCH_INFORMATION_SET_ENABLED = False
    agent._effective_time_search_setting = (
        lambda name, default: {
            "effective_depth": 1,
            "effective_nodes": 100,
        }.get(name, default)
    )
    actions = state.legal_actions("A")
    baseline = agent._select_rule_based_action(state, "A", actions)
    run_context = {}

    agent._time_limited_search_from_samples(
        state,
        "A",
        actions,
        baseline,
        [state],
        run_context=run_context,
        max_seconds_override=0.01,
        min_depth_override=5,
        max_nodes_multiplier_override=1.5,
        disable_stable_stop=True,
    )

    assert run_context["configured_max_depth"] == 5
    assert run_context["configured_max_nodes"] == 150


def test_select_action_records_search_without_changing_public_state() -> None:
    state = _initial_state()
    hands_before = {seat: list(hand) for seat, hand in state.hands.items()}
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent.TIME_SEARCH_MAX_SECONDS = 0.4
    agent.TIME_SEARCH_SAMPLE_COUNT = 4
    agent.TIME_SEARCH_MAX_DEPTH = 5
    agent.TIME_SEARCH_MAX_NODES = 20_000

    action = agent.select_action(state, "A", state.legal_actions("A"))
    search = agent._track[id(state)].get("last_time_limited_search")

    assert action in state.legal_actions("A")
    assert search is not None
    assert search["depth"] in (1, 3, 5)
    assert state.hands == hands_before


def test_enemy_second_attack_wait_requires_a_robust_inferred_win() -> None:
    state = _initial_state()
    state.phase = "receive"
    state.turn = "A"
    state.attacker = "B"
    state.current_attack = "5"

    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent._ensure_trackers(state)
    tracker = agent._track[id(state)]
    tracker["enemy_attack_counts"]["B"] = 2

    common = dict(
        baseline_action=("receive", "5", None),
        best_action=("pass", None, None),
        completed_depth=7,
        agreement=0.875,
        information_enabled=True,
        information_confidence=0.75,
        best_minimum=110000.0,
        baseline_minimum=105000.0,
        margin=5000.0,
    )
    assert agent._timed_search_enemy_third_attack_wait_is_safe(
        state,
        "A",
        tracker,
        **common,
    )

    preserve_baseline = dict(common)
    preserve_baseline["baseline_action"] = ("pass", None, None)
    assert agent._timed_search_enemy_third_attack_wait_is_safe(
        state,
        "A",
        tracker,
        **preserve_baseline,
    )

    unsafe = dict(common)
    unsafe["best_minimum"] = 1200.0
    assert not agent._timed_search_enemy_third_attack_wait_is_safe(
        state,
        "A",
        tracker,
        **unsafe,
    )

    tracker["enemy_attack_counts"]["B"] = 1
    assert not agent._timed_search_enemy_third_attack_wait_is_safe(
        state,
        "A",
        tracker,
        **common,
    )


def test_weak_first_receive_search_accepts_only_a_clear_robust_result() -> None:
    agent = RuleBasedAgent()
    agent._time_search_profile = "weak_first_receive"
    common = dict(
        baseline_action=("pass", None, None),
        best_action=("receive", "5", None),
        completed_depth=9,
        agreement=0.43,
        margin=38000.0,
        information_enabled=True,
        information_confidence=0.55,
    )

    assert agent._timed_search_weak_first_receive_is_decisive(**common)

    narrow = dict(common)
    narrow["margin"] = 1000.0
    assert not agent._timed_search_weak_first_receive_is_decisive(**narrow)

    uncertain = dict(common)
    uncertain["information_confidence"] = 0.30
    assert not agent._timed_search_weak_first_receive_is_decisive(**uncertain)


def test_zero_shi_stop_signal_uses_context_limited_search_thresholds() -> None:
    agent = RuleBasedAgent()
    agent._time_search_profile = "weak_first_receive"
    result = dict(
        baseline_action=("pass", None, None),
        best_action=("receive", "4", None),
        completed_depth=7,
        agreement=0.582,
        margin=87.32,
        information_enabled=True,
        information_confidence=0.593,
    )

    assert not agent._timed_search_weak_first_receive_is_decisive(**result)
    assert agent._timed_search_weak_first_receive_is_decisive(
        **result,
        zero_shi_stop_signal=True,
    )


def test_low_reentry_receive_uses_five_second_search_thresholds() -> None:
    agent = RuleBasedAgent()
    agent._time_search_profile = "low_reentry_receive"
    result = dict(
        baseline_action=("pass", None, None),
        best_action=("receive", "2", None),
        completed_depth=7,
        agreement=0.745,
        margin=336.51,
        information_enabled=True,
        information_confidence=0.640,
    )

    assert agent._timed_search_weak_first_receive_is_decisive(**result)

    narrow = dict(result)
    narrow["margin"] = 199.0
    assert not agent._timed_search_weak_first_receive_is_decisive(**narrow)

    observed_silver_receive = dict(result)
    observed_silver_receive.update(
        completed_depth=5,
        agreement=0.658,
        margin=202.33,
        information_confidence=0.628,
    )
    assert agent._timed_search_weak_first_receive_is_decisive(
        **observed_silver_receive
    )


def test_kyosha_pass_compare_requires_a_completed_depth_seven_result() -> None:
    agent = RuleBasedAgent()
    agent._time_search_profile = "kyosha_pass_compare"
    result = dict(
        baseline_action=("pass", None, None),
        best_action=("receive", "2", None),
        completed_depth=7,
        agreement=0.75,
        margin=1.0,
        information_enabled=True,
        information_confidence=0.60,
    )

    assert agent._timed_search_kyosha_pass_compare_is_decisive(**result)

    shallow = dict(result)
    shallow["completed_depth"] = 5
    assert not agent._timed_search_kyosha_pass_compare_is_decisive(**shallow)

    pass_best = dict(result)
    pass_best["best_action"] = ("pass", None, None)
    assert not agent._timed_search_kyosha_pass_compare_is_decisive(**pass_best)

    receive_baseline = dict(result)
    receive_baseline["baseline_action"] = ("receive", "2", None)
    receive_baseline["best_action"] = ("pass", None, None)
    assert agent._timed_search_kyosha_pass_compare_is_decisive(
        **receive_baseline
    )


def test_depth_seven_kyosha_comparison_receives_and_attacks_fourth_horse() -> None:
    clear_prediction_sample_cache()
    reset_time_search_budget_model()
    state = GoitaState(
        hands={
            "A": list("11244678"),
            "B": list("11133457"),
            "C": list("11233456"),
            "D": list("11122559"),
        },
        dealer="B",
    )
    agents = {player: RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)
        agent._ensure_trackers(state)

    def apply_public(player: str, action) -> None:
        action_type, block, attack = action
        if action_type == "pass":
            state.apply_pass(player)
        elif action_type == "receive":
            state.apply_receive(player, block)
        elif action_type == "attack":
            state.apply_attack(player, attack)
        else:
            state.apply_attack_after_block(player, block, attack)
        for agent in agents.values():
            agent.on_public_action(state, player, action)

    opening = (
        ("B", ("attack_after_block", "1", "3")),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("attack_after_block", "1", "3")),
        ("C", ("receive", "3", None)),
        ("C", ("attack", None, "6")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("attack_after_block", "1", "4")),
        ("D", ("receive", "9", None)),
        ("D", ("attack", None, "2")),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
    )
    for player, action in opening:
        apply_public(player, action)

    c_agent = agents["C"]
    legal = state.legal_actions("C")
    preview = copy.deepcopy(c_agent)
    baseline = preview._select_rule_based_action(state, "C", legal)
    receive = c_agent.select_action(state, "C", legal)
    search = c_agent._track[id(state)]["last_time_limited_search"]

    assert baseline == ("pass", None, None)
    assert receive == ("receive", "2", None)
    assert c_agent.last_decision_reason == "time_search"
    assert c_agent.last_score_fallback_detail.startswith("kyosha_pass_compare_")
    assert search["depth"] == 7
    assert search["decisive"] is True
    assert search["budget"]["effective_seconds"] == 10.0

    apply_public("C", receive)
    attack = c_agent.select_action(state, "C", state.legal_actions("C"))

    assert attack == ("attack", None, "3")


def test_kyosha_receive_plan_keeps_public_fourth_silver_for_followup() -> None:
    state = GoitaState(
        hands={
            "A": list("11234678"),
            "B": list("11112557"),
            "C": list("11134459"),
            "D": list("12233456"),
        },
        dealer="C",
    )
    agents = {player: RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)
        agent._ensure_trackers(state)

    def apply_public(player: str, action) -> None:
        action_type, block, attack = action
        if action_type == "pass":
            state.apply_pass(player)
        elif action_type == "receive":
            state.apply_receive(player, block)
        elif action_type == "attack":
            state.apply_attack(player, attack)
        else:
            state.apply_attack_after_block(player, block, attack)
        for agent in agents.values():
            agent.on_public_action(state, player, action)

    opening = (
        ("C", ("attack_after_block", "1", "4")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("attack_after_block", "1", "4")),
        ("D", ("receive", "4", None)),
        ("D", ("attack", None, "3")),
        ("A", ("receive", "3", None)),
        ("A", ("attack", None, "6")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("receive", "6", None)),
        ("D", ("attack", None, "2")),
    )
    for action_player, action in opening:
        apply_public(action_player, action)

    a_agent = agents["A"]
    legal = state.legal_actions("A")
    assert a_agent._should_compare_kyosha_pass_and_receive(state, "A", legal)
    preview = copy.deepcopy(a_agent)
    baseline = preview._select_rule_based_action(state, "A", legal)
    receive = a_agent.select_action(state, "A", legal)

    assert receive == ("receive", "2", None)
    assert (
        a_agent._track[id(state)]["pending_kyosha_receive_attack_piece"]
        == "4"
    )
    stored_plan = a_agent._lookup_conditional_response_plan(
        state,
        "A",
        legal,
        baseline,
    )
    assert stored_plan is not None
    assert stored_plan.action == receive
    assert stored_plan.followup_attack_piece == "4"

    apply_public("A", receive)
    attack = a_agent._select_rule_based_action(
        state,
        "A",
        state.legal_actions("A"),
    )

    assert attack == ("attack", None, "4")
    assert a_agent.last_score_fallback_detail == "kyosha_receive_followup_4"


def test_zero_shi_stop_signal_context_matches_enemy_reply_to_ally_shi() -> None:
    state = GoitaState(
        hands={
            "A": list("56117431"),
            "B": list("41141319"),
            "C": list("45337562"),
            "D": list("58121221"),
        },
        dealer="A",
    )
    agent = RuleBasedAgent()
    agent.bind_player("C")
    agent._ensure_trackers(state)

    actions = (
        ("A", ("attack_after_block", "3", "1")),
        ("B", ("receive", "1", None)),
        ("B", ("attack", None, "4")),
    )
    for player, action in actions:
        action_type, block, attack = action
        if action_type == "receive":
            state.apply_receive(player, block)
        elif action_type == "attack":
            state.apply_attack(player, attack)
        else:
            state.apply_attack_after_block(player, block, attack)
        agent.on_public_action(state, player, action)

    assert agent._timed_search_zero_shi_stop_signal_context(
        state,
        "C",
        agent._track[id(state)],
        baseline_action=("pass", None, None),
        best_action=("receive", "4", None),
    )


def test_weak_rank_search_receives_gold_to_continue_ally_shi_attack() -> None:
    clear_prediction_sample_cache()
    reset_time_search_budget_model()
    state = GoitaState(
        hands={
            "A": list("19731141"),
            "B": list("52315436"),
            "C": list("51711262"),
            "D": list("51124843"),
        },
        dealer="A",
    )
    agents = {player: RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)
        agent._ensure_trackers(state)

    def apply_public(action_player: str, action) -> None:
        action_type, block, attack = action
        if action_type == "pass":
            state.apply_pass(action_player)
        elif action_type == "receive":
            state.apply_receive(action_player, block)
        elif action_type == "attack":
            state.apply_attack(action_player, attack)
        else:
            state.apply_attack_after_block(action_player, block, attack)
        for agent in agents.values():
            agent.on_public_action(state, action_player, action)

    opening = (
        ("A", ("attack_after_block", "1", "1")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("receive", "1", None)),
        ("D", ("attack", None, "4")),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("attack_after_block", "3", "4")),
        ("A", ("receive", "4", None)),
        ("A", ("attack", None, "1")),
        ("B", ("receive", "1", None)),
        ("B", ("attack", None, "5")),
    )
    for action_player, action in opening:
        apply_public(action_player, action)

    c_agent = agents["C"]
    receive = c_agent.select_action(state, "C", state.legal_actions("C"))
    search = c_agent._track[id(state)]["last_time_limited_search"]

    assert receive == ("receive", "5", None)
    assert c_agent.last_decision_reason == "time_search"
    assert c_agent.last_score_fallback_detail.startswith("weak_first_receive_")
    assert search["decisive"] is True
    assert search["budget"]["effective_seconds"] == 5.0

    apply_public("C", receive)
    attack = c_agent.select_action(state, "C", state.legal_actions("C"))

    assert attack == ("attack", None, "1")


def test_one_second_search_keeps_ally_gold_pass_after_broader_sampling() -> None:
    clear_prediction_sample_cache()
    reset_time_search_budget_model()
    state = GoitaState(
        hands={
            "A": list("24115126"),
            "B": list("41167234"),
            "C": list("89513451"),
            "D": list("31135723"),
        },
        dealer="B",
    )
    agents = {player: RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)
        agent._ensure_trackers(state)

    def apply_public(player: str, action) -> None:
        action_type, block, attack = action
        if action_type == "pass":
            state.apply_pass(player)
        elif action_type == "receive":
            state.apply_receive(player, block)
        elif action_type == "attack":
            state.apply_attack(player, attack)
        else:
            state.apply_attack_after_block(player, block, attack)
        for agent in agents.values():
            agent.on_public_action(state, player, action)

    opening = (
        ("B", ("attack_after_block", "1", "4")),
        ("C", ("receive", "4", None)),
        ("C", ("attack", None, "5")),
        ("D", ("pass", None, None)),
    )
    for player, action in opening:
        apply_public(player, action)

    a_agent = agents["A"]
    legal = state.legal_actions("A")
    preview = copy.deepcopy(a_agent)
    baseline = preview._select_rule_based_action(state, "A", legal)
    chosen = a_agent.select_action(state, "A", legal)
    search = a_agent._track[id(state)]["last_time_limited_search"]

    assert baseline == ("pass", None, None)
    assert chosen == ("pass", None, None)
    assert search["samples"] == 80
    assert search["decisive"] is False
    assert a_agent.last_score_fallback_detail == "pass_base"


def test_receive_branch_compares_followup_attacks_through_the_final_score() -> None:
    state = GoitaState(
        hands={
            "A": list("1145"),
            "B": list("9281"),
            "C": list("3366"),
            "D": list("5577"),
        },
        dealer="A",
    )
    state.phase = "receive"
    state.turn = "B"
    state.attacker = "A"
    state.current_attack = "4"

    agent = RuleBasedAgent()
    agent.bind_player("B")
    agent._ensure_trackers(state)
    after_receive = agent._timed_search_apply(
        state,
        "B",
        ("receive", "8", None),
    )
    baseline_scores = dict(state.team_score)
    deadline = time.perf_counter() + 2.0
    stats = {"nodes": 0, "max_nodes": 100_000}
    followup_values = {}
    for action in after_receive.legal_actions("B"):
        child = agent._timed_search_apply(after_receive, "B", action)
        followup_values[action] = agent._timed_search_minimax(
            child,
            "B",
            baseline_scores,
            8,
            -float("inf"),
            float("inf"),
            deadline,
            stats,
            {},
        )

    lance = ("attack", None, "2")
    assert max(followup_values, key=followup_values.get) == lance
    assert followup_values[lance] >= 125_000.0
    assert followup_values[lance] > followup_values[("attack", None, "9")]


if __name__ == "__main__":
    test_rule_based_agent_uses_timed_search_mixin()
    test_terminal_summary_separates_leaf_value_components()
    test_generic_hint_only_reorders_root_actions()
    test_generic_hint_effect_is_measured_without_changing_legal_actions()
    test_tactical_hint_takes_priority_and_records_the_final_search_choice()
    test_tactical_hint_runs_an_isolated_paired_comparison()
    test_human_hint_runs_equal_budget_root_fixed_comparisons()
    test_generic_hint_narrowing_shadow_compares_after_depth_three()
    test_generic_narrowing_requires_a_clear_depth_three_gap()
    test_generic_narrowing_rejects_specialized_or_uncertain_searches()
    test_generic_narrowing_records_when_dictionary_move_is_outside_top_two()
    test_generic_narrowing_reserves_one_second_within_hard_limit()
    test_generic_narrowing_adds_twenty_five_percent_nodes()
    test_rule_search_authority_separates_proven_and_strategic_rules()
    test_strong_rule_requires_deep_agreed_search_before_override()
    test_strong_rule_runs_search_instead_of_stopping_it()
    test_proven_rule_still_skips_search()
    test_hidden_hand_sampling_does_not_read_actual_opponent_hands()
    test_time_limited_search_returns_a_completed_legal_result()
    test_comparison_search_applies_depth_and_node_overrides()
    test_select_action_records_search_without_changing_public_state()
    test_enemy_second_attack_wait_requires_a_robust_inferred_win()
    test_kyosha_pass_compare_requires_a_completed_depth_seven_result()
    test_depth_seven_kyosha_comparison_receives_and_attacks_fourth_horse()
    test_kyosha_receive_plan_keeps_public_fourth_silver_for_followup()
    test_one_second_search_keeps_ally_gold_pass_after_broader_sampling()
    test_receive_branch_compares_followup_attacks_through_the_final_score()
    print("TIMED_SEARCH_MODULE_TEST_OK")

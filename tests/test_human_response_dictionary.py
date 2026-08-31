from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from goita_ai2.current_ai.generic_response_store import (
    GenericResponsePatternStore,
    generic_response_pattern_snapshot,
    reset_generic_response_patterns,
)
from goita_ai2.current_ai.human_response_dictionary import (
    DEFAULT_HUMAN_RESPONSE_DICTIONARY_PATH,
    build_human_response_dictionary,
    reload_human_response_dictionary,
)
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


def _receive_state() -> GoitaState:
    state = GoitaState(
        hands={
            "A": list("11234678"),
            "B": list("11122345"),
            "C": list("11123459"),
            "D": list("11234567"),
        },
        dealer="B",
    )
    state.phase = "receive"
    state.turn = "A"
    state.attacker = "B"
    state.current_attack = "2"
    return state


def test_human_dictionary_shadow_does_not_change_live_action() -> None:
    state = _receive_state()
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent._ensure_trackers(state)
    actions = state.legal_actions("A")
    baseline = ("pass", None, None)
    selected = ("receive", "2", None)
    tactical = agent._tactical_response_pattern_payload(
        state,
        "A",
        actions,
        baseline,
    )
    audit = {
        "response_summary": {
            "thresholds": {
                "minimum_observations": 15,
                "minimum_distinct_matches": 8,
                "minimum_dominance": 0.70,
            },
        },
        "patterns": [{
            "features": tactical,
            "observations": 20,
            "distinct_matches": 10,
            "action_counts": {"receive_same": 18, "pass": 2},
        }],
    }
    dictionary = build_human_response_dictionary(audit)

    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "human-response-patterns.json"
        path.write_text(
            json.dumps(dictionary, ensure_ascii=False),
            encoding="utf-8",
        )
        reload_human_response_dictionary(path)
        try:
            reset_generic_response_patterns()
            hand_before = list(state.hands["A"])
            agent._compare_generic_response_shadow(
                state,
                "A",
                actions,
                baseline,
                selected,
            )
            snapshot = generic_response_pattern_snapshot()

            assert agent.last_generic_response_human_shadow["status"] == "match"
            assert snapshot["human_dictionary_patterns"] == 1
            assert snapshot["human_shadow_lookups"] == 1
            assert snapshot["human_shadow_recommendations"] == 1
            assert snapshot["human_shadow_matches"] == 1
            assert list(state.hands["A"]) == hand_before
            assert state.phase == "receive"
            assert selected == ("receive", "2", None)
        finally:
            reload_human_response_dictionary(
                DEFAULT_HUMAN_RESPONSE_DICTIONARY_PATH
            )


def test_deployable_dictionary_contains_only_aggregate_fields() -> None:
    payload = json.loads(
        DEFAULT_HUMAN_RESPONSE_DICTIONARY_PATH.read_text(encoding="utf-8")
    )
    encoded = json.dumps(payload, ensure_ascii=False)

    assert payload["pattern_count"] == 738
    assert payload["live_ai_affected"] is False
    assert payload["privacy"]["names_retained"] is False
    for private_key in (
        "played_at",
        "initial_hands",
        '"history"',
        '"match_id"',
        '"players"',
    ):
        assert private_key not in encoded


def test_human_shadow_metrics_survive_checkpoint() -> None:
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "generic-response-patterns.json"
        store = GenericResponsePatternStore(path=path)
        result = store.compare_human_shadow(
            recommendation={
                "status": "recommended",
                "pattern_key": "human-pattern-1",
                "anonymous_context": {
                    "attacker_relation": "enemy",
                    "attack_piece": "2",
                    "attack_stage": "first",
                    "hand_stage": "early",
                    "next_receiver_stage": "middle",
                    "same_piece": "one",
                    "royal_receive": False,
                    "followup_strength": "pair",
                    "reentry_width": "wide",
                    "shi_context": "not_shi",
                    "score_pressure": "normal",
                },
                "recommended_action": "receive_same",
                "observations": 20,
                "support": 18,
                "dominance": 0.9,
                "distinct_matches_lower_bound": 8,
            },
            actual_action="pass",
        )
        assert result["status"] == "mismatch"
        snapshot = store.snapshot()
        assert snapshot["human_mismatch_detail_count"] == 1
        assert snapshot["human_mismatch_details"][0]["pattern_id"] == (
            "human-pattern-1"[:10]
        )
        assert snapshot["human_mismatch_details"][0][
            "recommended_action"
        ] == "receive_same"
        assert snapshot["human_mismatch_details"][0]["actual_action"] == "pass"
        assert snapshot["human_mismatch_details"][0]["observations"] == 20
        assert snapshot["human_mismatch_details"][0]["support"] == 18
        assert snapshot["human_mismatch_details"][0]["dominance"] == 0.9
        assert snapshot["human_mismatch_details"][0][
            "distinct_matches_lower_bound"
        ] == 8
        assert snapshot["human_mismatch_details"][0]["anonymous_context"] == {
            "attacker_relation": "enemy",
            "attack_piece": "2",
            "attack_stage": "first",
            "hand_stage": "early",
            "next_receiver_stage": "middle",
            "same_piece": "one",
            "royal_receive": False,
            "followup_strength": "pair",
            "reentry_width": "wide",
            "shi_context": "not_shi",
            "score_pressure": "normal",
        }
        store.record_human_priority_pair(
            comparison_complete=True,
            selected_side="human",
            normal_depth=5,
            priority_depth=7,
            normal_elapsed_seconds=1.25,
            priority_elapsed_seconds=1.75,
            normal_nodes=1200,
            priority_nodes=1800,
            value_delta=250.0,
        )
        snapshot = store.snapshot()
        assert snapshot["human_pair_comparisons"] == 1
        assert snapshot["human_pair_completed"] == 1
        assert snapshot["human_pair_incomplete"] == 0
        assert snapshot["human_pair_human_selected"] == 1
        assert snapshot["human_pair_ai_selected"] == 0
        assert snapshot["human_pair_human_value_better"] == 1
        assert snapshot["human_pair_ai_value_better"] == 0
        assert snapshot["human_pair_average_normal_depth"] == 5.0
        assert snapshot["human_pair_average_priority_depth"] == 7.0
        assert snapshot["human_pair_average_normal_elapsed_seconds"] == 1.25
        assert snapshot["human_pair_average_priority_elapsed_seconds"] == 1.75
        assert snapshot["human_pair_average_normal_nodes"] == 1200.0
        assert snapshot["human_pair_average_priority_nodes"] == 1800.0
        assert snapshot["human_pair_average_value_delta"] == 250.0
        store.record_human_root_pair(
            comparison_complete=True,
            pattern_key="human-pattern-1",
            selected_side="human",
            common_depth=7,
            ai_depth=7,
            human_depth=9,
            ai_elapsed_seconds=4.8,
            human_elapsed_seconds=4.9,
            ai_nodes=12000,
            human_nodes=12500,
            value_delta=375.0,
            ai_terminal_win_rate=0.25,
            human_terminal_win_rate=0.5,
            ai_terminal_loss_rate=0.2,
            human_terminal_loss_rate=0.1,
            ai_terminal_point_swing=2.5,
            human_terminal_point_swing=7.5,
            human_action_label="pass",
            ai_action_label="receive_same",
            mean_value_delta=300.0,
            terminal_outcome_delta=125.0,
            terminal_score_delta=25.0,
            nonterminal_delta=150.0,
        )
        snapshot = store.snapshot()
        assert snapshot["human_root_comparisons"] == 1
        assert snapshot["human_root_completed"] == 1
        assert snapshot["human_root_human_better"] == 1
        assert snapshot["human_root_ai_better"] == 0
        assert snapshot["human_root_average_ai_depth"] == 7.0
        assert snapshot["human_root_average_human_depth"] == 9.0
        assert snapshot["human_root_average_common_depth"] == 7.0
        assert snapshot["human_root_average_value_delta"] == 375.0
        assert snapshot["human_root_average_ai_terminal_win_rate"] == 0.25
        assert snapshot["human_root_average_human_terminal_win_rate"] == 0.5
        assert snapshot["human_root_average_ai_terminal_loss_rate"] == 0.2
        assert snapshot["human_root_average_human_terminal_loss_rate"] == 0.1
        assert snapshot["human_root_average_ai_terminal_point_swing"] == 2.5
        assert snapshot["human_root_average_human_terminal_point_swing"] == 7.5
        assert snapshot["human_root_diag_completed"] == 1
        assert snapshot["human_root_diag_average_mean_value_delta"] == 300.0
        assert snapshot[
            "human_root_diag_average_terminal_outcome_delta"
        ] == 125.0
        assert snapshot[
            "human_root_diag_average_terminal_score_delta"
        ] == 25.0
        assert snapshot["human_root_diag_average_nonterminal_delta"] == 150.0
        assert snapshot["human_root_diag_action_pairs"] == [{
            "human_action": "pass",
            "ai_action": "receive_same",
            "comparisons": 1,
            "human_better": 1,
            "ai_better": 0,
        }]
        assert snapshot["human_root_pattern_detail_count"] == 1
        assert snapshot["human_root_pattern_details"] == [{
            "pattern_id": "human-pattern-1"[:10],
            "comparisons": 1,
            "completed": 1,
            "incomplete": 0,
            "human_better": 1,
            "ai_better": 0,
            "tied": 0,
            "average_value_delta": 375.0,
            "minimum_value_delta": 375.0,
            "maximum_value_delta": 375.0,
            "average_common_depth": 7.0,
            "terminal_loss_samples": 1,
            "average_ai_terminal_loss_rate": 0.2,
            "average_human_terminal_loss_rate": 0.1,
            "candidate": False,
            "candidate_reason": "insufficient_comparisons",
            "action_pairs": [{
                "human_action": "pass",
                "ai_action": "receive_same",
                "comparisons": 1,
            }],
            "last_seen_at": snapshot["human_root_pattern_details"][0][
                "last_seen_at"
            ],
        }]
        store.record_human_root_pair(
            comparison_complete=False,
            pattern_key="human-pattern-1",
            incomplete_reason="common_depth_below_five",
            human_stop_reason="node_limit",
            ai_stop_reason="time_limit",
            human_action_label="pass",
            ai_action_label="receive_same",
        )
        snapshot = store.snapshot()
        assert snapshot["human_root_comparisons"] == 2
        assert snapshot["human_root_incomplete"] == 1
        assert snapshot["human_root_incomplete_shallow"] == 1
        assert snapshot["human_root_diag_incomplete"] == 1
        assert snapshot["human_root_diag_human_stop_nodes"] == 1
        assert snapshot["human_root_diag_ai_stop_time"] == 1
        assert snapshot["human_root_pattern_details"][0]["comparisons"] == 2
        assert snapshot["human_root_pattern_details"][0]["completed"] == 1
        assert snapshot["human_root_pattern_details"][0]["incomplete"] == 1
        assert snapshot["human_root_pattern_details"][0]["action_pairs"][0][
            "comparisons"
        ] == 2
        assert store.checkpoint("human-shadow-test") is True
        saved_payload = json.loads(path.read_text(encoding="utf-8"))
        saved_pattern = saved_payload["human_root_pattern_details"][
            "human-pattern-1"
        ]
        assert set(saved_pattern) == {
            "pattern_key",
            "comparisons",
            "completed",
            "incomplete",
            "human_better",
            "ai_better",
            "tied",
            "value_delta_sum",
            "value_delta_min",
            "value_delta_max",
            "common_depth_sum",
            "terminal_loss_samples",
            "ai_terminal_loss_rate_sum",
            "human_terminal_loss_rate_sum",
            "action_pair_counts",
            "first_seen_at",
            "last_seen_at",
        }

        restored = GenericResponsePatternStore(path=path).snapshot()
        assert restored["human_shadow_lookups"] == 1
        assert restored["human_shadow_recommendations"] == 1
        assert restored["human_shadow_mismatches"] == 1
        assert restored["human_mismatch_detail_count"] == 1
        assert restored["human_mismatch_details"][0]["count"] == 1
        assert restored["human_mismatch_details"][0][
            "anonymous_context"
        ]["attack_piece"] == "2"
        assert restored["human_pair_completed"] == 1
        assert restored["human_pair_human_selected"] == 1
        assert restored["human_pair_average_value_delta"] == 250.0
        assert restored["human_root_completed"] == 1
        assert restored["human_root_human_better"] == 1
        assert restored["human_root_average_value_delta"] == 375.0
        assert restored["human_root_average_common_depth"] == 7.0
        assert restored["human_root_average_human_terminal_win_rate"] == 0.5
        assert restored["human_root_diag_completed"] == 1
        assert restored["human_root_diag_incomplete"] == 1
        assert restored["human_root_diag_human_stop_nodes"] == 1
        assert restored["human_root_diag_action_pairs"][0][
            "human_action"
        ] == "pass"
        assert restored["human_root_pattern_detail_count"] == 1
        assert restored["human_root_pattern_details"][0]["comparisons"] == 2
        assert restored["human_root_pattern_details"][0]["human_better"] == 1
        assert restored["human_root_pattern_details"][0][
            "average_value_delta"
        ] == 375.0
        assert restored["human_root_pattern_details"][0][
            "terminal_loss_samples"
        ] == 1


def test_human_root_candidate_filter_requires_repeatable_safe_gain() -> None:
    store = GenericResponsePatternStore()

    for _index in range(5):
        store.record_human_root_pair(
            comparison_complete=True,
            pattern_key="safe-human-pattern",
            selected_side="human",
            common_depth=5,
            value_delta=100.0,
            ai_terminal_loss_rate=0.2,
            human_terminal_loss_rate=0.1,
            human_action_label="pass",
            ai_action_label="receive_same",
        )

    snapshot = store.snapshot()
    safe_pattern = snapshot["human_root_pattern_details"][0]
    assert snapshot["human_root_candidate_min_comparisons"] == 5
    assert snapshot["human_root_candidate_count"] == 1
    assert safe_pattern["candidate"] is True
    assert safe_pattern["candidate_reason"] == "candidate"
    assert safe_pattern["terminal_loss_samples"] == 5
    assert safe_pattern["average_human_terminal_loss_rate"] == 0.1
    assert safe_pattern["average_ai_terminal_loss_rate"] == 0.2

    store.record_human_root_pair(
        comparison_complete=True,
        pattern_key="safe-human-pattern",
        selected_side="human",
        common_depth=5,
        value_delta=100.0,
        ai_terminal_loss_rate=0.0,
        human_terminal_loss_rate=1.0,
        human_action_label="pass",
        ai_action_label="receive_same",
    )

    snapshot = store.snapshot()
    unsafe_pattern = snapshot["human_root_pattern_details"][0]
    assert snapshot["human_root_candidate_count"] == 0
    assert snapshot["human_root_candidate_terminal_loss_increased"] == 1
    assert unsafe_pattern["candidate"] is False
    assert unsafe_pattern["candidate_reason"] == "terminal_loss_increased"


def test_restored_root_patterns_wait_for_new_terminal_loss_samples() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "generic-response-patterns.json"
        store = GenericResponsePatternStore(path=path)
        for _index in range(5):
            store.record_human_root_pair(
                comparison_complete=True,
                pattern_key="legacy-positive-pattern",
                selected_side="human",
                common_depth=5,
                value_delta=100.0,
                ai_terminal_loss_rate=0.2,
                human_terminal_loss_rate=0.1,
            )
        assert store.checkpoint("before-candidate-safety") is True

        payload = json.loads(path.read_text(encoding="utf-8"))
        legacy = payload["human_root_pattern_details"][
            "legacy-positive-pattern"
        ]
        legacy.pop("terminal_loss_samples")
        legacy.pop("ai_terminal_loss_rate_sum")
        legacy.pop("human_terminal_loss_rate_sum")
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        restored = GenericResponsePatternStore(path=path).snapshot()
        detail = restored["human_root_pattern_details"][0]
        assert restored["human_root_candidate_count"] == 0
        assert restored["human_root_candidate_insufficient_loss_data"] == 1
        assert detail["completed"] == 5
        assert detail["terminal_loss_samples"] == 0
        assert detail["candidate_reason"] == "insufficient_loss_data"


def test_old_root_comparison_metrics_reset_without_losing_other_data() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "generic-response-patterns.json"
        store = GenericResponsePatternStore(path=path)
        store.record_human_priority_pair(
            comparison_complete=True,
            selected_side="ai",
        )
        store.record_human_root_pair(
            comparison_complete=True,
            selected_side="ai",
            common_depth=5,
        )
        assert store.checkpoint("old-root-comparison") is True

        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["counters"].pop("human_root_comparison_version", None)
        path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

        restored = GenericResponsePatternStore(path=path).snapshot()
        assert restored["human_pair_completed"] == 1
        assert restored["human_root_comparisons"] == 0
        assert restored["human_root_completed"] == 0
        assert restored["human_root_average_common_depth"] == 0.0


def test_new_root_diagnostics_start_clean_without_resetting_root_totals() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "generic-response-patterns.json"
        store = GenericResponsePatternStore(path=path)
        store.record_human_root_pair(
            comparison_complete=True,
            selected_side="ai",
            common_depth=5,
            human_action_label="pass",
            ai_action_label="receive_same",
            mean_value_delta=-50.0,
        )
        assert store.checkpoint("pre-diagnostic-version") is True

        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["counters"].pop("human_root_diagnostic_version", None)
        path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

        restored = GenericResponsePatternStore(path=path).snapshot()
        assert restored["human_root_completed"] == 1
        assert restored["human_root_ai_better"] == 1
        assert restored["human_root_diag_comparisons"] == 0
        assert restored["human_root_diag_completed"] == 0
        assert restored["human_root_diag_action_pairs"] == []


if __name__ == "__main__":
    test_human_dictionary_shadow_does_not_change_live_action()
    test_deployable_dictionary_contains_only_aggregate_fields()
    test_human_shadow_metrics_survive_checkpoint()
    test_human_root_candidate_filter_requires_repeatable_safe_gain()
    test_restored_root_patterns_wait_for_new_terminal_loss_samples()
    test_old_root_comparison_metrics_reset_without_losing_other_data()
    test_new_root_diagnostics_start_clean_without_resetting_root_totals()
    print("HUMAN_RESPONSE_DICTIONARY_TEST_OK")

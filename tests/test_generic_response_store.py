from __future__ import annotations

import json
import tempfile
from pathlib import Path

from goita_ai2.current_ai.generic_response_store import (
    GenericResponsePatternStore,
    medium_response_pattern_payload,
    resolve_generic_response_store_path,
    tactical_response_pattern_payload,
)
from goita_ai2.current_ai.search_cache import _digest_payload


def _features() -> dict:
    return {
        "version": 1,
        "context": {
            "attacker_relation": "enemy",
            "attack_family": "middle",
        },
        "hand": {
            "size": "middle",
            "shi": "2",
            "middle_pairs": "1",
        },
    }


def test_store_aggregates_and_restores_anonymous_patterns() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "generic-patterns.json"
        store = GenericResponsePatternStore(path=path)
        store.record(
            pattern_key="pattern-a",
            features=_features(),
            action_label="receive_same",
            followup_label="middle_pair",
            source="default",
            depth=7,
            agreement=0.75,
            confidence=0.65,
            margin=120.0,
        )
        store.record(
            pattern_key="pattern-a",
            features=_features(),
            action_label="pass",
            followup_label="none",
            source="response_dictionary_background",
            depth=5,
            agreement=0.60,
            confidence=0.45,
            margin=40.0,
        )
        store.reject("depth")
        assert store.compare_shadow(
            pattern_key="pattern-a",
            actual_action="receive_same",
            min_observations=2,
            min_dominance=0.5,
        )["status"] == "ambiguous"

        assert store.checkpoint("test") is True
        restored = GenericResponsePatternStore(path=path)
        snapshot = restored.snapshot()
        pattern = restored.pattern("pattern-a")

        assert snapshot["considered"] == 3
        assert snapshot["recorded"] == 2
        assert snapshot["rejected"] == 1
        assert snapshot["pattern_count"] == 1
        assert snapshot["action_counts"] == {
            "receive_same": 1,
            "pass": 1,
        }
        assert snapshot["rejection_counts"] == {"depth": 1}
        assert snapshot["shadow_lookups"] == 1
        assert snapshot["shadow_ambiguous"] == 1
        assert pattern is not None
        assert pattern["observations"] == 2
        assert pattern["average_depth"] == 6.0
        assert pattern["routes"] == [
            {
                "action": "pass",
                "followup": "none",
                "observations": 1,
                "source_counts": {"response_dictionary_background": 1},
                "average_depth": 5.0,
                "average_agreement": 0.6,
                "average_confidence": 0.45,
                "average_margin": 40.0,
            },
            {
                "action": "receive_same",
                "followup": "middle_pair",
                "observations": 1,
                "source_counts": {"default": 1},
                "average_depth": 7.0,
                "average_agreement": 0.75,
                "average_confidence": 0.65,
                "average_margin": 120.0,
            },
        ]

        encoded = path.read_text(encoding="utf-8")
        parsed = json.loads(encoded)
        assert parsed["reason"] == "test"
        for forbidden in (
            "player_name",
            "room_id",
            "kifu",
            "exact_hand",
            "プレイヤーA",
        ):
            assert forbidden not in encoded


def test_store_path_uses_render_persistent_directory() -> None:
    assert resolve_generic_response_store_path({}) is None
    assert resolve_generic_response_store_path({
        "GOITA_PERSISTENT_DATA_DIR": "/var/data",
    }) == Path("/var/data/goita-ai/generic-response-patterns.json")
    assert resolve_generic_response_store_path({
        "GOITA_PERSISTENT_DATA_DIR": "/var/data",
        "GOITA_GENERIC_RESPONSE_PATTERN_PATH": "/tmp/custom.json",
    }) == Path("/tmp/custom.json")


def test_shadow_comparison_requires_support_and_never_selects_an_action() -> None:
    store = GenericResponsePatternStore()
    assert store.compare_shadow(
        pattern_key="missing",
        actual_action="pass",
    ) == {"status": "no_pattern"}

    for _index in range(4):
        store.record(
            pattern_key="pattern-b",
            features=_features(),
            action_label="receive_same",
            followup_label="middle_pair",
            source="default",
            depth=7,
            agreement=0.75,
            confidence=0.65,
            margin=120.0,
        )
    assert store.compare_shadow(
        pattern_key="pattern-b",
        actual_action="receive_same",
    )["status"] == "insufficient"

    store.record(
        pattern_key="pattern-b",
        features=_features(),
        action_label="receive_same",
        followup_label="middle_pair",
        source="default",
        depth=7,
        agreement=0.75,
        confidence=0.65,
        margin=120.0,
    )
    matched = store.compare_shadow(
        pattern_key="pattern-b",
        actual_action="receive_same",
    )
    mismatched = store.compare_shadow(
        pattern_key="pattern-b",
        actual_action="pass",
    )
    recommendation = store.recommendation("pattern-b")
    store.record_priority_query(recommendation)
    snapshot = store.snapshot()

    assert matched["status"] == "match"
    assert matched["recommended_action"] == "receive_same"
    assert matched["recommended_followup"] == "middle_pair"
    assert mismatched["status"] == "mismatch"
    assert snapshot["shadow_lookups"] == 4
    assert snapshot["shadow_no_pattern"] == 1
    assert snapshot["shadow_insufficient"] == 1
    assert snapshot["shadow_recommendations"] == 2
    assert snapshot["shadow_matches"] == 1
    assert snapshot["shadow_mismatches"] == 1
    assert snapshot["shadow_match_rate"] == 0.5
    assert recommendation["status"] == "recommended"
    assert recommendation["recommended_action"] == "receive_same"
    assert snapshot["priority_queries"] == 1
    assert snapshot["priority_hits"] == 1
    assert snapshot["priority_action_counts"] == {"receive_same": 1}


def test_medium_patterns_pool_detailed_variants_and_survive_restore() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "generic-patterns.json"
        store = GenericResponsePatternStore(path=path)
        medium_key = _digest_payload(medium_response_pattern_payload(_features()))

        for index in range(10):
            features = {**_features(), "detailed_variant": index}
            store.record(
                pattern_key=f"detail-{index}",
                features=features,
                action_label="receive_same",
                followup_label="middle_pair",
                source="default",
                depth=7,
                agreement=0.80,
                confidence=0.70,
                margin=100.0,
            )

        snapshot = store.snapshot()
        recommendation = store.recommendation(
            medium_key,
            granularity="medium",
            min_observations=10,
            min_dominance=0.70,
        )
        assert snapshot["pattern_count"] == 10
        assert snapshot["medium_pattern_count"] == 1
        assert snapshot["medium_patterns_10_plus"] == 1
        assert recommendation["status"] == "recommended"
        assert recommendation["granularity"] == "medium"
        assert recommendation["recommended_action"] == "receive_same"

        assert store.checkpoint("medium-backfill") is True
        restored = GenericResponsePatternStore(path=path)
        restored_medium = restored.pattern(
            medium_key,
            granularity="medium",
        )
        assert restored.snapshot()["medium_pattern_count"] == 1
        assert restored_medium is not None
        assert restored_medium["observations"] == 10


def test_shadow_uses_medium_pattern_when_detail_has_no_support() -> None:
    store = GenericResponsePatternStore()
    medium_key = _digest_payload(medium_response_pattern_payload(_features()))
    for index in range(5):
        store.record(
            pattern_key=f"detail-{index}",
            features={**_features(), "detailed_variant": index},
            action_label="pass",
            followup_label="none",
            source="default",
            depth=7,
            agreement=0.80,
            confidence=0.70,
            margin=100.0,
        )

    result = store.compare_shadow(
        pattern_key="unseen-detail",
        medium_pattern_key=medium_key,
        actual_action="pass",
        medium_min_observations=5,
        medium_min_dominance=0.70,
    )
    snapshot = store.snapshot()
    assert result["status"] == "match"
    assert result["granularity"] == "medium"
    assert snapshot["shadow_granularity_counts"] == {"medium": 1}


def test_tactical_patterns_backfill_and_compare_without_affecting_priority() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "generic-patterns.json"
        store = GenericResponsePatternStore(path=path)
        tactical_key = _digest_payload(
            tactical_response_pattern_payload(_features())
        )
        for index in range(10):
            store.record(
                pattern_key=f"tactical-detail-{index}",
                features={**_features(), "detailed_variant": index},
                action_label="receive_same",
                followup_label="middle_pair",
                source="default",
                depth=7,
                agreement=0.80,
                confidence=0.70,
                margin=100.0,
            )

        matched = store.compare_tactical_shadow(
            pattern_key=tactical_key,
            actual_action="receive_same",
        )
        mismatched = store.compare_tactical_shadow(
            pattern_key=tactical_key,
            actual_action="pass",
        )
        repeated_mismatch = store.compare_tactical_shadow(
            pattern_key=tactical_key,
            actual_action="pass",
        )
        store.record_tactical_priority_query("offered")
        store.record_tactical_priority_effect(
            reordered=True,
            baseline_disagreed=True,
            selected=True,
            completed_depth=7,
        )
        store.record_tactical_priority_pair(
            comparison_complete=True,
            action_matched=False,
            with_priority_selected=True,
            without_priority_selected=False,
            with_depth=7,
            without_depth=5,
            with_elapsed_seconds=1.25,
            without_elapsed_seconds=1.75,
            with_nodes=1200,
            without_nodes=1800,
            value_delta=250.0,
            margin_delta=75.0,
        )
        snapshot = store.snapshot()
        assert matched["status"] == "match"
        assert mismatched["status"] == "mismatch"
        assert repeated_mismatch["status"] == "mismatch"
        assert snapshot["tactical_pattern_count"] == 1
        assert snapshot["tactical_patterns_8_plus"] == 1
        assert snapshot["tactical_patterns_10_plus"] == 1
        assert snapshot["tactical_shadow_lookups"] == 3
        assert snapshot["tactical_shadow_recommendations"] == 3
        assert snapshot["tactical_shadow_matches"] == 1
        assert snapshot["tactical_shadow_mismatches"] == 2
        assert snapshot["tactical_shadow_match_rate"] == 0.33333
        assert snapshot["tactical_mismatch_detail_count"] == 1
        assert snapshot["tactical_priority_lookups"] == 1
        assert snapshot["tactical_priority_offered"] == 1
        assert snapshot["tactical_priority_effects"] == 1
        assert snapshot["tactical_priority_reordered"] == 1
        assert snapshot["tactical_priority_baseline_disagreements"] == 1
        assert snapshot["tactical_priority_selected"] == 1
        assert snapshot["tactical_priority_selected_rate"] == 1.0
        assert snapshot["tactical_priority_average_depth"] == 7.0
        assert snapshot["tactical_pair_comparisons"] == 1
        assert snapshot["tactical_pair_completed"] == 1
        assert snapshot["tactical_pair_incomplete"] == 0
        assert snapshot["tactical_pair_action_matches"] == 0
        assert snapshot["tactical_pair_action_changes"] == 1
        assert snapshot["tactical_pair_action_match_rate"] == 0.0
        assert snapshot["tactical_pair_with_priority_selected"] == 1
        assert snapshot["tactical_pair_without_priority_selected"] == 0
        assert snapshot["tactical_pair_average_with_depth"] == 7.0
        assert snapshot["tactical_pair_average_without_depth"] == 5.0
        assert snapshot["tactical_pair_average_with_elapsed_seconds"] == 1.25
        assert snapshot["tactical_pair_average_without_elapsed_seconds"] == 1.75
        assert snapshot["tactical_pair_average_with_nodes"] == 1200.0
        assert snapshot["tactical_pair_average_without_nodes"] == 1800.0
        assert snapshot["tactical_pair_average_value_delta"] == 250.0
        assert snapshot["tactical_pair_average_margin_delta"] == 75.0
        assert snapshot["tactical_mismatch_details"] == [{
            "pattern_id": tactical_key[:10],
            "recommended_action": "receive_same",
            "actual_action": "pass",
            "count": 2,
            "observations": 10,
            "support": 10,
            "dominance": 1.0,
            "anonymous_context": {
                "attacker_relation": "enemy",
                "attack_piece": "none",
                "attack_stage": "first",
                "hand_stage": "middle",
                "next_receiver_stage": "middle",
                "same_piece": "none",
                "royal_receive": False,
                "followup_strength": "open",
                "reentry_width": "closed",
                "shi_context": "not_shi",
                "score_pressure": "normal",
            },
            "last_seen_at": snapshot["tactical_mismatch_details"][0][
                "last_seen_at"
            ],
            "detail_id": snapshot["tactical_mismatch_details"][0][
                "detail_id"
            ],
        }]
        assert snapshot["priority_queries"] == 0

        assert store.checkpoint("tactical-backfill") is True
        restored = GenericResponsePatternStore(path=path)
        restored_snapshot = restored.snapshot()
        restored_pattern = restored.pattern(
            tactical_key,
            granularity="tactical",
        )
        assert restored_snapshot["tactical_pattern_count"] == 1
        assert restored_snapshot["tactical_mismatch_detail_count"] == 1
        assert restored_snapshot["tactical_priority_offered"] == 1
        assert restored_snapshot["tactical_priority_selected"] == 1
        assert restored_snapshot["tactical_pair_completed"] == 1
        assert restored_snapshot["tactical_pair_action_changes"] == 1
        assert restored_snapshot["tactical_pair_average_with_depth"] == 7.0
        assert restored_snapshot["tactical_mismatch_details"][0]["count"] == 2
        assert restored_snapshot["tactical_mismatch_details"][0][
            "recommended_action"
        ] == "receive_same"
        assert restored_snapshot["tactical_mismatch_details"][0][
            "actual_action"
        ] == "pass"
        assert restored_snapshot["tactical_mismatch_details"][0][
            "anonymous_context"
        ]["attacker_relation"] == "enemy"
        assert restored_pattern is not None
        assert restored_pattern["observations"] == 10


def test_priority_effect_metrics_compare_and_restore_without_board_data() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "generic-patterns.json"
        store = GenericResponsePatternStore(path=path)
        store.record_priority_effect(
            reordered=True,
            beam_preserved=False,
            comparison_complete=True,
            recommended_selected=True,
            action_changed=False,
            with_depth=7,
            without_depth=7,
            with_elapsed_seconds=1.25,
            without_elapsed_seconds=1.25,
            value_delta=0.0,
        )
        store.record_narrowing_shadow(
            status="insufficient_depth",
        )
        store.record_narrowing_shadow(
            status="no_reduction",
        )
        store.record_narrowing_shadow(
            status="compared",
            matched=True,
            priority_selected=True,
            full_candidates=4,
            kept_candidates=2,
            depth=7,
            actual_elapsed_seconds=4.0,
            estimated_elapsed_seconds=3.0,
            value_loss=0.0,
        )
        store.record_active_narrowing(status="low_confidence")
        store.record_active_narrowing(
            status="no_deepening",
            full_candidates=3,
            kept_candidates=2,
            completed_depth=3,
            no_deepening_reason="node_limit",
        )
        store.record_active_narrowing(
            status="deepened",
            full_candidates=4,
            kept_candidates=2,
            completed_depth=7,
            elapsed_seconds=2.5,
            priority_selected=True,
            continuation_extension_seconds=0.8,
            added_nodes=2500,
        )
        store.record_priority_effect(
            reordered=True,
            beam_preserved=True,
            comparison_complete=False,
            recommended_selected=False,
            action_changed=False,
            with_depth=5,
            without_depth=0,
            with_elapsed_seconds=0.8,
            without_elapsed_seconds=0.0,
            value_delta=0.0,
        )

        snapshot = store.snapshot()
        assert snapshot["priority_effect_comparisons"] == 2
        assert snapshot["priority_effect_exact"] == 1
        assert snapshot["priority_effect_incomplete"] == 1
        assert snapshot["priority_effect_reorders"] == 2
        assert snapshot["priority_effect_beam_preserved"] == 1
        assert snapshot["priority_effect_selected"] == 1
        assert snapshot["priority_effect_changed"] == 0
        assert snapshot["priority_effect_unchanged"] == 1
        assert snapshot["priority_effect_average_with_depth"] == 7.0
        assert snapshot["priority_effect_average_without_depth"] == 7.0
        assert snapshot["priority_effect_saved_seconds"] == 0.0
        assert snapshot["narrowing_shadow_considered"] == 3
        assert snapshot["narrowing_shadow_insufficient_depth"] == 1
        assert snapshot["narrowing_shadow_no_reduction"] == 1
        assert snapshot["narrowing_shadow_comparisons"] == 1
        assert snapshot["narrowing_shadow_matches"] == 1
        assert snapshot["narrowing_shadow_match_rate"] == 1.0
        assert snapshot["narrowing_shadow_average_full_candidates"] == 4.0
        assert snapshot["narrowing_shadow_average_kept_candidates"] == 2.0
        assert snapshot["narrowing_shadow_average_removed_candidates"] == 2.0
        assert snapshot["narrowing_shadow_estimated_saved_seconds"] == 1.0
        assert snapshot["active_narrowing_considered"] == 3
        assert snapshot["active_narrowing_applied"] == 2
        assert snapshot["active_narrowing_safety_rejected"] == 1
        assert snapshot["active_narrowing_rejected_low_confidence"] == 1
        assert snapshot["active_narrowing_deepened"] == 1
        assert snapshot["active_narrowing_no_deepening"] == 1
        assert snapshot["active_narrowing_no_deepening_node_limit"] == 1
        assert snapshot["active_narrowing_no_deepening_time_limit"] == 0
        assert snapshot["active_narrowing_no_deepening_other"] == 0
        assert snapshot["active_narrowing_average_full_candidates"] == 3.5
        assert snapshot["active_narrowing_average_kept_candidates"] == 2.0
        assert snapshot["active_narrowing_average_depth"] == 5.0
        assert snapshot["active_narrowing_average_elapsed_seconds"] == 1.25
        assert snapshot["active_narrowing_continuation_extensions"] == 1
        assert snapshot["active_narrowing_continuation_extension_seconds"] == 0.8
        assert snapshot["active_narrowing_node_extensions"] == 1
        assert snapshot["active_narrowing_added_nodes"] == 2500

        assert store.checkpoint("priority-effect") is True
        restored = GenericResponsePatternStore(path=path).snapshot()
        assert restored["priority_effect_comparisons"] == 2
        assert restored["priority_effect_exact"] == 1
        assert restored["priority_effect_beam_preserved"] == 1
        assert restored["narrowing_shadow_considered"] == 3
        assert restored["narrowing_shadow_matches"] == 1
        assert restored["narrowing_shadow_estimated_saved_seconds"] == 1.0
        assert restored["active_narrowing_considered"] == 3
        assert restored["active_narrowing_applied"] == 2
        assert restored["active_narrowing_safety_rejected"] == 1
        assert restored["active_narrowing_rejected_low_confidence"] == 1
        assert restored["active_narrowing_deepened"] == 1
        assert restored["active_narrowing_no_deepening"] == 1
        assert restored["active_narrowing_no_deepening_node_limit"] == 1
        assert restored["active_narrowing_continuation_extensions"] == 1
        assert restored["active_narrowing_continuation_extension_seconds"] == 0.8
        assert restored["active_narrowing_node_extensions"] == 1
        assert restored["active_narrowing_added_nodes"] == 2500


def test_human_targeted_priority_metrics_survive_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "generic-patterns.json"
        store = GenericResponsePatternStore(path=path)
        store.record_human_targeted_priority_query("offered")
        store.record_human_targeted_priority_effect(
            reordered=True,
            baseline_disagreed=True,
            selected=True,
            completed_depth=7,
        )
        store.record_human_targeted_priority_pair(
            comparison_complete=True,
            action_matched=False,
            with_priority_selected=True,
            without_priority_selected=False,
            with_depth=7,
            without_depth=5,
            with_elapsed_seconds=1.2,
            without_elapsed_seconds=1.5,
            with_nodes=1200,
            without_nodes=1500,
            value_delta=200.0,
            margin_delta=50.0,
        )

        snapshot = store.snapshot()
        assert snapshot["human_targeted_priority_lookups"] == 1
        assert snapshot["human_targeted_priority_offered"] == 1
        assert snapshot["human_targeted_priority_effects"] == 1
        assert snapshot["human_targeted_priority_reordered"] == 1
        assert snapshot[
            "human_targeted_priority_baseline_disagreements"
        ] == 1
        assert snapshot["human_targeted_priority_selected"] == 1
        assert snapshot["human_targeted_priority_selected_rate"] == 1.0
        assert snapshot["human_targeted_priority_average_depth"] == 7.0
        assert snapshot["human_targeted_pair_comparisons"] == 1
        assert snapshot["human_targeted_pair_completed"] == 1
        assert snapshot["human_targeted_pair_action_changes"] == 1
        assert snapshot["human_targeted_pair_with_priority_selected"] == 1
        assert snapshot["human_targeted_pair_average_with_depth"] == 7.0
        assert snapshot["human_targeted_pair_average_without_depth"] == 5.0
        assert snapshot["human_targeted_pair_average_value_delta"] == 200.0

        assert store.checkpoint("human-targeted-priority") is True
        restored = GenericResponsePatternStore(path=path).snapshot()
        assert restored["human_targeted_priority_lookups"] == 1
        assert restored["human_targeted_priority_offered"] == 1
        assert restored["human_targeted_priority_selected"] == 1
        assert restored["human_targeted_priority_average_depth"] == 7.0
        assert restored["human_targeted_pair_comparisons"] == 1
        assert restored["human_targeted_pair_completed"] == 1
        assert restored["human_targeted_pair_action_changes"] == 1
        assert restored["human_targeted_pair_average_margin_delta"] == 50.0


if __name__ == "__main__":
    test_store_aggregates_and_restores_anonymous_patterns()
    test_store_path_uses_render_persistent_directory()
    test_shadow_comparison_requires_support_and_never_selects_an_action()
    test_medium_patterns_pool_detailed_variants_and_survive_restore()
    test_shadow_uses_medium_pattern_when_detail_has_no_support()
    test_tactical_patterns_backfill_and_compare_without_affecting_priority()
    test_priority_effect_metrics_compare_and_restore_without_board_data()
    test_human_targeted_priority_metrics_survive_checkpoint()
    print("GENERIC_RESPONSE_STORE_TEST_OK")

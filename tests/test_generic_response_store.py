from __future__ import annotations

import json
import tempfile
from pathlib import Path

from goita_ai2.current_ai.generic_response_store import (
    GenericResponsePatternStore,
    resolve_generic_response_store_path,
)


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


if __name__ == "__main__":
    test_store_aggregates_and_restores_anonymous_patterns()
    test_store_path_uses_render_persistent_directory()
    test_shadow_comparison_requires_support_and_never_selects_an_action()
    print("GENERIC_RESPONSE_STORE_TEST_OK")

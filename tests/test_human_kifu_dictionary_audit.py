"""Tests the private raw-kifu audit and its anonymous output boundary."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from goita_ai2.current_ai.human_response_dictionary import (
    human_response_pattern_payload,
)
from goita_ai2.human_kifu_dictionary_audit import (
    _response_features,
    audit_human_kifu_archive,
)
from goita_ai2.kifu_validation import (
    iter_kifu_decisions,
    replay_validation_case,
)
from goita_ai2.rule_based import RuleBasedAgent


def _archive() -> dict:
    normal_round = {
        "round_index": 1,
        "hand": {
            "p0": "ししし香馬金金飛",
            "p1": "ししし銀銀銀飛玉",
            "p2": "し香馬馬銀金金角",
            "p3": "ししし香香馬角王",
        },
        "uchidashi": 0,
        "score": [0, 0],
        "game": [
            ["0", "し", "金"],
            ["0", "し", "金"],
            ["1", "王", "銀"],
            ["2", "銀", "馬"],
        ],
    }
    five_shi_round = {
        "round_index": 2,
        "hand": {
            "p0": "ししししし馬馬玉",
            "p1": "ししし香香銀金飛",
            "p2": "し香馬銀銀金角飛",
            "p3": "し香馬銀金金角王",
        },
        "uchidashi": 0,
        "score": [0, 0],
        "game": [["0", "し", "馬"]],
    }
    return {
        "schema_version": 1,
        "match_count": 1,
        "round_count": 2,
        "move_count": 5,
        "matches": [{
            "id": "private-match-id",
            "played_at": "2026-01-01 12:34:56",
            "players": {
                "p0": "Private Alpha",
                "p1": "Private Beta",
                "p2": "Private Gamma",
                "p3": "Private Delta",
            },
            "rounds": [normal_round, five_shi_round],
        }],
    }


def test_audit_reconstructs_passes_and_excludes_five_shi() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "raw.json"
        path.write_text(
            json.dumps(_archive(), ensure_ascii=False),
            encoding="utf-8",
        )
        report = audit_human_kifu_archive(
            path,
            min_observations=1,
            min_matches=1,
            min_dominance=0.0,
        )

    source = report["source_summary"]
    replay = report["replay_summary"]
    response = report["response_summary"]
    assert report["live_ai_affected"] is False
    assert source["five_shi_rounds"] == 1
    assert source["included_rounds"] == 1
    assert replay["reconstructed_decisions"] > replay["included_decisions"]
    assert replay["inferred_passes"] > 0
    assert response["observations"] == (
        replay["inferred_passes"] + replay["receives"]
    )
    assert response["pattern_count"] > 0
    assert response["reusable_candidate_count"] > 0


def test_audit_output_does_not_retain_private_source_data() -> None:
    archive = _archive()
    with TemporaryDirectory() as directory:
        path = Path(directory) / "raw.json"
        path.write_text(
            json.dumps(archive, ensure_ascii=False),
            encoding="utf-8",
        )
        report = audit_human_kifu_archive(
            path,
            min_observations=1,
            min_matches=1,
            min_dominance=0.0,
        )
    serialized = json.dumps(report, ensure_ascii=False)

    assert "Private Alpha" not in serialized
    assert "Private Beta" not in serialized
    assert "private-match-id" not in serialized
    assert "2026-01-01 12:34:56" not in serialized
    assert "ししし香馬金金飛" not in serialized
    assert '"history"' not in serialized
    assert report["privacy"] == {
        "raw_archive_is_private": True,
        "names_retained": False,
        "timestamps_retained": False,
        "match_ids_retained": False,
        "hands_retained": False,
        "move_histories_retained": False,
    }


def test_raw_and_live_positions_build_the_same_human_shadow_key() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "raw.json"
        path.write_text(
            json.dumps(_archive(), ensure_ascii=False),
            encoding="utf-8",
        )
        case = next(
            item
            for item in iter_kifu_decisions(path)
            if item["position"]["phase"] == "receive"
        )

    raw_state = replay_validation_case(case)
    raw_features = human_response_pattern_payload(
        _response_features(case, raw_state)
    )
    agent = RuleBasedAgent()
    live_state = replay_validation_case(case, agent)
    actions = live_state.legal_actions(str(case["player"]))
    live_features = human_response_pattern_payload(
        agent._tactical_response_pattern_payload(
            live_state,
            str(case["player"]),
            actions,
            tuple(case["actual_action"]),
        )
    )
    assert raw_features == live_features


if __name__ == "__main__":
    test_audit_reconstructs_passes_and_excludes_five_shi()
    test_audit_output_does_not_retain_private_source_data()
    test_raw_and_live_positions_build_the_same_human_shadow_key()
    print("HUMAN_KIFU_DICTIONARY_AUDIT_TEST_OK")

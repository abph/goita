"""Tests private-kifu extraction and information-set comparison.

Recorded combined moves are reconstructed into passes, receives, and attacks;
saved review cases can later gain human-approved regression expectations.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from goita_ai2.kifu_validation import (
    compare_validation_cases,
    extract_kifu_validation_cases,
    load_validation_cases,
    replay_validation_case,
    save_review_cases,
    validate_problem_cases,
    write_json,
)


ARCHIVE = {
    "schema_version": 1,
    "matches": [
        {
            "id": "test-0001",
            "rounds": [
                {
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
            ],
        }
    ],
}


def test_extracts_replayable_decisions_including_omitted_passes() -> None:
    with TemporaryDirectory() as directory:
        archive_path = Path(directory) / "kifu.json"
        archive_path.write_text(
            json.dumps(ARCHIVE, ensure_ascii=False),
            encoding="utf-8",
        )
        case_set = extract_kifu_validation_cases(archive_path, limit=4)

    assert case_set["case_count"] == 4
    assert case_set["scanned_decisions"] > case_set["case_count"]
    assert len({case["category"] for case in case_set["cases"]}) >= 2
    for case in case_set["cases"]:
        state = replay_validation_case(case)
        assert tuple(case["actual_action"]) in state.legal_actions(case["player"])


def test_compares_searches_and_replays_saved_human_expectation() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        archive_path = root / "kifu.json"
        archive_path.write_text(
            json.dumps(ARCHIVE, ensure_ascii=False),
            encoding="utf-8",
        )
        case_set = extract_kifu_validation_cases(archive_path, limit=1)
        comparison = compare_validation_cases(
            case_set,
            search_seconds=0.05,
            sample_count=2,
            max_depth=1,
            max_nodes=1_000,
        )

        assert comparison["summary"]["cases"] == 1
        assert comparison["summary"]["information_set_used"] == 1
        item = comparison["comparisons"][0]
        assert item["baseline_consistent"] is True

        review_input = copy.deepcopy(comparison)
        review_input["comparisons"][0]["information_set_changed_legacy"] = True
        problem_path = root / "problems.json"
        saved = save_review_cases(review_input, problem_path)
        assert saved["case_count"] == 1

        loaded = load_validation_cases(problem_path)
        expected = item["information_set_action"]
        loaded["cases"][0]["expectation"]["allowed_actions"] = [expected]
        write_json(problem_path, loaded)
        result = validate_problem_cases(
            load_validation_cases(problem_path),
            search_seconds=0.05,
            sample_count=2,
            max_depth=1,
            max_nodes=1_000,
        )

    assert result["checked_expectations"] == 1
    assert result["passed"] is True


if __name__ == "__main__":
    test_extracts_replayable_decisions_including_omitted_passes()
    test_compares_searches_and_replays_saved_human_expectation()
    print("KIFU_VALIDATION_TEST_OK")

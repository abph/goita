"""Tests the privacy boundary for the neural policy training dataset."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from goita_ai2.neural_training_data import build_training_dataset
from goita_ai2.neural_training_data import training_record
from goita_ai2.kifu_validation import iter_kifu_decisions, replay_validation_case
from goita_ai2.neural_policy import live_state_payload


def _archive() -> dict:
    return {
        "schema_version": 1,
        "matches": [{
            "id": "secret-match-id",
            "players": {
                "p0": "1222",
                "p1": "Private Beta",
                "p2": "Private Gamma",
                "p3": "Private Delta",
            },
            "rounds": [{
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
            }, {
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
            }],
        }],
    }


def test_builds_trainable_records_without_private_opponent_data() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.json"
        output = root / "decisions.jsonl"
        summary_path = root / "summary.json"
        source.write_text(json.dumps(_archive(), ensure_ascii=False), encoding="utf-8")
        summary = build_training_dataset(source, output, summary_path)
        records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert records
    assert summary["records"] == len(records)
    assert summary["target_player_records"] > 0
    assert {record["round_index"] for record in records} == {1}
    assert len({record["match_group"] for record in records}) == 1
    assert len({record["split"] for record in records}) == 1
    assert all(record["profile"] == "強化中AI2" for record in records)
    assert all(
        0 <= record["selected_action_index"] < len(record["legal_actions"])
        for record in records
    )

    serialized = json.dumps(records, ensure_ascii=False)
    assert "Private Beta" not in serialized
    assert "Private Gamma" not in serialized
    assert "secret-match-id" not in serialized
    assert "ししし銀銀銀飛玉" not in serialized


def test_hides_other_players_blocks_but_keeps_actors_own_knowledge() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.json"
        output = root / "decisions.jsonl"
        summary_path = root / "summary.json"
        source.write_text(json.dumps(_archive(), ensure_ascii=False), encoding="utf-8")
        build_training_dataset(source, output, summary_path)
        records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    own_history = next(
        record["state"]["history"]
        for record in records
        if record["is_target_player"] and record["state"]["history"]
    )
    assert own_history[0]["actor"] == "self"
    assert own_history[0]["type"] == "attack_after_block"
    assert own_history[0]["block"] == "1"
    assert own_history[0]["block_known"] is True

    other_view = next(
        record["state"]["history"]
        for record in records
        if not record["is_target_player"] and record["state"]["history"]
    )
    assert other_view[0]["actor"] != "self"
    assert other_view[0]["type"] == "attack_after_block"
    assert other_view[0]["block"] is None
    assert other_view[0]["block_known"] is False


def test_training_and_live_feature_payloads_match() -> None:
    with TemporaryDirectory() as directory:
        source = Path(directory) / "source.json"
        source.write_text(json.dumps(_archive(), ensure_ascii=False), encoding="utf-8")
        case = next(
            item
            for item in iter_kifu_decisions(source)
            if item["history"] and len(item["legal_actions"]) > 1
        )
    record = training_record(case, player_name="1222", target_player="1222")
    state = replay_validation_case(case)
    actor = str(case["player"])
    live = live_state_payload(
        state,
        actor,
        initial_hand=case["initial_hands"][actor],
        history=case["history"],
    )

    assert live == record["state"]


if __name__ == "__main__":
    test_builds_trainable_records_without_private_opponent_data()
    test_hides_other_players_blocks_but_keeps_actors_own_knowledge()
    test_training_and_live_feature_payloads_match()
    print("NEURAL_TRAINING_DATA_TEST_OK")

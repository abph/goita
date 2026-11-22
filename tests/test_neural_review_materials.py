"""Confirmed lessons stay synchronized without hidden information or model writes."""
import copy
import json

import pytest

from goita_ai2 import decision_check as dc
from goita_ai2 import neural_review_materials as materials
from goita_ai2.neural_policy import record_features
from goita_ai2.train_neural_policy import _expand_acceptable_actions, _safe_corrections, correction_fingerprint


RECEIVE = ["receive", "4", None]
KYOSHA = ["attack", None, "2"]
KAKU = ["attack", None, "6"]
PASS = ["pass", None, None]


@pytest.fixture
def setup(tmp_path):
    position = {
        "initial_hands": {"A": list("15319711"), "B": list("61231151"),
                          "C": list("48325174"), "D": list("34262145")},
        "dealer": "A", "player": "D", "history": [
            {"player": "A", "action": ["attack_after_block", "1", "1"]},
            {"player": "B", "action": ["receive", "1", None]},
            {"player": "B", "action": ["attack", None, "1"]},
            {"player": "C", "action": ["receive", "1", None]},
            {"player": "C", "action": ["attack", None, "4"]},
        ],
    }
    case = {"id": "D-material-test", "position": position, "fingerprint": dc.digest(position),
            "excluded": False, "answers": {"safe": {
                "confirmed": True, "reason": "相方への引き継ぎを作る。",
                "best": [[RECEIVE, KYOSHA]], "acceptable": [[RECEIVE, KAKU]], "avoid": [[PASS]],
            }}}
    work = tmp_path / "work"
    suite = {"format": dc.FORMAT, "schema_version": dc.VERSION, "cases": [case]}
    dc.atomic_json(dc.suite_path(work), suite)
    return work, tmp_path / "lessons.jsonl", tmp_path / "model.json", suite


def save(work, suite):
    for case in suite["cases"]:
        case["fingerprint"] = dc.digest(case["position"])
    dc.atomic_json(dc.suite_path(work), suite)


def test_registers_receive_and_attack_accepts_alternatives_and_is_idempotent(setup):
    work, path, model, _ = setup
    result = materials.synchronize(work, path, model)
    assert result["last_sync_changes"] == {"added": 2, "updated": 0, "removed": 0}
    rows = materials.read_records(path)
    root, attack = rows
    assert root["state"]["phase"] == "receive" and attack["state"]["phase"] == "attack"
    assert root["legal_actions"][root["selected_action_index"]]["block"] == "4"
    assert len(attack["acceptable_action_indices"]) == 2
    assert len(attack["best_action_indices"]) == 1
    assert len(_expand_acceptable_actions(rows)) == 3
    for row in rows:
        assert record_features(row)[0]
    before = path.read_bytes()
    again = materials.synchronize(work, path, model)
    assert path.read_bytes() == before
    assert again["last_sync_changes"] == {"added": 0, "updated": 0, "removed": 0}
    assert not again["needs_sync"]


def test_status_and_check_do_not_write_and_detect_missing_material(setup):
    work, path, model, _ = setup
    args = ["--workdir", str(work), "--corrections", str(path), "--model", str(model)]
    assert materials.main(["status", *args]) == 0
    assert materials.main(["check", *args]) == 1
    assert not path.exists() and not (work / "neural_materials_status.json").exists()
    materials.synchronize(work, path, model)
    assert materials.main(["check", *args]) == 0


def test_model_is_unchanged_and_reflection_requires_matching_training_hash(setup):
    work, path, model, _ = setup
    model.write_text('{"training":{}}', encoding="utf-8")
    before = model.read_bytes()
    assert materials.synchronize(work, path, model)["model_materials_status"] == "retraining_pending"
    assert model.read_bytes() == before
    rows = materials.read_records(path)
    dc.atomic_json(model, {"training": {"safe_corrections_sha256": correction_fingerprint(_safe_corrections(rows))}})
    assert materials.material_plan(work, path, model)[1]["model_materials_status"] == "matched"


def test_changed_answer_and_confirmation_withdrawal_replace_only_managed_lessons(setup):
    work, path, model, suite = setup
    materials.synchronize(work, path, model)
    manual = copy.deepcopy(materials.read_records(path)[0])
    manual["decision_id"] = "original-manual"
    manual["review"] = {"reason": "元からある教材"}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(manual, ensure_ascii=False) + "\n")
    suite["cases"][0]["answers"]["safe"]["reason"] = "説明を補足。"
    save(work, suite)
    assert materials.material_plan(work, path, model)[1]["needs_sync"]
    assert materials.synchronize(work, path, model)["last_sync_changes"]["updated"] == 2
    assert list((work / "neural_materials_backups").glob("*.jsonl"))
    suite["cases"][0]["answers"]["safe"]["confirmed"] = False
    save(work, suite)
    assert materials.synchronize(work, path, model)["last_sync_changes"]["removed"] == 2
    assert materials.read_records(path) == [manual]


def test_risk_and_unconfirmed_are_separate_and_five_shi_is_excluded(setup):
    work, path, model, suite = setup
    original = suite["cases"][0]
    original["answers"]["risk"] = copy.deepcopy(original["answers"]["safe"])
    unconfirmed = copy.deepcopy(original)
    unconfirmed.update(id="D-unconfirmed", answers={})
    excluded = copy.deepcopy(original)
    excluded["id"] = "D-five-shi"
    excluded["position"]["initial_hands"]["A"] = list("11111975")
    suite["cases"].extend([unconfirmed, excluded])
    save(work, suite)
    result = materials.synchronize(work, path, model)
    assert result["case_counts"] == {"registered": 2, "unconfirmed": 1, "excluded": 1}
    assert result["safe_records"] == result["risk_records"] == 2
    assert len(_safe_corrections(materials.read_records(path))) == 2


def test_bad_followup_does_not_label_its_good_receive_as_bad(setup):
    work, path, model, suite = setup
    suite["cases"][0]["answers"]["safe"]["avoid"].append([RECEIVE, ["attack", None, "5"]])
    save(work, suite)
    materials.synchronize(work, path, model)
    root, attack = materials.read_records(path)
    assert root["selected_action_index"] not in root["avoid_action_indices"]
    assert [root["legal_actions"][i]["type"] for i in root["avoid_action_indices"]] == ["pass"]
    assert [attack["legal_actions"][i]["attack"] for i in attack["avoid_action_indices"]] == ["5"]


def test_illegal_answer_or_manual_conflict_leaves_existing_material_untouched(setup):
    work, path, model, suite = setup
    materials.synchronize(work, path, model)
    before = path.read_bytes()
    suite["cases"][0]["answers"]["safe"]["best"] = [[RECEIVE, ["attack", None, "9"]]]
    save(work, suite)
    with pytest.raises(ValueError, match="合法"):
        materials.synchronize(work, path, model)
    assert path.read_bytes() == before
    suite["cases"][0]["answers"]["safe"]["best"] = [[RECEIVE, KYOSHA]]
    save(work, suite)
    manual = copy.deepcopy(materials.read_records(path)[0])
    manual.update(decision_id="manual-conflict", selected_action_index=0, acceptable_action_indices=[0], review={})
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(manual) + "\n")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="異なる正解"):
        materials.synchronize(work, path, model)
    assert path.read_bytes() == before


def test_nonpublic_hand_swap_cannot_change_exported_model_input(setup):
    work, path, model, suite = setup
    first, _ = materials.material_plan(work, path, model)
    hands = suite["cases"][0]["position"]["initial_hands"]
    hands["A"].remove("9"); hands["A"].append("2")
    hands["B"].remove("2"); hands["B"].append("9")
    save(work, suite)
    second, _ = materials.material_plan(work, path, model)
    for before, after in zip(first, second):
        assert before["state"] == after["state"]
        assert before["legal_actions"] == after["legal_actions"]
        assert "initial_hands" not in before["state"]
        assert all(item["block"] is None for item in before["state"]["history"] if item["type"] == "attack_after_block")


def test_missing_suite_cannot_erase_lessons_and_fingerprint_ignores_row_order(setup):
    work, path, model, _ = setup
    materials.synchronize(work, path, model)
    rows = materials.read_records(path)
    assert correction_fingerprint(rows) == correction_fingerprint(list(reversed(rows)))
    dc.suite_path(work).unlink()
    before = path.read_bytes()
    with pytest.raises(ValueError, match="問題集がありません"):
        materials.synchronize(work, path, model)
    assert path.read_bytes() == before

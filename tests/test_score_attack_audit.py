from backend.score_attack_audit import (
    ScoreAttackAuditStore,
    candidate_id,
)


def _record(candidate="candidate-1", default_status="eligible"):
    return {
        "candidate_id": candidate,
        "default_status": default_status,
        "match_id": "match-1",
        "round_index": 1,
        "score_ac": 20,
        "score_bd": 30,
        "winner": "A",
        "gained_score": 40,
        "reasons": [],
        "metadata": {"move_count": 8},
    }


def test_audit_store_preserves_manual_status_and_note(tmp_path):
    store = ScoreAttackAuditStore(tmp_path / "audit.sqlite3")
    first = store.sync(_record())
    assert first["status"] == "eligible"
    changed = store.set_status("candidate-1", "excluded", note="不自然なパス")
    assert changed["status"] == "excluded"
    assert changed["decision_note"] == "不自然なパス"
    assert store.history("candidate-1")[0]["status"] == "excluded"
    assert store.history("candidate-1")[0]["note"] == "不自然なパス"

    rescanned = store.sync({**_record(), "reasons": ["自動検証結果を更新"]})
    assert rescanned["status"] == "excluded"
    assert rescanned["reasons"] == ["自動検証結果を更新"]
    assert store.list("excluded")[0]["candidate_id"] == "candidate-1"


def test_invalid_or_out_of_range_round_cannot_be_restored_as_eligible(tmp_path):
    store = ScoreAttackAuditStore(tmp_path / "audit.sqlite3")
    store.sync(_record("invalid", "invalid"))
    store.sync(_record("out-of-range", "out_of_range"))
    for identifier in ("invalid", "out-of-range"):
        try:
            store.set_status(identifier, "eligible")
        except ValueError:
            pass
        else:
            raise AssertionError("対象外の棋譜を採用に戻せてはいけません")


def test_review_round_can_be_approved_for_score_attack(tmp_path):
    store = ScoreAttackAuditStore(tmp_path / "audit.sqlite3")
    store.sync(_record("review", "review"))
    assert store.set_status("review", "eligible")["status"] == "eligible"


def test_candidate_id_changes_when_round_source_changes():
    match = {"id": "match-1"}
    first = {"round_index": 1, "score": [20, 30]}
    second = {"round_index": 1, "score": [20, 40]}
    assert candidate_id(match, first, 0) != candidate_id(match, second, 0)
    assert candidate_id(match, first, 0) == candidate_id(match, first, 0)

"""Replay boundaries, human answer integrity and truthful before/after grading."""
import copy
import json
from pathlib import Path

import pytest

from goita_ai2 import decision_check as dc
from goita_ai2.decision_check_report import write_review


HANDS = {"A": list("15319711"), "B": list("61231151"),
         "C": list("48325174"), "D": list("34262145")}
RECEIVE_KYOSHA = [["receive", "4", None], ["attack", None, "2"]]
RECEIVE_KAKU = [["receive", "4", None], ["attack", None, "6"]]
PASS = [["pass", None, None]]
OPTIONS = {"style": "safe", "seed": 1222, "rank_table": "production",
           "seconds": None, "no_search": False, "cases": None}


def report():
    return {"format": "sorou-goita-ai-review", "schema_version": 2,
            "initial_hands": copy.deepcopy(HANDS), "dealer": "A",
            "target": {"seat": "D", "decision_log_index": 6, "action": PASS[0],
                       "label": "D手番4"},
            "log": ["Game start. dealer=A", "A: block 1 -> attack 1 (hidden)",
                    "B: receive 1", "B: attack 1", "C: receive 1", "C: attack 4",
                    "D: pass [AI:old/reason]", "A: pass", "B: pass"],
            "human_review": {"preferred_action": "銀を受けて香か角", "reason": "味方にし攻めの継続が難しいと伝える"},
            "settings_at_capture": {"D": {"TIME_SEARCH_ENABLED": True,
                                           "TIME_SEARCH_MAX_SECONDS": 1,
                                           "GENERIC_RESPONSE_NARROWING_ENABLED": True}}}


def write_report(path, value=None):
    path.write_text(json.dumps(value or report(), ensure_ascii=False), encoding="utf-8")
    return path


def prepare(tmp_path, *, second=False):
    path = write_report(tmp_path / "report.json")
    paths = [path]
    if second:
        other = report()
        other["target"] = {"seat": "C", "decision_log_index": 5,
                           "action": ["attack", None, "4"], "label": "Cの攻め"}
        paths.append(write_report(tmp_path / "other.json", other))
    work = tmp_path / "work"
    dc.import_reports(work, paths)
    return work


def answer_doc(cases):
    return {"format": dc.ANSWER_FORMAT, "schema_version": 1, "cases": [
        {"id": c["id"], "fingerprint": c["fingerprint"], "style": "safe", "answer": {
            "confirmed": True, "reason": "公開情報から相方への引き継ぎを考える。",
            "best": [RECEIVE_KYOSHA], "acceptable": [RECEIVE_KAKU], "avoid": [PASS]}}
        for c in cases]}


def fake_evaluator(route):
    def evaluate(case, options):
        return {"status": "ok", "route": copy.deepcopy(route), "seconds": 0,
                "steps": [{"reason": "actual", "detail": "selected", "action": route[0]}],
                "conditions": {"style": options["style"], "seed": options["seed"],
                               "no_search": options["no_search"]}}
    return evaluate


def test_import_keeps_only_prefix_for_replay_and_does_not_certify_prose(tmp_path):
    work = prepare(tmp_path)
    c = dc.load_suite(work)["cases"][0]
    assert len(c["position"]["history"]) == 5
    state = dc.replay(c["position"])
    assert (state.turn, state.phase, state.current_attack) == ("D", "receive", "4")
    assert c["answers"] == {}
    assert c["original"]["log"].endswith("old/reason]")
    assert RECEIVE_KYOSHA in c["legal_routes"] and RECEIVE_KAKU in c["legal_routes"]
    assert [RECEIVE_KYOSHA[0]] not in c["legal_routes"]
    assert PASS in c["legal_routes"]


@pytest.mark.parametrize("mutation", [
    lambda r: r["target"].update(seat="A"),
    lambda r: r["target"].update(action=["receive", "4", None]),
    lambda r: r["log"].__setitem__(3, "B: attack 9"),
    lambda r: r["log"].__setitem__(3, "B: unsupported action"),
    lambda r: r["initial_hands"]["A"].pop(),
])
def test_corrupt_replay_is_rejected(mutation):
    r = report()
    mutation(r)
    with pytest.raises(ValueError):
        dc.extract_report(r)


def test_unknown_report_schema_is_not_guessed():
    r = report()
    r["schema_version"] = 99
    with pytest.raises(ValueError, match="バージョン"):
        dc.extract_report(r)


def test_five_shi_excluded_and_other_import_errors_visible(tmp_path):
    r = report()
    r["initial_hands"]["A"].remove("3")
    r["initial_hands"]["B"].remove("1")
    r["initial_hands"]["A"].append("1")
    r["initial_hands"]["B"].append("3")
    path = write_report(tmp_path / "five.json", r)
    invalid = tmp_path / "broken.json"
    invalid.write_text("{", encoding="utf-8")
    work = tmp_path / "work"
    counts = dc.import_reports(work, [path, invalid])
    assert counts["added"] == 1 and counts["excluded"] == 1 and len(counts["errors"]) == 1
    c = dc.load_suite(work)["cases"][0]
    assert c["excluded"]
    called = []
    run = dc.run_suite(work, OPTIONS, evaluator=lambda *a: called.append(a), progress=lambda _: None)
    assert called == [] and run["summary"] == {"excluded": 1}


def test_reimport_preserves_certified_answers_and_original_log(tmp_path):
    work = prepare(tmp_path)
    dc.import_answers(work, answer_doc(dc.load_suite(work)["cases"]))
    before = dc.load_suite(work)
    r = report()
    r["human_review"]["reason"] = "別の提案"
    path = write_report(tmp_path / "new.json", r)
    counts = dc.import_reports(work, [path])
    after = dc.load_suite(work)
    assert counts["existing"] == 1
    assert len(after["cases"]) == 1
    assert after["cases"][0]["answers"] == before["cases"][0]["answers"]
    assert after["cases"][0]["original"] == before["cases"][0]["original"]
    assert len(after["cases"][0]["sources"]) == 2


@pytest.mark.parametrize("mutation", [
    lambda e: e.update(fingerprint="stale"),
    lambda e: e.update(style="unknown"),
    lambda e: e["answer"].update(confirmed=False),
    lambda e: e["answer"].update(reason=""),
    lambda e: e["answer"].update(best=[]),
    lambda e: e["answer"].update(avoid=[RECEIVE_KYOSHA]),
    lambda e: e["answer"].update(best=[[["attack", None, "9"]]]),
])
def test_invalid_answers_are_atomic(tmp_path, mutation):
    work = prepare(tmp_path)
    before = dc.suite_path(work).read_bytes()
    doc = answer_doc(dc.load_suite(work)["cases"])
    mutation(doc["cases"][0])
    with pytest.raises(ValueError):
        dc.import_answers(work, doc)
    assert dc.suite_path(work).read_bytes() == before


def test_answer_for_one_style_does_not_grade_other_style(tmp_path):
    work = prepare(tmp_path)
    dc.import_answers(work, answer_doc(dc.load_suite(work)["cases"]))
    opts = dict(OPTIONS, style="risk")
    run = dc.run_suite(work, opts, evaluator=fake_evaluator(RECEIVE_KYOSHA), progress=lambda _: None)
    assert run["results"][0]["grade"] == "unconfirmed"


def test_batch_answer_failure_does_not_commit_earlier_valid_entry(tmp_path):
    work = prepare(tmp_path, second=True)
    before = dc.suite_path(work).read_bytes()
    doc = answer_doc(dc.load_suite(work)["cases"])
    doc["cases"][1]["fingerprint"] = "stale"
    with pytest.raises(ValueError):
        dc.import_answers(work, doc)
    assert dc.suite_path(work).read_bytes() == before


def test_real_before_after_changes_and_answer_changes_are_distinguished(tmp_path):
    work = prepare(tmp_path)
    dc.import_answers(work, answer_doc(dc.load_suite(work)["cases"]))
    def run(route, **kw):
        return dc.run_suite(work, dict(OPTIONS, **kw), evaluator=fake_evaluator(route), progress=lambda _: None)["results"][0]
    assert run(PASS)["change"] == "first"
    assert run(RECEIVE_KYOSHA)["change"] == "improved"
    assert run(RECEIVE_KYOSHA)["change"] == "maintained"
    row = run(RECEIVE_KAKU)
    assert row["change"] == "regressed" and row["route_changed"]
    assert row["previous"]["route"] == RECEIVE_KYOSHA
    assert run(PASS)["grade"] == "avoid"
    assert run(RECEIVE_KYOSHA, seed=99)["change"] == "conditions_changed"
    doc = answer_doc(dc.load_suite(work)["cases"])
    doc["cases"][0]["answer"]["reason"] = "評価の基準を変更した。"
    dc.import_answers(work, doc)
    assert run(RECEIVE_KYOSHA, seed=99)["change"] == "conditions_changed"


def test_unlisted_is_not_silently_a_wrong_answer(tmp_path):
    work = prepare(tmp_path)
    dc.import_answers(work, answer_doc(dc.load_suite(work)["cases"]))
    route = [["receive", "4", None], ["attack", None, "3"]]
    run = dc.run_suite(work, OPTIONS, evaluator=fake_evaluator(route), progress=lambda _: None)
    assert run["summary"] == {"unlisted": 1}


def test_error_continues_and_does_not_replace_baseline(tmp_path):
    work = prepare(tmp_path, second=True)
    c = dc.load_suite(work)["cases"][0]
    dc.run_suite(work, dict(OPTIONS, cases=[c["id"]]), evaluator=fake_evaluator(PASS), progress=lambda _: None)
    baseline = (work / "latest-safe.json").read_bytes()
    calls = []
    def evaluate(case, options):
        calls.append(case["id"])
        if case["id"] == c["id"]:
            raise ValueError("test error")
        return fake_evaluator(case["legal_routes"][0])(case, options)
    run = dc.run_suite(work, OPTIONS, evaluator=evaluate, progress=lambda _: None)
    assert len(calls) == 2
    assert run["status"] == "incomplete" and run["summary"]["error"] == 1
    assert (work / "latest-safe.json").read_bytes() == baseline


def test_interrupt_is_saved_without_replacing_baseline(tmp_path):
    work = prepare(tmp_path)
    def stop(*args):
        raise KeyboardInterrupt
    run = dc.run_suite(work, OPTIONS, evaluator=stop, progress=lambda _: None)
    assert run["status"] == "interrupted"
    assert (work / "runs" / (run["run_id"] + ".json")).exists()
    assert not (work / "latest-safe.json").exists()


def test_subset_runs_preserve_other_cases_baseline(tmp_path):
    work = prepare(tmp_path, second=True)
    cases = dc.load_suite(work)["cases"]
    for c in cases:
        dc.run_suite(work, dict(OPTIONS, cases=[c["id"]]),
                     evaluator=fake_evaluator(c["legal_routes"][0]), progress=lambda _: None)
    assert set(dc.previous_results(work, "safe")) == {c["id"] for c in cases}


def test_reason_changes_are_recorded_even_when_route_stays_the_same(tmp_path):
    work = prepare(tmp_path)
    dc.import_answers(work, answer_doc(dc.load_suite(work)["cases"]))
    dc.run_suite(work, OPTIONS, evaluator=fake_evaluator(RECEIVE_KYOSHA), progress=lambda _: None)
    def different_reason(case, opts):
        result = fake_evaluator(RECEIVE_KYOSHA)(case, opts)
        result["steps"][0]["detail"] = "updated actual reason"
        return result
    row = dc.run_suite(work, OPTIONS, evaluator=different_reason, progress=lambda _: None)["results"][0]
    assert row["reason_changed"] and not row["route_changed"]
    assert row["change"] == "maintained"
    assert row["previous"]["code_sha256"] and row["previous"]["run_id"]


def test_source_change_during_run_is_not_a_clean_baseline(tmp_path, monkeypatch):
    work = prepare(tmp_path)
    identities = iter([{"sha256": "before"}, {"sha256": "after"}])
    monkeypatch.setattr(dc, "code_identity", lambda: next(identities))
    run = dc.run_suite(work, OPTIONS, evaluator=fake_evaluator(PASS), progress=lambda _: None)
    assert run["status"] == "incomplete" and not (work / "latest-safe.json").exists()


class FakeAgent:
    TIME_SEARCH_ENABLED = True
    TIME_SEARCH_MAX_SECONDS = 1
    GENERIC_RESPONSE_NARROWING_ENABLED = False

    def __init__(self, name):
        self.name = name
        self.events = []
        self._track = {}

    def bind_player(self, player):
        self.player = player

    def _ensure_trackers(self, state):
        self._track[id(state)] = {}

    def on_public_action(self, state, actor, action):
        self.events.append((actor, action))

    def select_action(self, state, player, actions):
        expected = 5 if state.phase == "receive" else 6
        assert len(self.events) == expected  # target pass and future are not observed
        assert not self.TIME_SEARCH_BACKGROUND_ENABLED
        assert self._relative_hand_rank_table == {}
        assert self.GENERIC_RESPONSE_TACTICAL_PAIRED_COMPARISON_ENABLED
        self.last_decision_reason = "receive_reason" if state.phase == "receive" else "attack_reason"
        return ("receive", "4", None) if state.phase == "receive" else ("attack", None, "2")


def test_evaluation_replays_public_events_and_keeps_both_actual_reasons():
    c = dc.extract_report(report())
    result = dc.evaluate(c, OPTIONS, FakeAgent)
    assert result["route"] == RECEIVE_KYOSHA
    assert [s["reason"] for s in result["steps"]] == ["receive_reason", "attack_reason"]
    assert result["conditions"]["settings"]["TIME_SEARCH_BACKGROUND_ENABLED"] is False


def test_html_imported_content_cannot_close_script(tmp_path):
    c = dc.extract_report(report())
    c["label"] = '</script><script>alert("oops")</script>'
    path = tmp_path / "review.html"
    write_review(path, {"format": dc.FORMAT, "cases": [c]})
    text = path.read_text(encoding="utf-8")
    assert c["label"] not in text and "\\u003c/script>" in text


def test_latest_ai2_smoke_receive_and_attack():
    from goita_ai2.current_ai import generic_response_store as stores
    old_store = stores._GENERIC_RESPONSE_PATTERN_STORE
    old_snapshot = old_store.snapshot(detail_limit=0)
    c = dc.extract_report(report())
    result = dc.evaluate(c, dict(OPTIONS, no_search=True))
    assert result["route"] == RECEIVE_KYOSHA
    assert result["steps"][0]["detail"] == "receive_limited_shi_support_1_followup_2"
    assert result["steps"][1]["detail"] == "attack_limited_shi_support_2"
    assert stores._GENERIC_RESPONSE_PATTERN_STORE is old_store
    after = old_store.snapshot(detail_limit=0)
    old_snapshot.pop("uptime_seconds", None)
    after.pop("uptime_seconds", None)
    assert after == old_snapshot

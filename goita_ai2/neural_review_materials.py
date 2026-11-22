"""Register confirmed AI判断チェック answers as neural teaching material, without training."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import uuid
from collections import Counter
from pathlib import Path

from goita_ai2 import decision_check as dc
from goita_ai2.neural_policy import live_state_payload
from goita_ai2.train_neural_policy import (
    DEFAULT_CORRECTIONS, DEFAULT_MODEL, _safe_corrections, correction_fingerprint,
)

SOURCE = "ai_decision_check"


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    ids = [record["decision_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("教材のdecision_idが重複しています")
    return records


def source_key(workdir: Path) -> str:
    path = dc.suite_path(workdir).resolve()
    try:
        return path.relative_to(dc.ROOT).as_posix()
    except ValueError:
        return "external-suite-" + dc.digest(str(path))[:16]


def build_case_records(case: dict, style: str, suite_key: str) -> list[dict]:
    answer = case["answers"][style]
    if not isinstance(answer.get("reason"), str) or not answer["reason"].strip() or not answer.get("best"):
        raise ValueError(f"{case['id']}: 確定した最善手と理由が必要です")
    position = case["position"]
    legal_routes = dc.legal_routes(position)
    nodes: dict[tuple, dict] = {}
    seen = set()
    for grade in ("best", "acceptable", "avoid"):
        for route in answer.get(grade, []):
            token = dc.digest(route)
            if token in seen or route not in legal_routes:
                raise ValueError(f"{case['id']}: 重複または合法でない正解手順があります")
            seen.add(token)
            if grade == "avoid":
                continue
            prefix = ()
            for raw in route:
                action = dc.action_tuple(raw)
                node = nodes.setdefault(prefix, {"best": set(), "acceptable": set()})
                node[grade].add(action)
                prefix += (action,)

    records = []
    for prefix, labels in nodes.items():
        state = dc.replay(position)
        player = position["player"]
        history = copy.deepcopy(position["history"])
        for action in prefix:
            dc.apply_action(state, player, action)
            history.append({"player": player, "action": list(action)})
        legal = state.legal_actions(player)
        preferred = labels["best"] or labels["acceptable"]
        accepted = labels["best"] | labels["acceptable"]
        avoided = set()
        for route in answer.get("avoid", []):
            # A rejected receive+attack pair rejects the final attack at its
            # own node, not every receive route sharing the same first step.
            actions = tuple(dc.action_tuple(a) for a in route)
            if actions[:-1] == prefix:
                avoided.add(actions[-1])
        if accepted & avoided:
            raise ValueError(f"{case['id']}: 同じ行動が許容手と避けたい手に重複しています")
        indices = lambda choices: [i for i, action in enumerate(legal) if action in choices]
        records.append({
            "schema_version": 1, "profile": "強化中AI2", "thinking_style": style,
            "decision_id": f"decision-check-{case['id']}-{style}-{dc.digest(prefix)[:12]}",
            "match_group": "human-review-" + dc.digest(position["initial_hands"])[:16],
            "split": "train", "is_target_player": True,
            "category": "human_review_decision_check",
            "state": live_state_payload(state, player, initial_hand=position["initial_hands"][player], history=history),
            "legal_actions": [{"type": a[0], "block": a[1], "attack": a[2]} for a in legal],
            "selected_action_index": indices(preferred)[0],
            "acceptable_action_indices": indices(accepted),
            "best_action_indices": indices(labels["best"]),
            "avoid_action_indices": indices(avoided),
            "review": {
                "reason": answer["reason"].strip(), "thinking_style": style,
                "registration_source": SOURCE, "source_suite": suite_key,
                "case_id": case["id"], "case_fingerprint": case["fingerprint"],
                "answer_sha256": dc.digest(answer), "route_prefix": [list(a) for a in prefix],
                "best_routes": copy.deepcopy(answer["best"]),
                "acceptable_routes": copy.deepcopy(answer.get("acceptable", [])),
                "avoid_routes": copy.deepcopy(answer.get("avoid", [])),
            },
        })
    return records


def material_plan(workdir: Path, corrections: Path, model: Path) -> tuple[list[dict], dict]:
    if not dc.suite_path(workdir).exists():
        raise ValueError("AI判断チェックの問題集がありません")
    suite = dc.load_suite(workdir)
    existing = read_records(corrections)
    owner = source_key(workdir)
    managed = lambda r: (r.get("review", {}).get("registration_source") == SOURCE
                         and r.get("review", {}).get("source_suite") == owner)
    retained = [r for r in existing if not managed(r)]
    desired, cases = [], []
    for case in suite["cases"]:
        five_shi = any(Counter(hand)["1"] >= 5 for hand in case["position"]["initial_hands"].values())
        if case["excluded"] or five_shi:
            cases.append({"id": case["id"], "status": "excluded", "records": 0})
            continue
        confirmed = False
        for style in dc.STYLES:
            answer = case.get("answers", {}).get(style, {})
            if answer.get("confirmed") is not True:
                continue
            confirmed = True
            rows = build_case_records(case, style, owner)
            desired.extend(rows)
            before = [r for r in existing if managed(r) and r["review"].get("case_id") == case["id"]
                      and r.get("thinking_style") == style]
            cases.append({"id": case["id"], "style": style,
                          "status": "registered" if before == rows else "needs_sync", "records": len(rows)})
        if not confirmed:
            cases.append({"id": case["id"], "status": "unconfirmed", "records": 0})
    planned = retained + desired
    if len({r["decision_id"] for r in planned}) != len(planned):
        raise ValueError("既存教材と生成教材のIDが競合しています")
    # Keep legacy/manual lessons intact, but never silently add conflicting
    # labels for an identical observable input in the same thinking style.
    labels_by_input = {}
    for record in planned:
        style = record.get("thinking_style", record.get("review", {}).get("thinking_style", "safe"))
        key = dc.digest([style, record["state"], record["legal_actions"]])
        labels = frozenset(record.get("acceptable_action_indices") or [record["selected_action_index"]])
        if key in labels_by_input and labels != labels_by_input[key]:
            raise ValueError("同じ公開局面の教材に異なる正解があります。既存教材と正解を確認してください")
        labels_by_input[key] = labels
    safe = _safe_corrections(planned)
    training_hash = None
    if model.exists():
        training_hash = dc.read_json(model).get("training", {}).get("safe_corrections_sha256")
    old = {r["decision_id"]: r for r in existing if managed(r)}
    new = {r["decision_id"]: r for r in desired}
    summary = {
        "format": "sorou-goita-neural-materials-status", "schema_version": 1,
        "cases": cases, "case_counts": dict(Counter(c["status"] for c in cases)),
        "added": len(new.keys() - old.keys()), "removed": len(old.keys() - new.keys()),
        "updated": sum(old[k] != new[k] for k in old.keys() & new.keys()),
        "total_records": len(planned), "safe_records": len(safe),
        "risk_records": len(planned) - len(safe), "needs_sync": existing != planned,
        "model_materials_status": "matched" if training_hash == correction_fingerprint(safe) else "retraining_pending",
        "note": "教材登録のみ。再学習・モデル差し替え・AI判断の実行は行いません。",
    }
    return planned, summary


def synchronize(workdir: Path, corrections: Path, model: Path) -> dict:
    records, summary = material_plan(workdir, corrections, model)
    if summary["needs_sync"]:
        if corrections.exists():
            original = corrections.read_bytes()
            backups = workdir / "neural_materials_backups"
            backups.mkdir(parents=True, exist_ok=True)
            backup = backups / (dc.digest(original.decode("utf-8-sig")) + ".jsonl")
            if not backup.exists():
                backup.write_bytes(original)
        corrections.parent.mkdir(parents=True, exist_ok=True)
        temp = corrections.with_name(corrections.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            temp.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                                    + "\n" for r in records), encoding="utf-8")
            os.replace(temp, corrections)
        finally:
            temp.unlink(missing_ok=True)
    _, after = material_plan(workdir, corrections, model)
    after["last_sync_changes"] = {k: summary[k] for k in ("added", "updated", "removed")}
    dc.atomic_json(workdir / "neural_materials_status.json", after)
    return after


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("sync", "status", "check"))
    parser.add_argument("--workdir", type=Path, default=dc.DEFAULT_WORKDIR)
    parser.add_argument("--corrections", type=Path, default=dc.ROOT / DEFAULT_CORRECTIONS)
    parser.add_argument("--model", type=Path, default=dc.ROOT / DEFAULT_MODEL)
    args = parser.parse_args(argv)
    try:
        if args.command == "sync":
            result = synchronize(args.workdir, args.corrections, args.model)
        else:
            _, result = material_plan(args.workdir, args.corrections, args.model)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if args.command == "check" and result["needs_sync"] else 0
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(f"教材登録エラー: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

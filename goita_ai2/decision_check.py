"""Local, private decision regression suite for the latest 強化中AI2.

Run ``python -m goita_ai2.decision_check --help``. Imported prose is evidence,
never an automatically approved answer. No server, training or network calls.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.ai_review_report import SEARCH_FIELDS, parse_action
from goita_ai2.constants import ALL_SEATS, PIECE_KANJI, PIECE_TOTALS
from goita_ai2.state import GoitaState

FORMAT = "sorou-goita-ai-decision-check"
ANSWER_FORMAT = FORMAT + "-answers"
VERSION = 1
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKDIR = ROOT / "private_data" / "ai_decision_check"
STYLES = ("safe", "risk")
GRADES = ("best", "acceptable", "avoid", "unlisted")
TACTICAL_FLAGS = (
    "GENERIC_RESPONSE_TACTICAL_PRIORITY_ENABLED",
    "GENERIC_RESPONSE_TACTICAL_PAIRED_COMPARISON_ENABLED",
    "GENERIC_RESPONSE_HUMAN_TARGETED_PAIRED_COMPARISON_ENABLED",
    "GENERIC_RESPONSE_HUMAN_PAIRED_COMPARISON_ENABLED",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSONの先頭はオブジェクトにしてください: {path.name}")
    return value


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def suite_path(workdir: Path) -> Path:
    return workdir / "suite.json"


def load_suite(workdir: Path) -> dict:
    path = suite_path(workdir)
    if not path.exists():
        return {"format": FORMAT, "schema_version": VERSION, "cases": []}
    suite = read_json(path)
    if suite.get("format") != FORMAT or suite.get("schema_version") != VERSION:
        raise ValueError("対応していない問題集形式です")
    ids = set()
    for case in suite["cases"]:
        if case["id"] in ids or digest(case["position"]) != case["fingerprint"]:
            raise ValueError("問題集の局面ID重複または再現情報の変更を検出しました。元レポートから取り込み直してください")
        ids.add(case["id"])
    return suite


def action_tuple(raw: Any) -> tuple:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError("行動は [種類, 受け/伏せ, 攻め] の3項目です")
    kind, block, attack = raw
    expected = {"pass": (False, False), "receive": (True, False),
                "attack": (False, True), "attack_after_block": (True, True)}
    if kind not in expected:
        raise ValueError("不明な行動です")
    for piece, required in zip((block, attack), expected[kind]):
        if (required and piece not in PIECE_TOTALS) or (not required and piece is not None):
            raise ValueError("行動の駒指定が不正です")
    return kind, block, attack


def apply_action(state: GoitaState, seat: str, raw: Any, agent=None) -> None:
    action = action_tuple(raw)
    if state.finished or seat != state.turn or action not in state.legal_actions(seat):
        raise ValueError(f"{seat}の合法手ではありません: {action}")
    kind, block, attack = action
    if kind == "pass":
        state.apply_pass(seat)
    elif kind == "receive":
        state.apply_receive(seat, block)
    elif kind == "attack":
        state.apply_attack(seat, attack)
    else:
        state.apply_attack_after_block(seat, block, attack)
    if agent is not None:
        agent.on_public_action(state, seat, action)


def replay(position: dict, agent=None) -> GoitaState:
    hands = position["initial_hands"]
    if set(hands) != set(ALL_SEATS) or any(len(hands[s]) != 8 for s in ALL_SEATS):
        raise ValueError("初期手駒はA〜D各8枚が必要です")
    if Counter(p for hand in hands.values() for p in hand) != Counter(PIECE_TOTALS):
        raise ValueError("初期手駒の全体枚数が一致しません")
    if position["dealer"] not in ALL_SEATS or position["player"] not in ALL_SEATS:
        raise ValueError("親または対象席が不正です")
    state = GoitaState(copy.deepcopy(hands), dealer=position["dealer"])
    state.team_score = {k: int(position.get("initial_score", {}).get(k, 0)) for k in ("AC", "BD")}
    if agent is not None:
        agent.bind_player(position["player"])
        agent._ensure_trackers(state)
    for item in position["history"]:
        apply_action(state, item["player"], item["action"], agent)
    if state.finished or state.turn != position["player"]:
        raise ValueError("対象手番の直前を再現できません")
    return state


def legal_routes(position: dict) -> list:
    state = replay(position)
    seat = position["player"]
    routes = []
    for action in state.legal_actions(seat):
        if action[0] == "receive":
            after = copy.deepcopy(state)
            apply_action(after, seat, action)
            if not after.finished:
                routes.extend([[list(action), list(next_action)]
                               for next_action in after.legal_actions(seat)])
                continue
        routes.append([list(action)])
    return sorted(routes, key=lambda route: json.dumps(route))


def route_label(route: list) -> str:
    labels = []
    for kind, block, attack in route:
        if kind == "pass":
            labels.append("パス")
        elif kind == "receive":
            labels.append(f"{PIECE_KANJI[block]}で受ける")
        elif kind == "attack":
            labels.append(f"{PIECE_KANJI[attack]}で攻める")
        else:
            labels.append(f"{PIECE_KANJI[block]}を伏せる → {PIECE_KANJI[attack]}で攻める")
    return " → ".join(labels)


def extract_report(report: dict) -> dict:
    if report.get("format") != "sorou-goita-ai-review":
        raise ValueError("デバッグルームのAI思考修正レポートJSONを指定してください。旧『理解度確認』の出力は局面再現情報が不足します")
    if report.get("schema_version") not in (1, 2):
        raise ValueError("対応していないAI思考修正レポートのバージョンです")
    target = report["target"]
    index = target["decision_log_index"]
    logs = report["log"]
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(logs):
        raise ValueError("対象のログ番号が不正です")
    parsed = parse_action(logs[index])
    if parsed is None or parsed[0] != target["seat"]:
        raise ValueError("対象の席とログが一致しません")
    if tuple(target["action"]) != parsed[1]:
        raise ValueError("対象の行動とログが一致しません")
    history = []
    for line in logs[:index]:
        event = parse_action(line)
        if event:
            history.append({"player": event[0], "action": list(event[1])})
        elif any(str(line).startswith(s + ":") for s in ALL_SEATS):
            raise ValueError("対象より前の行動ログを解釈できません")
    # Only the prefix is passed to the evaluator. Never replay the target or future.
    position = {"initial_hands": {s: list(report["initial_hands"][s]) for s in ALL_SEATS},
                "dealer": report["dealer"], "player": target["seat"], "history": history}
    # Scores were not included in early reports; absence is made explicit.
    score = report.get("initial_score")
    if isinstance(score, dict):
        position["initial_score"] = {k: int(score.get(k, 0)) for k in ("AC", "BD")}
    state = replay(position)
    apply_action(copy.deepcopy(state), target["seat"], parsed[1])
    fingerprint = digest(position)
    captured_route = [list(parsed[1])]
    if parsed[1][0] == "receive":
        for line in logs[index + 1:]:
            event = parse_action(line)
            if event:
                if event[0] == target["seat"] and event[1][0] == "attack":
                    captured_route.append(list(event[1]))
                break
    return {
        "id": "D-" + fingerprint[:16], "fingerprint": fingerprint,
        "label": target.get("label") or f"{target['seat']} 手番{target.get('turn_number', '?')}",
        "position": position, "legal_routes": legal_routes(position),
        "excluded": "誰かの初期手駒が5し以上" if any(h.count("1") >= 5 for h in position["initial_hands"].values()) else None,
        "original": {"route": captured_route, "log": logs[index],
                     "human_review": report.get("human_review", {}),
                     "versions": report.get("versions", {})},
        "capture_settings": report.get("settings_at_capture", {}).get(target["seat"], {}),
        "score_recorded": "initial_score" in position,
        "answers": {}, "sources": [],
    }


def import_reports(workdir: Path, paths: list[Path]) -> dict:
    suite = load_suite(workdir)
    by_id = {c["id"]: c for c in suite["cases"]}
    counts = {"added": 0, "existing": 0, "excluded": 0, "errors": []}
    files = []
    for path in paths:
        files.extend(sorted(path.glob("goita_ai_review_*.json")) if path.is_dir() else [path])
    if not files:
        raise ValueError("読み込むレポートがありません")
    for path in dict.fromkeys(files):
        try:
            report = read_json(path)
            case = extract_report(report)
            report_hash = digest(report)
            source = {"name": path.name, "sha256": report_hash}
            old = by_id.get(case["id"])
            if old:
                if old["fingerprint"] != case["fingerprint"]:
                    raise ValueError("局面IDが衝突しました")
                if source not in old["sources"]:
                    old["sources"].append(source)
                counts["existing"] += 1
            else:
                case["sources"].append(source)
                suite["cases"].append(case)
                by_id[case["id"]] = case
                counts["added"] += 1
            counts["excluded"] += bool(case["excluded"])
            atomic_json(workdir / "sources" / (report_hash + ".json"), report)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            counts["errors"].append({"file": path.name, "error": str(exc)})
    suite["updated_at"] = now()
    atomic_json(suite_path(workdir), suite)
    return counts


def import_answers(workdir: Path, document: dict) -> int:
    if document.get("format") != ANSWER_FORMAT or document.get("schema_version") != VERSION:
        raise ValueError("AI判断チェックから書き出した回答JSONを指定してください")
    suite = load_suite(workdir)
    by_id = {c["id"]: c for c in suite["cases"]}
    seen = set()
    for entry in document["cases"]:
        key = (entry["id"], entry["style"])
        if key in seen:
            raise ValueError("回答が重複しています")
        seen.add(key)
        case = by_id.get(entry["id"])
        if case is None or entry["fingerprint"] != case["fingerprint"]:
            raise ValueError("未知の局面または古い局面への回答です")
        if entry["style"] not in STYLES or case["excluded"]:
            raise ValueError("対象外の思考スタイルまたは除外局面です")
        answer = entry["answer"]
        if (answer.get("confirmed") is not True or not isinstance(answer.get("reason"), str)
                or not answer["reason"].strip()):
            raise ValueError("正解の確定と理由が必要です")
        groups = {}
        used = set()
        for category in GRADES[:3]:
            routes = answer.get(category, [])
            groups[category] = []
            for route in routes:
                token = digest(route)
                if token in used or route not in legal_routes(case["position"]):
                    raise ValueError("重複した手順または合法でない手順があります")
                used.add(token)
                groups[category].append(route)
        if not groups["best"]:
            raise ValueError("最善手を1つ以上選択してください")
        groups.update(confirmed=True, reason=str(answer["reason"]).strip())
        case["answers"][entry["style"]] = groups
    suite["updated_at"] = now()
    atomic_json(suite_path(workdir), suite)
    return len(seen)


def judge(route: list, answer: dict | None) -> str:
    if not answer or not answer.get("confirmed"):
        return "unconfirmed"
    return next((g for g in GRADES[:3] if route in answer[g]), "unlisted")


def comparison(result: dict, previous: dict | None) -> str:
    if result["status"] != "ok":
        return "error"
    if result["grade"] == "unconfirmed":
        return "unconfirmed"
    if previous is None:
        return "first"
    if (previous.get("answer_hash") != result["answer_hash"]
            or previous.get("conditions_hash") != result["conditions_hash"]):
        return "conditions_changed"
    if previous.get("status") != "ok":
        return "first"
    before, after = previous["grade"], result["grade"]
    if before == "unlisted" or after == "unlisted":
        return "needs_review"
    rank = {"best": 2, "acceptable": 1, "avoid": 0}
    if before not in rank:
        return "first"
    if rank[after] > rank[before]:
        return "improved"
    if rank[after] < rank[before]:
        return "regressed"
    return "maintained" if after != "avoid" else "needs_review"


def code_identity() -> dict:
    files = set()
    for directory in (ROOT / "goita_ai2", ROOT / "goita_ai2" / "current_ai",
                      ROOT / "goita_ai2" / "experimental_ai2"):
        files.update(directory.glob("*.py"))
    for directory in (ROOT / "goita_ai2" / "current_ai" / "data",
                      ROOT / "goita_ai2" / "experimental_ai2" / "data"):
        files.update(directory.rglob("*.json"))
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(files)}
    return {"sha256": digest(hashes), "files": hashes}


def configure(agent, case: dict, options: dict) -> dict:
    settings = case.get("capture_settings", {})
    for name in SEARCH_FIELDS:
        if name not in settings:
            continue
        value = settings[name]
        # Treat imported settings as data; only known, bounded scalar values.
        current = getattr(agent, name, None)
        if isinstance(current, bool):
            if not isinstance(value, bool):
                raise ValueError(f"設定 {name} は真偽値が必要です")
        elif isinstance(current, (int, float)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"設定 {name} は有限数が必要です")
            if isinstance(current, int) and not float(value).is_integer():
                raise ValueError(f"設定 {name} は整数が必要です")
            limit = 1000000 if name.endswith("MAX_NODES") else 1000 if name.endswith("SAMPLE_COUNT") else 30 if name.endswith("DEPTH") else 20
            if value <= 0 or value > limit:
                raise ValueError(f"設定 {name} の値が範囲外です")
        else:
            continue
        setattr(agent, name, int(value) if isinstance(current, int) and not isinstance(current, bool) else value)
    if options.get("seconds") is not None:
        agent.TIME_SEARCH_MAX_SECONDS = options["seconds"]
    if options.get("no_search"):
        agent.TIME_SEARCH_ENABLED = False
    agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    narrowing = bool(getattr(agent, "GENERIC_RESPONSE_NARROWING_ENABLED", False))
    for name in TACTICAL_FLAGS:
        setattr(agent, name, narrowing)
    agent.GENERIC_RESPONSE_HUMAN_TARGETED_PRIORITY_ENABLED = True
    agent.GENERIC_RESPONSE_HUMAN_TARGETED_LIVE_ENABLED = True
    agent.final_shi_risk = options["style"] == "risk"
    rank_path = ROOT / "results" / "relative_hand_all_4745_500_combined.csv"
    if options["rank_table"] == "production":
        agent._relative_hand_rank_table = {}
    elif not rank_path.is_file():
        raise ValueError("ローカルの相対ランク表がありません")
    effective = {name: getattr(agent, name, None) for name in
                 (*SEARCH_FIELDS, *TACTICAL_FLAGS, "TIME_SEARCH_BACKGROUND_ENABLED",
                  "GENERIC_RESPONSE_HUMAN_TARGETED_PRIORITY_ENABLED", "GENERIC_RESPONSE_HUMAN_TARGETED_LIVE_ENABLED")}
    return {"settings": effective, "style": options["style"], "seed": options["seed"],
            "rank_table": options["rank_table"], "rank_table_sha256":
            hashlib.sha256(rank_path.read_bytes()).hexdigest() if options["rank_table"] == "local" else None,
            "python": platform.python_version(), "platform": platform.platform(),
            "settings_basis": "capture_plus_service_flags_background_off",
            "generic_response_store": "fresh_in_memory_per_case",
            "score_recorded": case.get("score_recorded", False)}


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return str(value)


def evaluate(case: dict, options: dict, agent_factory=None) -> dict:
    if agent_factory is None:
        from goita_ai2.experimental_ai2 import RuleBasedAgent
        from goita_ai2.current_ai import generic_response_store as stores
        previous_store = stores._GENERIC_RESPONSE_PATTERN_STORE
        previous_model = (RuleBasedAgent._shared_neural_model, RuleBasedAgent._shared_neural_error)
        # Live aggregate dictionaries must neither influence later cases nor be
        # updated by this local check. The shipped human dictionary stays enabled.
        stores._GENERIC_RESPONSE_PATTERN_STORE = stores.GenericResponsePatternStore()
        RuleBasedAgent._shared_neural_model = RuleBasedAgent._shared_neural_error = None
        try:
            return _evaluate_with_agent(case, options, RuleBasedAgent)
        finally:
            stores._GENERIC_RESPONSE_PATTERN_STORE = previous_store
            RuleBasedAgent._shared_neural_model, RuleBasedAgent._shared_neural_error = previous_model
    return _evaluate_with_agent(case, options, agent_factory)


def _evaluate_with_agent(case: dict, options: dict, agent_factory) -> dict:
    agent = agent_factory(name="AI判断チェック・強化中AI2")
    conditions = configure(agent, case, options)
    rng_state = random.getstate()
    seed = int(digest([case["id"], options["seed"]])[:16], 16)
    random.seed(seed)
    started = time.perf_counter()
    try:
        state = replay(case["position"], agent)
        seat = case["position"]["player"]
        route, steps = [], []
        while True:
            action = action_tuple(agent.select_action(state, seat, state.legal_actions(seat)))
            route.append(list(action))
            tracker = getattr(agent, "_track", {}).get(id(state), {})
            steps.append(json_safe({
                "action": action, "reason": getattr(agent, "last_decision_reason", None),
                "detail": getattr(agent, "last_score_fallback_detail", None),
                "neural": copy.deepcopy(getattr(agent, "last_neural_shadow", {})),
                "candidates": copy.deepcopy(getattr(agent, "last_attack_candidate_snapshot", {})),
                "candidate_scores": copy.deepcopy(getattr(agent, "last_attack_candidate_scores", [])),
                "search": copy.deepcopy(tracker.get("last_time_limited_search", {})),
                "performance": copy.deepcopy(getattr(agent, "last_performance_metrics", {})),
            }))
            apply_action(state, seat, action, agent)
            if action[0] != "receive" or state.finished:
                break
        if route not in legal_routes(case["position"]):
            raise ValueError("AIが不正な手順を返しました")
        return {"status": "ok", "route": route, "steps": steps,
                "seconds": round(time.perf_counter() - started, 4), "conditions": conditions}
    finally:
        random.setstate(rng_state)


def previous_results(workdir: Path, style: str) -> dict:
    latest = workdir / f"latest-{style}.json"
    if not latest.exists():
        return {}
    # The pointer contains an ID, not an imported arbitrary file path.
    run_id = read_json(latest)["run_id"]
    if not isinstance(run_id, str) or any(c not in "0123456789abcdef-" for c in run_id):
        raise ValueError("前回結果のIDが不正です")
    previous = read_json(workdir / "runs" / (run_id + ".json"))
    if previous.get("status") != "complete":
        return {}
    return {r["id"]: r for r in previous["results"] if r.get("status") != "excluded"}


def run_suite(workdir: Path, options: dict, *, evaluator=None, progress=print) -> dict:
    suite = load_suite(workdir)
    cases = suite["cases"]
    if options.get("cases"):
        unknown = set(options["cases"]) - {c["id"] for c in cases}
        if unknown:
            raise ValueError("指定した局面IDがありません: " + ", ".join(sorted(unknown)))
        cases = [c for c in cases if c["id"] in options["cases"]]
    if not cases:
        raise ValueError("問題集が空です。先にレポートを取り込んでください")
    previous = previous_results(workdir, options["style"])
    run_id = str(uuid.uuid4())
    path = workdir / "runs" / (run_id + ".json")
    result = {"format": FORMAT + "-run", "schema_version": VERSION,
              "run_id": run_id, "started_at": now(), "status": "running",
              "ai": "experimental_ai2", "ai_label": "強化中AI2", "options": options,
              "code": code_identity(), "total": len(cases), "results": [],
              "notes": ["時間制限付き探索のため、同条件でも実行時間により判断が変わる場合があります。",
                        "riskは現在の終盤し攻めのリスク選択のみ。AI全体の性格切替ではありません。",
                        "当時の乱数・キャッシュは未記録のため、当時の完全再現ではなく最新AIの比較です。",
                        "初期得点が未記録の旧レポートはAC/BDとも0点として再現します。"]}
    atomic_json(path, result)
    evaluator = evaluator or evaluate
    try:
        for number, case in enumerate(cases, 1):
            progress(f"AI判断チェック {number}/{len(cases)}: {case['id']}")
            answer = case["answers"].get(options["style"])
            row = {"id": case["id"], "label": case["label"], "fingerprint": case["fingerprint"],
                   "original": case["original"], "answer": answer, "answer_hash": digest(answer),
                   "code_sha256": result["code"]["sha256"], "run_id": run_id,
                   "executed_at": now()}
            if case["excluded"]:
                row.update(status="excluded", grade="excluded", change="excluded", error=case["excluded"])
            else:
                try:
                    row.update(evaluator(case, options))
                    row["grade"] = judge(row["route"], answer)
                    row["conditions_hash"] = digest(row["conditions"])
                    row["change"] = comparison(row, previous.get(case["id"]))
                    prev = previous.get(case["id"])
                    row["previous"] = {key: prev.get(key) for key in
                                       ("route", "grade", "steps", "code_sha256", "run_id", "executed_at")} if prev else None
                    row["route_changed"] = prev is not None and prev.get("route") != row["route"]
                    row["reason_changed"] = prev is not None and [
                        (step.get("reason"), step.get("detail")) for step in prev.get("steps", [])
                    ] != [(step.get("reason"), step.get("detail")) for step in row["steps"]]
                except Exception as exc:
                    row.update(status="error", grade="error", change="error",
                               error=f"{type(exc).__name__}: {exc}")
            result["results"].append(row)
            atomic_json(path, result)
        result["status"] = "complete" if not any(r["status"] == "error" for r in result["results"]) else "incomplete"
    except KeyboardInterrupt:
        result["status"] = "interrupted"
    result["finished_at"] = now()
    result["summary"] = dict(Counter(r["grade"] for r in result["results"]))
    result["changes"] = dict(Counter(r["change"] for r in result["results"]))
    # Do not use a run spanning source/model edits as a clean baseline.
    if code_identity()["sha256"] != result["code"]["sha256"]:
        result["status"] = "incomplete"
        result["notes"].append("実行中にソースまたはモデルが変更されました。前回基準には採用しません。")
    atomic_json(path, result)
    if result["status"] == "complete":
        # A subset run must not discard other successful cases' last baseline.
        merged = dict(previous)
        merged.update({r["id"]: r for r in result["results"] if r["status"] == "ok"})
        baseline = dict(result, results=list(merged.values()))
        baseline_id = str(uuid.uuid4())
        atomic_json(workdir / "runs" / (baseline_id + ".json"), baseline)
        atomic_json(workdir / f"latest-{options['style']}.json", {"run_id": baseline_id})
    from goita_ai2.decision_check_report import write_results
    write_results(workdir / "result.html", result)
    progress("AI判断チェックを実行しました。対象=強化中AI2 / " + options["style"]
             + f" / {len(result['results'])}/{len(cases)}件 / 状態={result['status']} / "
             + json.dumps(result["summary"], ensure_ascii=False))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="AI判断チェック：ローカルでレポートを問題集にして最新の強化中AI2を確認")
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR, help="非公開の保存先")
    sub = parser.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import", help="AI思考修正レポートをまとめて取り込む")
    imp.add_argument("paths", type=Path, nargs="+")
    sub.add_parser("review", help="正解を確認するHTMLを作る")
    ans = sub.add_parser("answers", help="確定した回答JSONを取り込む")
    ans.add_argument("path", type=Path)
    run = sub.add_parser("run", help="最新の強化中AI2で一括確認")
    run.add_argument("--style", choices=STYLES, default="safe")
    run.add_argument("--seed", type=int, default=1222)
    run.add_argument("--rank-table", choices=("production", "local"), default="production")
    run.add_argument("--seconds", type=float, help="各探索の基本時間を上書き（局面全体の上限ではありません）")
    run.add_argument("--no-search", action="store_true", help="動作確認用。通常の採点と比較されません")
    run.add_argument("--case", dest="cases", action="append", help="局面IDを指定。複数回指定可")
    args = parser.parse_args(argv)
    try:
        if args.command == "import":
            counts = import_reports(args.workdir, args.paths)
            print(json.dumps(counts, ensure_ascii=False))
            from goita_ai2.decision_check_report import write_review
            write_review(args.workdir / "review.html", load_suite(args.workdir))
            print(args.workdir / "review.html")
            return 1 if counts["errors"] else 0
        if args.command == "review":
            from goita_ai2.decision_check_report import write_review
            write_review(args.workdir / "review.html", load_suite(args.workdir))
            print(args.workdir / "review.html")
        elif args.command == "answers":
            print(f"回答{import_answers(args.workdir, read_json(args.path))}件を取り込みました。")
            from goita_ai2.decision_check_report import write_review
            write_review(args.workdir / "review.html", load_suite(args.workdir))
        else:
            if args.seconds is not None and (not math.isfinite(args.seconds) or not 0 < args.seconds <= 20):
                raise ValueError("--secondsは0より大きく20以下にしてください")
            options = {key: getattr(args, key) for key in ("style", "seed", "rank_table", "seconds", "no_search", "cases")}
            result = run_suite(args.workdir, options)
            print(args.workdir / "result.html")
            return 0 if result["status"] == "complete" else 1
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

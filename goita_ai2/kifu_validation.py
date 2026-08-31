"""Builds private AI validation cases from the local Goita archive.

It reconstructs omitted passes, compares information-set and legacy search,
and stores review cases under results without publishing the source archive.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from goita_ai2.constants import ALL_SEATS, PIECE_TOTALS
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


Action = Tuple[str, Optional[str], Optional[str]]

KANJI_TO_PIECE = {
    "し": "1",
    "香": "2",
    "馬": "3",
    "銀": "4",
    "金": "5",
    "角": "6",
    "飛": "7",
    "玉": "8",
    "王": "9",
}
ROYAL_KANJI = "王"
DEFAULT_CASE_PATH = Path("results/ai_validation/real_positions.json")
DEFAULT_COMPARISON_PATH = Path("results/ai_validation/search_comparison.json")
DEFAULT_PROBLEM_PATH = Path("results/ai_validation/problem_cases.json")


def _seat(player_id: str) -> str:
    index = int(str(player_id).removeprefix("p"))
    return ALL_SEATS[index]


def _hand_digits(hand: str) -> List[str]:
    return [KANJI_TO_PIECE[piece] for piece in str(hand)]


def _raw_piece_candidates(raw_piece: str, hand: Sequence[str]) -> Tuple[str, ...]:
    if raw_piece == ROYAL_KANJI:
        return tuple(piece for piece in ("9", "8") if piece in hand)
    return (KANJI_TO_PIECE[raw_piece],)


def _apply_action(state: GoitaState, player: str, action: Action) -> None:
    action_type, block, attack = action
    if action_type == "pass":
        state.apply_pass(player)
    elif action_type == "receive":
        if block is None:
            raise ValueError("receive requires a block")
        state.apply_receive(player, block)
    elif action_type == "attack":
        if attack is None:
            raise ValueError("attack requires an attack piece")
        state.apply_attack(player, attack)
    elif action_type == "attack_after_block":
        if block is None or attack is None:
            raise ValueError("attack_after_block requires two pieces")
        state.apply_attack_after_block(player, block, attack)
    else:
        raise ValueError(f"unknown action type: {action_type}")


def _resolve_recorded_move(
    state: GoitaState,
    actor: str,
    raw_block: str,
    raw_attack: str,
) -> Tuple[Action, ...]:
    block_candidates = _raw_piece_candidates(raw_block, state.hands[actor])
    if state.phase == "receive":
        for block in block_candidates:
            if not state.can_receive(actor, block):
                continue
            preview = copy.deepcopy(state)
            preview.apply_receive(actor, block)
            for attack in _raw_piece_candidates(raw_attack, preview.hands[actor]):
                action = ("attack", None, attack)
                if action in preview.legal_actions(actor):
                    return (("receive", block, None), action)
        raise ValueError(
            f"recorded receive {raw_block}/{raw_attack} is not legal for {actor}"
        )

    legal = set(state.legal_actions(actor))
    for block in block_candidates:
        for attack in _raw_piece_candidates(raw_attack, state.hands[actor]):
            action = ("attack_after_block", block, attack)
            if action in legal:
                return (action,)
    raise ValueError(
        f"recorded block/attack {raw_block}/{raw_attack} is not legal for {actor}"
    )


def _phase_label(hand_size: int) -> str:
    if hand_size >= 7:
        return "opening"
    if hand_size >= 4:
        return "middle"
    return "endgame"


def _action_label(action: Action) -> str:
    if action[0] == "attack_after_block":
        return "attack"
    return str(action[0])


def _case_category(state: GoitaState, action: Action) -> str:
    return f"{_phase_label(len(state.hands[state.turn]))}_{_action_label(action)}"


def _case_id(
    match_id: str,
    round_index: int,
    decision_index: int,
    player: str,
    action: Action,
) -> str:
    return (
        f"kifu-{match_id}-r{int(round_index):02d}-d{int(decision_index):03d}"
        f"-{player}-{action[0]}"
    )


def _case_record(
    *,
    match_id: str,
    round_index: int,
    decision_index: int,
    initial_hands: Dict[str, List[str]],
    dealer: str,
    initial_score: Dict[str, int],
    state: GoitaState,
    history: Sequence[Dict[str, Any]],
    action: Action,
) -> Dict[str, Any]:
    player = state.turn
    category = _case_category(state, action)
    if state.attacker is None:
        attacker_relation = "none"
    elif state.attacker == player:
        attacker_relation = "self"
    elif ({state.attacker, player} <= {"A", "C"}) or (
        {state.attacker, player} <= {"B", "D"}
    ):
        attacker_relation = "ally"
    else:
        attacker_relation = "enemy"
    return {
        "id": _case_id(match_id, round_index, decision_index, player, action),
        "source": {
            "match_id": str(match_id),
            "round_index": int(round_index),
            "decision_index": int(decision_index),
        },
        "category": category,
        "dealer": dealer,
        "initial_score": dict(initial_score),
        "initial_hands": copy.deepcopy(initial_hands),
        "history": copy.deepcopy(list(history)),
        "player": player,
        "actual_action": list(action),
        "position": {
            "phase": state.phase,
            "hand_size": len(state.hands[player]),
            "hand": list(state.hands[player]),
            "current_attack": state.current_attack,
            "attacker": state.attacker,
            "attacker_relation": attacker_relation,
            "legal_action_count": len(state.legal_actions(player)),
        },
    }


def iter_kifu_decisions(kifu_path: Path) -> Iterator[Dict[str, Any]]:
    """Yield every reconstructed decision, including passes omitted by kifu."""
    archive = json.loads(kifu_path.read_text(encoding="utf-8"))
    matches = archive.get("matches", []) if isinstance(archive, dict) else []
    for match in matches:
        match_id = str(match.get("id", "unknown"))
        for round_item in match.get("rounds", []):
            round_index = int(round_item.get("round_index", 0))
            initial_hands = {
                _seat(player_id): _hand_digits(hand)
                for player_id, hand in round_item.get("hand", {}).items()
            }
            if tuple(sorted(initial_hands)) != tuple(ALL_SEATS):
                continue
            dealer = ALL_SEATS[int(round_item.get("uchidashi", 0))]
            score = list(round_item.get("score", [0, 0]))
            initial_score = {
                "AC": int(score[0]) if score else 0,
                "BD": int(score[1]) if len(score) > 1 else 0,
            }
            state = GoitaState(copy.deepcopy(initial_hands), dealer=dealer)
            state.team_score = dict(initial_score)
            history: List[Dict[str, Any]] = []
            decision_index = 0

            def emit(action: Action) -> Dict[str, Any]:
                nonlocal decision_index
                decision_index += 1
                return _case_record(
                    match_id=match_id,
                    round_index=round_index,
                    decision_index=decision_index,
                    initial_hands=initial_hands,
                    dealer=dealer,
                    initial_score=initial_score,
                    state=state,
                    history=history,
                    action=action,
                )

            def commit(player: str, action: Action) -> None:
                _apply_action(state, player, action)
                history.append({"player": player, "action": list(action)})

            for raw_move in round_item.get("game", []):
                actor = ALL_SEATS[int(raw_move[0])]
                pass_guard = 0
                while state.turn != actor:
                    pass_guard += 1
                    if pass_guard > len(ALL_SEATS):
                        raise ValueError(
                            f"cannot reconstruct passes for {match_id}/{round_index}"
                        )
                    action = ("pass", None, None)
                    yield emit(action)
                    commit(state.turn, action)

                for action in _resolve_recorded_move(
                    state,
                    actor,
                    str(raw_move[1]),
                    str(raw_move[2]),
                ):
                    yield emit(action)
                    commit(actor, action)


def _selection_rank(case: Dict[str, Any]) -> str:
    source = case["source"]
    payload = (
        f"{source['match_id']}:{source['round_index']}:"
        f"{source['decision_index']}:{case['player']}"
    )
    return hashlib.sha256(payload.encode("ascii", errors="ignore")).hexdigest()


def extract_kifu_validation_cases(
    kifu_path: Path,
    *,
    limit: int = 18,
) -> Dict[str, Any]:
    """Select a deterministic, phase/action-balanced private case set."""
    limit = max(1, int(limit))
    category_count = 9
    per_category = max(1, (limit + category_count - 1) // category_count)
    buckets: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    scanned = 0
    category_totals: Counter = Counter()
    for case in iter_kifu_decisions(kifu_path):
        scanned += 1
        category = str(case["category"])
        category_totals[category] += 1
        bucket = buckets[category]
        bucket.append((_selection_rank(case), case))
        bucket.sort(key=lambda item: item[0])
        del bucket[per_category:]

    balanced = [item for bucket in buckets.values() for item in bucket]
    balanced.sort(key=lambda item: (item[1]["category"], item[0]))
    selected = [case for _rank, case in balanced[:limit]]
    if len(selected) < limit:
        selected_ids = {case["id"] for case in selected}
        extras = sorted(
            (
                (_selection_rank(case), case)
                for case in iter_kifu_decisions(kifu_path)
                if case["id"] not in selected_ids
            ),
            key=lambda item: item[0],
        )
        selected.extend(case for _rank, case in extras[: limit - len(selected)])
    selected.sort(key=lambda case: (case["category"], case["id"]))
    return {
        "schema_version": 1,
        "source": {
            "kind": "private_kifu_archive",
            "path": str(kifu_path),
            "content_not_for_publication": True,
        },
        "scanned_decisions": scanned,
        "category_totals": dict(sorted(category_totals.items())),
        "case_count": len(selected),
        "cases": selected,
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_validation_cases(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError("validation case file must contain a cases list")
    return data


def replay_validation_case(
    case: Dict[str, Any],
    agent: Optional[RuleBasedAgent] = None,
) -> GoitaState:
    state = GoitaState(
        copy.deepcopy(case["initial_hands"]),
        dealer=str(case["dealer"]),
    )
    state.team_score = {
        "AC": int(case.get("initial_score", {}).get("AC", 0)),
        "BD": int(case.get("initial_score", {}).get("BD", 0)),
    }
    if agent is not None:
        agent.bind_player(str(case["player"]))
        agent._ensure_trackers(state)
    for item in case.get("history", []):
        player = str(item["player"])
        raw_action = item["action"]
        action: Action = (raw_action[0], raw_action[1], raw_action[2])
        _apply_action(state, player, action)
        if agent is not None:
            agent.on_public_action(state, player, action)
    if state.turn != case["player"]:
        raise ValueError(f"case {case.get('id')} does not replay to its player")
    return state


def _configure_comparison_agent(
    *,
    information_set: bool,
    search_seconds: float,
    sample_count: int,
    max_depth: int,
    max_nodes: int,
) -> RuleBasedAgent:
    agent = RuleBasedAgent(name="kifu-information-set-comparison")
    agent.TIME_SEARCH_INFORMATION_SET_ENABLED = bool(information_set)
    agent.TIME_SEARCH_CACHE_ENABLED = False
    agent.TIME_SEARCH_PREDICTION_CACHE_ENABLED = False
    agent.TIME_SEARCH_ADAPTIVE_BUDGET_ENABLED = False
    agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    agent.TIME_SEARCH_MAX_SECONDS = max(0.01, float(search_seconds))
    agent.TIME_SEARCH_SAMPLE_COUNT = max(1, int(sample_count))
    agent.TIME_SEARCH_MAX_DEPTH = max(1, int(max_depth))
    agent.TIME_SEARCH_MAX_NODES = max(1, int(max_nodes))
    return agent


def _inference_summary(agent: RuleBasedAgent, state: GoitaState) -> Dict[str, Any]:
    tracker = agent._track.get(id(state), {})
    estimates = tracker.get("estimated_current_hands", {})
    player = str(agent.me)
    per_player = {}
    for seat in ALL_SEATS:
        if seat == player:
            continue
        exact = constrained = 0
        for piece in PIECE_TOTALS:
            item = estimates.get(seat, {}).get(piece, {})
            minimum = int(item.get("min", 0))
            maximum = int(item.get("max", PIECE_TOTALS[piece]))
            if minimum == maximum:
                exact += 1
            if minimum > 0 or maximum < int(PIECE_TOTALS[piece]):
                constrained += 1
        per_player[seat] = {
            "exact_piece_types": exact,
            "constrained_piece_types": constrained,
        }
    joint = tracker.get("joint_hand_inference", {})
    return {
        "players": per_player,
        "joint_feasible": bool(joint.get("feasible", False)),
        "joint_solution_count": int(joint.get("solution_count", 0)),
    }


def compare_validation_case(
    case: Dict[str, Any],
    *,
    search_seconds: float,
    sample_count: int,
    max_depth: int,
    max_nodes: int,
) -> Dict[str, Any]:
    info_agent = _configure_comparison_agent(
        information_set=True,
        search_seconds=search_seconds,
        sample_count=sample_count,
        max_depth=max_depth,
        max_nodes=max_nodes,
    )
    legacy_agent = _configure_comparison_agent(
        information_set=False,
        search_seconds=search_seconds,
        sample_count=sample_count,
        max_depth=max_depth,
        max_nodes=max_nodes,
    )
    info_state = replay_validation_case(case, info_agent)
    legacy_state = replay_validation_case(case, legacy_agent)
    player = str(case["player"])
    legal_actions = info_state.legal_actions(player)
    if legal_actions != legacy_state.legal_actions(player):
        raise ValueError(f"comparison replay differs for {case.get('id')}")
    actual: Action = tuple(case["actual_action"])  # type: ignore[assignment]
    if actual not in legal_actions:
        raise ValueError(f"recorded action is illegal in {case.get('id')}")

    info_baseline = info_agent._select_rule_based_action(
        info_state,
        player,
        legal_actions,
    )
    legacy_baseline = legacy_agent._select_rule_based_action(
        legacy_state,
        player,
        legal_actions,
    )
    tracker = info_agent._track[id(info_state)]
    generated_started = time.perf_counter()
    samples = info_agent._timed_search_sample_states(
        info_state,
        player,
        tracker,
        max(1, int(sample_count)),
    )
    sample_seconds = time.perf_counter() - generated_started

    info_started = time.perf_counter()
    info_result = info_agent._time_limited_search_from_samples(
        info_state,
        player,
        legal_actions,
        info_baseline,
        samples,
    )
    info_seconds = time.perf_counter() - info_started
    legacy_started = time.perf_counter()
    legacy_result = legacy_agent._time_limited_search_from_samples(
        legacy_state,
        player,
        legal_actions,
        legacy_baseline,
        samples,
    )
    legacy_seconds = time.perf_counter() - legacy_started

    info_action = info_result.action if info_result is not None else info_baseline
    legacy_action = legacy_result.action if legacy_result is not None else legacy_baseline
    return {
        "case_id": case["id"],
        "source": copy.deepcopy(case["source"]),
        "category": case["category"],
        "player": player,
        "hand_size": len(info_state.hands[player]),
        "legal_action_count": len(legal_actions),
        "actual_action": list(actual),
        "attacker_relation": str(
            case.get("position", {}).get("attacker_relation", "none")
        ),
        "rule_baseline_action": list(info_baseline),
        "information_set_action": list(info_action),
        "legacy_action": list(legacy_action),
        "information_set_changed_legacy": info_action != legacy_action,
        "information_set_matches_actual": info_action == actual,
        "legacy_matches_actual": legacy_action == actual,
        "baseline_consistent": info_baseline == legacy_baseline,
        "sample_generation_seconds": round(sample_seconds, 6),
        "information_set_seconds": round(info_seconds, 6),
        "legacy_seconds": round(legacy_seconds, 6),
        "information_set_result": info_result.as_dict() if info_result else None,
        "legacy_result": legacy_result.as_dict() if legacy_result else None,
        "inference": _inference_summary(info_agent, info_state),
        "case": copy.deepcopy(case),
    }


def compare_validation_cases(
    case_set: Dict[str, Any],
    *,
    search_seconds: float = 0.25,
    sample_count: int = 12,
    max_depth: int = 3,
    max_nodes: int = 50_000,
) -> Dict[str, Any]:
    comparisons = [
        compare_validation_case(
            case,
            search_seconds=search_seconds,
            sample_count=sample_count,
            max_depth=max_depth,
            max_nodes=max_nodes,
        )
        for case in case_set.get("cases", [])
    ]
    changed = sum(item["information_set_changed_legacy"] for item in comparisons)
    info_actual = sum(item["information_set_matches_actual"] for item in comparisons)
    legacy_actual = sum(item["legacy_matches_actual"] for item in comparisons)
    information_set_used = sum(
        bool((item.get("information_set_result") or {}).get("information_set"))
        for item in comparisons
    )
    exact_type_counts = [
        sum(
            int(player_item.get("exact_piece_types", 0))
            for player_item in item.get("inference", {}).get("players", {}).values()
        )
        for item in comparisons
    ]
    near_determined = sum(count >= 15 for count in exact_type_counts)
    multi_candidate = sum(
        int((item.get("information_set_result") or {}).get("candidate_count", 0)) > 1
        for item in comparisons
    )
    ally_context = sum(item.get("attacker_relation") == "ally" for item in comparisons)
    category_summary: Dict[str, Dict[str, int]] = {}
    for item in comparisons:
        target = category_summary.setdefault(
            str(item["category"]),
            {"cases": 0, "changed": 0, "information_set_matches_actual": 0},
        )
        target["cases"] += 1
        target["changed"] += int(item["information_set_changed_legacy"])
        target["information_set_matches_actual"] += int(
            item["information_set_matches_actual"]
        )
    count = len(comparisons)
    return {
        "schema_version": 1,
        "settings": {
            "search_seconds": float(search_seconds),
            "sample_count": int(sample_count),
            "max_depth": int(max_depth),
            "max_nodes": int(max_nodes),
        },
        "summary": {
            "cases": count,
            "information_set_used": information_set_used,
            "changed_from_legacy": changed,
            "changed_rate": round(changed / count if count else 0.0, 5),
            "information_set_matches_recorded": info_actual,
            "legacy_matches_recorded": legacy_actual,
            "near_determined_cases": near_determined,
            "multi_candidate_cases": multi_candidate,
            "ally_attack_response_cases": ally_context,
            "recorded_match_is_not_strength_score": True,
        },
        "categories": dict(sorted(category_summary.items())),
        "comparisons": comparisons,
    }


def _problem_reasons(comparison: Dict[str, Any]) -> List[str]:
    reasons = []
    info_result = comparison.get("information_set_result") or {}
    if comparison.get("information_set_changed_legacy"):
        reasons.append("changed_from_legacy")
    if not comparison.get("information_set_matches_actual"):
        reasons.append("differs_from_recorded_action")
    if not info_result:
        reasons.append("search_did_not_complete")
    elif not info_result.get("information_set"):
        reasons.append("information_set_fallback")
    if float(info_result.get("information_confidence", 1.0)) < 0.15:
        reasons.append("low_information_confidence")
    return reasons


def save_review_cases(
    comparison_report: Dict[str, Any],
    path: Path,
    *,
    limit: int = 30,
) -> Dict[str, Any]:
    """Merge review-worthy positions without overwriting human expectations."""
    existing: Dict[str, Any] = {
        "schema_version": 1,
        "private": True,
        "cases": [],
    }
    if path.exists():
        existing = load_validation_cases(path)
        existing.setdefault("private", True)
    by_id = {item["case"]["id"]: item for item in existing.get("cases", [])}
    candidates = []
    for comparison in comparison_report.get("comparisons", []):
        reasons = _problem_reasons(comparison)
        if not reasons:
            continue
        priority = (
            0 if "search_did_not_complete" in reasons else
            1 if "information_set_fallback" in reasons else
            2 if "changed_from_legacy" in reasons else 3
        )
        candidates.append((priority, str(comparison["case_id"]), comparison, reasons))
    candidates.sort(key=lambda item: (item[0], item[1]))
    for _priority, _case_id_value, comparison, reasons in candidates[: max(0, int(limit))]:
        case_id = str(comparison["case_id"])
        previous = by_id.get(case_id, {})
        by_id[case_id] = {
            "case": copy.deepcopy(comparison["case"]),
            "review": {
                "reasons": reasons,
                "actual_action": comparison["actual_action"],
                "information_set_action": comparison["information_set_action"],
                "legacy_action": comparison["legacy_action"],
            },
            "expectation": copy.deepcopy(previous.get("expectation", {
                "allowed_actions": [],
                "forbidden_actions": [],
                "note": "",
            })),
        }
    stored = {
        "schema_version": 1,
        "private": True,
        "case_count": len(by_id),
        "cases": [by_id[key] for key in sorted(by_id)],
    }
    write_json(path, stored)
    return stored


def validate_problem_cases(
    problem_set: Dict[str, Any],
    *,
    search_seconds: float = 0.25,
    sample_count: int = 12,
    max_depth: int = 3,
    max_nodes: int = 50_000,
) -> Dict[str, Any]:
    results = []
    for item in problem_set.get("cases", []):
        expectation = item.get("expectation", {})
        allowed = {tuple(action) for action in expectation.get("allowed_actions", [])}
        forbidden = {tuple(action) for action in expectation.get("forbidden_actions", [])}
        comparison = compare_validation_case(
            item["case"],
            search_seconds=search_seconds,
            sample_count=sample_count,
            max_depth=max_depth,
            max_nodes=max_nodes,
        )
        action = tuple(comparison["information_set_action"])
        passed = (not allowed or action in allowed) and action not in forbidden
        results.append({
            "case_id": comparison["case_id"],
            "action": list(action),
            "has_expectation": bool(allowed or forbidden),
            "passed": passed,
        })
    checked = [item for item in results if item["has_expectation"]]
    return {
        "cases": len(results),
        "checked_expectations": len(checked),
        "passed": all(item["passed"] for item in checked),
        "results": results,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kifu", type=Path, default=Path("frontend/kifu_data.json"))
    parser.add_argument("--limit", type=int, default=18)
    parser.add_argument("--search-seconds", type=float, default=0.25)
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-nodes", type=int, default=50_000)
    parser.add_argument("--cases-output", type=Path, default=DEFAULT_CASE_PATH)
    parser.add_argument("--comparison-output", type=Path, default=DEFAULT_COMPARISON_PATH)
    parser.add_argument("--problems-output", type=Path, default=DEFAULT_PROBLEM_PATH)
    args = parser.parse_args(argv)

    case_set = extract_kifu_validation_cases(args.kifu, limit=args.limit)
    write_json(args.cases_output, case_set)
    comparison = compare_validation_cases(
        case_set,
        search_seconds=args.search_seconds,
        sample_count=args.sample_count,
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
    )
    write_json(args.comparison_output, comparison)
    problems = save_review_cases(comparison, args.problems_output)
    problem_regression = validate_problem_cases(
        problems,
        search_seconds=args.search_seconds,
        sample_count=args.sample_count,
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
    )
    summary = {
        "7-1": {
            "scanned_decisions": case_set["scanned_decisions"],
            "selected_cases": case_set["case_count"],
            "output": str(args.cases_output),
        },
        "7-2": comparison["summary"],
        "7-4": {
            "stored_review_cases": problems["case_count"],
            "checked_expectations": problem_regression["checked_expectations"],
            "expectations_passed": problem_regression["passed"],
            "output": str(args.problems_output),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

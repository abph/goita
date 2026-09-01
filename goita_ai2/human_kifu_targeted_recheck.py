"""Recheck promising human receive routes against the current AI offline.

The raw kifu archive remains local and private.  This command reconstructs
positions only in memory and writes a privacy-safe aggregate report containing
anonymous pattern IDs and comparison totals.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


DEFAULT_KIFU_PATH = Path("frontend/kifu_data_raw.json")
DEFAULT_OUTPUT_PATH = Path(
    "results/kifu_audit/human_receive_targeted_recheck.json"
)
RECEIVE_ACTIONS = frozenset({"receive_same", "receive_royal"})


def _load_json(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def prioritized_pattern_ids(metrics: Mapping[str, object]) -> list[str]:
    """Return promising receive-versus-pass patterns in review order."""
    generic = dict(metrics.get("generic_patterns", {}) or {})
    ordered: list[str] = []
    seen = set()

    promising = []
    for raw in generic.get("human_root_pattern_details", []) or []:
        if not isinstance(raw, Mapping):
            continue
        if not bool(raw.get("candidate_scope_eligible", False)):
            continue
        human_better = max(0, int(raw.get("human_better", 0)))
        ai_better = max(0, int(raw.get("ai_better", 0)))
        value_delta = float(raw.get("average_value_delta", 0.0))
        if human_better <= ai_better or value_delta <= 0.0:
            continue
        promising.append((
            -max(0, int(raw.get("comparisons", 0))),
            -human_better,
            -value_delta,
            str(raw.get("pattern_id", "")),
        ))
    for _comparisons, _wins, _value, pattern_id in sorted(promising):
        if pattern_id and pattern_id not in seen:
            ordered.append(pattern_id)
            seen.add(pattern_id)

    mismatches = []
    for raw in generic.get("human_mismatch_details", []) or []:
        if not isinstance(raw, Mapping):
            continue
        if str(raw.get("recommended_action", "")) not in RECEIVE_ACTIONS:
            continue
        if str(raw.get("actual_action", "")) != "pass":
            continue
        mismatches.append((
            -max(0, int(raw.get("count", 0))),
            -max(0, int(raw.get("observations", 0))),
            str(raw.get("pattern_id", "")),
        ))
    for _count, _observations, pattern_id in sorted(mismatches):
        if pattern_id and pattern_id not in seen:
            ordered.append(pattern_id)
            seen.add(pattern_id)
    return ordered


def _round_robin_candidates(
    buckets: Mapping[str, Sequence[dict]],
    pattern_order: Iterable[str],
    *,
    per_pattern: int,
    max_evaluations: int,
) -> list[dict]:
    selected: list[dict] = []
    limit_per_pattern = max(1, int(per_pattern))
    total_limit = max(1, int(max_evaluations))
    for pattern_id in pattern_order:
        candidates = sorted(
            buckets.get(pattern_id, ()),
            key=lambda case: (
                0 if tuple(case.get("actual_action", ()))[0] == "pass" else 1,
                str(case.get("id", "")),
            ),
        )
        selected.extend(candidates[:limit_per_pattern])
        if len(selected) >= total_limit:
            return selected[:total_limit]
    return selected[:total_limit]


def _comparison_summary(snapshot: Mapping[str, object]) -> dict:
    keys = (
        "human_root_focus_considered",
        "human_root_focus_eligible",
        "human_root_focus_skipped",
        "human_root_comparisons",
        "human_root_completed",
        "human_root_incomplete",
        "human_root_incomplete_shallow",
        "human_root_human_better",
        "human_root_ai_better",
        "human_root_tied",
        "human_root_average_common_depth",
        "human_root_average_value_delta",
        "human_root_candidate_count",
    )
    return {key: snapshot.get(key, 0) for key in keys}


def _direct_route_comparison(
    agent,
    state,
    player: str,
    human_label: str,
    *,
    comparison_seconds: float,
) -> dict:
    """Compare a forced human receive with a forced AI pass on shared samples."""
    actions = list(state.legal_actions(player))
    human_actions = agent._generic_response_actions_for_label(
        state,
        actions,
        human_label,
    )
    ai_actions = agent._generic_response_actions_for_label(
        state,
        actions,
        "pass",
    )
    tracker = agent._track.get(id(state))
    if not tracker or not human_actions or not ai_actions:
        return {"completed": False, "reason": "actions_unavailable"}
    samples = agent._timed_search_sample_states(
        state,
        player,
        tracker,
        int(agent.TIME_SEARCH_SAMPLE_COUNT),
    )
    if not samples:
        return {"completed": False, "reason": "samples_unavailable"}

    human_baseline = max(
        human_actions,
        key=lambda action: agent._timed_search_rule_prior(state, player, action),
    )
    ai_baseline = ai_actions[0]
    human_context: Dict[str, object] = {}
    ai_context: Dict[str, object] = {}
    seconds = max(0.1, float(comparison_seconds))
    node_multiplier = float(
        agent.GENERIC_RESPONSE_HUMAN_ROOT_COMPARISON_NODE_MULTIPLIER
    )
    saved_suppression = getattr(
        agent,
        "_suppress_response_dictionary_metrics",
        False,
    )
    agent._suppress_response_dictionary_metrics = True
    try:
        ai_result = agent._time_limited_search_from_samples(
            state,
            player,
            list(ai_actions),
            ai_baseline,
            samples,
            cancel_event=None,
            tactical_priority_enabled=False,
            record_priority_metrics=False,
            run_context=ai_context,
            forced_priority_action=ai_baseline,
            max_seconds_override=seconds,
            min_depth_override=5,
            max_nodes_multiplier_override=node_multiplier,
            disable_stable_stop=True,
        )
        human_result = agent._time_limited_search_from_samples(
            state,
            player,
            list(human_actions),
            human_baseline,
            samples,
            cancel_event=None,
            tactical_priority_enabled=False,
            record_priority_metrics=False,
            run_context=human_context,
            forced_priority_action=human_baseline,
            max_seconds_override=seconds,
            min_depth_override=5,
            max_nodes_multiplier_override=node_multiplier,
            disable_stable_stop=True,
        )
    except Exception as error:
        return {
            "completed": False,
            "reason": "search_error",
            "error_type": type(error).__name__,
        }
    finally:
        agent._suppress_response_dictionary_metrics = saved_suppression

    human_depths = dict(human_context.get("depth_results", {}) or {})
    ai_depths = dict(ai_context.get("depth_results", {}) or {})
    common_depths = sorted(
        int(depth)
        for depth in set(human_depths) & set(ai_depths)
        if int(depth) >= 5
    )
    if human_result is None or ai_result is None or not common_depths:
        return {
            "completed": False,
            "reason": "common_depth_below_five",
            "human_stop_reason": str(human_context.get("stop_reason", "")),
            "ai_stop_reason": str(ai_context.get("stop_reason", "")),
        }
    depth = common_depths[-1]
    human = dict(human_depths[depth])
    ai = dict(ai_depths[depth])
    value_delta = float(human.get("evaluation_value", 0.0)) - float(
        ai.get("evaluation_value", 0.0)
    )
    return {
        "completed": True,
        "selected_side": (
            "human" if value_delta > 1e-6
            else "ai" if value_delta < -1e-6
            else "tied"
        ),
        "common_depth": depth,
        "value_delta": round(value_delta, 3),
        "human_terminal_loss_rate": round(
            float(human.get("terminal_loss_rate", 0.0)),
            5,
        ),
        "ai_terminal_loss_rate": round(
            float(ai.get("terminal_loss_rate", 0.0)),
            5,
        ),
        "human_elapsed_seconds": round(float(human_result.elapsed_seconds), 5),
        "ai_elapsed_seconds": round(float(ai_result.elapsed_seconds), 5),
        "human_nodes": int(human_result.nodes),
        "ai_nodes": int(ai_result.nodes),
    }


def _summarize_direct_comparisons(
    comparisons: Sequence[Mapping[str, object]],
) -> dict:
    completed = [item for item in comparisons if item.get("completed")]
    values = [float(item.get("value_delta", 0.0)) for item in completed]
    return {
        "attempted": len(comparisons),
        "completed": len(completed),
        "incomplete": len(comparisons) - len(completed),
        "human_better": sum(
            item.get("selected_side") == "human" for item in completed
        ),
        "ai_better": sum(
            item.get("selected_side") == "ai" for item in completed
        ),
        "tied": sum(item.get("selected_side") == "tied" for item in completed),
        "average_common_depth": round(
            sum(int(item.get("common_depth", 0)) for item in completed)
            / max(1, len(completed)),
            3,
        ),
        "average_value_delta": round(
            sum(values) / max(1, len(values)),
            3,
        ),
    }


def run_targeted_recheck(
    kifu_path: Path,
    *,
    metrics_path: Optional[Path] = None,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    max_evaluations: int = 30,
    per_pattern: int = 5,
    search_seconds: float = 1.0,
    comparison_seconds: float = 5.0,
    scan_only: bool = False,
) -> dict:
    metrics = _load_json(metrics_path) if metrics_path is not None else {}
    priority_ids = prioritized_pattern_ids(metrics)

    # Configure a fresh aggregate store before importing the current AI.  The
    # temporary checkpoint is discarded; only the sanitized report is kept.
    with tempfile.TemporaryDirectory(prefix="goita-human-recheck-") as temp_dir:
        os.environ["GOITA_GENERIC_RESPONSE_PATTERN_PATH"] = str(
            Path(temp_dir) / "aggregate.json"
        )
        from goita_ai2.current_ai.agent import RuleBasedAgent
        from goita_ai2.current_ai.generic_response_store import (
            generic_response_pattern_snapshot,
            reset_generic_response_patterns,
        )
        from goita_ai2.current_ai.human_response_dictionary import (
            human_response_dictionary,
        )
        from goita_ai2.human_kifu_dictionary_audit import (
            _response_action_label,
            _response_features,
        )
        from goita_ai2.kifu_validation import (
            iter_kifu_decisions,
            replay_validation_case,
        )

        reset_generic_response_patterns()
        dictionary = human_response_dictionary()
        priority_set = set(priority_ids)
        buckets: Dict[str, list[dict]] = defaultdict(list)
        scan = {
            "reconstructed_decisions": 0,
            "response_decisions": 0,
            "recorded_receive_decisions": 0,
            "recorded_pass_decisions": 0,
            "dictionary_receive_matches": 0,
            "priority_pattern_matches": 0,
        }

        for case in iter_kifu_decisions(Path(kifu_path)):
            scan["reconstructed_decisions"] += 1
            if case.get("position", {}).get("phase") != "receive":
                continue
            action = tuple(case.get("actual_action", ()))
            if len(action) != 3 or action[0] not in ("pass", "receive"):
                continue
            scan["response_decisions"] += 1
            scan[f"recorded_{action[0]}_decisions"] += 1
            state = replay_validation_case(case)
            recommendation = dictionary.recommendation(
                _response_features(case, state)
            )
            if (
                recommendation.get("status") != "recommended"
                or recommendation.get("recommended_action") not in RECEIVE_ACTIONS
            ):
                continue
            scan["dictionary_receive_matches"] += 1
            pattern_id = str(recommendation.get("pattern_key", ""))[:10]
            if priority_set and pattern_id not in priority_set:
                continue
            scan["priority_pattern_matches"] += 1
            buckets[pattern_id].append(case)

        if not priority_ids:
            priority_ids = sorted(
                buckets,
                key=lambda pattern_id: (-len(buckets[pattern_id]), pattern_id),
            )
        selected = _round_robin_candidates(
            buckets,
            priority_ids,
            per_pattern=per_pattern,
            max_evaluations=max_evaluations,
        )

        action_counts: Dict[str, int] = defaultdict(int)
        source_action_counts: Dict[str, int] = defaultdict(int)
        evaluated_patterns: Dict[str, int] = defaultdict(int)
        direct_comparisons: list[dict] = []
        direct_pattern_results: Dict[str, list[dict]] = defaultdict(list)
        if not scan_only:
            for index, case in enumerate(selected, start=1):
                agent = RuleBasedAgent(name=f"human-targeted-recheck-{index}")
                agent.TIME_SEARCH_BACKGROUND_ENABLED = False
                agent.TIME_SEARCH_CACHE_ENABLED = False
                agent.TIME_SEARCH_PREDICTION_CACHE_ENABLED = False
                agent.TIME_SEARCH_ADAPTIVE_BUDGET_ENABLED = False
                agent.TIME_SEARCH_MAX_SECONDS = max(0.05, float(search_seconds))
                agent.TIME_SEARCH_HARD_MAX_SECONDS = max(
                    1.0,
                    float(search_seconds) + (2.0 * float(comparison_seconds)) + 1.0,
                )
                agent.GENERIC_RESPONSE_NARROWING_ENABLED = False
                agent.GENERIC_RESPONSE_TACTICAL_PRIORITY_ENABLED = False
                agent.GENERIC_RESPONSE_TACTICAL_PAIRED_COMPARISON_ENABLED = False
                # The offline command compares every reproduced pass directly,
                # including passes selected by specialized rules.
                agent.GENERIC_RESPONSE_HUMAN_PAIRED_COMPARISON_ENABLED = False
                agent.GENERIC_RESPONSE_HUMAN_ROOT_COMPARISON_SECONDS = max(
                    0.1,
                    float(comparison_seconds),
                )
                state = replay_validation_case(case, agent)
                chosen = agent.select_action(
                    state,
                    str(case["player"]),
                    state.legal_actions(str(case["player"])),
                )
                chosen_label = _response_action_label(chosen, state.current_attack)
                action_counts[chosen_label] += 1
                source_action_counts[
                    _response_action_label(
                        tuple(case.get("actual_action", ())),
                        state.current_attack,
                    )
                ] += 1
                recommendation = dictionary.recommendation(
                    _response_features(case, state)
                )
                pattern_id = str(recommendation.get("pattern_key", ""))[:10]
                evaluated_patterns[pattern_id] += 1
                if chosen_label == "pass":
                    comparison = _direct_route_comparison(
                        agent,
                        state,
                        str(case["player"]),
                        str(recommendation.get("recommended_action", "")),
                        comparison_seconds=comparison_seconds,
                    )
                    direct_comparisons.append(comparison)
                    direct_pattern_results[pattern_id].append(comparison)
                print(
                    f"[{index}/{len(selected)}] {pattern_id}: {chosen_label}",
                    flush=True,
                )

        snapshot = generic_response_pattern_snapshot(detail_limit=None)
        relevant_details = [
            dict(item)
            for item in snapshot.get("human_root_pattern_details", []) or []
            if str(item.get("pattern_id", "")) in evaluated_patterns
        ]
        relevant_details.sort(
            key=lambda item: (
                -int(item.get("comparisons", 0)),
                str(item.get("pattern_id", "")),
            )
        )
        direct_patterns = []
        for pattern_id, comparisons in direct_pattern_results.items():
            summary = _summarize_direct_comparisons(comparisons)
            completed = max(0, int(summary["completed"]))
            human_better = max(0, int(summary["human_better"]))
            ai_better = max(0, int(summary["ai_better"]))
            human_win_rate = human_better / max(1, completed)
            candidate = bool(
                completed >= 5
                and human_better >= 3
                and human_win_rate >= 0.60
                and human_better > ai_better
                and float(summary["average_value_delta"]) > 0.0
            )
            direct_patterns.append({
                "pattern_id": pattern_id,
                **summary,
                "human_win_rate": round(human_win_rate, 5),
                "candidate": candidate,
            })
        direct_patterns.sort(
            key=lambda item: (-int(item["completed"]), item["pattern_id"])
        )
        report = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "purpose": "offline_human_receive_targeted_recheck",
            "live_ai_affected": False,
            "privacy": {
                "names_retained": False,
                "match_ids_retained": False,
                "hands_retained": False,
                "move_histories_retained": False,
            },
            "settings": {
                "metrics_priority_used": metrics_path is not None,
                "priority_pattern_count": len(priority_ids),
                "max_evaluations": max(1, int(max_evaluations)),
                "per_pattern": max(1, int(per_pattern)),
                "search_seconds": max(0.05, float(search_seconds)),
                "comparison_seconds": max(0.1, float(comparison_seconds)),
                "scan_only": bool(scan_only),
            },
            "scan_summary": {
                **scan,
                "matched_pattern_count": len(buckets),
                "selected_for_evaluation": len(selected),
            },
            "evaluation_summary": {
                "evaluated": 0 if scan_only else sum(action_counts.values()),
                "recorded_human_actions": dict(
                    sorted(source_action_counts.items())
                ),
                "current_ai_actions": dict(sorted(action_counts.items())),
                "evaluated_pattern_count": len(evaluated_patterns),
                "evaluated_patterns": dict(sorted(evaluated_patterns.items())),
            },
            "comparison_summary": _comparison_summary(snapshot),
            "pattern_results": relevant_details,
            "direct_comparison_summary": _summarize_direct_comparisons(
                direct_comparisons
            ),
            "direct_pattern_results": direct_patterns,
        }
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kifu", type=Path, default=DEFAULT_KIFU_PATH)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-evaluations", type=int, default=30)
    parser.add_argument("--per-pattern", type=int, default=5)
    parser.add_argument("--search-seconds", type=float, default=1.0)
    parser.add_argument("--comparison-seconds", type=float, default=5.0)
    parser.add_argument("--scan-only", action="store_true")
    args = parser.parse_args(argv)
    report = run_targeted_recheck(
        args.kifu,
        metrics_path=args.metrics,
        output_path=args.output,
        max_evaluations=args.max_evaluations,
        per_pattern=args.per_pattern,
        search_seconds=args.search_seconds,
        comparison_seconds=args.comparison_seconds,
        scan_only=args.scan_only,
    )
    print(json.dumps({
        "output": str(args.output),
        "scan_summary": report["scan_summary"],
        "evaluation_summary": report["evaluation_summary"],
        "comparison_summary": report["comparison_summary"],
        "direct_comparison_summary": report["direct_comparison_summary"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

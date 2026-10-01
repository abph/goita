"""Evaluate Current AI, neural, and hybrid choices on human-reviewed disagreements.

This dataset is deliberately small and biased toward reported mistakes.  Its
rates are regression signals for reviewed positions, not estimates of overall
playing strength.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from goita_ai2.neural_policy import NeuralPolicyModel, action_tuple


Action = tuple[str, Optional[str], Optional[str]]
DEFAULT_CASES = Path(
    "goita_ai2/experimental_ai2/data/reviewed_policy_evaluation.jsonl"
)
DEFAULT_MODEL = Path("goita_ai2/experimental_ai2/data/neural_policy.json")
DEFAULT_REPORT = Path(
    "private_data/neural_training/reviewed_policy_evaluation.json"
)
POLICIES = ("current", "neural", "hybrid")


def load_reviewed_cases(path: Path = DEFAULT_CASES) -> list[Dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    decision_ids: set[str] = set()
    for record in records:
        decision_id = str(record.get("decision_id", ""))
        if not decision_id or decision_id in decision_ids:
            raise ValueError("reviewed decision ids must be present and unique")
        decision_ids.add(decision_id)
        legal = {
            action_tuple(item) for item in record.get("legal_actions", [])
        }
        acceptable = {
            action_tuple(item)
            for item in record.get("human_acceptable_actions", [])
        }
        if not acceptable or not acceptable <= legal:
            raise ValueError(
                f"human acceptable actions must be legal: {decision_id}"
            )
        captured = dict(record.get("policy_at_review", {}) or {})
        for policy in POLICIES:
            action = action_tuple(captured.get(policy, {}))
            if action not in legal:
                raise ValueError(
                    f"captured {policy} action must be legal: {decision_id}"
                )
    return records


def _metric(matches: int, count: int) -> Dict[str, Any]:
    return {
        "matches": int(matches),
        "records": int(count),
        "match_rate": round(matches / count, 6) if count else None,
    }


def _captured_summary(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for policy in POLICIES:
        matches = sum(
            action_tuple(record["policy_at_review"][policy])
            in {
                action_tuple(item)
                for item in record["human_acceptable_actions"]
            }
            for record in records
        )
        result[policy] = _metric(matches, len(records))
    return result


def _grouped_summary(
    records: Sequence[Mapping[str, Any]],
    details_by_id: Mapping[str, Mapping[str, Any]],
    field: str,
) -> Dict[str, Any]:
    grouped: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get(field, "unknown"))].append(record)
    result: Dict[str, Any] = {}
    for key, items in sorted(grouped.items()):
        live_details = [
            details_by_id[str(item["decision_id"])] for item in items
        ]
        live = [
            bool(item["live_neural_matches"])
            for item in live_details
            if item["live_neural_matches"] is not None
        ]
        result[key] = {
            "records": len(items),
            "captured_at_review": _captured_summary(items),
            "live_neural": _metric(sum(live), len(live)),
        }
    return result


def evaluate_reviewed_cases(
    records: Sequence[Mapping[str, Any]],
    *,
    neural_model: Optional[NeuralPolicyModel] = None,
) -> Dict[str, Any]:
    """Return regression metrics for the fixed human-reviewed positions."""
    live_matches = 0
    live_count = 0
    details = []
    outcomes = Counter()
    for record in records:
        acceptable = {
            action_tuple(item)
            for item in record["human_acceptable_actions"]
        }
        captured = {
            policy: action_tuple(record["policy_at_review"][policy])
            for policy in POLICIES
        }
        current_match = captured["current"] in acceptable
        neural_match = captured["neural"] in acceptable
        if current_match and neural_match:
            outcomes["both_match"] += 1
        elif current_match:
            outcomes["current_only"] += 1
        elif neural_match:
            outcomes["neural_only"] += 1
        else:
            outcomes["neither"] += 1

        live_action = None
        live_match = None
        live_margin = None
        if neural_model is not None:
            legal = [
                action_tuple(item) for item in record.get("legal_actions", [])
            ]
            state = record.get("state")
            if legal and isinstance(state, Mapping):
                ranked = neural_model.rank_actions(state, legal)
                live_action = ranked[0][0]
                live_match = live_action in acceptable
                live_margin = (
                    round(float(ranked[0][1] - ranked[1][1]), 6)
                    if len(ranked) > 1
                    else None
                )
                live_count += 1
                live_matches += int(live_match)

        details.append({
            "decision_id": str(record["decision_id"]),
            "category": str(record.get("category", "unknown")),
            "training_usage": str(
                record.get("training_usage", "unknown")
            ),
            "captured_matches": {
                policy: captured[policy] in acceptable
                for policy in POLICIES
            },
            "live_neural_action": list(live_action) if live_action else None,
            "live_neural_matches": live_match,
            "live_neural_margin": live_margin,
        })

    details_by_id = {
        str(item["decision_id"]): item for item in details
    }
    return {
        "schema_version": 1,
        "scope": "human_reviewed_disagreement_regression_only",
        "warning": (
            "These cases were selected because a reported disagreement was "
            "reviewed; the rates do not measure overall playing strength."
        ),
        "records": len(records),
        "composition": dict(sorted(Counter(
            str(record.get("training_usage", "unknown"))
            for record in records
        ).items())),
        "captured_at_review": _captured_summary(records),
        "captured_disagreement_outcomes": {
            key: int(outcomes.get(key, 0))
            for key in ("current_only", "neural_only", "both_match", "neither")
        },
        "live_neural": _metric(live_matches, live_count),
        "by_category": _grouped_summary(
            records,
            details_by_id,
            "category",
        ),
        "by_training_usage": _grouped_summary(
            records,
            details_by_id,
            "training_usage",
        ),
        "details": details,
    }


def write_evaluation_report(
    cases_path: Path = DEFAULT_CASES,
    model_path: Optional[Path] = DEFAULT_MODEL,
    report_path: Path = DEFAULT_REPORT,
) -> Dict[str, Any]:
    records = load_reviewed_cases(cases_path)
    model = (
        NeuralPolicyModel.load(model_path)
        if model_path is not None and model_path.exists()
        else None
    )
    report = evaluate_reviewed_cases(records, neural_model=model)
    report["cases_path"] = str(cases_path)
    report["model_path"] = str(model_path) if model_path is not None else None
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = write_evaluation_report(args.cases, args.model, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

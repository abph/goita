"""Loads privacy-safe human response patterns for shadow comparison only."""

from __future__ import annotations

import json
import threading
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, Optional

from goita_ai2.current_ai.search_cache import _digest_payload


HUMAN_RESPONSE_DICTIONARY_SCHEMA_VERSION = 1
DEFAULT_HUMAN_RESPONSE_DICTIONARY_PATH = (
    Path(__file__).with_name("data") / "human-response-patterns.json"
)


def human_response_pattern_payload(
    tactical: Mapping[str, object],
) -> Dict[str, object]:
    """Return the shared anonymous shape available in kifu and live play."""
    attack_piece = str(tactical.get("attack_piece", "none"))
    return {
        "version": HUMAN_RESPONSE_DICTIONARY_SCHEMA_VERSION,
        "granularity": "human_response_shadow",
        "attacker_relation": str(
            tactical.get("attacker_relation", "none")
        ),
        "attack_piece": attack_piece,
        "attack_stage": str(tactical.get("attack_stage", "first")),
        "hand_stage": str(tactical.get("hand_stage", "middle")),
        "next_receiver_stage": str(
            tactical.get("next_receiver_stage", "middle")
        ),
        "same_piece": str(tactical.get("same_piece", "none")),
        "royal_receive": bool(tactical.get("royal_receive", False)),
        "followup_strength": str(
            tactical.get("followup_strength", "open")
        ),
        "reentry_width": str(tactical.get("reentry_width", "closed")),
        # Raw kifu cannot reproduce every live shi signal. Relation and piece
        # remain separate, while all shi-specific signal states share one key.
        "shi_context": "shi" if attack_piece == "1" else "not_shi",
        "score_pressure": str(tactical.get("score_pressure", "normal")),
    }


def build_human_response_dictionary(
    audit_report: Mapping[str, object],
) -> Dict[str, object]:
    """Collapse an audit report into a deterministic deployable dictionary."""
    summary = dict(audit_report.get("response_summary", {}) or {})
    thresholds = dict(summary.get("thresholds", {}) or {})
    min_observations = max(
        1,
        int(thresholds.get("minimum_observations", 15)),
    )
    min_matches = max(
        1,
        int(thresholds.get("minimum_distinct_matches", 8)),
    )
    min_dominance = max(
        0.0,
        min(1.0, float(thresholds.get("minimum_dominance", 0.70))),
    )
    merged: Dict[str, dict] = {}
    for raw in audit_report.get("patterns", []):
        if not isinstance(raw, Mapping):
            continue
        features = human_response_pattern_payload(
            dict(raw.get("features", {}) or {})
        )
        key = _digest_payload(features)
        entry = merged.setdefault(
            key,
            {
                "pattern_key": key,
                "features": features,
                "observations": 0,
                "distinct_matches_lower_bound": 0,
                "action_counts": Counter(),
            },
        )
        entry["observations"] += max(0, int(raw.get("observations", 0)))
        entry["distinct_matches_lower_bound"] = max(
            int(entry["distinct_matches_lower_bound"]),
            max(0, int(raw.get("distinct_matches", 0))),
        )
        entry["action_counts"].update({
            str(action): max(0, int(count))
            for action, count in dict(raw.get("action_counts", {}) or {}).items()
        })

    patterns = []
    for entry in merged.values():
        action_counts = Counter(entry["action_counts"])
        if not action_counts:
            continue
        recommended, support = sorted(
            action_counts.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )[0]
        observations = max(1, int(entry["observations"]))
        dominance = int(support) / observations
        if (
            observations < min_observations
            or int(entry["distinct_matches_lower_bound"]) < min_matches
            or dominance < min_dominance
        ):
            continue
        patterns.append({
            "pattern_key": str(entry["pattern_key"]),
            "features": dict(entry["features"]),
            "observations": observations,
            "distinct_matches_lower_bound": int(
                entry["distinct_matches_lower_bound"]
            ),
            "recommended_action": str(recommended),
            "support": int(support),
            "dominance": round(dominance, 5),
            "action_counts": dict(sorted(action_counts.items())),
        })
    patterns.sort(
        key=lambda item: (
            -int(item["distinct_matches_lower_bound"]),
            -int(item["observations"]),
            str(item["pattern_key"]),
        )
    )
    return {
        "schema_version": HUMAN_RESPONSE_DICTIONARY_SCHEMA_VERSION,
        "purpose": "human_response_shadow_comparison",
        "live_ai_affected": False,
        "privacy": {
            "names_retained": False,
            "timestamps_retained": False,
            "match_ids_retained": False,
            "hands_retained": False,
            "move_histories_retained": False,
        },
        "thresholds": {
            "minimum_observations": min_observations,
            "minimum_distinct_matches": min_matches,
            "minimum_dominance": min_dominance,
        },
        "source_pattern_count": len(
            list(audit_report.get("patterns", []))
        ),
        "merged_pattern_count": len(merged),
        "pattern_count": len(patterns),
        "patterns": patterns,
    }


class HumanResponseDictionary:
    """Read-only aggregate dictionary; it never selects a live action."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path or DEFAULT_HUMAN_RESPONSE_DICTIONARY_PATH)
        self._patterns: Dict[str, dict] = {}
        self._load_error = ""
        self._load()

    def _load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if (
                not isinstance(payload, dict)
                or int(payload.get("schema_version", 0))
                != HUMAN_RESPONSE_DICTIONARY_SCHEMA_VERSION
            ):
                raise ValueError("unsupported human response dictionary")
            self._patterns = {
                str(raw["pattern_key"]): dict(raw)
                for raw in payload.get("patterns", [])
                if isinstance(raw, dict) and raw.get("pattern_key")
            }
            self._load_error = ""
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
            self._patterns = {}
            self._load_error = str(error)

    def recommendation(self, tactical: Mapping[str, object]) -> dict:
        features = human_response_pattern_payload(tactical)
        raw = self._patterns.get(_digest_payload(features))
        if raw is None:
            return {"status": "no_pattern"}
        return {
            "status": "recommended",
            "recommended_action": str(raw.get("recommended_action", "other")),
            "observations": max(0, int(raw.get("observations", 0))),
            "support": max(0, int(raw.get("support", 0))),
            "dominance": max(
                0.0,
                min(1.0, float(raw.get("dominance", 0.0))),
            ),
            "distinct_matches_lower_bound": max(
                0,
                int(raw.get("distinct_matches_lower_bound", 0)),
            ),
        }

    def snapshot(self) -> dict:
        return {
            "human_dictionary_loaded": not bool(self._load_error),
            "human_dictionary_patterns": len(self._patterns),
            "human_dictionary_error": self._load_error,
            "human_dictionary_live_ai_affected": False,
        }


_HUMAN_RESPONSE_DICTIONARY = HumanResponseDictionary()
_HUMAN_RESPONSE_DICTIONARY_LOCK = threading.Lock()


def human_response_dictionary() -> HumanResponseDictionary:
    return _HUMAN_RESPONSE_DICTIONARY


def reload_human_response_dictionary(
    path: Optional[Path] = None,
) -> HumanResponseDictionary:
    global _HUMAN_RESPONSE_DICTIONARY
    with _HUMAN_RESPONSE_DICTIONARY_LOCK:
        _HUMAN_RESPONSE_DICTIONARY = HumanResponseDictionary(path)
        return _HUMAN_RESPONSE_DICTIONARY

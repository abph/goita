"""Aggregates generalized response-pattern evidence across rounds.
Only anonymous tactical features and search-quality totals are retained, and
optional atomic checkpoints keep the aggregate on a persistent Render disk.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, Optional

from goita_ai2.current_ai.search_cache import _digest_payload


GENERIC_RESPONSE_STORE_SCHEMA_VERSION = 1
GENERIC_RESPONSE_STORE_FILENAME = "generic-response-patterns.json"
TACTICAL_MISMATCH_DETAIL_LIMIT = 250
HUMAN_MISMATCH_DETAIL_LIMIT = 250
_LOGGER = logging.getLogger(__name__)


def medium_response_pattern_payload(
    detailed: Mapping[str, object],
) -> Dict[str, object]:
    """Collapse a detailed public pattern into a reusable tactical class."""
    context = dict(detailed.get("context", {}) or {})
    hand = dict(detailed.get("hand", {}) or {})
    after = dict(detailed.get("after_same_receive", {}) or {})
    followup = dict(after.get("followup", {}) or {})
    receive_modes = tuple(
        sorted(str(item) for item in detailed.get("legal", {}).get(
            "receive_modes",
            (),
        ))
    )

    def present(value) -> bool:
        return str(value or "0") != "0"

    same_piece = str(hand.get("same_piece", "0"))
    attack_count = str(context.get("attacker_attack_count", "0"))
    if present(followup.get("fourth_followups")):
        followup_strength = "fourth"
    elif present(followup.get("scarce_followups")):
        followup_strength = "scarce"
    elif present(followup.get("pair_followups")):
        followup_strength = "pair"
    else:
        followup_strength = "open"

    return {
        "version": 1,
        "granularity": "medium",
        "attacker_relation": str(context.get("attacker_relation", "none")),
        "attack_family": str(context.get("attack_family", "none")),
        "attack_stage": "first" if attack_count in ("0", "1") else "later",
        "hand_stage": str(hand.get("size", "middle")),
        "same_piece": "multiple" if same_piece in ("2", "3+") else "one",
        "royal_receive": "royal" in receive_modes,
        "followup_strength": followup_strength,
    }


def tactical_response_pattern_payload(
    detailed: Mapping[str, object],
) -> Dict[str, object]:
    """Keep tactically meaningful detail without retaining an exact position."""
    context = dict(detailed.get("context", {}) or {})
    legal = dict(detailed.get("legal", {}) or {})
    hand = dict(detailed.get("hand", {}) or {})
    after = dict(detailed.get("after_same_receive", {}) or {})
    followup = dict(after.get("followup", {}) or {})
    signals = dict(detailed.get("signals", {}) or {})
    score = dict(detailed.get("score", {}) or {})
    receive_modes = {
        str(item) for item in legal.get("receive_modes", ())
    }

    same_piece = str(hand.get("same_piece", "0"))
    if same_piece == "0":
        same_piece_class = "none"
    elif same_piece == "1":
        same_piece_class = "one"
    else:
        same_piece_class = "multiple"

    attack_count = str(context.get("attacker_attack_count", "0"))
    if str(followup.get("fourth_followups", "0")) != "0":
        followup_strength = "fourth"
    elif str(followup.get("scarce_followups", "0")) != "0":
        followup_strength = "scarce"
    elif str(followup.get("pair_followups", "0")) != "0":
        followup_strength = "pair"
    else:
        followup_strength = "open"

    reentry = str(followup.get("reentry_width", "0"))
    if reentry == "0":
        reentry_width = "closed"
    elif reentry == "1":
        reentry_width = "narrow"
    else:
        reentry_width = "wide"

    attack_family = str(context.get("attack_family", "none"))
    if attack_family != "shi":
        shi_context = "not_shi"
    elif bool(signals.get("shi_attack_mode", False)) or str(
        signals.get("ally_shi", "unknown")
    ) in ("returned_shi", "sashikomi"):
        shi_context = "continue"
    elif str(signals.get("ally_shi", "unknown")) == "weak":
        shi_context = "stop"
    elif bool(signals.get("enemy_rejected_shi", False)):
        shi_context = "enemy_rejected"
    else:
        shi_context = "neutral"

    own_near = str(score.get("own_match_point", "normal")) == "near"
    enemy_near = str(score.get("enemy_match_point", "normal")) == "near"
    if own_near and enemy_near:
        score_pressure = "both_near"
    elif own_near:
        score_pressure = "own_near"
    elif enemy_near:
        score_pressure = "enemy_near"
    else:
        score_pressure = "normal"

    return {
        "version": 1,
        "granularity": "tactical",
        "attacker_relation": str(context.get("attacker_relation", "none")),
        "attack_piece": str(context.get("attack_piece", "none")),
        "attack_stage": "first" if attack_count in ("0", "1") else "later",
        "hand_stage": str(hand.get("size", "middle")),
        "next_receiver_stage": str(
            context.get("next_receiver_hand", "middle")
        ),
        "same_piece": same_piece_class,
        "royal_receive": "royal" in receive_modes,
        "followup_strength": followup_strength,
        "reentry_width": reentry_width,
        "shi_context": shi_context,
        "score_pressure": score_pressure,
    }


def resolve_generic_response_store_path(
    environ: Mapping[str, str],
) -> Optional[Path]:
    explicit = str(
        environ.get("GOITA_GENERIC_RESPONSE_PATTERN_PATH", "") or ""
    ).strip()
    if explicit:
        return Path(explicit)
    persistent_directory = str(
        environ.get("GOITA_PERSISTENT_DATA_DIR", "") or ""
    ).strip()
    if persistent_directory:
        return (
            Path(persistent_directory)
            / "goita-ai"
            / GENERIC_RESPONSE_STORE_FILENAME
        )
    return None


class GenericResponsePatternStore:
    """Thread-safe aggregate store with optional JSON persistence."""

    def __init__(
        self,
        *,
        path: Optional[Path] = None,
        max_patterns: int = 5_000,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.max_patterns = max(100, int(max_patterns))
        self._lock = threading.RLock()
        self._checkpoint_lock = threading.Lock()
        self._started_at = time.time()
        self._patterns: Dict[str, dict] = {}
        self._tactical_patterns: Dict[str, dict] = {}
        self._medium_patterns: Dict[str, dict] = {}
        self._tactical_mismatch_details: Dict[str, dict] = {}
        self._human_mismatch_details: Dict[str, dict] = {}
        self._counters = self._empty_counters()
        self._last_checkpoint_error = ""
        self._restore()

    @staticmethod
    def _empty_counters() -> Dict[str, object]:
        return {
            "considered": 0,
            "recorded": 0,
            "evicted_patterns": 0,
            "shadow_lookups": 0,
            "shadow_no_pattern": 0,
            "shadow_insufficient": 0,
            "shadow_ambiguous": 0,
            "shadow_recommendations": 0,
            "shadow_matches": 0,
            "shadow_mismatches": 0,
            "tactical_shadow_lookups": 0,
            "tactical_shadow_no_pattern": 0,
            "tactical_shadow_insufficient": 0,
            "tactical_shadow_ambiguous": 0,
            "tactical_shadow_recommendations": 0,
            "tactical_shadow_matches": 0,
            "tactical_shadow_mismatches": 0,
            "human_shadow_lookups": 0,
            "human_shadow_no_pattern": 0,
            "human_shadow_recommendations": 0,
            "human_shadow_matches": 0,
            "human_shadow_mismatches": 0,
            "human_pair_comparisons": 0,
            "human_pair_completed": 0,
            "human_pair_incomplete": 0,
            "human_pair_human_selected": 0,
            "human_pair_ai_selected": 0,
            "human_pair_other_selected": 0,
            "human_pair_human_value_better": 0,
            "human_pair_ai_value_better": 0,
            "human_pair_value_tied": 0,
            "human_pair_normal_depth_sum": 0.0,
            "human_pair_priority_depth_sum": 0.0,
            "human_pair_normal_elapsed_sum": 0.0,
            "human_pair_priority_elapsed_sum": 0.0,
            "human_pair_normal_nodes_sum": 0,
            "human_pair_priority_nodes_sum": 0,
            "human_pair_value_delta_sum": 0.0,
            "tactical_priority_lookups": 0,
            "tactical_priority_no_recommendation": 0,
            "tactical_priority_rejected_action": 0,
            "tactical_priority_rejected_context": 0,
            "tactical_priority_no_legal_followup": 0,
            "tactical_priority_offered": 0,
            "tactical_priority_effects": 0,
            "tactical_priority_reordered": 0,
            "tactical_priority_baseline_disagreements": 0,
            "tactical_priority_selected": 0,
            "tactical_priority_depth_sum": 0,
            "tactical_pair_comparisons": 0,
            "tactical_pair_completed": 0,
            "tactical_pair_incomplete": 0,
            "tactical_pair_action_matches": 0,
            "tactical_pair_action_changes": 0,
            "tactical_pair_with_priority_selected": 0,
            "tactical_pair_without_priority_selected": 0,
            "tactical_pair_with_depth_sum": 0.0,
            "tactical_pair_without_depth_sum": 0.0,
            "tactical_pair_with_elapsed_sum": 0.0,
            "tactical_pair_without_elapsed_sum": 0.0,
            "tactical_pair_with_nodes_sum": 0,
            "tactical_pair_without_nodes_sum": 0,
            "tactical_pair_value_delta_sum": 0.0,
            "tactical_pair_margin_delta_sum": 0.0,
            "priority_queries": 0,
            "priority_hits": 0,
            "priority_effect_comparisons": 0,
            "priority_effect_exact": 0,
            "priority_effect_incomplete": 0,
            "priority_effect_reorders": 0,
            "priority_effect_beam_preserved": 0,
            "priority_effect_selected": 0,
            "priority_effect_changed": 0,
            "priority_effect_unchanged": 0,
            "priority_effect_with_depth_sum": 0.0,
            "priority_effect_without_depth_sum": 0.0,
            "priority_effect_with_elapsed_sum": 0.0,
            "priority_effect_without_elapsed_sum": 0.0,
            "priority_effect_saved_seconds_sum": 0.0,
            "priority_effect_value_delta_sum": 0.0,
            "narrowing_shadow_considered": 0,
            "narrowing_shadow_insufficient_depth": 0,
            "narrowing_shadow_no_reduction": 0,
            "narrowing_shadow_incomplete": 0,
            "narrowing_shadow_comparisons": 0,
            "narrowing_shadow_matches": 0,
            "narrowing_shadow_mismatches": 0,
            "narrowing_shadow_priority_selected": 0,
            "narrowing_shadow_full_candidates_sum": 0.0,
            "narrowing_shadow_kept_candidates_sum": 0.0,
            "narrowing_shadow_removed_candidates_sum": 0.0,
            "narrowing_shadow_depth_sum": 0.0,
            "narrowing_shadow_actual_elapsed_sum": 0.0,
            "narrowing_shadow_estimated_elapsed_sum": 0.0,
            "narrowing_shadow_estimated_saved_sum": 0.0,
            "narrowing_shadow_value_loss_sum": 0.0,
            "active_narrowing_considered": 0,
            "active_narrowing_applied": 0,
            "active_narrowing_insufficient_depth": 0,
            "active_narrowing_no_reduction": 0,
            "active_narrowing_safety_rejected": 0,
            "active_narrowing_rejected_specialized_profile": 0,
            "active_narrowing_rejected_information_unavailable": 0,
            "active_narrowing_rejected_low_confidence": 0,
            "active_narrowing_rejected_priority_not_top_two": 0,
            "active_narrowing_rejected_insufficient_margin": 0,
            "active_narrowing_incomplete": 0,
            "active_narrowing_deepened": 0,
            "active_narrowing_no_deepening": 0,
            "active_narrowing_no_deepening_time_limit": 0,
            "active_narrowing_no_deepening_node_limit": 0,
            "active_narrowing_no_deepening_other": 0,
            "active_narrowing_priority_selected": 0,
            "active_narrowing_continuation_extensions": 0,
            "active_narrowing_node_extensions": 0,
            "active_narrowing_added_nodes_sum": 0,
            "active_narrowing_full_candidates_sum": 0.0,
            "active_narrowing_kept_candidates_sum": 0.0,
            "active_narrowing_depth_sum": 0.0,
            "active_narrowing_elapsed_sum": 0.0,
            "active_narrowing_continuation_extension_seconds_sum": 0.0,
            "action_counts": Counter(),
            "followup_counts": Counter(),
            "source_counts": Counter(),
            "rejection_counts": Counter(),
            "shadow_recommended_actions": Counter(),
            "shadow_actual_actions": Counter(),
            "shadow_match_actions": Counter(),
            "shadow_granularity_counts": Counter(),
            "priority_action_counts": Counter(),
            "priority_granularity_counts": Counter(),
            "depth_sum": 0.0,
            "agreement_sum": 0.0,
            "confidence_sum": 0.0,
            "margin_sum": 0.0,
        }

    def reset(self) -> None:
        with self._lock:
            self._started_at = time.time()
            self._patterns = {}
            self._tactical_patterns = {}
            self._medium_patterns = {}
            self._tactical_mismatch_details = {}
            self._human_mismatch_details = {}
            self._counters = self._empty_counters()
            self._last_checkpoint_error = ""

    def reject(self, reason: str) -> None:
        with self._lock:
            self._counters["considered"] += 1
            self._counters["rejection_counts"][str(reason or "other")] += 1

    def _record_pattern_locked(
        self,
        table: Dict[str, dict],
        *,
        pattern_key: str,
        features: Mapping[str, object],
        action_label: str,
        followup_label: str,
        source: str,
        depth: int,
        agreement: float,
        confidence: float,
        margin: float,
        now: float,
    ) -> None:
        key = str(pattern_key)
        entry = table.get(key)
        if entry is None:
            if len(table) >= self.max_patterns:
                evicted_key = min(
                    table,
                    key=lambda existing_key: (
                        int(table[existing_key].get("observations", 0)),
                        float(table[existing_key].get("last_seen_at", 0.0)),
                    ),
                )
                table.pop(evicted_key, None)
                self._counters["evicted_patterns"] += 1
            entry = {
                "features": dict(features),
                "observations": 0,
                "action_counts": Counter(),
                "followup_counts": Counter(),
                "source_counts": Counter(),
                "routes": {},
                "depth_sum": 0.0,
                "agreement_sum": 0.0,
                "confidence_sum": 0.0,
                "margin_sum": 0.0,
                "first_seen_at": now,
                "last_seen_at": now,
            }
            table[key] = entry

        entry["observations"] += 1
        entry["action_counts"][str(action_label)] += 1
        entry["followup_counts"][str(followup_label)] += 1
        entry["source_counts"][str(source)] += 1
        entry["depth_sum"] += max(0, int(depth))
        entry["agreement_sum"] += max(0.0, min(1.0, float(agreement)))
        entry["confidence_sum"] += max(0.0, min(1.0, float(confidence)))
        entry["margin_sum"] += float(margin)
        entry["last_seen_at"] = now

        route_key = f"{action_label}|{followup_label}"
        route = entry["routes"].get(route_key)
        if route is None:
            route = {
                "action": str(action_label),
                "followup": str(followup_label),
                "observations": 0,
                "source_counts": Counter(),
                "depth_sum": 0.0,
                "agreement_sum": 0.0,
                "confidence_sum": 0.0,
                "margin_sum": 0.0,
            }
            entry["routes"][route_key] = route
        route["observations"] += 1
        route["source_counts"][str(source)] += 1
        route["depth_sum"] += max(0, int(depth))
        route["agreement_sum"] += max(0.0, min(1.0, float(agreement)))
        route["confidence_sum"] += max(
            0.0,
            min(1.0, float(confidence)),
        )
        route["margin_sum"] += float(margin)

    def record(
        self,
        *,
        pattern_key: str,
        features: Mapping[str, object],
        action_label: str,
        followup_label: str,
        source: str,
        depth: int,
        agreement: float,
        confidence: float,
        margin: float,
    ) -> None:
        now = time.time()
        with self._lock:
            self._counters["considered"] += 1
            self._counters["recorded"] += 1
            self._counters["action_counts"][str(action_label)] += 1
            self._counters["followup_counts"][str(followup_label)] += 1
            self._counters["source_counts"][str(source)] += 1
            self._counters["depth_sum"] += max(0, int(depth))
            self._counters["agreement_sum"] += max(
                0.0, min(1.0, float(agreement))
            )
            self._counters["confidence_sum"] += max(
                0.0, min(1.0, float(confidence))
            )
            self._counters["margin_sum"] += float(margin)

            self._record_pattern_locked(
                self._patterns,
                pattern_key=str(pattern_key),
                features=features,
                action_label=action_label,
                followup_label=followup_label,
                source=source,
                depth=depth,
                agreement=agreement,
                confidence=confidence,
                margin=margin,
                now=now,
            )
            tactical_features = tactical_response_pattern_payload(features)
            self._record_pattern_locked(
                self._tactical_patterns,
                pattern_key=_digest_payload(tactical_features),
                features=tactical_features,
                action_label=action_label,
                followup_label=followup_label,
                source=source,
                depth=depth,
                agreement=agreement,
                confidence=confidence,
                margin=margin,
                now=now,
            )
            medium_features = medium_response_pattern_payload(features)
            self._record_pattern_locked(
                self._medium_patterns,
                pattern_key=_digest_payload(medium_features),
                features=medium_features,
                action_label=action_label,
                followup_label=followup_label,
                source=source,
                depth=depth,
                agreement=agreement,
                confidence=confidence,
                margin=margin,
                now=now,
            )

    @staticmethod
    def _counter_dict(value) -> Dict[str, int]:
        if not isinstance(value, Mapping):
            return {}
        return {
            str(key): max(0, int(count))
            for key, count in value.items()
        }

    def _merge_pattern_entry_locked(
        self,
        table: Dict[str, dict],
        *,
        pattern_key: str,
        features: Mapping[str, object],
        raw: Mapping[str, object],
    ) -> None:
        key = str(pattern_key)
        entry = table.get(key)
        if entry is None:
            entry = {
                "features": dict(features),
                "observations": 0,
                "action_counts": Counter(),
                "followup_counts": Counter(),
                "source_counts": Counter(),
                "routes": {},
                "depth_sum": 0.0,
                "agreement_sum": 0.0,
                "confidence_sum": 0.0,
                "margin_sum": 0.0,
                "first_seen_at": float(raw.get("first_seen_at", 0.0)),
                "last_seen_at": float(raw.get("last_seen_at", 0.0)),
            }
            table[key] = entry

        entry["observations"] += max(0, int(raw.get("observations", 0)))
        entry["action_counts"].update(raw.get("action_counts", {}))
        entry["followup_counts"].update(raw.get("followup_counts", {}))
        entry["source_counts"].update(raw.get("source_counts", {}))
        for name in (
            "depth_sum",
            "agreement_sum",
            "confidence_sum",
            "margin_sum",
        ):
            entry[name] += float(raw.get(name, 0.0))
        first_seen = float(raw.get("first_seen_at", 0.0))
        last_seen = float(raw.get("last_seen_at", 0.0))
        if entry["first_seen_at"] <= 0.0 or (
            first_seen > 0.0 and first_seen < entry["first_seen_at"]
        ):
            entry["first_seen_at"] = first_seen
        entry["last_seen_at"] = max(entry["last_seen_at"], last_seen)

        for route_key, raw_route in raw.get("routes", {}).items():
            route = entry["routes"].get(route_key)
            if route is None:
                route = {
                    "action": str(raw_route.get("action", "other")),
                    "followup": str(raw_route.get("followup", "none")),
                    "observations": 0,
                    "source_counts": Counter(),
                    "depth_sum": 0.0,
                    "agreement_sum": 0.0,
                    "confidence_sum": 0.0,
                    "margin_sum": 0.0,
                }
                entry["routes"][str(route_key)] = route
            route["observations"] += max(
                0,
                int(raw_route.get("observations", 0)),
            )
            route["source_counts"].update(
                raw_route.get("source_counts", {})
            )
            for name in (
                "depth_sum",
                "agreement_sum",
                "confidence_sum",
                "margin_sum",
            ):
                route[name] += float(raw_route.get(name, 0.0))

    def _rebuild_medium_patterns_locked(self) -> None:
        self._medium_patterns = {}
        for raw in self._patterns.values():
            detailed_features = dict(raw.get("features", {}) or {})
            medium_features = medium_response_pattern_payload(detailed_features)
            self._merge_pattern_entry_locked(
                self._medium_patterns,
                pattern_key=_digest_payload(medium_features),
                features=medium_features,
                raw=raw,
            )

    def _rebuild_tactical_patterns_locked(self) -> None:
        self._tactical_patterns = {}
        for raw in self._patterns.values():
            detailed_features = dict(raw.get("features", {}) or {})
            tactical_features = tactical_response_pattern_payload(
                detailed_features
            )
            self._merge_pattern_entry_locked(
                self._tactical_patterns,
                pattern_key=_digest_payload(tactical_features),
                features=tactical_features,
                raw=raw,
            )

    def _serializable_locked(self) -> dict:
        def serialize_patterns(table: Dict[str, dict]) -> dict:
            patterns = {}
            for key, raw in table.items():
                routes = {
                    route_key: {
                        **route,
                        "source_counts": dict(route["source_counts"]),
                    }
                    for route_key, route in raw.get("routes", {}).items()
                }
                patterns[key] = {
                    **raw,
                    "action_counts": dict(raw["action_counts"]),
                    "followup_counts": dict(raw["followup_counts"]),
                    "source_counts": dict(raw["source_counts"]),
                    "routes": routes,
                }
            return patterns

        return {
            "schema_version": GENERIC_RESPONSE_STORE_SCHEMA_VERSION,
            "saved_at": time.time(),
            "started_at": self._started_at,
            "max_patterns": self.max_patterns,
            "counters": {
                **self._counters,
                "action_counts": dict(self._counters["action_counts"]),
                "followup_counts": dict(self._counters["followup_counts"]),
                "source_counts": dict(self._counters["source_counts"]),
                "rejection_counts": dict(self._counters["rejection_counts"]),
                "shadow_granularity_counts": dict(
                    self._counters["shadow_granularity_counts"]
                ),
                "priority_granularity_counts": dict(
                    self._counters["priority_granularity_counts"]
                ),
            },
            "patterns": serialize_patterns(self._patterns),
            "tactical_patterns": serialize_patterns(self._tactical_patterns),
            "medium_patterns": serialize_patterns(self._medium_patterns),
            "tactical_mismatch_details": self._tactical_mismatch_details,
            "human_mismatch_details": self._human_mismatch_details,
        }

    def checkpoint(self, reason: str = "manual") -> bool:
        if self.path is None:
            return False
        with self._lock:
            payload = self._serializable_locked()
        payload["reason"] = str(reason or "manual")
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        try:
            with self._checkpoint_lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                    json.dump(
                        payload,
                        handle,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
            with self._lock:
                self._last_checkpoint_error = ""
            return True
        except (OSError, TypeError, ValueError) as error:
            with self._lock:
                self._last_checkpoint_error = str(error)
            _LOGGER.error(
                "Unable to save generic response patterns to %s: %s",
                self.path,
                error,
            )
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            return False

    def _restore(self) -> bool:
        if self.path is None or not self.path.is_file():
            return False
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            self._last_checkpoint_error = str(error)
            return False
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != GENERIC_RESPONSE_STORE_SCHEMA_VERSION
        ):
            return False

        counters = payload.get("counters", {})
        patterns = payload.get("patterns", {})
        mismatch_details = payload.get("tactical_mismatch_details", {})
        human_mismatch_details = payload.get("human_mismatch_details", {})
        if not isinstance(counters, dict) or not isinstance(patterns, dict):
            return False
        if not isinstance(mismatch_details, dict):
            mismatch_details = {}
        if not isinstance(human_mismatch_details, dict):
            human_mismatch_details = {}
        restored_patterns = {}
        restored_mismatch_details = {}
        restored_human_mismatch_details = {}
        try:
            for key, raw in patterns.items():
                if not isinstance(key, str) or not isinstance(raw, dict):
                    continue
                restored_routes = {}
                for route_key, route in raw.get("routes", {}).items():
                    if not isinstance(route_key, str) or not isinstance(route, dict):
                        continue
                    restored_routes[route_key] = {
                        "action": str(route.get("action", "other")),
                        "followup": str(route.get("followup", "none")),
                        "observations": max(
                            0,
                            int(route.get("observations", 0)),
                        ),
                        "source_counts": Counter(
                            self._counter_dict(route.get("source_counts", {}))
                        ),
                        "depth_sum": float(route.get("depth_sum", 0.0)),
                        "agreement_sum": float(
                            route.get("agreement_sum", 0.0)
                        ),
                        "confidence_sum": float(
                            route.get("confidence_sum", 0.0)
                        ),
                        "margin_sum": float(route.get("margin_sum", 0.0)),
                    }
                restored_patterns[key] = {
                    "features": dict(raw.get("features", {})),
                    "observations": max(0, int(raw.get("observations", 0))),
                    "action_counts": Counter(
                        self._counter_dict(raw.get("action_counts", {}))
                    ),
                    "followup_counts": Counter(
                        self._counter_dict(raw.get("followup_counts", {}))
                    ),
                    "source_counts": Counter(
                        self._counter_dict(raw.get("source_counts", {}))
                    ),
                    "routes": restored_routes,
                    "depth_sum": float(raw.get("depth_sum", 0.0)),
                    "agreement_sum": float(raw.get("agreement_sum", 0.0)),
                    "confidence_sum": float(raw.get("confidence_sum", 0.0)),
                    "margin_sum": float(raw.get("margin_sum", 0.0)),
                    "first_seen_at": float(raw.get("first_seen_at", 0.0)),
                    "last_seen_at": float(raw.get("last_seen_at", 0.0)),
                }
            for key, raw in mismatch_details.items():
                if not isinstance(key, str) or not isinstance(raw, dict):
                    continue
                pattern_key = str(raw.get("pattern_key", ""))
                recommended_action = str(
                    raw.get("recommended_action", "other")
                )
                actual_action = str(raw.get("actual_action", "other"))
                if not pattern_key:
                    continue
                restored_mismatch_details[key] = {
                    "pattern_key": pattern_key,
                    "recommended_action": recommended_action,
                    "actual_action": actual_action,
                    "count": max(0, int(raw.get("count", 0))),
                    "observations": max(
                        0,
                        int(raw.get("observations", 0)),
                    ),
                    "support": max(0, int(raw.get("support", 0))),
                    "dominance": max(
                        0.0,
                        min(1.0, float(raw.get("dominance", 0.0))),
                    ),
                    "first_seen_at": float(raw.get("first_seen_at", 0.0)),
                    "last_seen_at": float(raw.get("last_seen_at", 0.0)),
                }
            for key, raw in human_mismatch_details.items():
                if not isinstance(key, str) or not isinstance(raw, dict):
                    continue
                pattern_key = str(raw.get("pattern_key", ""))
                if not pattern_key:
                    continue
                restored_human_mismatch_details[key] = {
                    "pattern_key": pattern_key,
                    "anonymous_context": dict(
                        raw.get("anonymous_context", {}) or {}
                    ),
                    "recommended_action": str(
                        raw.get("recommended_action", "other")
                    ),
                    "actual_action": str(raw.get("actual_action", "other")),
                    "count": max(0, int(raw.get("count", 0))),
                    "observations": max(
                        0,
                        int(raw.get("observations", 0)),
                    ),
                    "support": max(0, int(raw.get("support", 0))),
                    "dominance": max(
                        0.0,
                        min(1.0, float(raw.get("dominance", 0.0))),
                    ),
                    "distinct_matches_lower_bound": max(
                        0,
                        int(raw.get("distinct_matches_lower_bound", 0)),
                    ),
                    "first_seen_at": float(raw.get("first_seen_at", 0.0)),
                    "last_seen_at": float(raw.get("last_seen_at", 0.0)),
                }
            restored_counters = self._empty_counters()
            for name in (
                "considered",
                "recorded",
                "evicted_patterns",
                "shadow_lookups",
                "shadow_no_pattern",
                "shadow_insufficient",
                "shadow_ambiguous",
                "shadow_recommendations",
                "shadow_matches",
                "shadow_mismatches",
                "tactical_shadow_lookups",
                "tactical_shadow_no_pattern",
                "tactical_shadow_insufficient",
                "tactical_shadow_ambiguous",
                "tactical_shadow_recommendations",
                "tactical_shadow_matches",
                "tactical_shadow_mismatches",
                "human_shadow_lookups",
                "human_shadow_no_pattern",
                "human_shadow_recommendations",
                "human_shadow_matches",
                "human_shadow_mismatches",
                "human_pair_comparisons",
                "human_pair_completed",
                "human_pair_incomplete",
                "human_pair_human_selected",
                "human_pair_ai_selected",
                "human_pair_other_selected",
                "human_pair_human_value_better",
                "human_pair_ai_value_better",
                "human_pair_value_tied",
                "human_pair_normal_nodes_sum",
                "human_pair_priority_nodes_sum",
                "tactical_priority_lookups",
                "tactical_priority_no_recommendation",
                "tactical_priority_rejected_action",
                "tactical_priority_rejected_context",
                "tactical_priority_no_legal_followup",
                "tactical_priority_offered",
                "tactical_priority_effects",
                "tactical_priority_reordered",
                "tactical_priority_baseline_disagreements",
                "tactical_priority_selected",
                "tactical_priority_depth_sum",
                "tactical_pair_comparisons",
                "tactical_pair_completed",
                "tactical_pair_incomplete",
                "tactical_pair_action_matches",
                "tactical_pair_action_changes",
                "tactical_pair_with_priority_selected",
                "tactical_pair_without_priority_selected",
                "tactical_pair_with_nodes_sum",
                "tactical_pair_without_nodes_sum",
                "priority_queries",
                "priority_hits",
                "priority_effect_comparisons",
                "priority_effect_exact",
                "priority_effect_incomplete",
                "priority_effect_reorders",
                "priority_effect_beam_preserved",
                "priority_effect_selected",
                "priority_effect_changed",
                "priority_effect_unchanged",
                "narrowing_shadow_considered",
                "narrowing_shadow_insufficient_depth",
                "narrowing_shadow_no_reduction",
                "narrowing_shadow_incomplete",
                "narrowing_shadow_comparisons",
                "narrowing_shadow_matches",
                "narrowing_shadow_mismatches",
                "narrowing_shadow_priority_selected",
                "active_narrowing_considered",
                "active_narrowing_applied",
                "active_narrowing_insufficient_depth",
                "active_narrowing_no_reduction",
                "active_narrowing_safety_rejected",
                "active_narrowing_rejected_specialized_profile",
                "active_narrowing_rejected_information_unavailable",
                "active_narrowing_rejected_low_confidence",
                "active_narrowing_rejected_priority_not_top_two",
                "active_narrowing_rejected_insufficient_margin",
                "active_narrowing_incomplete",
                "active_narrowing_deepened",
                "active_narrowing_no_deepening",
                "active_narrowing_no_deepening_time_limit",
                "active_narrowing_no_deepening_node_limit",
                "active_narrowing_no_deepening_other",
                "active_narrowing_priority_selected",
                "active_narrowing_continuation_extensions",
                "active_narrowing_node_extensions",
                "active_narrowing_added_nodes_sum",
            ):
                restored_counters[name] = max(0, int(counters.get(name, 0)))
            for name in (
                "depth_sum",
                "agreement_sum",
                "confidence_sum",
                "margin_sum",
                "priority_effect_with_depth_sum",
                "priority_effect_without_depth_sum",
                "priority_effect_with_elapsed_sum",
                "priority_effect_without_elapsed_sum",
                "priority_effect_saved_seconds_sum",
                "priority_effect_value_delta_sum",
                "tactical_pair_with_depth_sum",
                "tactical_pair_without_depth_sum",
                "tactical_pair_with_elapsed_sum",
                "tactical_pair_without_elapsed_sum",
                "tactical_pair_value_delta_sum",
                "tactical_pair_margin_delta_sum",
                "human_pair_normal_depth_sum",
                "human_pair_priority_depth_sum",
                "human_pair_normal_elapsed_sum",
                "human_pair_priority_elapsed_sum",
                "human_pair_value_delta_sum",
                "narrowing_shadow_full_candidates_sum",
                "narrowing_shadow_kept_candidates_sum",
                "narrowing_shadow_removed_candidates_sum",
                "narrowing_shadow_depth_sum",
                "narrowing_shadow_actual_elapsed_sum",
                "narrowing_shadow_estimated_elapsed_sum",
                "narrowing_shadow_estimated_saved_sum",
                "narrowing_shadow_value_loss_sum",
                "active_narrowing_full_candidates_sum",
                "active_narrowing_kept_candidates_sum",
                "active_narrowing_depth_sum",
                "active_narrowing_elapsed_sum",
                "active_narrowing_continuation_extension_seconds_sum",
            ):
                restored_counters[name] = float(counters.get(name, 0.0))
            for name in (
                "action_counts",
                "followup_counts",
                "source_counts",
                "rejection_counts",
                "shadow_recommended_actions",
                "shadow_actual_actions",
                "shadow_match_actions",
                "shadow_granularity_counts",
                "priority_action_counts",
                "priority_granularity_counts",
            ):
                restored_counters[name] = Counter(
                    self._counter_dict(counters.get(name, {}))
                )
        except (TypeError, ValueError):
            return False

        with self._lock:
            self._patterns = restored_patterns
            self._rebuild_tactical_patterns_locked()
            self._rebuild_medium_patterns_locked()
            self._tactical_mismatch_details = restored_mismatch_details
            self._human_mismatch_details = restored_human_mismatch_details
            self._counters = restored_counters
            self._started_at = float(payload.get("started_at", time.time()))
            self._last_checkpoint_error = ""
        return True

    def pattern(
        self,
        pattern_key: str,
        *,
        granularity: str = "detailed",
    ) -> Optional[dict]:
        with self._lock:
            if granularity == "tactical":
                table = self._tactical_patterns
            elif granularity == "medium":
                table = self._medium_patterns
            else:
                table = self._patterns
            raw = table.get(str(pattern_key))
            if raw is None:
                return None
            observations = int(raw["observations"])
            routes = []
            for route in raw.get("routes", {}).values():
                route_observations = int(route.get("observations", 0))
                routes.append({
                    "action": str(route.get("action", "other")),
                    "followup": str(route.get("followup", "none")),
                    "observations": route_observations,
                    "source_counts": dict(route.get("source_counts", {})),
                    "average_depth": round(
                        float(route.get("depth_sum", 0.0)) / route_observations
                        if route_observations else 0.0,
                        3,
                    ),
                    "average_agreement": round(
                        float(route.get("agreement_sum", 0.0))
                        / route_observations
                        if route_observations else 0.0,
                        5,
                    ),
                    "average_confidence": round(
                        float(route.get("confidence_sum", 0.0))
                        / route_observations
                        if route_observations else 0.0,
                        5,
                    ),
                    "average_margin": round(
                        float(route.get("margin_sum", 0.0)) / route_observations
                        if route_observations else 0.0,
                        3,
                    ),
                })
            routes.sort(
                key=lambda route: (
                    -int(route["observations"]),
                    str(route["action"]),
                    str(route["followup"]),
                )
            )
            return {
                "features": dict(raw["features"]),
                "observations": observations,
                "action_counts": dict(raw["action_counts"]),
                "followup_counts": dict(raw["followup_counts"]),
                "source_counts": dict(raw["source_counts"]),
                "routes": routes,
                "average_depth": round(
                    float(raw["depth_sum"]) / observations
                    if observations else 0.0,
                    3,
                ),
                "average_agreement": round(
                    float(raw["agreement_sum"]) / observations
                    if observations else 0.0,
                    5,
                ),
                "average_confidence": round(
                    float(raw["confidence_sum"]) / observations
                    if observations else 0.0,
                    5,
                ),
            }

    def _recommendation_locked(
        self,
        pattern_key: str,
        *,
        granularity: str,
        min_observations: int,
        min_dominance: float,
    ) -> dict:
        if granularity == "tactical":
            table = self._tactical_patterns
        elif granularity == "medium":
            table = self._medium_patterns
        else:
            table = self._patterns
        raw = table.get(str(pattern_key))
        if raw is None:
            return {
                "status": "no_pattern",
                "granularity": granularity,
            }

        observations = max(0, int(raw.get("observations", 0)))
        if observations < max(1, int(min_observations)):
            return {
                "status": "insufficient",
                "granularity": granularity,
                "observations": observations,
            }

        action_counts = {
            str(action): max(0, int(count))
            for action, count in raw.get("action_counts", {}).items()
        }
        ranked = sorted(
            action_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if not ranked or ranked[0][1] <= 0:
            return {
                "status": "insufficient",
                "granularity": granularity,
                "observations": observations,
            }
        top_action, top_count = ranked[0]
        second_count = ranked[1][1] if len(ranked) > 1 else 0
        dominance = top_count / max(1, observations)
        if (
            top_count == second_count
            or dominance < max(0.0, min(1.0, float(min_dominance)))
        ):
            return {
                "status": "ambiguous",
                "granularity": granularity,
                "observations": observations,
                "dominance": round(dominance, 5),
            }

        recommended_routes = [
            route
            for route in raw.get("routes", {}).values()
            if str(route.get("action", "other")) == top_action
        ]
        recommended_routes.sort(
            key=lambda route: (
                -int(route.get("observations", 0)),
                str(route.get("followup", "none")),
            )
        )
        followup = (
            str(recommended_routes[0].get("followup", "none"))
            if recommended_routes else "none"
        )
        return {
            "status": "recommended",
            "granularity": granularity,
            "recommended_action": top_action,
            "recommended_followup": followup,
            "observations": observations,
            "support": top_count,
            "dominance": round(dominance, 5),
        }

    def recommendation(
        self,
        pattern_key: str,
        *,
        granularity: str = "detailed",
        min_observations: int = 5,
        min_dominance: float = 0.60,
    ) -> dict:
        """Return a stable generic recommendation without changing counters."""
        with self._lock:
            return dict(self._recommendation_locked(
                pattern_key,
                granularity=granularity,
                min_observations=min_observations,
                min_dominance=min_dominance,
            ))

    def record_priority_query(self, recommendation: Optional[dict]) -> None:
        """Count a recommendation actually offered to the search engine."""
        with self._lock:
            self._counters["priority_queries"] += 1
            if not recommendation or recommendation.get("status") != "recommended":
                return
            action = str(recommendation.get("recommended_action", "other"))
            granularity = str(recommendation.get("granularity", "detailed"))
            self._counters["priority_hits"] += 1
            self._counters["priority_action_counts"][action] += 1
            self._counters["priority_granularity_counts"][granularity] += 1

    def record_tactical_priority_query(self, status: str) -> None:
        """Count one debug-only tactical priority lookup."""
        counter_name = {
            "no_recommendation": "tactical_priority_no_recommendation",
            "rejected_action": "tactical_priority_rejected_action",
            "rejected_context": "tactical_priority_rejected_context",
            "no_legal_followup": "tactical_priority_no_legal_followup",
            "offered": "tactical_priority_offered",
        }.get(str(status), "tactical_priority_rejected_context")
        with self._lock:
            self._counters["tactical_priority_lookups"] += 1
            self._counters[counter_name] += 1

    def record_tactical_priority_effect(
        self,
        *,
        reordered: bool,
        baseline_disagreed: bool,
        selected: bool,
        completed_depth: int,
    ) -> None:
        """Record aggregate-only effects of an offered tactical hint."""
        with self._lock:
            counters = self._counters
            counters["tactical_priority_effects"] += 1
            if reordered:
                counters["tactical_priority_reordered"] += 1
            if baseline_disagreed:
                counters["tactical_priority_baseline_disagreements"] += 1
            if selected:
                counters["tactical_priority_selected"] += 1
            counters["tactical_priority_depth_sum"] += max(
                0,
                int(completed_depth),
            )

    def record_tactical_priority_pair(
        self,
        *,
        comparison_complete: bool,
        action_matched: bool = False,
        with_priority_selected: bool = False,
        without_priority_selected: bool = False,
        with_depth: int = 0,
        without_depth: int = 0,
        with_elapsed_seconds: float = 0.0,
        without_elapsed_seconds: float = 0.0,
        with_nodes: int = 0,
        without_nodes: int = 0,
        value_delta: float = 0.0,
        margin_delta: float = 0.0,
    ) -> None:
        """Record a tactical-on/off comparison without storing its position."""
        with self._lock:
            counters = self._counters
            counters["tactical_pair_comparisons"] += 1
            if not comparison_complete:
                counters["tactical_pair_incomplete"] += 1
                return

            counters["tactical_pair_completed"] += 1
            if action_matched:
                counters["tactical_pair_action_matches"] += 1
            else:
                counters["tactical_pair_action_changes"] += 1
            if with_priority_selected:
                counters["tactical_pair_with_priority_selected"] += 1
            if without_priority_selected:
                counters["tactical_pair_without_priority_selected"] += 1
            counters["tactical_pair_with_depth_sum"] += max(
                0,
                int(with_depth),
            )
            counters["tactical_pair_without_depth_sum"] += max(
                0,
                int(without_depth),
            )
            counters["tactical_pair_with_elapsed_sum"] += max(
                0.0,
                float(with_elapsed_seconds),
            )
            counters["tactical_pair_without_elapsed_sum"] += max(
                0.0,
                float(without_elapsed_seconds),
            )
            counters["tactical_pair_with_nodes_sum"] += max(
                0,
                int(with_nodes),
            )
            counters["tactical_pair_without_nodes_sum"] += max(
                0,
                int(without_nodes),
            )
            counters["tactical_pair_value_delta_sum"] += float(value_delta)
            counters["tactical_pair_margin_delta_sum"] += float(margin_delta)

    def record_human_priority_pair(
        self,
        *,
        comparison_complete: bool,
        selected_side: str = "other",
        normal_depth: int = 0,
        priority_depth: int = 0,
        normal_elapsed_seconds: float = 0.0,
        priority_elapsed_seconds: float = 0.0,
        normal_nodes: int = 0,
        priority_nodes: int = 0,
        value_delta: float = 0.0,
    ) -> None:
        """Record a human-hint comparison without retaining its position."""
        with self._lock:
            counters = self._counters
            counters["human_pair_comparisons"] += 1
            if not comparison_complete:
                counters["human_pair_incomplete"] += 1
                return

            counters["human_pair_completed"] += 1
            side = str(selected_side)
            if side == "human":
                counters["human_pair_human_selected"] += 1
            elif side == "ai":
                counters["human_pair_ai_selected"] += 1
            else:
                counters["human_pair_other_selected"] += 1
            delta = float(value_delta)
            if delta > 1e-6:
                counters["human_pair_human_value_better"] += 1
            elif delta < -1e-6:
                counters["human_pair_ai_value_better"] += 1
            else:
                counters["human_pair_value_tied"] += 1
            counters["human_pair_normal_depth_sum"] += max(
                0,
                int(normal_depth),
            )
            counters["human_pair_priority_depth_sum"] += max(
                0,
                int(priority_depth),
            )
            counters["human_pair_normal_elapsed_sum"] += max(
                0.0,
                float(normal_elapsed_seconds),
            )
            counters["human_pair_priority_elapsed_sum"] += max(
                0.0,
                float(priority_elapsed_seconds),
            )
            counters["human_pair_normal_nodes_sum"] += max(
                0,
                int(normal_nodes),
            )
            counters["human_pair_priority_nodes_sum"] += max(
                0,
                int(priority_nodes),
            )
            counters["human_pair_value_delta_sum"] += delta

    def record_priority_effect(
        self,
        *,
        reordered: bool,
        beam_preserved: bool,
        comparison_complete: bool,
        recommended_selected: bool,
        action_changed: bool,
        with_depth: int,
        without_depth: int,
        with_elapsed_seconds: float,
        without_elapsed_seconds: float,
        value_delta: float,
    ) -> None:
        """Aggregate one paired search-order comparison without storing a board."""
        with self._lock:
            counters = self._counters
            counters["priority_effect_comparisons"] += 1
            if reordered:
                counters["priority_effect_reorders"] += 1
            if beam_preserved:
                counters["priority_effect_beam_preserved"] += 1
            if recommended_selected:
                counters["priority_effect_selected"] += 1
            if not comparison_complete:
                counters["priority_effect_incomplete"] += 1
                return

            counters["priority_effect_exact"] += 1
            if action_changed:
                counters["priority_effect_changed"] += 1
            else:
                counters["priority_effect_unchanged"] += 1
            counters["priority_effect_with_depth_sum"] += max(0, int(with_depth))
            counters["priority_effect_without_depth_sum"] += max(
                0,
                int(without_depth),
            )
            with_elapsed = max(0.0, float(with_elapsed_seconds))
            without_elapsed = max(0.0, float(without_elapsed_seconds))
            counters["priority_effect_with_elapsed_sum"] += with_elapsed
            counters["priority_effect_without_elapsed_sum"] += without_elapsed
            counters["priority_effect_saved_seconds_sum"] += (
                without_elapsed - with_elapsed
            )
            counters["priority_effect_value_delta_sum"] += float(value_delta)

    def record_narrowing_shadow(
        self,
        *,
        status: str,
        matched: bool = False,
        priority_selected: bool = False,
        full_candidates: int = 0,
        kept_candidates: int = 0,
        depth: int = 0,
        actual_elapsed_seconds: float = 0.0,
        estimated_elapsed_seconds: float = 0.0,
        value_loss: float = 0.0,
    ) -> None:
        """Aggregate a depth-three hypothetical narrowing comparison."""
        with self._lock:
            counters = self._counters
            counters["narrowing_shadow_considered"] += 1
            normalized_status = str(status or "incomplete")
            if normalized_status == "insufficient_depth":
                counters["narrowing_shadow_insufficient_depth"] += 1
                return
            if normalized_status == "no_reduction":
                counters["narrowing_shadow_no_reduction"] += 1
                return
            if normalized_status != "compared":
                counters["narrowing_shadow_incomplete"] += 1
                return

            full_count = max(0, int(full_candidates))
            kept_count = max(0, min(full_count, int(kept_candidates)))
            actual_elapsed = max(0.0, float(actual_elapsed_seconds))
            estimated_elapsed = max(
                0.0,
                min(actual_elapsed, float(estimated_elapsed_seconds)),
            )
            counters["narrowing_shadow_comparisons"] += 1
            if matched:
                counters["narrowing_shadow_matches"] += 1
            else:
                counters["narrowing_shadow_mismatches"] += 1
            if priority_selected:
                counters["narrowing_shadow_priority_selected"] += 1
            counters["narrowing_shadow_full_candidates_sum"] += full_count
            counters["narrowing_shadow_kept_candidates_sum"] += kept_count
            counters["narrowing_shadow_removed_candidates_sum"] += max(
                0,
                full_count - kept_count,
            )
            counters["narrowing_shadow_depth_sum"] += max(0, int(depth))
            counters["narrowing_shadow_actual_elapsed_sum"] += actual_elapsed
            counters["narrowing_shadow_estimated_elapsed_sum"] += (
                estimated_elapsed
            )
            counters["narrowing_shadow_estimated_saved_sum"] += max(
                0.0,
                actual_elapsed - estimated_elapsed,
            )
            counters["narrowing_shadow_value_loss_sum"] += max(
                0.0,
                float(value_loss),
            )

    def record_active_narrowing(
        self,
        *,
        status: str,
        full_candidates: int = 0,
        kept_candidates: int = 0,
        completed_depth: int = 0,
        elapsed_seconds: float = 0.0,
        priority_selected: bool = False,
        continuation_extension_seconds: float = 0.0,
        added_nodes: int = 0,
        no_deepening_reason: str = "",
    ) -> None:
        """Aggregate live narrowing without retaining hands or actions."""
        with self._lock:
            counters = self._counters
            counters["active_narrowing_considered"] += 1
            normalized_status = str(status or "incomplete")
            if normalized_status == "insufficient_depth":
                counters["active_narrowing_insufficient_depth"] += 1
                return
            if normalized_status == "no_reduction":
                counters["active_narrowing_no_reduction"] += 1
                return
            rejection_counters = {
                "specialized_profile": (
                    "active_narrowing_rejected_specialized_profile"
                ),
                "information_unavailable": (
                    "active_narrowing_rejected_information_unavailable"
                ),
                "low_confidence": (
                    "active_narrowing_rejected_low_confidence"
                ),
                "priority_not_top_two": (
                    "active_narrowing_rejected_priority_not_top_two"
                ),
                "insufficient_margin": (
                    "active_narrowing_rejected_insufficient_margin"
                ),
            }
            if normalized_status == "safety_rejected":
                counters["active_narrowing_safety_rejected"] += 1
                return
            if normalized_status in rejection_counters:
                counters["active_narrowing_safety_rejected"] += 1
                counters[rejection_counters[normalized_status]] += 1
                return
            if normalized_status not in ("deepened", "no_deepening"):
                counters["active_narrowing_incomplete"] += 1
                return

            full_count = max(0, int(full_candidates))
            kept_count = max(0, min(full_count, int(kept_candidates)))
            counters["active_narrowing_applied"] += 1
            counters[f"active_narrowing_{normalized_status}"] += 1
            if normalized_status == "no_deepening":
                reason = str(no_deepening_reason or "other")
                if reason not in ("time_limit", "node_limit"):
                    reason = "other"
                counters[f"active_narrowing_no_deepening_{reason}"] += 1
            if priority_selected:
                counters["active_narrowing_priority_selected"] += 1
            extension_seconds = max(
                0.0,
                float(continuation_extension_seconds),
            )
            if extension_seconds > 0.0:
                counters["active_narrowing_continuation_extensions"] += 1
                counters[
                    "active_narrowing_continuation_extension_seconds_sum"
                ] += extension_seconds
            normalized_added_nodes = max(0, int(added_nodes))
            if normalized_added_nodes > 0:
                counters["active_narrowing_node_extensions"] += 1
                counters["active_narrowing_added_nodes_sum"] += (
                    normalized_added_nodes
                )
            counters["active_narrowing_full_candidates_sum"] += full_count
            counters["active_narrowing_kept_candidates_sum"] += kept_count
            counters["active_narrowing_depth_sum"] += max(
                0,
                int(completed_depth),
            )
            counters["active_narrowing_elapsed_sum"] += max(
                0.0,
                float(elapsed_seconds),
            )

    def compare_shadow(
        self,
        *,
        pattern_key: str,
        medium_pattern_key: Optional[str] = None,
        actual_action: str,
        min_observations: int = 5,
        min_dominance: float = 0.60,
        medium_min_observations: int = 5,
        medium_min_dominance: float = 0.70,
    ) -> dict:
        """Compare a stored recommendation without affecting gameplay."""
        with self._lock:
            self._counters["shadow_lookups"] += 1
            recommendation = self._recommendation_locked(
                pattern_key,
                granularity="detailed",
                min_observations=min_observations,
                min_dominance=min_dominance,
            )
            if (
                recommendation.get("status") != "recommended"
                and medium_pattern_key is not None
            ):
                medium_recommendation = self._recommendation_locked(
                    medium_pattern_key,
                    granularity="medium",
                    min_observations=medium_min_observations,
                    min_dominance=medium_min_dominance,
                )
                if medium_recommendation.get("status") == "recommended":
                    recommendation = medium_recommendation
                elif recommendation.get("status") == "no_pattern":
                    recommendation = medium_recommendation
            status = str(recommendation.get("status", "insufficient"))
            if status == "no_pattern":
                self._counters["shadow_no_pattern"] += 1
                return {"status": "no_pattern"}
            if status == "insufficient":
                self._counters["shadow_insufficient"] += 1
                return recommendation
            if status == "ambiguous":
                self._counters["shadow_ambiguous"] += 1
                return recommendation

            actual = str(actual_action)
            top_action = str(recommendation["recommended_action"])
            matched = actual == top_action
            self._counters["shadow_recommendations"] += 1
            self._counters["shadow_recommended_actions"][top_action] += 1
            self._counters["shadow_actual_actions"][actual] += 1
            granularity = str(recommendation.get("granularity", "detailed"))
            self._counters["shadow_granularity_counts"][granularity] += 1
            if matched:
                self._counters["shadow_matches"] += 1
                self._counters["shadow_match_actions"][top_action] += 1
            else:
                self._counters["shadow_mismatches"] += 1

            return {
                **recommendation,
                "status": "match" if matched else "mismatch",
                "actual_action": actual,
            }

    def compare_tactical_shadow(
        self,
        *,
        pattern_key: str,
        actual_action: str,
        min_observations: int = 8,
        min_dominance: float = 0.70,
    ) -> dict:
        """Measure tactical-class advice without affecting search order."""
        with self._lock:
            counters = self._counters
            counters["tactical_shadow_lookups"] += 1
            recommendation = self._recommendation_locked(
                pattern_key,
                granularity="tactical",
                min_observations=min_observations,
                min_dominance=min_dominance,
            )
            status = str(recommendation.get("status", "insufficient"))
            if status == "no_pattern":
                counters["tactical_shadow_no_pattern"] += 1
                return recommendation
            if status == "insufficient":
                counters["tactical_shadow_insufficient"] += 1
                return recommendation
            if status == "ambiguous":
                counters["tactical_shadow_ambiguous"] += 1
                return recommendation

            actual = str(actual_action)
            recommended = str(recommendation["recommended_action"])
            matched = actual == recommended
            counters["tactical_shadow_recommendations"] += 1
            if matched:
                counters["tactical_shadow_matches"] += 1
            else:
                counters["tactical_shadow_mismatches"] += 1
                detail_key = _digest_payload({
                    "pattern_key": str(pattern_key),
                    "recommended_action": recommended,
                    "actual_action": actual,
                })
                detail = self._tactical_mismatch_details.get(detail_key)
                now = time.time()
                if detail is None:
                    if (
                        len(self._tactical_mismatch_details)
                        >= TACTICAL_MISMATCH_DETAIL_LIMIT
                    ):
                        evicted_key = min(
                            self._tactical_mismatch_details,
                            key=lambda key: (
                                int(self._tactical_mismatch_details[key].get(
                                    "count",
                                    0,
                                )),
                                float(self._tactical_mismatch_details[key].get(
                                    "last_seen_at",
                                    0.0,
                                )),
                            ),
                        )
                        self._tactical_mismatch_details.pop(evicted_key, None)
                    detail = {
                        "pattern_key": str(pattern_key),
                        "recommended_action": recommended,
                        "actual_action": actual,
                        "count": 0,
                        "first_seen_at": now,
                    }
                    self._tactical_mismatch_details[detail_key] = detail
                detail["count"] = int(detail.get("count", 0)) + 1
                detail["observations"] = max(
                    0,
                    int(recommendation.get("observations", 0)),
                )
                detail["support"] = max(
                    0,
                    int(recommendation.get("support", 0)),
                )
                detail["dominance"] = max(
                    0.0,
                    min(1.0, float(recommendation.get("dominance", 0.0))),
                )
                detail["last_seen_at"] = now
            return {
                **recommendation,
                "status": "match" if matched else "mismatch",
                "actual_action": actual,
            }

    def compare_human_shadow(
        self,
        *,
        recommendation: Mapping[str, object],
        actual_action: str,
    ) -> dict:
        """Compare aggregate human advice without affecting the live action."""
        with self._lock:
            counters = self._counters
            counters["human_shadow_lookups"] += 1
            status = str(recommendation.get("status", "no_pattern"))
            if status != "recommended":
                counters["human_shadow_no_pattern"] += 1
                return dict(recommendation)

            actual = str(actual_action)
            recommended = str(
                recommendation.get("recommended_action", "other")
            )
            matched = actual == recommended
            counters["human_shadow_recommendations"] += 1
            if matched:
                counters["human_shadow_matches"] += 1
            else:
                counters["human_shadow_mismatches"] += 1
                anonymous_context = {
                    key: value
                    for key, value in dict(
                        recommendation.get("anonymous_context", {}) or {}
                    ).items()
                    if key in (
                        "attacker_relation",
                        "attack_piece",
                        "attack_stage",
                        "hand_stage",
                        "next_receiver_stage",
                        "same_piece",
                        "royal_receive",
                        "followup_strength",
                        "reentry_width",
                        "shi_context",
                        "score_pressure",
                    )
                }
                pattern_key = str(
                    recommendation.get("pattern_key", "")
                    or _digest_payload(anonymous_context)
                )
                detail_key = _digest_payload({
                    "pattern_key": pattern_key,
                    "recommended_action": recommended,
                    "actual_action": actual,
                })
                detail = self._human_mismatch_details.get(detail_key)
                now = time.time()
                if detail is None:
                    if (
                        len(self._human_mismatch_details)
                        >= HUMAN_MISMATCH_DETAIL_LIMIT
                    ):
                        evicted_key = min(
                            self._human_mismatch_details,
                            key=lambda key: (
                                int(self._human_mismatch_details[key].get(
                                    "count",
                                    0,
                                )),
                                float(self._human_mismatch_details[key].get(
                                    "last_seen_at",
                                    0.0,
                                )),
                            ),
                        )
                        self._human_mismatch_details.pop(evicted_key, None)
                    detail = {
                        "pattern_key": pattern_key,
                        "anonymous_context": anonymous_context,
                        "recommended_action": recommended,
                        "actual_action": actual,
                        "count": 0,
                        "first_seen_at": now,
                    }
                    self._human_mismatch_details[detail_key] = detail
                detail["count"] = int(detail.get("count", 0)) + 1
                detail["observations"] = max(
                    0,
                    int(recommendation.get("observations", 0)),
                )
                detail["support"] = max(
                    0,
                    int(recommendation.get("support", 0)),
                )
                detail["dominance"] = max(
                    0.0,
                    min(1.0, float(recommendation.get("dominance", 0.0))),
                )
                detail["distinct_matches_lower_bound"] = max(
                    0,
                    int(recommendation.get(
                        "distinct_matches_lower_bound",
                        0,
                    )),
                )
                detail["last_seen_at"] = now
            return {
                **recommendation,
                "status": "match" if matched else "mismatch",
                "actual_action": actual,
            }

    def snapshot(self) -> dict:
        with self._lock:
            counters = self._counters
            recorded = int(counters["recorded"])
            observation_counts = [
                int(raw.get("observations", 0))
                for raw in self._patterns.values()
            ]
            medium_observation_counts = [
                int(raw.get("observations", 0))
                for raw in self._medium_patterns.values()
            ]
            tactical_observation_counts = [
                int(raw.get("observations", 0))
                for raw in self._tactical_patterns.values()
            ]
            tactical_mismatch_details = []
            for detail_key, raw in self._tactical_mismatch_details.items():
                pattern_key = str(raw.get("pattern_key", ""))
                recommended_action = str(
                    raw.get("recommended_action", "other")
                )
                pattern = self._tactical_patterns.get(pattern_key, {})
                observations = max(
                    0,
                    int(pattern.get(
                        "observations",
                        raw.get("observations", 0),
                    )),
                )
                support = max(
                    0,
                    int(pattern.get("action_counts", {}).get(
                        recommended_action,
                        raw.get("support", 0),
                    )),
                )
                pattern_features = dict(pattern.get("features", {}) or {})
                anonymous_context = {
                    key: pattern_features.get(key)
                    for key in (
                        "attacker_relation",
                        "attack_piece",
                        "attack_stage",
                        "hand_stage",
                        "next_receiver_stage",
                        "same_piece",
                        "royal_receive",
                        "followup_strength",
                        "reentry_width",
                        "shi_context",
                        "score_pressure",
                    )
                    if key in pattern_features
                }
                tactical_mismatch_details.append({
                    "pattern_id": pattern_key[:10],
                    "recommended_action": recommended_action,
                    "actual_action": str(raw.get("actual_action", "other")),
                    "count": max(0, int(raw.get("count", 0))),
                    "observations": observations,
                    "support": support,
                    "dominance": round(
                        support / max(1, observations),
                        5,
                    ),
                    "anonymous_context": anonymous_context,
                    "last_seen_at": float(raw.get("last_seen_at", 0.0)),
                    "detail_id": detail_key[:10],
                })
            tactical_mismatch_details.sort(
                key=lambda item: (
                    -int(item["count"]),
                    -float(item["last_seen_at"]),
                    str(item["detail_id"]),
                )
            )
            human_mismatch_details = []
            for detail_key, raw in self._human_mismatch_details.items():
                human_mismatch_details.append({
                    "pattern_id": str(raw.get("pattern_key", ""))[:10],
                    "recommended_action": str(
                        raw.get("recommended_action", "other")
                    ),
                    "actual_action": str(raw.get("actual_action", "other")),
                    "count": max(0, int(raw.get("count", 0))),
                    "observations": max(
                        0,
                        int(raw.get("observations", 0)),
                    ),
                    "support": max(0, int(raw.get("support", 0))),
                    "dominance": round(
                        max(
                            0.0,
                            min(1.0, float(raw.get("dominance", 0.0))),
                        ),
                        5,
                    ),
                    "distinct_matches_lower_bound": max(
                        0,
                        int(raw.get("distinct_matches_lower_bound", 0)),
                    ),
                    "anonymous_context": dict(
                        raw.get("anonymous_context", {}) or {}
                    ),
                    "last_seen_at": float(raw.get("last_seen_at", 0.0)),
                    "detail_id": detail_key[:10],
                })
            human_mismatch_details.sort(
                key=lambda item: (
                    -int(item["count"]),
                    -float(item["last_seen_at"]),
                    str(item["detail_id"]),
                )
            )
            return {
                "considered": int(counters["considered"]),
                "recorded": recorded,
                "rejected": max(
                    0,
                    int(counters["considered"]) - recorded,
                ),
                "pattern_count": len(self._patterns),
                "patterns_5_plus": sum(count >= 5 for count in observation_counts),
                "patterns_20_plus": sum(count >= 20 for count in observation_counts),
                "patterns_50_plus": sum(count >= 50 for count in observation_counts),
                "medium_pattern_count": len(self._medium_patterns),
                "medium_patterns_5_plus": sum(
                    count >= 5 for count in medium_observation_counts
                ),
                "medium_patterns_10_plus": sum(
                    count >= 10 for count in medium_observation_counts
                ),
                "medium_patterns_20_plus": sum(
                    count >= 20 for count in medium_observation_counts
                ),
                "tactical_pattern_count": len(self._tactical_patterns),
                "tactical_patterns_8_plus": sum(
                    count >= 8 for count in tactical_observation_counts
                ),
                "tactical_patterns_10_plus": sum(
                    count >= 10 for count in tactical_observation_counts
                ),
                "tactical_patterns_20_plus": sum(
                    count >= 20 for count in tactical_observation_counts
                ),
                "action_counts": dict(counters["action_counts"]),
                "followup_counts": dict(counters["followup_counts"]),
                "source_counts": dict(counters["source_counts"]),
                "rejection_counts": dict(counters["rejection_counts"]),
                "shadow_lookups": int(counters["shadow_lookups"]),
                "shadow_no_pattern": int(counters["shadow_no_pattern"]),
                "shadow_insufficient": int(counters["shadow_insufficient"]),
                "shadow_ambiguous": int(counters["shadow_ambiguous"]),
                "shadow_recommendations": int(
                    counters["shadow_recommendations"]
                ),
                "shadow_matches": int(counters["shadow_matches"]),
                "shadow_mismatches": int(counters["shadow_mismatches"]),
                "shadow_match_rate": round(
                    int(counters["shadow_matches"])
                    / max(1, int(counters["shadow_recommendations"])),
                    5,
                ),
                "shadow_recommended_actions": dict(
                    counters["shadow_recommended_actions"]
                ),
                "shadow_actual_actions": dict(
                    counters["shadow_actual_actions"]
                ),
                "shadow_match_actions": dict(
                    counters["shadow_match_actions"]
                ),
                "shadow_granularity_counts": dict(
                    counters["shadow_granularity_counts"]
                ),
                "tactical_shadow_lookups": int(
                    counters["tactical_shadow_lookups"]
                ),
                "tactical_shadow_no_pattern": int(
                    counters["tactical_shadow_no_pattern"]
                ),
                "tactical_shadow_insufficient": int(
                    counters["tactical_shadow_insufficient"]
                ),
                "tactical_shadow_ambiguous": int(
                    counters["tactical_shadow_ambiguous"]
                ),
                "tactical_shadow_recommendations": int(
                    counters["tactical_shadow_recommendations"]
                ),
                "tactical_shadow_matches": int(
                    counters["tactical_shadow_matches"]
                ),
                "tactical_shadow_mismatches": int(
                    counters["tactical_shadow_mismatches"]
                ),
                "tactical_shadow_match_rate": round(
                    int(counters["tactical_shadow_matches"])
                    / max(
                        1,
                        int(counters["tactical_shadow_recommendations"]),
                    ),
                    5,
                ),
                "tactical_mismatch_details": tactical_mismatch_details[:50],
                "tactical_mismatch_detail_count": len(
                    self._tactical_mismatch_details
                ),
                "human_shadow_lookups": int(
                    counters["human_shadow_lookups"]
                ),
                "human_shadow_no_pattern": int(
                    counters["human_shadow_no_pattern"]
                ),
                "human_shadow_recommendations": int(
                    counters["human_shadow_recommendations"]
                ),
                "human_shadow_matches": int(
                    counters["human_shadow_matches"]
                ),
                "human_shadow_mismatches": int(
                    counters["human_shadow_mismatches"]
                ),
                "human_shadow_match_rate": round(
                    int(counters["human_shadow_matches"])
                    / max(
                        1,
                        int(counters["human_shadow_recommendations"]),
                    ),
                    5,
                ),
                "human_mismatch_details": human_mismatch_details[:50],
                "human_mismatch_detail_count": len(
                    self._human_mismatch_details
                ),
                "human_pair_comparisons": int(
                    counters["human_pair_comparisons"]
                ),
                "human_pair_completed": int(
                    counters["human_pair_completed"]
                ),
                "human_pair_incomplete": int(
                    counters["human_pair_incomplete"]
                ),
                "human_pair_human_selected": int(
                    counters["human_pair_human_selected"]
                ),
                "human_pair_ai_selected": int(
                    counters["human_pair_ai_selected"]
                ),
                "human_pair_other_selected": int(
                    counters["human_pair_other_selected"]
                ),
                "human_pair_human_value_better": int(
                    counters["human_pair_human_value_better"]
                ),
                "human_pair_ai_value_better": int(
                    counters["human_pair_ai_value_better"]
                ),
                "human_pair_value_tied": int(
                    counters["human_pair_value_tied"]
                ),
                "human_pair_average_normal_depth": round(
                    float(counters["human_pair_normal_depth_sum"])
                    / max(1, int(counters["human_pair_completed"])),
                    3,
                ),
                "human_pair_average_priority_depth": round(
                    float(counters["human_pair_priority_depth_sum"])
                    / max(1, int(counters["human_pair_completed"])),
                    3,
                ),
                "human_pair_average_normal_elapsed_seconds": round(
                    float(counters["human_pair_normal_elapsed_sum"])
                    / max(1, int(counters["human_pair_completed"])),
                    5,
                ),
                "human_pair_average_priority_elapsed_seconds": round(
                    float(counters["human_pair_priority_elapsed_sum"])
                    / max(1, int(counters["human_pair_completed"])),
                    5,
                ),
                "human_pair_average_normal_nodes": round(
                    int(counters["human_pair_normal_nodes_sum"])
                    / max(1, int(counters["human_pair_completed"])),
                    2,
                ),
                "human_pair_average_priority_nodes": round(
                    int(counters["human_pair_priority_nodes_sum"])
                    / max(1, int(counters["human_pair_completed"])),
                    2,
                ),
                "human_pair_average_value_delta": round(
                    float(counters["human_pair_value_delta_sum"])
                    / max(1, int(counters["human_pair_completed"])),
                    3,
                ),
                "tactical_priority_lookups": int(
                    counters["tactical_priority_lookups"]
                ),
                "tactical_priority_no_recommendation": int(
                    counters["tactical_priority_no_recommendation"]
                ),
                "tactical_priority_rejected_action": int(
                    counters["tactical_priority_rejected_action"]
                ),
                "tactical_priority_rejected_context": int(
                    counters["tactical_priority_rejected_context"]
                ),
                "tactical_priority_no_legal_followup": int(
                    counters["tactical_priority_no_legal_followup"]
                ),
                "tactical_priority_offered": int(
                    counters["tactical_priority_offered"]
                ),
                "tactical_priority_effects": int(
                    counters["tactical_priority_effects"]
                ),
                "tactical_priority_reordered": int(
                    counters["tactical_priority_reordered"]
                ),
                "tactical_priority_baseline_disagreements": int(
                    counters["tactical_priority_baseline_disagreements"]
                ),
                "tactical_priority_selected": int(
                    counters["tactical_priority_selected"]
                ),
                "tactical_priority_selected_rate": round(
                    int(counters["tactical_priority_selected"])
                    / max(1, int(counters["tactical_priority_effects"])),
                    5,
                ),
                "tactical_priority_average_depth": round(
                    int(counters["tactical_priority_depth_sum"])
                    / max(1, int(counters["tactical_priority_effects"])),
                    3,
                ),
                "tactical_pair_comparisons": int(
                    counters["tactical_pair_comparisons"]
                ),
                "tactical_pair_completed": int(
                    counters["tactical_pair_completed"]
                ),
                "tactical_pair_incomplete": int(
                    counters["tactical_pair_incomplete"]
                ),
                "tactical_pair_action_matches": int(
                    counters["tactical_pair_action_matches"]
                ),
                "tactical_pair_action_changes": int(
                    counters["tactical_pair_action_changes"]
                ),
                "tactical_pair_action_match_rate": round(
                    int(counters["tactical_pair_action_matches"])
                    / max(1, int(counters["tactical_pair_completed"])),
                    5,
                ),
                "tactical_pair_with_priority_selected": int(
                    counters["tactical_pair_with_priority_selected"]
                ),
                "tactical_pair_without_priority_selected": int(
                    counters["tactical_pair_without_priority_selected"]
                ),
                "tactical_pair_average_with_depth": round(
                    float(counters["tactical_pair_with_depth_sum"])
                    / max(1, int(counters["tactical_pair_completed"])),
                    3,
                ),
                "tactical_pair_average_without_depth": round(
                    float(counters["tactical_pair_without_depth_sum"])
                    / max(1, int(counters["tactical_pair_completed"])),
                    3,
                ),
                "tactical_pair_average_with_elapsed_seconds": round(
                    float(counters["tactical_pair_with_elapsed_sum"])
                    / max(1, int(counters["tactical_pair_completed"])),
                    5,
                ),
                "tactical_pair_average_without_elapsed_seconds": round(
                    float(counters["tactical_pair_without_elapsed_sum"])
                    / max(1, int(counters["tactical_pair_completed"])),
                    5,
                ),
                "tactical_pair_average_with_nodes": round(
                    int(counters["tactical_pair_with_nodes_sum"])
                    / max(1, int(counters["tactical_pair_completed"])),
                    1,
                ),
                "tactical_pair_average_without_nodes": round(
                    int(counters["tactical_pair_without_nodes_sum"])
                    / max(1, int(counters["tactical_pair_completed"])),
                    1,
                ),
                "tactical_pair_average_value_delta": round(
                    float(counters["tactical_pair_value_delta_sum"])
                    / max(1, int(counters["tactical_pair_completed"])),
                    3,
                ),
                "tactical_pair_average_margin_delta": round(
                    float(counters["tactical_pair_margin_delta_sum"])
                    / max(1, int(counters["tactical_pair_completed"])),
                    3,
                ),
                "priority_queries": int(counters["priority_queries"]),
                "priority_hits": int(counters["priority_hits"]),
                "priority_hit_rate": round(
                    int(counters["priority_hits"])
                    / max(1, int(counters["priority_queries"])),
                    5,
                ),
                "priority_action_counts": dict(
                    counters["priority_action_counts"]
                ),
                "priority_granularity_counts": dict(
                    counters["priority_granularity_counts"]
                ),
                "priority_effect_comparisons": int(
                    counters["priority_effect_comparisons"]
                ),
                "priority_effect_exact": int(counters["priority_effect_exact"]),
                "priority_effect_incomplete": int(
                    counters["priority_effect_incomplete"]
                ),
                "priority_effect_reorders": int(
                    counters["priority_effect_reorders"]
                ),
                "priority_effect_beam_preserved": int(
                    counters["priority_effect_beam_preserved"]
                ),
                "priority_effect_selected": int(
                    counters["priority_effect_selected"]
                ),
                "priority_effect_changed": int(
                    counters["priority_effect_changed"]
                ),
                "priority_effect_unchanged": int(
                    counters["priority_effect_unchanged"]
                ),
                "priority_effect_change_rate": round(
                    int(counters["priority_effect_changed"])
                    / max(1, int(counters["priority_effect_exact"])),
                    5,
                ),
                "priority_effect_average_with_depth": round(
                    float(counters["priority_effect_with_depth_sum"])
                    / max(1, int(counters["priority_effect_exact"])),
                    3,
                ),
                "priority_effect_average_without_depth": round(
                    float(counters["priority_effect_without_depth_sum"])
                    / max(1, int(counters["priority_effect_exact"])),
                    3,
                ),
                "priority_effect_average_with_elapsed_seconds": round(
                    float(counters["priority_effect_with_elapsed_sum"])
                    / max(1, int(counters["priority_effect_exact"])),
                    5,
                ),
                "priority_effect_average_without_elapsed_seconds": round(
                    float(counters["priority_effect_without_elapsed_sum"])
                    / max(1, int(counters["priority_effect_exact"])),
                    5,
                ),
                "priority_effect_saved_seconds": round(
                    float(counters["priority_effect_saved_seconds_sum"]),
                    5,
                ),
                "priority_effect_average_value_delta": round(
                    float(counters["priority_effect_value_delta_sum"])
                    / max(1, int(counters["priority_effect_exact"])),
                    3,
                ),
                "narrowing_shadow_considered": int(
                    counters["narrowing_shadow_considered"]
                ),
                "narrowing_shadow_insufficient_depth": int(
                    counters["narrowing_shadow_insufficient_depth"]
                ),
                "narrowing_shadow_no_reduction": int(
                    counters["narrowing_shadow_no_reduction"]
                ),
                "narrowing_shadow_incomplete": int(
                    counters["narrowing_shadow_incomplete"]
                ),
                "narrowing_shadow_comparisons": int(
                    counters["narrowing_shadow_comparisons"]
                ),
                "narrowing_shadow_matches": int(
                    counters["narrowing_shadow_matches"]
                ),
                "narrowing_shadow_mismatches": int(
                    counters["narrowing_shadow_mismatches"]
                ),
                "narrowing_shadow_match_rate": round(
                    int(counters["narrowing_shadow_matches"])
                    / max(1, int(counters["narrowing_shadow_comparisons"])),
                    5,
                ),
                "narrowing_shadow_priority_selected": int(
                    counters["narrowing_shadow_priority_selected"]
                ),
                "narrowing_shadow_average_full_candidates": round(
                    float(counters["narrowing_shadow_full_candidates_sum"])
                    / max(1, int(counters["narrowing_shadow_comparisons"])),
                    3,
                ),
                "narrowing_shadow_average_kept_candidates": round(
                    float(counters["narrowing_shadow_kept_candidates_sum"])
                    / max(1, int(counters["narrowing_shadow_comparisons"])),
                    3,
                ),
                "narrowing_shadow_average_removed_candidates": round(
                    float(counters["narrowing_shadow_removed_candidates_sum"])
                    / max(1, int(counters["narrowing_shadow_comparisons"])),
                    3,
                ),
                "narrowing_shadow_average_depth": round(
                    float(counters["narrowing_shadow_depth_sum"])
                    / max(1, int(counters["narrowing_shadow_comparisons"])),
                    3,
                ),
                "narrowing_shadow_average_actual_elapsed_seconds": round(
                    float(counters["narrowing_shadow_actual_elapsed_sum"])
                    / max(1, int(counters["narrowing_shadow_comparisons"])),
                    5,
                ),
                "narrowing_shadow_average_estimated_elapsed_seconds": round(
                    float(counters["narrowing_shadow_estimated_elapsed_sum"])
                    / max(1, int(counters["narrowing_shadow_comparisons"])),
                    5,
                ),
                "narrowing_shadow_estimated_saved_seconds": round(
                    float(counters["narrowing_shadow_estimated_saved_sum"]),
                    5,
                ),
                "narrowing_shadow_average_value_loss": round(
                    float(counters["narrowing_shadow_value_loss_sum"])
                    / max(1, int(counters["narrowing_shadow_comparisons"])),
                    3,
                ),
                "active_narrowing_considered": int(
                    counters["active_narrowing_considered"]
                ),
                "active_narrowing_applied": int(
                    counters["active_narrowing_applied"]
                ),
                "active_narrowing_insufficient_depth": int(
                    counters["active_narrowing_insufficient_depth"]
                ),
                "active_narrowing_no_reduction": int(
                    counters["active_narrowing_no_reduction"]
                ),
                "active_narrowing_safety_rejected": int(
                    counters["active_narrowing_safety_rejected"]
                ),
                "active_narrowing_rejected_specialized_profile": int(
                    counters["active_narrowing_rejected_specialized_profile"]
                ),
                "active_narrowing_rejected_information_unavailable": int(
                    counters[
                        "active_narrowing_rejected_information_unavailable"
                    ]
                ),
                "active_narrowing_rejected_low_confidence": int(
                    counters["active_narrowing_rejected_low_confidence"]
                ),
                "active_narrowing_rejected_priority_not_top_two": int(
                    counters[
                        "active_narrowing_rejected_priority_not_top_two"
                    ]
                ),
                "active_narrowing_rejected_insufficient_margin": int(
                    counters[
                        "active_narrowing_rejected_insufficient_margin"
                    ]
                ),
                "active_narrowing_incomplete": int(
                    counters["active_narrowing_incomplete"]
                ),
                "active_narrowing_deepened": int(
                    counters["active_narrowing_deepened"]
                ),
                "active_narrowing_no_deepening": int(
                    counters["active_narrowing_no_deepening"]
                ),
                "active_narrowing_no_deepening_time_limit": int(
                    counters["active_narrowing_no_deepening_time_limit"]
                ),
                "active_narrowing_no_deepening_node_limit": int(
                    counters["active_narrowing_no_deepening_node_limit"]
                ),
                "active_narrowing_no_deepening_other": int(
                    counters["active_narrowing_no_deepening_other"]
                ),
                "active_narrowing_deepening_rate": round(
                    int(counters["active_narrowing_deepened"])
                    / max(1, int(counters["active_narrowing_applied"])),
                    5,
                ),
                "active_narrowing_priority_selected": int(
                    counters["active_narrowing_priority_selected"]
                ),
                "active_narrowing_continuation_extensions": int(
                    counters["active_narrowing_continuation_extensions"]
                ),
                "active_narrowing_continuation_extension_seconds": round(
                    float(
                        counters[
                            "active_narrowing_continuation_extension_seconds_sum"
                        ]
                    ),
                    5,
                ),
                "active_narrowing_node_extensions": int(
                    counters["active_narrowing_node_extensions"]
                ),
                "active_narrowing_added_nodes": int(
                    counters["active_narrowing_added_nodes_sum"]
                ),
                "active_narrowing_average_full_candidates": round(
                    float(counters["active_narrowing_full_candidates_sum"])
                    / max(1, int(counters["active_narrowing_applied"])),
                    3,
                ),
                "active_narrowing_average_kept_candidates": round(
                    float(counters["active_narrowing_kept_candidates_sum"])
                    / max(1, int(counters["active_narrowing_applied"])),
                    3,
                ),
                "active_narrowing_average_depth": round(
                    float(counters["active_narrowing_depth_sum"])
                    / max(1, int(counters["active_narrowing_applied"])),
                    3,
                ),
                "active_narrowing_average_elapsed_seconds": round(
                    float(counters["active_narrowing_elapsed_sum"])
                    / max(1, int(counters["active_narrowing_applied"])),
                    5,
                ),
                "average_depth": round(
                    float(counters["depth_sum"]) / recorded if recorded else 0.0,
                    3,
                ),
                "average_agreement": round(
                    float(counters["agreement_sum"]) / recorded
                    if recorded else 0.0,
                    5,
                ),
                "average_confidence": round(
                    float(counters["confidence_sum"]) / recorded
                    if recorded else 0.0,
                    5,
                ),
                "average_margin": round(
                    float(counters["margin_sum"]) / recorded
                    if recorded else 0.0,
                    3,
                ),
                "evicted_patterns": int(counters["evicted_patterns"]),
                "max_patterns": self.max_patterns,
                "persistent": self.path is not None,
                "checkpoint_error": self._last_checkpoint_error,
                "uptime_seconds": round(
                    max(0.0, time.time() - self._started_at),
                    3,
                ),
            }


def _positive_env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return max(1, int(default))


_GENERIC_RESPONSE_PATTERN_STORE = GenericResponsePatternStore(
    path=resolve_generic_response_store_path(os.environ),
    max_patterns=_positive_env_int(
        "GOITA_GENERIC_RESPONSE_PATTERN_MAX_PATTERNS",
        5_000,
    ),
)


def generic_response_pattern_store() -> GenericResponsePatternStore:
    return _GENERIC_RESPONSE_PATTERN_STORE


def generic_response_pattern_snapshot() -> dict:
    snapshot = _GENERIC_RESPONSE_PATTERN_STORE.snapshot()
    from goita_ai2.current_ai.human_response_dictionary import (
        human_response_dictionary,
    )

    snapshot.update(human_response_dictionary().snapshot())
    return snapshot


def checkpoint_generic_response_patterns(reason: str = "manual") -> bool:
    return _GENERIC_RESPONSE_PATTERN_STORE.checkpoint(reason)


def reset_generic_response_patterns() -> None:
    _GENERIC_RESPONSE_PATTERN_STORE.reset()

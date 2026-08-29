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
        self._medium_patterns: Dict[str, dict] = {}
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
            "priority_queries": 0,
            "priority_hits": 0,
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
            self._medium_patterns = {}
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
            "medium_patterns": serialize_patterns(self._medium_patterns),
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
        if not isinstance(counters, dict) or not isinstance(patterns, dict):
            return False
        restored_patterns = {}
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
                "priority_queries",
                "priority_hits",
            ):
                restored_counters[name] = max(0, int(counters.get(name, 0)))
            for name in (
                "depth_sum",
                "agreement_sum",
                "confidence_sum",
                "margin_sum",
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
            self._rebuild_medium_patterns_locked()
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
            table = (
                self._medium_patterns
                if granularity == "medium"
                else self._patterns
            )
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
        table = (
            self._medium_patterns
            if granularity == "medium"
            else self._patterns
        )
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
    return _GENERIC_RESPONSE_PATTERN_STORE.snapshot()


def checkpoint_generic_response_patterns(reason: str = "manual") -> bool:
    return _GENERIC_RESPONSE_PATTERN_STORE.checkpoint(reason)


def reset_generic_response_patterns() -> None:
    _GENERIC_RESPONSE_PATTERN_STORE.reset()

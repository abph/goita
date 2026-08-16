"""Runs speculative timed search between server-side turns.

Public pass continuations and a small number of inference-sampled actions are
projected without reading real opposing hands. Results enter the exact cache
and are usable only when the actual public path matches the projection.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import threading
import time
from collections import deque
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .persistence import resolve_adaptive_value_storage


Action = Tuple[str, Optional[str], Optional[str]]
LOGGER = logging.getLogger(__name__)


@dataclass
class _ProjectedBranch:
    worker: Any
    state: Any
    path: Tuple[Action, ...]
    score: float
    pass_count: int
    label: str
    admission_reason: str = "not_evaluated"


def _positive_env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return max(1, int(default))


class BackgroundSearchDiagnostics:
    """Keeps a bounded process-wide trail of speculative-search decisions."""

    def __init__(self, max_events: int) -> None:
        self._lock = threading.RLock()
        self._events = deque(maxlen=max(16, int(max_events)))
        self._sequence = 0

    def record(self, event: Dict[str, Any]) -> None:
        with self._lock:
            self._sequence += 1
            payload = dict(event)
            payload["sequence"] = self._sequence
            self._events.append(payload)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "limit": self._events.maxlen,
                "events": [dict(event) for event in self._events],
            }


_BACKGROUND_DIAGNOSTICS = BackgroundSearchDiagnostics(
    _positive_env_int("GOITA_BACKGROUND_DIAGNOSTIC_EVENTS_GLOBAL", 256)
)


class BackgroundSearchRuntime:
    """Bounds speculative work globally and gives foreground search priority."""

    def __init__(self, *, max_workers: int, max_pending: int) -> None:
        self.max_workers = max(1, int(max_workers))
        self.max_pending = max(self.max_workers, int(max_pending))
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="goita-ai-prefetch",
        )
        self._lock = threading.RLock()
        self._futures = set()
        self._pending = 0
        self._active = 0
        self._foreground_active = 0
        self._counters = {
            "accepted": 0,
            "rejected_busy": 0,
            "rejected_full": 0,
            "started": 0,
            "finished": 0,
            "cancelled_queued": 0,
            "submit_errors": 0,
            "max_pending_seen": 0,
            "max_active_seen": 0,
        }

    def submit_with_reason(
        self,
        fn: Callable[..., None],
        *args,
    ) -> Tuple[Optional[Future], str]:
        with self._lock:
            if self._foreground_active > 0:
                self._counters["rejected_busy"] += 1
                return None, "foreground_busy"
            if self._pending >= self.max_pending:
                self._counters["rejected_full"] += 1
                return None, "pending_full"
            self._pending += 1
            self._counters["accepted"] += 1
            self._counters["max_pending_seen"] = max(
                self._counters["max_pending_seen"],
                self._pending,
            )

        try:
            future = self._executor.submit(self._run_task, fn, args)
        except Exception:
            with self._lock:
                self._pending = max(0, self._pending - 1)
                self._counters["submit_errors"] += 1
            return None, "submit_error"

        with self._lock:
            self._futures.add(future)
        future.add_done_callback(self._on_future_done)
        return future, "accepted"

    def submit(self, fn: Callable[..., None], *args) -> Optional[Future]:
        future, _reason = self.submit_with_reason(fn, *args)
        return future

    def _run_task(self, fn: Callable[..., None], args: tuple) -> None:
        with self._lock:
            self._active += 1
            self._counters["started"] += 1
            self._counters["max_active_seen"] = max(
                self._counters["max_active_seen"],
                self._active,
            )
        try:
            fn(*args)
        finally:
            with self._lock:
                self._active = max(0, self._active - 1)
                self._pending = max(0, self._pending - 1)
                self._counters["finished"] += 1

    def _on_future_done(self, future: Future) -> None:
        with self._lock:
            self._futures.discard(future)
            if future.cancelled():
                self._pending = max(0, self._pending - 1)
                self._counters["cancelled_queued"] += 1

    def foreground_started(self) -> None:
        with self._lock:
            self._foreground_active += 1
            queued = [
                future
                for future in self._futures
                if not future.running() and not future.done()
            ]
        for future in queued:
            future.cancel()

    def foreground_finished(self) -> None:
        with self._lock:
            self._foreground_active = max(0, self._foreground_active - 1)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                **self._counters,
                "pending": self._pending,
                "active": self._active,
                "foreground_active": self._foreground_active,
                "max_workers": self.max_workers,
                "max_pending": self.max_pending,
            }

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)


class BackgroundSearchValueModel:
    """Learns recent branch value by kind and stage-and-distance context.

    Lifetime counters remain available for diagnostics, while admission uses
    exponentially decayed evidence so recent games can outweigh old results.
    """

    SCHEMA_VERSION = 3
    LEGACY_SCHEMA_VERSIONS = (1, 2)
    _PROTECTED_KIND_KEYS = frozenset(("all_pass", "current_turn"))

    def __init__(
        self,
        *,
        checkpoint_path: Optional[str] = None,
        checkpoint_operations: int = 100,
        decay_half_life_operations: int = 4096,
        decay_interval_operations: int = 64,
        max_kind_entries: int = 64,
        max_context_entries: int = 512,
        checkpoint_generations: int = 3,
    ) -> None:
        self._lock = threading.RLock()
        self._checkpoint_lock = threading.Lock()
        self._kind_stats: Dict[str, Dict[str, Any]] = {}
        self._context_stats: Dict[str, Dict[str, Any]] = {}
        self._checkpoint_path = str(checkpoint_path or "").strip()
        self._checkpoint_operations = max(1, int(checkpoint_operations))
        self._decay_half_life_operations = max(
            1,
            int(decay_half_life_operations),
        )
        self._decay_interval_operations = max(
            1,
            min(
                int(decay_interval_operations),
                self._decay_half_life_operations,
            ),
        )
        self._max_kind_entries = max(1, int(max_kind_entries))
        self._max_context_entries = max(1, int(max_context_entries))
        self._checkpoint_generations = max(0, int(checkpoint_generations))
        self._learning_operations = 0
        self._last_decay_operation = 0
        self._decay_events = 0
        self._decayed_scheduled = 0.0
        self._decayed_cache_hits = 0.0
        self._kind_evictions = 0
        self._context_evictions = 0
        self._restore_pruned_kind = 0
        self._restore_pruned_context = 0
        self._orphan_cache_hits = 0
        self._revision = 0
        self._saved_revision = 0
        self._loaded = False
        self._loaded_source = ""
        self._loaded_generation = 0
        self._restore_errors: List[str] = []
        self._recovery_count = 0
        self._last_checkpoint_error = ""
        self.restore()

    @staticmethod
    def _normalize_key(key: str) -> str:
        return str(key or "unknown").strip()[:160] or "unknown"

    @staticmethod
    def _new_bucket() -> Dict[str, Any]:
        return {
            "scheduled": 0,
            "cache_hits": 0,
            "suppressed": 0,
            "probes": 0,
            "effective_scheduled": 0.0,
            "effective_cache_hits": 0.0,
            "last_seen_operation": 0,
        }

    @staticmethod
    def _eviction_priority(
        key: str,
        bucket: Dict[str, Any],
        *,
        scope: str,
    ) -> Tuple[int, float, int, int, str]:
        protected = (
            1
            if scope == "kind"
            and key in BackgroundSearchValueModel._PROTECTED_KIND_KEYS
            else 0
        )
        return (
            protected,
            max(0.0, float(bucket.get("effective_scheduled", 0.0))),
            max(0, int(bucket.get("last_seen_operation", 0))),
            max(0, int(bucket.get("scheduled", 0))),
            key,
        )

    def _entry_limit(self, scope: str) -> int:
        return (
            self._max_kind_entries
            if scope == "kind"
            else self._max_context_entries
        )

    def _record_eviction_locked(self, scope: str, *, restoring: bool) -> None:
        if scope == "kind":
            self._kind_evictions += 1
            if restoring:
                self._restore_pruned_kind += 1
        else:
            self._context_evictions += 1
            if restoring:
                self._restore_pruned_context += 1

    def _evict_one_locked(
        self,
        table: Dict[str, Dict[str, Any]],
        *,
        scope: str,
        restoring: bool = False,
    ) -> bool:
        if not table:
            return False
        victim = min(
            table,
            key=lambda key: self._eviction_priority(
                key,
                table[key],
                scope=scope,
            ),
        )
        table.pop(victim, None)
        self._record_eviction_locked(scope, restoring=restoring)
        return True

    def _prune_table_locked(
        self,
        table: Dict[str, Dict[str, Any]],
        *,
        scope: str,
        restoring: bool = False,
    ) -> int:
        removed = 0
        limit = self._entry_limit(scope)
        while len(table) > limit:
            if not self._evict_one_locked(
                table,
                scope=scope,
                restoring=restoring,
            ):
                break
            removed += 1
        return removed

    def _ensure_bucket_locked(
        self,
        table: Dict[str, Dict[str, Any]],
        key: str,
        *,
        scope: str,
    ) -> Dict[str, Any]:
        normalized = self._normalize_key(key)
        bucket = table.get(normalized)
        if bucket is not None:
            return bucket
        bucket = self._new_bucket()
        table[normalized] = bucket
        return bucket

    def reset(self) -> None:
        with self._lock:
            self._kind_stats.clear()
            self._context_stats.clear()
            self._learning_operations = 0
            self._last_decay_operation = 0
            self._decay_events = 0
            self._decayed_scheduled = 0.0
            self._decayed_cache_hits = 0.0
            self._kind_evictions = 0
            self._context_evictions = 0
            self._restore_pruned_kind = 0
            self._restore_pruned_context = 0
            self._orphan_cache_hits = 0
            self._revision += 1

    def _apply_decay_locked(self) -> bool:
        elapsed = self._learning_operations - self._last_decay_operation
        if elapsed < self._decay_interval_operations:
            return False
        factor = 0.5 ** (
            float(elapsed) / float(self._decay_half_life_operations)
        )
        decayed_scheduled = 0.0
        decayed_cache_hits = 0.0
        for table in (self._kind_stats, self._context_stats):
            for bucket in table.values():
                scheduled = max(
                    0.0,
                    float(bucket.get("effective_scheduled", 0.0)),
                )
                cache_hits = max(
                    0.0,
                    float(bucket.get("effective_cache_hits", 0.0)),
                )
                next_scheduled = scheduled * factor
                next_cache_hits = cache_hits * factor
                bucket["effective_scheduled"] = next_scheduled
                bucket["effective_cache_hits"] = min(
                    next_scheduled,
                    next_cache_hits,
                )
                decayed_scheduled += scheduled - next_scheduled
                decayed_cache_hits += cache_hits - next_cache_hits
        self._last_decay_operation = self._learning_operations
        self._decay_events += 1
        self._decayed_scheduled += decayed_scheduled
        self._decayed_cache_hits += decayed_cache_hits
        return True

    def _record_scheduled_bucket_locked(self, bucket: Dict[str, Any]) -> None:
        bucket["scheduled"] = int(bucket.get("scheduled", 0)) + 1
        bucket["effective_scheduled"] = (
            max(0.0, float(bucket.get("effective_scheduled", 0.0))) + 1.0
        )
        bucket["last_seen_operation"] = self._learning_operations

    def _record_cache_hit_bucket_locked(self, bucket: Dict[str, Any]) -> None:
        scheduled = max(0, int(bucket.get("scheduled", 0)))
        cache_hits = max(0, int(bucket.get("cache_hits", 0)))
        effective_scheduled = max(
            0.0,
            float(bucket.get("effective_scheduled", 0.0)),
        )
        effective_cache_hits = min(
            effective_scheduled,
            max(0.0, float(bucket.get("effective_cache_hits", 0.0))),
        )
        unmatched = max(0, scheduled - cache_hits)
        available_weight = max(0.0, effective_scheduled - effective_cache_hits)
        hit_weight = available_weight / unmatched if unmatched else 0.0
        bucket["cache_hits"] = cache_hits + 1
        bucket["effective_cache_hits"] = min(
            effective_scheduled,
            effective_cache_hits + hit_weight,
        )
        bucket["last_seen_operation"] = self._learning_operations

    def _record_revision_locked(self) -> bool:
        self._revision += 1
        return (
            bool(self._checkpoint_path)
            and self._revision - self._saved_revision
            >= self._checkpoint_operations
        )

    def record_scheduled(self, kind: str, context: Optional[str] = None) -> None:
        should_checkpoint = False
        with self._lock:
            self._learning_operations += 1
            self._apply_decay_locked()
            self._record_scheduled_bucket_locked(
                self._ensure_bucket_locked(
                    self._kind_stats,
                    kind,
                    scope="kind",
                )
            )
            self._prune_table_locked(self._kind_stats, scope="kind")
            if context:
                self._record_scheduled_bucket_locked(
                    self._ensure_bucket_locked(
                        self._context_stats,
                        context,
                        scope="context",
                    )
                )
                self._prune_table_locked(
                    self._context_stats,
                    scope="context",
                )
            should_checkpoint = self._record_revision_locked()
        if should_checkpoint:
            self.checkpoint("operation_interval")

    def record_cache_hit(self, kind: str, context: Optional[str] = None) -> None:
        should_checkpoint = False
        with self._lock:
            kind_bucket = self._kind_stats.get(self._normalize_key(kind))
            context_bucket = (
                self._context_stats.get(self._normalize_key(context))
                if context
                else None
            )
            if kind_bucket is not None:
                self._record_cache_hit_bucket_locked(kind_bucket)
            else:
                self._orphan_cache_hits += 1
            if context_bucket is not None:
                self._record_cache_hit_bucket_locked(
                    context_bucket
                )
            should_checkpoint = self._record_revision_locked()
        if should_checkpoint:
            self.checkpoint("operation_interval")

    def admission_decision(
        self,
        kind: str,
        *,
        context: Optional[str] = None,
        enabled: bool,
        min_scheduled: int,
        context_min_scheduled: int = 8,
        min_hit_rate: float,
        probe_interval: int,
    ) -> Tuple[bool, Dict[str, Any]]:
        kind = str(kind or "unknown")
        if not enabled:
            return True, {"reason": "adaptive_disabled", "scope": "none"}
        if kind in ("all_pass", "current_turn"):
            return True, {"reason": "protected_branch", "scope": "kind"}
        with self._lock:
            kind_bucket = self._kind_stats.get(self._normalize_key(kind))
            context_bucket = (
                self._context_stats.get(self._normalize_key(context))
                if context
                else None
            )
            context_effective_scheduled = (
                max(
                    0.0,
                    float(
                        context_bucket.get("effective_scheduled", 0.0)
                    ),
                )
                if context_bucket is not None
                else 0.0
            )
            if context_effective_scheduled >= max(
                1,
                int(context_min_scheduled),
            ):
                bucket = context_bucket
                required_scheduled = max(1, int(context_min_scheduled))
                scope = "context"
            else:
                bucket = kind_bucket or self._new_bucket()
                required_scheduled = max(1, int(min_scheduled))
                scope = "kind"
            scheduled = int(bucket["scheduled"])
            cache_hits = int(bucket["cache_hits"])
            effective_scheduled = max(
                0.0,
                float(bucket.get("effective_scheduled", scheduled)),
            )
            effective_cache_hits = min(
                effective_scheduled,
                max(
                    0.0,
                    float(bucket.get("effective_cache_hits", cache_hits)),
                ),
            )
            effective_hit_rate = (
                effective_cache_hits / effective_scheduled
                if effective_scheduled > 1e-12
                else 0.0
            )
            details = {
                "scope": scope,
                "scheduled": scheduled,
                "cache_hits": cache_hits,
                "effective_scheduled": round(effective_scheduled, 5),
                "effective_cache_hits": round(effective_cache_hits, 5),
                "hit_rate": round(effective_hit_rate, 5),
                "lifetime_hit_rate": round(
                    cache_hits / max(1, scheduled),
                    5,
                ),
                "required_scheduled": required_scheduled,
                "minimum_hit_rate": max(0.0, float(min_hit_rate)),
            }
            if effective_scheduled < required_scheduled:
                return True, {**details, "reason": "observation_window"}
            hit_rate = effective_hit_rate
            if hit_rate >= max(0.0, float(min_hit_rate)):
                return True, {**details, "reason": "hit_rate_ok"}
            bucket["suppressed"] += 1
            self._revision += 1
            interval = max(1, int(probe_interval))
            if bucket["suppressed"] % interval == 0:
                bucket["probes"] += 1
                return True, {
                    **details,
                    "reason": "periodic_probe",
                    "suppressed": int(bucket["suppressed"]),
                    "probe_interval": interval,
                }
            return False, {
                **details,
                "reason": "low_hit_rate",
                "suppressed": int(bucket["suppressed"]),
                "probe_interval": interval,
            }

    def should_admit(
        self,
        kind: str,
        *,
        context: Optional[str] = None,
        enabled: bool,
        min_scheduled: int,
        context_min_scheduled: int = 8,
        min_hit_rate: float,
        probe_interval: int,
    ) -> bool:
        admitted, _details = self.admission_decision(
            kind,
            context=context,
            enabled=enabled,
            min_scheduled=min_scheduled,
            context_min_scheduled=context_min_scheduled,
            min_hit_rate=min_hit_rate,
            probe_interval=probe_interval,
        )
        return admitted

    @staticmethod
    def _snapshot_table(
        table: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, float]]:
        snapshot: Dict[str, Dict[str, float]] = {}
        for key, bucket in sorted(table.items()):
            scheduled = max(0, int(bucket.get("scheduled", 0)))
            cache_hits = max(0, int(bucket.get("cache_hits", 0)))
            effective_scheduled = max(
                0.0,
                float(bucket.get("effective_scheduled", scheduled)),
            )
            effective_cache_hits = min(
                effective_scheduled,
                max(
                    0.0,
                    float(bucket.get("effective_cache_hits", cache_hits)),
                ),
            )
            snapshot[key] = {
                "scheduled": scheduled,
                "cache_hits": cache_hits,
                "suppressed": max(0, int(bucket.get("suppressed", 0))),
                "probes": max(0, int(bucket.get("probes", 0))),
                "last_seen_operation": max(
                    0,
                    int(bucket.get("last_seen_operation", 0)),
                ),
                "effective_scheduled": round(effective_scheduled, 5),
                "effective_cache_hits": round(effective_cache_hits, 5),
                "hit_rate": round(
                    effective_cache_hits / effective_scheduled
                    if effective_scheduled > 1e-12
                    else 0.0,
                    5,
                ),
                "lifetime_hit_rate": round(
                    cache_hits / scheduled if scheduled else 0.0,
                    5,
                ),
            }
        return snapshot

    @staticmethod
    def _serialize_table(
        table: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        return {
            key: {
                "scheduled": max(0, int(bucket.get("scheduled", 0))),
                "cache_hits": max(0, int(bucket.get("cache_hits", 0))),
                "suppressed": max(0, int(bucket.get("suppressed", 0))),
                "probes": max(0, int(bucket.get("probes", 0))),
                "last_seen_operation": max(
                    0,
                    int(bucket.get("last_seen_operation", 0)),
                ),
                "effective_scheduled": max(
                    0.0,
                    float(bucket.get("effective_scheduled", 0.0)),
                ),
                "effective_cache_hits": max(
                    0.0,
                    float(bucket.get("effective_cache_hits", 0.0)),
                ),
            }
            for key, bucket in sorted(table.items())
        }

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "by_kind": self._snapshot_table(self._kind_stats),
                "by_context": self._snapshot_table(self._context_stats),
                "decay": {
                    "half_life_operations": self._decay_half_life_operations,
                    "interval_operations": self._decay_interval_operations,
                    "learning_operations": self._learning_operations,
                    "last_decay_operation": self._last_decay_operation,
                    "pending_operations": (
                        self._learning_operations - self._last_decay_operation
                    ),
                    "events": self._decay_events,
                    "decayed_scheduled": round(self._decayed_scheduled, 5),
                    "decayed_cache_hits": round(self._decayed_cache_hits, 5),
                },
                "capacity": {
                    "max_kind_entries": self._max_kind_entries,
                    "max_context_entries": self._max_context_entries,
                    "kind_entries": len(self._kind_stats),
                    "context_entries": len(self._context_stats),
                    "kind_evictions": self._kind_evictions,
                    "context_evictions": self._context_evictions,
                    "restore_pruned_kind": self._restore_pruned_kind,
                    "restore_pruned_context": self._restore_pruned_context,
                    "orphan_cache_hits": self._orphan_cache_hits,
                },
                "persistence": {
                    "enabled": bool(self._checkpoint_path),
                    "path": self._checkpoint_path,
                    "loaded": self._loaded,
                    "loaded_source": self._loaded_source,
                    "loaded_generation": self._loaded_generation,
                    "recovery_used": self._loaded_generation > 0,
                    "recovery_count": self._recovery_count,
                    "generation_limit": self._checkpoint_generations,
                    "restore_errors": list(self._restore_errors),
                    "revision": self._revision,
                    "saved_revision": self._saved_revision,
                    "last_error": self._last_checkpoint_error,
                },
            }

    @staticmethod
    def _restore_table(
        value: Any,
        *,
        schema_version: int,
    ) -> Dict[str, Dict[str, Any]]:
        if not isinstance(value, dict):
            return {}
        restored: Dict[str, Dict[str, Any]] = {}
        for key, raw_bucket in value.items():
            key = str(key or "").strip()
            if not key or len(key) > 160 or not isinstance(raw_bucket, dict):
                continue
            bucket: Dict[str, Any] = {}
            for field in ("scheduled", "cache_hits", "suppressed", "probes"):
                try:
                    bucket[field] = max(0, int(raw_bucket.get(field, 0)))
                except (TypeError, ValueError):
                    bucket[field] = 0
            try:
                bucket["last_seen_operation"] = max(
                    0,
                    int(
                        raw_bucket.get("last_seen_operation", 0)
                        if schema_version >= 3
                        else 0
                    ),
                )
            except (TypeError, ValueError):
                bucket["last_seen_operation"] = 0
            for field, fallback in (
                ("effective_scheduled", bucket["scheduled"]),
                ("effective_cache_hits", bucket["cache_hits"]),
            ):
                try:
                    bucket[field] = max(
                        0.0,
                        float(
                            raw_bucket.get(field, fallback)
                            if schema_version >= 2
                            else fallback
                        ),
                    )
                except (TypeError, ValueError):
                    bucket[field] = float(fallback)
            bucket["effective_cache_hits"] = min(
                bucket["effective_scheduled"],
                bucket["effective_cache_hits"],
            )
            restored[key] = bucket
        return restored

    @staticmethod
    def _generation_path(path: Path, generation: int) -> Path:
        return path.with_name(f"{path.name}.bak.{max(1, int(generation))}")

    def _checkpoint_candidates(self, path: Path) -> List[Tuple[int, Path]]:
        return [(0, path)] + [
            (generation, self._generation_path(path, generation))
            for generation in range(1, self._checkpoint_generations + 1)
        ]

    def _decode_checkpoint(self, path: Path) -> Dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("checkpoint root must be an object")
        schema_version = int(payload.get("schema_version", 0))
        if schema_version not in (
            self.SCHEMA_VERSION,
            *self.LEGACY_SCHEMA_VERSIONS,
        ):
            raise ValueError("unsupported schema version")
        kind_stats = self._restore_table(
            payload.get("by_kind"),
            schema_version=schema_version,
        )
        context_stats = self._restore_table(
            payload.get("by_context"),
            schema_version=schema_version,
        )
        revision = max(0, int(payload.get("revision", 0)))
        raw_decay = payload.get("decay", {})
        if not isinstance(raw_decay, dict):
            raw_decay = {}
        learning_operations = max(
            0,
            int(raw_decay.get("learning_operations", 0)),
        )
        last_decay_operation = min(
            learning_operations,
            max(0, int(raw_decay.get("last_decay_operation", 0))),
        )
        raw_capacity = payload.get("capacity", {})
        if not isinstance(raw_capacity, dict):
            raw_capacity = {}
        return {
            "schema_version": schema_version,
            "kind_stats": kind_stats,
            "context_stats": context_stats,
            "revision": revision,
            "learning_operations": learning_operations,
            "last_decay_operation": last_decay_operation,
            "decay_events": max(0, int(raw_decay.get("events", 0))),
            "decayed_scheduled": max(
                0.0,
                float(raw_decay.get("decayed_scheduled", 0.0)),
            ),
            "decayed_cache_hits": max(
                0.0,
                float(raw_decay.get("decayed_cache_hits", 0.0)),
            ),
            "kind_evictions": max(
                0,
                int(raw_capacity.get("kind_evictions", 0)),
            ),
            "context_evictions": max(
                0,
                int(raw_capacity.get("context_evictions", 0)),
            ),
            "restore_pruned_kind": max(
                0,
                int(raw_capacity.get("restore_pruned_kind", 0)),
            ),
            "restore_pruned_context": max(
                0,
                int(raw_capacity.get("restore_pruned_context", 0)),
            ),
            "orphan_cache_hits": max(
                0,
                int(raw_capacity.get("orphan_cache_hits", 0)),
            ),
        }

    def restore(self) -> bool:
        if not self._checkpoint_path:
            return False
        path = Path(self._checkpoint_path)
        candidates = self._checkpoint_candidates(path)
        existing = [(generation, candidate) for generation, candidate in candidates if candidate.is_file()]
        if not existing:
            return False
        errors: List[str] = []
        if not path.is_file():
            errors.append("generation 0: checkpoint missing")
        for generation, candidate in existing:
            try:
                restored = self._decode_checkpoint(candidate)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"generation {generation}: {exc}")
                continue
            with self._lock:
                self._kind_stats = restored["kind_stats"]
                self._context_stats = restored["context_stats"]
                self._revision = restored["revision"]
                self._saved_revision = restored["revision"]
                self._learning_operations = restored["learning_operations"]
                self._last_decay_operation = restored["last_decay_operation"]
                self._decay_events = restored["decay_events"]
                self._decayed_scheduled = restored["decayed_scheduled"]
                self._decayed_cache_hits = restored["decayed_cache_hits"]
                self._kind_evictions = restored["kind_evictions"]
                self._context_evictions = restored["context_evictions"]
                self._restore_pruned_kind = restored["restore_pruned_kind"]
                self._restore_pruned_context = restored[
                    "restore_pruned_context"
                ]
                self._orphan_cache_hits = restored["orphan_cache_hits"]
                pruned = self._prune_table_locked(
                    self._kind_stats,
                    scope="kind",
                    restoring=True,
                )
                pruned += self._prune_table_locked(
                    self._context_stats,
                    scope="context",
                    restoring=True,
                )
                if (
                    restored["schema_version"] != self.SCHEMA_VERSION
                    or pruned
                    or generation > 0
                ):
                    self._revision += 1
                self._loaded = True
                self._loaded_source = str(candidate)
                self._loaded_generation = generation
                self._restore_errors = list(errors)
                if generation > 0:
                    self._recovery_count += 1
                self._last_checkpoint_error = ""
            return True
        with self._lock:
            self._restore_errors = errors
            self._last_checkpoint_error = "; ".join(errors)
        return False

    def _rotate_checkpoint_generations(self, path: Path) -> Optional[Path]:
        if self._checkpoint_generations <= 0 or not path.is_file():
            return None
        for generation in range(self._checkpoint_generations, 1, -1):
            source = self._generation_path(path, generation - 1)
            if source.is_file():
                os.replace(source, self._generation_path(path, generation))
        backup_temporary = path.with_name(
            f".{path.name}.{os.getpid()}.backup.tmp"
        )
        try:
            shutil.copyfile(path, backup_temporary)
            os.replace(backup_temporary, self._generation_path(path, 1))
        except OSError:
            try:
                backup_temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return backup_temporary

    def checkpoint(self, reason: str = "manual") -> bool:
        if not self._checkpoint_path:
            return False
        with self._lock:
            revision = self._revision
            payload = {
                "schema_version": self.SCHEMA_VERSION,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "reason": str(reason or "manual"),
                "revision": revision,
                "generation_limit": self._checkpoint_generations,
                "decay": {
                    "learning_operations": self._learning_operations,
                    "last_decay_operation": self._last_decay_operation,
                    "events": self._decay_events,
                    "decayed_scheduled": self._decayed_scheduled,
                    "decayed_cache_hits": self._decayed_cache_hits,
                },
                "capacity": {
                    "max_kind_entries": self._max_kind_entries,
                    "max_context_entries": self._max_context_entries,
                    "kind_entries": len(self._kind_stats),
                    "context_entries": len(self._context_stats),
                    "kind_evictions": self._kind_evictions,
                    "context_evictions": self._context_evictions,
                    "restore_pruned_kind": self._restore_pruned_kind,
                    "restore_pruned_context": self._restore_pruned_context,
                    "orphan_cache_hits": self._orphan_cache_hits,
                },
                "by_kind": self._serialize_table(self._kind_stats),
                "by_context": self._serialize_table(self._context_stats),
            }
        path = Path(self._checkpoint_path)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        backup_temporary: Optional[Path] = None
        try:
            with self._checkpoint_lock:
                with self._lock:
                    if path.is_file() and revision < self._saved_revision:
                        return True
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary.write_text(
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    encoding="utf-8",
                )
                backup_temporary = self._rotate_checkpoint_generations(path)
                os.replace(temporary, path)
                with self._lock:
                    self._saved_revision = max(self._saved_revision, revision)
                    self._last_checkpoint_error = ""
            return True
        except (OSError, TypeError, ValueError) as exc:
            with self._lock:
                self._last_checkpoint_error = str(exc)
            for pending in (temporary, backup_temporary):
                if pending is None:
                    continue
                try:
                    pending.unlink(missing_ok=True)
                except OSError:
                    pass
            return False


_BACKGROUND_RUNTIME = BackgroundSearchRuntime(
    max_workers=_positive_env_int("GOITA_BACKGROUND_SEARCH_WORKERS", 2),
    max_pending=_positive_env_int("GOITA_BACKGROUND_SEARCH_MAX_PENDING", 4),
)
_BACKGROUND_VALUE_STORAGE = resolve_adaptive_value_storage(os.environ)
if _BACKGROUND_VALUE_STORAGE.warning:
    LOGGER.warning(
        "AI adaptive-value storage warning: %s (path=%s)",
        _BACKGROUND_VALUE_STORAGE.warning,
        _BACKGROUND_VALUE_STORAGE.path,
    )
_BACKGROUND_VALUE_MODEL = BackgroundSearchValueModel(
    checkpoint_path=_BACKGROUND_VALUE_STORAGE.path,
    checkpoint_operations=_positive_env_int(
        "GOITA_AI_ADAPTIVE_VALUE_CHECKPOINT_OPERATIONS",
        100,
    ),
    decay_half_life_operations=_positive_env_int(
        "GOITA_AI_ADAPTIVE_VALUE_DECAY_HALF_LIFE_OPERATIONS",
        4096,
    ),
    decay_interval_operations=_positive_env_int(
        "GOITA_AI_ADAPTIVE_VALUE_DECAY_INTERVAL_OPERATIONS",
        64,
    ),
    max_kind_entries=_positive_env_int(
        "GOITA_AI_ADAPTIVE_VALUE_MAX_KIND_ENTRIES",
        64,
    ),
    max_context_entries=_positive_env_int(
        "GOITA_AI_ADAPTIVE_VALUE_MAX_CONTEXT_ENTRIES",
        512,
    ),
    checkpoint_generations=_positive_env_int(
        "GOITA_AI_ADAPTIVE_VALUE_GENERATIONS",
        3,
    ),
)


def background_search_foreground_started() -> None:
    _BACKGROUND_RUNTIME.foreground_started()


def background_search_foreground_finished() -> None:
    _BACKGROUND_RUNTIME.foreground_finished()


def background_search_runtime_snapshot() -> Dict[str, Any]:
    snapshot = _BACKGROUND_RUNTIME.snapshot()
    snapshot["adaptive_value"] = _BACKGROUND_VALUE_MODEL.snapshot()
    snapshot["adaptive_value"]["persistence"].update(
        _BACKGROUND_VALUE_STORAGE.snapshot()
    )
    snapshot["diagnostics"] = _BACKGROUND_DIAGNOSTICS.snapshot()
    return snapshot


def reset_background_search_value_model() -> None:
    """Reset helper for deterministic tests and benchmark comparisons."""
    _BACKGROUND_VALUE_MODEL.reset()


def checkpoint_background_search_value_model(reason: str = "manual") -> bool:
    return _BACKGROUND_VALUE_MODEL.checkpoint(reason)


class SearchCancellationToken:
    """A deepcopy-safe cancellation flag shared by one speculative worker."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def __deepcopy__(self, memo: Dict[int, Any]) -> "SearchCancellationToken":
        memo[id(self)] = self
        return self

    def cancel(self) -> None:
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()


class BackgroundSearchController:
    """Owns generation state without being copied into strategy previews."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.generation = 0
        self.futures: Dict[int, Future] = {}
        self.cancel_tokens: Dict[int, SearchCancellationToken] = {}
        self.branch_paths: Dict[int, Tuple[Action, ...]] = {}
        self.branch_labels: Dict[int, str] = {}
        self.branch_kinds: Dict[int, str] = {}
        self.outcomes_by_kind: Dict[str, Dict[str, int]] = {}
        self.diagnostic_events = deque(
            maxlen=max(
                16,
                _positive_env_int("GOITA_BACKGROUND_DIAGNOSTIC_EVENTS", 64),
            )
        )
        self.diagnostic_sequence = 0
        self.counters: Dict[str, int] = {
            "prefetch_calls": 0,
            "branch_candidates": 0,
            "scheduled": 0,
            "projected_non_pass": 0,
            "completed": 0,
            "cancelled": 0,
            "skipped": 0,
            "errors": 0,
            "cache_ready": 0,
            "path_mismatches": 0,
            "path_action_matches": 0,
            "path_completions": 0,
            "throttled": 0,
            "adaptive_skipped": 0,
            "background_cache_hits": 0,
        }
        self.last_status = "idle"
        self.last_pass_count = 0
        self.last_cache_key: Optional[str] = None
        self.last_paths: List[Tuple[Action, ...]] = []

    def __deepcopy__(self, memo: Dict[int, Any]) -> "BackgroundSearchController":
        memo[id(self)] = self
        return self

    def count_outcome(self, kind: str, outcome: str, amount: int = 1) -> None:
        bucket = self.outcomes_by_kind.setdefault(
            str(kind or "unknown"),
            {
                "scheduled": 0,
                "matches": 0,
                "mismatches": 0,
                "completions": 0,
                "cache_ready": 0,
                "cancelled": 0,
                "skipped": 0,
                "errors": 0,
                "cache_hits": 0,
            },
        )
        bucket[outcome] = int(bucket.get(outcome, 0)) + max(0, int(amount))

    def record_event(
        self,
        event: str,
        reason: str,
        **details: Any,
    ) -> None:
        with self.lock:
            self.diagnostic_sequence += 1
            payload = {
                "sequence": self.diagnostic_sequence,
                "timestamp": round(time.time(), 3),
                "event": str(event or "unknown"),
                "reason": str(reason or "unspecified"),
                **details,
            }
            self.diagnostic_events.append(payload)
        _BACKGROUND_DIAGNOSTICS.record(payload)


class BackgroundSearchMixin:
    """Projects bounded public branches and fills the exact search cache."""

    def _initialize_background_search(self) -> None:
        self._background_search_controller = BackgroundSearchController()

    def _background_search_conditions(self) -> Dict[str, Any]:
        return {
            "search_seconds": float(getattr(self, "TIME_SEARCH_MAX_SECONDS", 0.0)),
            "search_samples": int(getattr(self, "TIME_SEARCH_SAMPLE_COUNT", 0)),
            "search_max_depth": int(getattr(self, "TIME_SEARCH_MAX_DEPTH", 0)),
            "search_max_nodes": int(getattr(self, "TIME_SEARCH_MAX_NODES", 0)),
            "inference_samples": int(
                getattr(self, "TIME_SEARCH_BACKGROUND_INFERENCE_SAMPLES", 0)
            ),
            "max_branches": int(
                getattr(self, "TIME_SEARCH_BACKGROUND_MAX_BRANCHES", 0)
            ),
        }

    def _background_search_clone(self, state):
        source_id = id(state)
        future_state = copy.deepcopy(state)
        prediction_states = getattr(self, "_prediction_rollforward_states", [])
        worker = copy.deepcopy(self, {id(prediction_states): []})
        worker._prediction_cache_rollforward_enabled = False
        worker._prediction_rollforward_states = []
        worker._prediction_rollforward_key = None

        source_tracker = self._track.get(source_id)
        worker._track = {}
        if source_tracker is not None:
            worker._track[id(future_state)] = copy.deepcopy(source_tracker)

        initial_hand = self._my_initial_hands_by_state_id.get(source_id)
        worker._my_initial_hands_by_state_id = {}
        if initial_hand is not None:
            worker._my_initial_hands_by_state_id[id(future_state)] = list(initial_hand)
        return worker, future_state

    def _background_clone_branch(self, worker, state):
        state_id = id(state)
        branch_state = copy.deepcopy(state)
        branch_worker = copy.deepcopy(worker)
        branch_worker._prediction_cache_rollforward_enabled = False
        branch_worker._prediction_rollforward_states = []
        branch_worker._prediction_rollforward_key = None
        tracker = worker._track.get(state_id)
        branch_worker._track = {}
        if tracker is not None:
            branch_worker._track[id(branch_state)] = copy.deepcopy(tracker)
        initial_hand = worker._my_initial_hands_by_state_id.get(state_id)
        branch_worker._my_initial_hands_by_state_id = {}
        if initial_hand is not None:
            branch_worker._my_initial_hands_by_state_id[id(branch_state)] = list(
                initial_hand
            )
        return branch_worker, branch_state

    @staticmethod
    def _background_apply_public_action(worker, state, actor: str, action: Action) -> None:
        action_type, block, attack = action
        if action_type == "pass":
            state.apply_pass(actor)
        elif action_type == "receive":
            state.apply_receive(actor, block)
        elif action_type == "attack":
            state.apply_attack(actor, attack)
        else:
            state.apply_attack_after_block(actor, block, attack)
        worker.on_public_action(state, actor, action)

    def _background_public_action(self, actor: str, action: Action) -> Action:
        """Return only the portion of an action observable by this AI."""
        action_type, block, attack = action
        if action_type == "attack_after_block" and actor != self.me:
            return (action_type, None, attack)
        return action

    def _background_search_pass_projection(self, worker, state) -> Optional[int]:
        """Advance only through legal public passes until this AI acts."""
        if self.me is None:
            return None

        pass_count = 0
        max_passes = max(0, int(self.TIME_SEARCH_BACKGROUND_MAX_PASSES))
        while state.turn != self.me:
            if state.finished or state.phase != "receive" or pass_count >= max_passes:
                return None
            actor = state.turn
            pass_action: Action = ("pass", None, None)
            if pass_action not in state.legal_actions(actor):
                return None
            self._background_apply_public_action(worker, state, actor, pass_action)
            pass_count += 1
        return pass_count

    def _background_pass_branch(self, state) -> Optional[_ProjectedBranch]:
        worker, future_state = self._background_search_clone(state)
        pass_count = self._background_search_pass_projection(worker, future_state)
        if pass_count is None:
            return None
        path = tuple(("pass", None, None) for _ in range(pass_count))
        return _ProjectedBranch(
            worker=worker,
            state=future_state,
            path=path,
            score=0.0,
            pass_count=pass_count,
            label="all_pass" if path else "current_turn",
            admission_reason="protected_branch",
        )

    def _background_sampled_branches(self, state, limit: int) -> List[_ProjectedBranch]:
        """Vote on likely public paths across several inferred hidden deals."""
        if limit <= 0 or self.me is None:
            return []
        worker, public_state = self._background_search_clone(state)
        tracker = worker._track.get(id(public_state))
        if tracker is None:
            return []
        sample_count = max(
            1,
            int(getattr(self, "TIME_SEARCH_BACKGROUND_INFERENCE_SAMPLES", 1)),
        )
        allowed_actions = {
            str(action_type)
            for action_type in getattr(
                self,
                "TIME_SEARCH_BACKGROUND_ALLOWED_SAMPLED_ACTIONS",
                ("pass", "attack"),
            )
        }
        sampled_states = worker._timed_search_sample_states(
            public_state,
            self.me,
            tracker,
            sample_count,
        )
        if not sampled_states:
            return []

        initial_hand = worker._my_initial_hands_by_state_id.get(id(public_state))
        votes: Dict[Action, List[_ProjectedBranch]] = {}
        for sampled in sampled_states:
            branch = self._background_project_sample(
                worker,
                tracker,
                initial_hand,
                sampled,
            )
            if (
                branch is not None
                and branch.path
                and branch.path[0][0] in allowed_actions
            ):
                votes.setdefault(branch.path[0], []).append(branch)

        ranked_votes = sorted(
            votes.items(),
            key=lambda item: (
                len(item[1]),
                sum(branch.score for branch in item[1]) / len(item[1]),
                repr(item[0]),
            ),
            reverse=True,
        )
        selected: List[_ProjectedBranch] = []
        for _first_action, branches in ranked_votes[:limit]:
            representative = max(branches, key=lambda branch: branch.score)
            representative.score += len(branches) * 100_000.0
            selected.append(representative)
        return selected

    def _background_project_sample(
        self,
        worker,
        tracker: dict,
        initial_hand: Optional[List[str]],
        sampled,
    ) -> Optional[_ProjectedBranch]:
        """Return the highest-priority path for one inferred hidden deal."""
        sampled_worker = copy.deepcopy(worker)
        sampled_worker._track = {id(sampled): copy.deepcopy(tracker)}
        sampled_worker._my_initial_hands_by_state_id = {}
        if initial_hand is not None:
            sampled_worker._my_initial_hands_by_state_id[id(sampled)] = list(
                initial_hand
            )
        beam: List[_ProjectedBranch] = [
            _ProjectedBranch(sampled_worker, sampled, tuple(), 0.0, 0, "sampled")
        ]
        completed: List[_ProjectedBranch] = []
        max_actions = max(1, int(self.TIME_SEARCH_BACKGROUND_MAX_ACTIONS))
        branch_width = max(1, int(self.TIME_SEARCH_BACKGROUND_BRANCH_WIDTH))

        for _depth in range(max_actions):
            expanded: List[_ProjectedBranch] = []
            for node in beam:
                if node.state.finished:
                    continue
                if node.state.turn == self.me:
                    if node.path and any(action[0] != "pass" for action in node.path):
                        completed.append(node)
                    continue
                actor = node.state.turn
                actions = list(node.state.legal_actions(actor))
                actions.sort(
                    key=lambda action: node.worker._timed_search_action_priority(
                        node.state, actor, action
                    ),
                    reverse=True,
                )
                for action in actions[:branch_width]:
                    branch_worker, branch_state = self._background_clone_branch(
                        node.worker, node.state
                    )
                    try:
                        priority = branch_worker._timed_search_action_priority(
                            branch_state, actor, action
                        )
                        self._background_apply_public_action(
                            branch_worker, branch_state, actor, action
                        )
                    except Exception:
                        continue
                    public_action = self._background_public_action(actor, action)
                    path = node.path + (public_action,)
                    expanded.append(
                        _ProjectedBranch(
                            branch_worker,
                            branch_state,
                            path,
                            node.score + float(priority),
                            sum(1 for item in path if item[0] == "pass"),
                            "sampled",
                        )
                    )
            expanded.sort(key=lambda branch: branch.score, reverse=True)
            beam = expanded[:branch_width]
            if not beam:
                break

        for node in beam:
            if (
                node.state.turn == self.me
                and node.path
                and any(action[0] != "pass" for action in node.path)
            ):
                completed.append(node)

        return max(completed, key=lambda branch: branch.score, default=None)

    def _background_search_projections(self, state) -> List[_ProjectedBranch]:
        max_branches = max(1, int(self.TIME_SEARCH_BACKGROUND_MAX_BRANCHES))
        projections: List[_ProjectedBranch] = []
        pass_branch = self._background_pass_branch(state)
        if pass_branch is not None:
            projections.append(pass_branch)
        remaining = max(0, max_branches - len(projections))
        sampled = self._background_sampled_branches(state, remaining)
        for branch in sampled:
            kind = self._background_branch_kind(branch)
            context = self._background_branch_context(branch)
            admitted, admission = _BACKGROUND_VALUE_MODEL.admission_decision(
                kind,
                context=context,
                enabled=bool(
                    getattr(self, "TIME_SEARCH_BACKGROUND_ADAPTIVE_ENABLED", True)
                ),
                min_scheduled=int(
                    getattr(self, "TIME_SEARCH_BACKGROUND_ADAPTIVE_MIN_SCHEDULED", 16)
                ),
                context_min_scheduled=int(
                    getattr(
                        self,
                        "TIME_SEARCH_BACKGROUND_ADAPTIVE_CONTEXT_MIN_SCHEDULED",
                        8,
                    )
                ),
                min_hit_rate=float(
                    getattr(self, "TIME_SEARCH_BACKGROUND_ADAPTIVE_MIN_HIT_RATE", 0.06)
                ),
                probe_interval=int(
                    getattr(self, "TIME_SEARCH_BACKGROUND_ADAPTIVE_PROBE_INTERVAL", 8)
                ),
            )
            branch.admission_reason = str(admission.get("reason", "unknown"))
            if admitted:
                projections.append(branch)
            else:
                controller = self._background_search_controller
                with controller.lock:
                    controller.counters["adaptive_skipped"] += 1
                controller.record_event(
                    "branch_suppressed",
                    branch.admission_reason,
                    seat=self.me,
                    branch_kind=kind,
                    branch_context=context,
                    path_length=len(branch.path),
                    pass_count=branch.pass_count,
                    **{
                        key: value
                        for key, value in admission.items()
                        if key != "reason"
                    },
                )
        return projections[:max_branches]

    def _background_search_worker(
        self,
        generation: int,
        token: SearchCancellationToken,
        worker,
        future_state,
        branch: _ProjectedBranch,
    ) -> None:
        if token.is_set():
            self._finish_background_search(
                generation,
                "cancelled",
                branch,
                None,
                reason="cancelled_before_start",
            )
            return

        try:
            worker._time_search_cancel_event = token
            worker._time_search_background_branch_kind = self._background_branch_kind(
                branch
            )
            worker._time_search_background_branch_context = (
                self._background_branch_context(branch)
            )
            actions = future_state.legal_actions(self.me)
            if len(actions) < 2 or future_state.finished:
                self._finish_background_search(
                    generation,
                    "skipped",
                    branch,
                    None,
                    reason=(
                        "projected_state_finished"
                        if future_state.finished
                        else "single_legal_action"
                    ),
                )
                return
            worker.select_action(future_state, self.me, actions)
            cache_key = worker.last_time_search_cache_key
            if token.is_set():
                status = "cancelled"
                reason = "cancelled_during_search"
            elif cache_key:
                status = "cache_ready"
                reason = "exact_result_cached"
            else:
                status = "skipped"
                reason = "no_cacheable_result"
            self._finish_background_search(
                generation,
                status,
                branch,
                cache_key,
                reason=reason,
            )
        except Exception as exc:
            self._finish_background_search(
                generation,
                "errors",
                branch,
                None,
                reason="worker_exception",
                error_type=type(exc).__name__,
            )

    def _finish_background_search(
        self,
        generation: int,
        status: str,
        branch: _ProjectedBranch,
        cache_key: Optional[str],
        *,
        reason: str = "unspecified",
        error_type: Optional[str] = None,
    ) -> None:
        controller = self._background_search_controller
        with controller.lock:
            controller.counters[status] = int(controller.counters.get(status, 0)) + 1
            if status in ("cache_ready", "skipped"):
                controller.counters["completed"] += 1
            if generation == controller.generation:
                controller.last_status = status
                controller.last_cache_key = cache_key
            branch_kind = self._background_branch_kind(branch)
            controller.count_outcome(branch_kind, status)
        controller.record_event(
            "branch_finished",
            reason,
            seat=self.me,
            generation=generation,
            status=status,
            branch_kind=branch_kind,
            branch_context=self._background_branch_context(branch),
            path_length=len(branch.path),
            pass_count=branch.pass_count,
            cache_ready=bool(cache_key),
            error_type=error_type,
            effective_budget=dict(
                getattr(branch.worker, "last_time_search_budget", None) or {}
            ),
        )
        from goita_ai2.current_ai.telemetry import record_ai_background_finish

        record_ai_background_finish(status)

    def _background_future_done(
        self,
        generation: int,
        branch_index: int,
        branch: _ProjectedBranch,
        future: Future,
    ) -> None:
        if future.cancelled():
            self._finish_background_search(
                generation,
                "cancelled",
                branch,
                None,
                reason="cancelled_while_queued",
            )
        controller = self._background_search_controller
        with controller.lock:
            if controller.futures.get(branch_index) is future:
                controller.futures.pop(branch_index, None)
                controller.cancel_tokens.pop(branch_index, None)

    def prefetch_next_turn(self, state) -> bool:
        """Schedule bounded exact-cache prefetches for likely public branches."""
        controller = self._background_search_controller
        if (
            not self.TIME_SEARCH_BACKGROUND_ENABLED
            or not self.TIME_SEARCH_ENABLED
            or not self.TIME_SEARCH_CACHE_ENABLED
            or self.me is None
            or state.finished
        ):
            if not self.TIME_SEARCH_BACKGROUND_ENABLED:
                reason = "background_disabled"
            elif not self.TIME_SEARCH_ENABLED:
                reason = "timed_search_disabled"
            elif not self.TIME_SEARCH_CACHE_ENABLED:
                reason = "cache_disabled"
            elif self.me is None:
                reason = "seat_unbound"
            else:
                reason = "round_finished"
            controller.record_event(
                "prefetch_skipped",
                reason,
                seat=self.me,
            )
            return False

        self._ensure_trackers(state)
        projections = self._background_search_projections(state)
        if not projections:
            controller.record_event(
                "prefetch_skipped",
                "no_projected_branch",
                seat=self.me,
                **self._background_search_conditions(),
            )
            return False

        accepted = 0
        throttled = 0
        with controller.lock:
            superseded = sum(
                1 for future in controller.futures.values() if not future.done()
            )
            controller.generation += 1
            generation = controller.generation
            for token in list(controller.cancel_tokens.values()):
                token.cancel()
            for future in list(controller.futures.values()):
                future.cancel()
            controller.futures.clear()
            controller.cancel_tokens.clear()
            controller.branch_paths.clear()
            controller.branch_labels.clear()
            controller.branch_kinds.clear()
            controller.counters["prefetch_calls"] += 1
            controller.counters["branch_candidates"] += len(projections)
            controller.last_status = "scheduled"
            controller.last_pass_count = projections[0].pass_count
            controller.last_cache_key = None
            controller.last_paths = [branch.path for branch in projections]
            if superseded:
                controller.record_event(
                    "prefetch_cancelled",
                    "superseded_by_new_prefetch",
                    seat=self.me,
                    generation=generation,
                    branch_count=superseded,
                )
            for branch_index, branch in enumerate(projections):
                token = SearchCancellationToken()
                future, submit_reason = _BACKGROUND_RUNTIME.submit_with_reason(
                    self._background_search_worker,
                    generation,
                    token,
                    branch.worker,
                    branch.state,
                    branch,
                )
                if future is None:
                    controller.counters["throttled"] += 1
                    throttled += 1
                    controller.record_event(
                        "branch_throttled",
                        submit_reason,
                        seat=self.me,
                        generation=generation,
                        branch_kind=self._background_branch_kind(branch),
                        branch_context=self._background_branch_context(branch),
                        path_length=len(branch.path),
                        pass_count=branch.pass_count,
                        **self._background_search_conditions(),
                    )
                    continue
                accepted += 1
                controller.counters["scheduled"] += 1
                if any(action[0] != "pass" for action in branch.path):
                    controller.counters["projected_non_pass"] += 1
                controller.futures[branch_index] = future
                controller.cancel_tokens[branch_index] = token
                controller.branch_paths[branch_index] = branch.path
                controller.branch_labels[branch_index] = branch.label
                branch_kind = self._background_branch_kind(branch)
                branch_context = self._background_branch_context(branch)
                controller.branch_kinds[branch_index] = branch_kind
                controller.count_outcome(branch_kind, "scheduled")
                _BACKGROUND_VALUE_MODEL.record_scheduled(
                    branch_kind,
                    branch_context,
                )
                controller.record_event(
                    "branch_scheduled",
                    branch.admission_reason,
                    seat=self.me,
                    generation=generation,
                    branch_index=branch_index,
                    branch_kind=branch_kind,
                    branch_context=branch_context,
                    path=[list(action) for action in branch.path],
                    path_length=len(branch.path),
                    pass_count=branch.pass_count,
                    **self._background_search_conditions(),
                )
                future.add_done_callback(
                    lambda completed, current_generation=generation,
                    current_index=branch_index, projected=branch: self._background_future_done(
                        current_generation,
                        current_index,
                        projected,
                        completed,
                    )
                )
            if accepted == 0:
                controller.last_status = "throttled"
            controller.record_event(
                "prefetch_summary",
                "scheduled" if accepted else "all_branches_throttled",
                seat=self.me,
                generation=generation,
                candidates=len(projections),
                scheduled=accepted,
                throttled=throttled,
                **self._background_search_conditions(),
            )
        from goita_ai2.current_ai.telemetry import record_ai_background_prefetch

        record_ai_background_prefetch(
            candidates=len(projections),
            scheduled=accepted,
            throttled=throttled,
        )
        return accepted > 0

    def retain_background_search_for_action(self, action: Action) -> bool:
        """Cancel projected paths whose next public action did not occur."""
        controller = self._background_search_controller
        public_action = action
        if action[0] == "attack_after_block":
            public_action = (action[0], None, action[2])
        retained = False
        matches = 0
        mismatches = 0
        completions = 0
        with controller.lock:
            for branch_index, path in list(controller.branch_paths.items()):
                if not path:
                    controller.branch_paths.pop(branch_index, None)
                    controller.branch_labels.pop(branch_index, None)
                    controller.branch_kinds.pop(branch_index, None)
                    continue
                branch_kind = controller.branch_kinds.get(branch_index, "unknown")
                if path and path[0] == public_action:
                    retained = True
                    matches += 1
                    remaining = path[1:]
                    controller.branch_paths[branch_index] = remaining
                    if not remaining:
                        completions += 1
                        controller.count_outcome(branch_kind, "completions")
                    controller.count_outcome(branch_kind, "matches")
                    continue
                token = controller.cancel_tokens.get(branch_index)
                future = controller.futures.get(branch_index)
                if token is not None:
                    token.cancel()
                if future is not None:
                    future.cancel()
                controller.branch_paths.pop(branch_index, None)
                controller.branch_labels.pop(branch_index, None)
                controller.branch_kinds.pop(branch_index, None)
                controller.counters["path_mismatches"] += 1
                mismatches += 1
                controller.count_outcome(branch_kind, "mismatches")
            controller.counters["path_action_matches"] += matches
            controller.counters["path_completions"] += completions
            if retained:
                controller.last_status = "path_retained"
            elif controller.branch_paths:
                controller.last_status = "path_mismatch"
        if matches or mismatches or completions:
            controller.record_event(
                "path_observed",
                "prediction_matched" if matches else "prediction_mismatch",
                seat=self.me,
                action=list(public_action),
                matches=matches,
                mismatches=mismatches,
                completions=completions,
            )
        from goita_ai2.current_ai.telemetry import record_ai_background_paths

        if matches or mismatches or completions:
            record_ai_background_paths(
                matches=matches,
                mismatches=mismatches,
                completions=completions,
            )
        return retained

    def record_background_cache_hit(
        self,
        kind: Optional[str],
        context: Optional[str] = None,
        *,
        cached_compute_ms: float = 0.0,
    ) -> None:
        """Attribute a foreground cache use to its speculative branch kind."""
        branch_kind = str(kind or "unknown")
        controller = self._background_search_controller
        with controller.lock:
            controller.counters["background_cache_hits"] += 1
            controller.count_outcome(branch_kind, "cache_hits")
        controller.record_event(
            "background_cache_hit",
            "foreground_reused_exact_result",
            seat=self.me,
            branch_kind=branch_kind,
            branch_context=context,
            cached_compute_ms=round(max(0.0, float(cached_compute_ms)), 3),
        )
        _BACKGROUND_VALUE_MODEL.record_cache_hit(branch_kind, context)

    def cancel_background_search(self, reason: str = "manual") -> None:
        controller = self._background_search_controller
        with controller.lock:
            cancelled_count = sum(
                1 for future in controller.futures.values() if not future.done()
            )
            controller.generation += 1
            for token in list(controller.cancel_tokens.values()):
                token.cancel()
            for future in list(controller.futures.values()):
                future.cancel()
            controller.branch_paths.clear()
            controller.branch_labels.clear()
            controller.branch_kinds.clear()
            controller.last_status = "cancelled"
        controller.record_event(
            "prefetch_cancelled",
            str(reason or "manual"),
            seat=self.me,
            generation=controller.generation,
            branch_count=cancelled_count,
        )

    def wait_for_background_search(self, timeout: float = 2.0) -> bool:
        """Wait helper for tests and diagnostics; gameplay never calls this."""
        with self._background_search_controller.lock:
            futures = list(self._background_search_controller.futures.values())
        if not futures:
            return True
        deadline = time.monotonic() + max(0.01, float(timeout))
        for future in futures:
            try:
                future.result(timeout=max(0.01, deadline - time.monotonic()))
            except CancelledError:
                continue
            except TimeoutError:
                return False
            except Exception:
                return False
        return True

    def background_search_snapshot(self) -> Dict[str, Any]:
        controller = self._background_search_controller
        with controller.lock:
            return {
                **controller.counters,
                "generation": controller.generation,
                "last_status": controller.last_status,
                "last_pass_count": controller.last_pass_count,
                "last_cache_key": controller.last_cache_key,
                "last_paths": [list(path) for path in controller.last_paths],
                "diagnostic_event_limit": controller.diagnostic_events.maxlen,
                "diagnostic_events": [
                    dict(event) for event in controller.diagnostic_events
                ],
                "outcomes_by_kind": {
                    kind: dict(values)
                    for kind, values in sorted(controller.outcomes_by_kind.items())
                },
                "running_count": sum(
                    1 for future in controller.futures.values() if not future.done()
                ),
                "running": any(
                    not future.done() for future in controller.futures.values()
                ),
                "runtime": background_search_runtime_snapshot(),
            }

    @staticmethod
    def _background_branch_kind(branch: _ProjectedBranch) -> str:
        if branch.label == "all_pass":
            return "all_pass"
        if not branch.path:
            return str(branch.label or "current_turn")
        return f"{branch.label}:{branch.path[0][0]}"

    def _background_branch_context(self, branch: _ProjectedBranch) -> str:
        hand_size = (
            len(branch.state.hands.get(self.me, []))
            if self.me is not None
            else 0
        )
        if hand_size >= 6:
            stage = "early"
        elif hand_size >= 3:
            stage = "middle"
        else:
            stage = "endgame"
        distance = len(branch.path)
        distance_label = str(distance) if distance < 4 else "4_plus"
        return (
            f"{self._background_branch_kind(branch)}"
            f"|{stage}|distance_{distance_label}"
        )

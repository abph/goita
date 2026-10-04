"""Caches inferred hidden-hand samples generated from public information.

Entries store compact immutable assignments instead of mutable game states.
The cache is process-wide, bounded, and safe to share between foreground and
speculative searches because its keys never include real opposing hands.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple


@dataclass(frozen=True)
class PredictionSample:
    opponent_hands: Tuple[Tuple[str, Tuple[str, ...]], ...]
    opponent_hidden: Tuple[Tuple[str, Tuple[str, ...]], ...]
    opponent_had_both_kings: Tuple[Tuple[str, bool], ...]
    last_block: Optional[str]


@dataclass
class _PredictionEntry:
    samples: Tuple[PredictionSample, ...]
    created_at: float
    last_accessed_at: float
    source: str


class PredictionSampleCache:
    """Thread-safe LRU cache that can replace entries with larger sample sets."""

    def __init__(
        self,
        *,
        max_entries: int,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_entries = max(1, int(max_entries))
        self.ttl_seconds = max(0.01, float(ttl_seconds))
        self._clock = clock
        self._lock = threading.RLock()
        self._entries: "OrderedDict[str, _PredictionEntry]" = OrderedDict()
        self._inflight: Dict[str, threading.Event] = {}
        self._counters = {
            "hits": 0,
            "misses": 0,
            "partial_misses": 0,
            "partial_hits": 0,
            "stores": 0,
            "replacements": 0,
            "evictions": 0,
            "expired": 0,
            "inflight_claims": 0,
            "inflight_waits": 0,
            "inflight_hits": 0,
            "inflight_timeouts": 0,
            "generated_samples": 0,
            "reused_samples": 0,
            "rollforward_stores": 0,
            "rollforward_samples": 0,
        }

    def _prune_expired(self, now: float) -> None:
        expired = [
            key
            for key, entry in self._entries.items()
            if now - entry.created_at >= self.ttl_seconds
        ]
        for key in expired:
            self._entries.pop(key, None)
            self._counters["expired"] += 1

    def get(
        self,
        key: str,
        count: int,
        *,
        count_miss: bool = True,
    ) -> Optional[Tuple[PredictionSample, ...]]:
        requested = max(1, int(count))
        with self._lock:
            now = self._clock()
            self._prune_expired(now)
            entry = self._entries.get(str(key))
            if entry is None:
                if count_miss:
                    self._counters["misses"] += 1
                return None
            if len(entry.samples) < requested:
                if count_miss:
                    self._counters["partial_misses"] += 1
                return None
            entry.last_accessed_at = now
            self._entries.move_to_end(str(key))
            self._counters["hits"] += 1
            self._counters["reused_samples"] += requested
            return entry.samples[:requested]

    def get_available(
        self,
        key: str,
        count: int,
    ) -> Optional[Tuple[PredictionSample, ...]]:
        """Return a smaller surviving set so the caller can generate only its deficit."""
        requested = max(1, int(count))
        with self._lock:
            now = self._clock()
            self._prune_expired(now)
            entry = self._entries.get(str(key))
            if entry is None or not entry.samples:
                return None
            available = entry.samples[:requested]
            entry.last_accessed_at = now
            self._entries.move_to_end(str(key))
            self._counters["partial_hits"] += 1
            self._counters["reused_samples"] += len(available)
            return available

    def claim(self, key: str) -> Tuple[bool, threading.Event]:
        with self._lock:
            existing = self._inflight.get(str(key))
            if existing is not None:
                self._counters["inflight_waits"] += 1
                return False, existing
            event = threading.Event()
            self._inflight[str(key)] = event
            self._counters["inflight_claims"] += 1
            return True, event

    def wait(
        self,
        key: str,
        event: threading.Event,
        count: int,
        timeout_seconds: float,
        cancel_event=None,
    ) -> Optional[Tuple[PredictionSample, ...]]:
        deadline = self._clock() + max(0.0, float(timeout_seconds))
        while not event.is_set():
            if cancel_event is not None and cancel_event.is_set():
                return None
            remaining = deadline - self._clock()
            if remaining <= 0.0:
                with self._lock:
                    self._counters["inflight_timeouts"] += 1
                return None
            event.wait(min(0.02, remaining))
        result = self.get(key, count, count_miss=False)
        if result is not None:
            with self._lock:
                self._counters["inflight_hits"] += 1
        return result

    def _store_locked(
        self,
        key: str,
        samples: Tuple[PredictionSample, ...],
        *,
        source: str,
        generated_count: Optional[int] = None,
    ) -> None:
        if not samples:
            return
        now = self._clock()
        self._prune_expired(now)
        existing = self._entries.get(str(key))
        if existing is not None and len(samples) <= len(existing.samples):
            return
        if existing is not None:
            self._entries.pop(str(key), None)
            self._counters["replacements"] += 1
        self._entries[str(key)] = _PredictionEntry(
            samples=tuple(samples),
            created_at=now,
            last_accessed_at=now,
            source=str(source),
        )
        self._counters["stores"] += 1
        if source == "rollforward":
            self._counters["rollforward_stores"] += 1
            self._counters["rollforward_samples"] += len(samples)
        else:
            self._counters["generated_samples"] += max(
                0,
                int(len(samples) if generated_count is None else generated_count),
            )
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self._counters["evictions"] += 1

    def store_rollforward(
        self,
        key: str,
        samples: Tuple[PredictionSample, ...],
    ) -> None:
        with self._lock:
            self._store_locked(str(key), tuple(samples), source="rollforward")

    def finish(
        self,
        key: str,
        samples: Tuple[PredictionSample, ...],
        *,
        generated_count: Optional[int] = None,
    ) -> None:
        try:
            if samples:
                with self._lock:
                    self._store_locked(
                        str(key),
                        tuple(samples),
                        source="generated",
                        generated_count=generated_count,
                    )
        finally:
            with self._lock:
                event = self._inflight.pop(str(key), None)
                if event is not None:
                    event.set()

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            inflight = list(self._inflight.values())
            self._inflight.clear()
            for event in inflight:
                event.set()
            for key in self._counters:
                self._counters[key] = 0

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            self._prune_expired(self._clock())
            hits = int(self._counters["hits"])
            misses = int(self._counters["misses"])
            partial = int(self._counters["partial_misses"])
            lookups = hits + misses + partial
            reuse_lookups = hits + int(self._counters["partial_hits"])
            return {
                **self._counters,
                "lookups": lookups,
                "hit_rate": round(hits / lookups if lookups else 0.0, 5),
                "reuse_rate": round(
                    reuse_lookups / lookups if lookups else 0.0,
                    5,
                ),
                "size": len(self._entries),
                "inflight": len(self._inflight),
                "max_entries": self.max_entries,
                "ttl_seconds": self.ttl_seconds,
                "sample_counts": sorted(
                    (len(entry.samples) for entry in self._entries.values()),
                    reverse=True,
                ),
            }


_PREDICTION_SAMPLE_CACHE = PredictionSampleCache(
    max_entries=64,
    ttl_seconds=300.0,
)


def prediction_sample_cache_snapshot() -> Dict[str, Any]:
    return _PREDICTION_SAMPLE_CACHE.snapshot()


def clear_prediction_sample_cache() -> None:
    _PREDICTION_SAMPLE_CACHE.clear()


class PredictionCacheMixin:
    """Provides current-AI access to the shared prediction sample cache."""

    def _initialize_prediction_cache(self) -> None:
        self.last_prediction_cache_hit = False
        self.last_prediction_cache_key: Optional[str] = None
        self.last_prediction_cache_samples = 0
        self._prediction_rollforward_states = []
        self._prediction_rollforward_key: Optional[str] = None
        self._prediction_cache_rollforward_enabled = True

    def _prediction_cache_get(self, key: str, count: int):
        if not self.TIME_SEARCH_PREDICTION_CACHE_ENABLED:
            return None
        return _PREDICTION_SAMPLE_CACHE.get(key, count)

    def _prediction_cache_claim(self, key: str):
        return _PREDICTION_SAMPLE_CACHE.claim(key)

    def _prediction_cache_get_available(self, key: str, count: int):
        if not self.TIME_SEARCH_PREDICTION_CACHE_ENABLED:
            return None
        return _PREDICTION_SAMPLE_CACHE.get_available(key, count)

    def _prediction_cache_wait(
        self,
        key: str,
        event: threading.Event,
        count: int,
        timeout_seconds: float,
        cancel_event=None,
    ):
        return _PREDICTION_SAMPLE_CACHE.wait(
            key,
            event,
            count,
            timeout_seconds,
            cancel_event=cancel_event,
        )

    def _prediction_cache_finish(
        self,
        key: str,
        samples,
        *,
        generated_count: Optional[int] = None,
    ) -> None:
        _PREDICTION_SAMPLE_CACHE.finish(
            key,
            tuple(samples),
            generated_count=generated_count,
        )

    def _prediction_cache_store_rollforward(self, key: str, samples) -> None:
        _PREDICTION_SAMPLE_CACHE.store_rollforward(key, tuple(samples))

    def prediction_sample_cache_snapshot(self) -> Dict[str, Any]:
        return prediction_sample_cache_snapshot()

    def clear_prediction_sample_cache(self) -> None:
        clear_prediction_sample_cache()

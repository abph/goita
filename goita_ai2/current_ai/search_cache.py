"""Provides public-information keys and a bounded timed-search cache.

Keys intentionally exclude opponents' real hands and hidden blocks. The cache
uses exact inference and policy fingerprints so stale or incompatible search
results cannot be applied to a later decision.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from goita_ai2.constants import ALL_SEATS


Action = Tuple[str, Optional[str], Optional[str]]
_TRACKER_CACHE_EXCLUSIONS = frozenset(
    {
        "last_time_limited_search",
        "piece_inference_revision",
        "last_piece_inference_reason",
    }
)
_SEARCH_POLICY_ATTRIBUTES = (
    "TIME_SEARCH_MAX_SECONDS",
    "TIME_SEARCH_SAMPLE_COUNT",
    "TIME_SEARCH_ROOT_BEAM",
    "TIME_SEARCH_BRANCH_BEAM",
    "TIME_SEARCH_MAX_DEPTH",
    "TIME_SEARCH_MAX_NODES",
    "TIME_SEARCH_ADAPTIVE_BUDGET_ENABLED",
    "TIME_SEARCH_ADAPTIVE_BUDGET_WARMUP",
    "TIME_SEARCH_ADAPTIVE_MIN_SECONDS",
    "TIME_SEARCH_ADAPTIVE_MIN_SAMPLES",
    "TIME_SEARCH_ADAPTIVE_EWMA_ALPHA",
    "TIME_SEARCH_PREDICTION_CACHE_ENABLED",
    "TIME_SEARCH_RULE_PRIOR_WEIGHT",
    "TIME_SEARCH_BASELINE_PRIOR",
    "TIME_SEARCH_STABLE_MARGIN",
    "TIME_SEARCH_OVERRIDE_MARGIN",
    "TIME_SEARCH_OVERRIDE_AGREEMENT",
    "TIME_SEARCH_EARLY_OVERRIDE_MIN_DEPTH",
    "TIME_SEARCH_INFORMATION_SET_ENABLED",
    "TIME_SEARCH_INFORMATION_SET_ACTION_PRIOR_WEIGHT",
    "TIME_SEARCH_INFORMATION_SET_ACTION_PRIOR_CAP",
    "WEAK_FIRST_RECEIVE_SEARCH_MAX_SECONDS",
    "WEAK_FIRST_RECEIVE_SEARCH_SAMPLE_COUNT",
    "WEAK_FIRST_RECEIVE_SEARCH_MAX_DEPTH",
    "WEAK_FIRST_RECEIVE_SEARCH_MAX_NODES",
    "WEAK_FIRST_RECEIVE_SEARCH_TARGET_DEPTH",
    "WEAK_FIRST_RECEIVE_SEARCH_MIN_OVERRIDE_DEPTH",
    "WEAK_FIRST_RECEIVE_SEARCH_OVERRIDE_AGREEMENT",
    "WEAK_FIRST_RECEIVE_SEARCH_OVERRIDE_MARGIN",
    "WEAK_FIRST_RECEIVE_SEARCH_MIN_CONFIDENCE",
    "KYOSHA_PASS_COMPARE_MAX_SECONDS",
    "KYOSHA_PASS_COMPARE_SAMPLE_COUNT",
    "KYOSHA_PASS_COMPARE_MAX_DEPTH",
    "KYOSHA_PASS_COMPARE_MAX_NODES",
    "KYOSHA_PASS_COMPARE_TARGET_DEPTH",
    "KYOSHA_PASS_COMPARE_MIN_AGREEMENT",
    "KYOSHA_PASS_COMPARE_MIN_CONFIDENCE",
    "KYOSHA_PASS_COMPARE_MIN_MARGIN",
    "ZERO_SHI_STOP_SIGNAL_OVERRIDE_MARGIN",
    "ZERO_SHI_STOP_SIGNAL_OVERRIDE_AGREEMENT",
    "ZERO_SHI_STOP_SIGNAL_MIN_CONFIDENCE",
)


def _canonical(value: Any) -> Any:
    """Convert strategy state into deterministic JSON-compatible values."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return round(value, 6)
    if is_dataclass(value):
        return _canonical(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _TRACKER_CACHE_EXCLUSIONS
        }
    if isinstance(value, (set, frozenset)):
        items = [_canonical(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return {"type": type(value).__qualname__, "repr": repr(value)}


def _digest_payload(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(
        _canonical(payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SearchPositionKey:
    """An exact cache identity plus small fields useful in diagnostics."""

    digest: str
    player: str
    phase: str
    hand_size: int
    legal_action_count: int


@dataclass
class _CacheEntry:
    result: Any
    created_at: float
    last_accessed_at: float
    source: str = "foreground"
    compute_seconds: float = 0.0
    branch_kind: Optional[str] = None
    branch_context: Optional[str] = None
    hits: int = 0


@dataclass
class _InflightEntry:
    event: threading.Event


class TimedSearchCache:
    """Thread-safe LRU cache with TTL and lightweight usage statistics."""

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
        self._entries: "OrderedDict[SearchPositionKey, _CacheEntry]" = OrderedDict()
        self._inflight: Dict[SearchPositionKey, _InflightEntry] = {}
        self._lock = threading.RLock()
        self._counters = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "replacements": 0,
            "evictions": 0,
            "expired": 0,
            "inflight_claims": 0,
            "inflight_waits": 0,
            "inflight_hits": 0,
            "inflight_timeouts": 0,
        }

    def __deepcopy__(self, memo: Dict[int, Any]) -> "TimedSearchCache":
        # Rule preview clones must share, rather than copy, synchronization state.
        memo[id(self)] = self
        return self

    def _prune_expired(self, now: float) -> None:
        expired = [
            key
            for key, entry in self._entries.items()
            if now - entry.created_at >= self.ttl_seconds
        ]
        for key in expired:
            self._entries.pop(key, None)
            self._counters["expired"] += 1

    def get(self, key: SearchPositionKey) -> Optional[Any]:
        with self._lock:
            now = self._clock()
            self._prune_expired(now)
            entry = self._entries.get(key)
            if entry is None:
                self._counters["misses"] += 1
                return None
            entry.last_accessed_at = now
            entry.hits += 1
            self._entries.move_to_end(key)
            self._counters["hits"] += 1
            return entry.result

    @staticmethod
    def _result_quality(result: Any) -> tuple:
        return (
            int(getattr(result, "depth", 0)),
            bool(getattr(result, "decisive", False)),
            int(getattr(result, "samples", 0)),
            int(getattr(result, "nodes", 0)),
        )

    def put(
        self,
        key: SearchPositionKey,
        result: Any,
        *,
        source: str = "foreground",
        compute_seconds: float = 0.0,
        branch_kind: Optional[str] = None,
        branch_context: Optional[str] = None,
    ) -> bool:
        with self._lock:
            now = self._clock()
            self._prune_expired(now)
            existing = self._entries.get(key)
            if existing is not None:
                if self._result_quality(existing.result) > self._result_quality(result):
                    self._entries.move_to_end(key)
                    return False
                self._entries.pop(key)
                self._counters["replacements"] += 1
            self._entries[key] = _CacheEntry(
                result=result,
                created_at=now,
                last_accessed_at=now,
                source=str(source or "foreground"),
                compute_seconds=max(0.0, float(compute_seconds)),
                branch_kind=str(branch_kind) if branch_kind else None,
                branch_context=str(branch_context) if branch_context else None,
            )
            self._counters["stores"] += 1
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self._counters["evictions"] += 1
            return True

    def claim_compute(self, key: SearchPositionKey) -> Tuple[bool, threading.Event]:
        """Claim one computation, or return the event owned by another caller."""
        with self._lock:
            entry = self._inflight.get(key)
            if entry is not None:
                self._counters["inflight_waits"] += 1
                return False, entry.event
            event = threading.Event()
            self._inflight[key] = _InflightEntry(event=event)
            self._counters["inflight_claims"] += 1
            return True, event

    def wait_for_compute(
        self,
        key: SearchPositionKey,
        event: threading.Event,
        timeout_seconds: float,
        cancel_event=None,
    ) -> Tuple[bool, Optional[Any]]:
        """Wait for the owner while allowing speculative callers to cancel."""
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        completed = event.is_set()
        while not completed:
            if cancel_event is not None and cancel_event.is_set():
                return False, None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                with self._lock:
                    self._counters["inflight_timeouts"] += 1
                return False, None
            completed = event.wait(min(0.02, remaining))

        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return True, None
            entry.last_accessed_at = self._clock()
            entry.hits += 1
            self._entries.move_to_end(key)
            self._counters["hits"] += 1
            self._counters["inflight_hits"] += 1
            return True, entry.result

    def finish_compute(
        self,
        key: SearchPositionKey,
        result: Optional[Any],
        *,
        source: str = "foreground",
        compute_seconds: float = 0.0,
        branch_kind: Optional[str] = None,
        branch_context: Optional[str] = None,
    ) -> None:
        """Publish the result and release every waiter for this exact key."""
        try:
            if result is not None:
                self.put(
                    key,
                    result,
                    source=source,
                    compute_seconds=compute_seconds,
                    branch_kind=branch_kind,
                    branch_context=branch_context,
                )
        finally:
            with self._lock:
                entry = self._inflight.pop(key, None)
                if entry is not None:
                    entry.event.set()

    def entry_metadata(self, key: SearchPositionKey) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            return {
                "source": entry.source,
                "compute_seconds": entry.compute_seconds,
                "branch_kind": entry.branch_kind,
                "branch_context": entry.branch_context,
                "age_seconds": max(0.0, self._clock() - entry.created_at),
                "hits": entry.hits,
            }

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            inflight = list(self._inflight.values())
            self._inflight.clear()
            for entry in inflight:
                entry.event.set()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            now = self._clock()
            self._prune_expired(now)
            hits = int(self._counters["hits"])
            misses = int(self._counters["misses"])
            lookups = hits + misses
            return {
                **self._counters,
                "lookups": lookups,
                "hit_rate": round(hits / lookups if lookups else 0.0, 5),
                "size": len(self._entries),
                "inflight": len(self._inflight),
                "max_entries": self.max_entries,
                "ttl_seconds": self.ttl_seconds,
            }


class SearchCacheMixin:
    """Builds safe position keys and owns one bounded cache per AI seat."""

    def _initialize_search_cache(self) -> None:
        self._time_search_cache = TimedSearchCache(
            max_entries=int(self.TIME_SEARCH_CACHE_MAX_ENTRIES),
            ttl_seconds=float(self.TIME_SEARCH_CACHE_TTL_SECONDS),
        )
        self.last_time_search_cache_hit = False
        self.last_time_search_cache_key: Optional[str] = None
        self.last_time_search_cache_source: Optional[str] = None
        self.last_time_search_cached_compute_ms = 0.0
        self.last_time_search_cache_branch_kind: Optional[str] = None
        self.last_time_search_cache_branch_context: Optional[str] = None

    def _search_policy_fingerprint(self) -> Dict[str, Any]:
        return {
            name: getattr(self, name, None)
            for name in _SEARCH_POLICY_ATTRIBUTES
        }

    def _timed_search_cache_key(
        self,
        state,
        player: str,
        tr: dict,
        actions: Iterable[Action],
        baseline_action: Action,
    ) -> SearchPositionKey:
        actions_list = list(actions)
        tracker_public = {
            key: value
            for key, value in tr.items()
            if key not in _TRACKER_CACHE_EXCLUSIONS
        }
        payload = {
            "version": 2,
            "player": player,
            "state": {
                "dealer": state.dealer,
                "turn": state.turn,
                "phase": state.phase,
                "attacker": state.attacker,
                "current_attack": state.current_attack,
                "king_block_used": int(state.king_block_used),
                "score": dict(state.team_score),
                "hand_sizes": {
                    seat: len(state.hands[seat])
                    for seat in ALL_SEATS
                },
                "my_hand": sorted(state.hands[player]),
                "my_hidden_blocks": sorted(state.face_down_hidden[player]),
                "my_had_both_kings": bool(state.had_both_kings.get(player, False)),
            },
            "tracker": tracker_public,
            "legal_actions": sorted(tuple(action) for action in actions_list),
            "baseline_action": tuple(baseline_action),
            "policy": self._search_policy_fingerprint(),
            "search_profile": str(getattr(self, "_time_search_profile", "default")),
        }
        return SearchPositionKey(
            digest=_digest_payload(payload),
            player=player,
            phase=str(state.phase),
            hand_size=len(state.hands[player]),
            legal_action_count=len(actions_list),
        )

    def _get_cached_timed_search(self, key: SearchPositionKey) -> Optional[Any]:
        if not self.TIME_SEARCH_CACHE_ENABLED:
            return None
        result = self._time_search_cache.get(key)
        if result is not None:
            self._adopt_timed_search_cache_metadata(key)
        return result

    def _adopt_timed_search_cache_metadata(self, key: SearchPositionKey) -> None:
        metadata = self._time_search_cache.entry_metadata(key) or {}
        self.last_time_search_cache_source = metadata.get("source")
        self.last_time_search_cached_compute_ms = (
            max(0.0, float(metadata.get("compute_seconds", 0.0))) * 1000.0
        )
        self.last_time_search_cache_branch_kind = metadata.get("branch_kind")
        self.last_time_search_cache_branch_context = metadata.get("branch_context")

    def _claim_timed_search_compute(
        self,
        key: SearchPositionKey,
    ) -> Tuple[bool, threading.Event]:
        return self._time_search_cache.claim_compute(key)

    def _wait_for_timed_search_compute(
        self,
        key: SearchPositionKey,
        event: threading.Event,
        timeout_seconds: float,
        cancel_event=None,
    ) -> Tuple[bool, Optional[Any]]:
        completed, result = self._time_search_cache.wait_for_compute(
            key,
            event,
            timeout_seconds,
            cancel_event=cancel_event,
        )
        if result is not None:
            self._adopt_timed_search_cache_metadata(key)
        return completed, result

    def _finish_timed_search_compute(
        self,
        key: SearchPositionKey,
        result: Optional[Any],
        *,
        source: str,
        compute_seconds: float,
        branch_kind: Optional[str] = None,
        branch_context: Optional[str] = None,
    ) -> None:
        self._time_search_cache.finish_compute(
            key,
            result,
            source=source,
            compute_seconds=compute_seconds,
            branch_kind=branch_kind,
            branch_context=branch_context,
        )

    def time_search_cache_snapshot(self) -> Dict[str, Any]:
        return self._time_search_cache.snapshot()

    def clear_time_search_cache(self) -> None:
        self._time_search_cache.clear()

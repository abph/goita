"""Stores reusable responses for publicly identical receive positions.
Each entry keeps pass/receive together with the planned follow-up attack, while
strict public-state keys and legality checks prevent stale plans from playing.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

from goita_ai2.constants import ALL_SEATS
from goita_ai2.current_ai.search_cache import _digest_payload


Action = Tuple[str, Optional[str], Optional[str]]

_RUNTIME_COUNTER_NAMES = (
    "hits",
    "misses",
    "stores",
    "invalid",
    "evictions",
    "expired",
    "pass_hits",
    "receive_hits",
    "background_hits",
    "foreground_hits",
    "followup_hits",
    "followup_unavailable",
)
_RUNTIME_LOCK = threading.RLock()
_RUNTIME_COUNTERS = {name: 0 for name in _RUNTIME_COUNTER_NAMES}
_RUNTIME_ESTIMATED_SAVED_MS = 0.0


def _record_runtime_counter(name: str, amount: int = 1) -> None:
    with _RUNTIME_LOCK:
        _RUNTIME_COUNTERS[name] += int(amount)


def _record_runtime_saved_ms(amount: float) -> None:
    global _RUNTIME_ESTIMATED_SAVED_MS
    with _RUNTIME_LOCK:
        _RUNTIME_ESTIMATED_SAVED_MS += max(0.0, float(amount))


def reset_conditional_response_runtime() -> None:
    """Reset process-wide metrics for tests and explicit diagnostics."""
    global _RUNTIME_ESTIMATED_SAVED_MS
    with _RUNTIME_LOCK:
        for name in _RUNTIME_COUNTER_NAMES:
            _RUNTIME_COUNTERS[name] = 0
        _RUNTIME_ESTIMATED_SAVED_MS = 0.0


@dataclass(frozen=True)
class ConditionalResponsePlan:
    """One validated root response and its immediate attack continuation."""

    action: Action
    followup_attack_piece: Optional[str]
    baseline_action: Action
    source: str
    depth: int
    agreement: float
    information_confidence: float
    margin: float
    cache_source: str
    cache_branch_kind: Optional[str]
    cache_branch_context: Optional[str]
    cached_compute_ms: float


@dataclass
class _ConditionalResponseEntry:
    plan: ConditionalResponsePlan
    created_at: float
    last_accessed_at: float
    hits: int = 0


class ConditionalResponseDictionary:
    """Thread-safe LRU dictionary shared with background-search clones."""

    def __init__(self, *, max_entries: int, ttl_seconds: float) -> None:
        self.max_entries = max(1, int(max_entries))
        self.ttl_seconds = max(0.01, float(ttl_seconds))
        self._entries: "OrderedDict[str, _ConditionalResponseEntry]" = OrderedDict()
        self._lock = threading.RLock()
        self._counters = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "invalid": 0,
            "evictions": 0,
            "expired": 0,
            "pass_hits": 0,
            "receive_hits": 0,
            "background_hits": 0,
            "foreground_hits": 0,
            "followup_hits": 0,
            "followup_unavailable": 0,
        }
        self._estimated_saved_ms = 0.0

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def _prune(self, now: float) -> None:
        expired = [
            key
            for key, entry in self._entries.items()
            if now - entry.created_at >= self.ttl_seconds
        ]
        for key in expired:
            self._entries.pop(key, None)
            self._counters["expired"] += 1
        if expired:
            _record_runtime_counter("expired", len(expired))

    def get(self, key: str) -> Optional[ConditionalResponsePlan]:
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            entry = self._entries.get(key)
            if entry is None:
                self._counters["misses"] += 1
                _record_runtime_counter("misses")
                return None
            entry.last_accessed_at = now
            self._entries.move_to_end(key)
            return entry.plan

    def put(self, key: str, plan: ConditionalResponsePlan) -> None:
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            self._entries.pop(key, None)
            self._entries[key] = _ConditionalResponseEntry(plan, now, now)
            self._counters["stores"] += 1
            _record_runtime_counter("stores")
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self._counters["evictions"] += 1
                _record_runtime_counter("evictions")

    def mark_invalid(self, key: str) -> None:
        with self._lock:
            self._entries.pop(key, None)
            self._counters["invalid"] += 1
            _record_runtime_counter("invalid")

    def record_reuse(self, plan: ConditionalResponsePlan) -> None:
        with self._lock:
            self._counters["hits"] += 1
            _record_runtime_counter("hits")
            counter = (
                "receive_hits"
                if plan.action[0] == "receive"
                else "pass_hits"
            )
            self._counters[counter] += 1
            _record_runtime_counter(counter)
            source_counter = (
                "background_hits"
                if plan.cache_source == "background"
                else "foreground_hits"
            )
            self._counters[source_counter] += 1
            _record_runtime_counter(source_counter)
            saved_ms = max(0.0, float(plan.cached_compute_ms))
            self._estimated_saved_ms += saved_ms
            _record_runtime_saved_ms(saved_ms)

    def record_followup(self, *, used: bool) -> None:
        with self._lock:
            counter = "followup_hits" if used else "followup_unavailable"
            self._counters[counter] += 1
            _record_runtime_counter(counter)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            self._prune(time.monotonic())
            hits = int(self._counters["hits"])
            misses = int(self._counters["misses"])
            invalid = int(self._counters["invalid"])
            lookups = hits + misses + invalid
            return {
                **self._counters,
                "lookups": lookups,
                "hit_rate": round(hits / lookups if lookups else 0.0, 5),
                "estimated_saved_ms": round(self._estimated_saved_ms, 3),
                "estimated_saved_seconds": round(
                    self._estimated_saved_ms / 1000.0,
                    4,
                ),
                "size": len(self._entries),
                "max_entries": self.max_entries,
                "ttl_seconds": self.ttl_seconds,
            }


class ConditionalResponseMixin:
    """Builds, validates, stores, and applies conditional receive plans."""

    def _initialize_conditional_response_dictionary(self) -> None:
        self._conditional_response_dictionary = ConditionalResponseDictionary(
            max_entries=int(self.CONDITIONAL_RESPONSE_MAX_ENTRIES),
            ttl_seconds=float(self.CONDITIONAL_RESPONSE_TTL_SECONDS),
        )
        self.last_conditional_response_key: Optional[str] = None
        self.last_conditional_response_hit = False

    def _conditional_response_key(
        self,
        state,
        player: str,
        actions: Iterable[Action],
        baseline_action: Action,
    ) -> str:
        """Return an exact public-information key without opponent hands."""
        tracker = self._track.get(id(state)) or {}
        payload = {
            "version": 1,
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
            "public": {
                "seen": tracker.get("public_seen_counts", {}),
                "attack_counts": tracker.get("enemy_attack_counts", {}),
                "my_attack_count": int(tracker.get("my_attack_count", 0)),
                "my_attacks": tracker.get("my_attack_history", ()),
                "ally_attacks": tracker.get("ally_past_attacks", set()),
                "enemy_attacks": tracker.get("enemy_past_attacks", set()),
                "ally_shi_signal": tracker.get("ally_shi_signal"),
                "shi_attack_mode": bool(tracker.get("shi_attack_mode")),
                "estimated_hands": tracker.get("estimated_current_hands", {}),
                "count_caps": tracker.get("current_piece_count_caps", {}),
                "hand_models": tracker.get("public_hand_models", {}),
            },
            "legal_actions": sorted(tuple(action) for action in actions),
            "baseline_action": tuple(baseline_action),
            "policy": self._search_policy_fingerprint(),
        }
        return _digest_payload(payload)

    @staticmethod
    def _conditional_response_plan_is_legal(
        state,
        player: str,
        actions: Iterable[Action],
        plan: ConditionalResponsePlan,
    ) -> bool:
        legal_actions = set(actions)
        if plan.action not in legal_actions:
            return False
        if plan.action[0] != "receive":
            return plan.followup_attack_piece is None
        if plan.action[1] != state.current_attack:
            return False
        followup = plan.followup_attack_piece
        if followup is None:
            return True
        remaining = list(state.hands[player])
        try:
            remaining.remove(str(plan.action[1]))
        except ValueError:
            return False
        return followup in remaining

    def _lookup_conditional_response_plan(
        self,
        state,
        player: str,
        actions: Iterable[Action],
        baseline_action: Action,
    ) -> Optional[ConditionalResponsePlan]:
        self.last_conditional_response_hit = False
        self.last_conditional_response_key = None
        if (
            not self.CONDITIONAL_RESPONSE_ENABLED
            or state.phase != "receive"
        ):
            return None
        actions_list = list(actions)
        if not any(action[0] == "pass" for action in actions_list):
            return None
        key = self._conditional_response_key(
            state,
            player,
            actions_list,
            baseline_action,
        )
        self.last_conditional_response_key = key
        plan = self._conditional_response_dictionary.get(key)
        if plan is None:
            return None
        if not self._conditional_response_plan_is_legal(
            state,
            player,
            actions_list,
            plan,
        ):
            self._conditional_response_dictionary.mark_invalid(key)
            return None
        self.last_conditional_response_hit = True
        self._conditional_response_dictionary.record_reuse(plan)
        self._compare_generic_response_shadow(
            state,
            player,
            actions_list,
            baseline_action,
            plan.action,
        )
        self._record_generic_response_plan_reuse(
            state,
            player,
            actions_list,
            baseline_action,
            plan,
        )
        self.last_time_search_cache_hit = True
        self.last_time_search_cache_key = key
        self.last_time_search_cache_source = plan.cache_source
        self.last_time_search_cache_branch_kind = plan.cache_branch_kind
        self.last_time_search_cache_branch_context = plan.cache_branch_context
        self.last_time_search_cached_compute_ms = plan.cached_compute_ms
        return plan

    def _remember_conditional_response_plan(
        self,
        state,
        player: str,
        actions: Iterable[Action],
        baseline_action: Action,
        selected_action: Action,
        search_result,
        *,
        source: str,
    ) -> Optional[ConditionalResponsePlan]:
        if (
            not self.CONDITIONAL_RESPONSE_ENABLED
            or state.phase != "receive"
            or selected_action[0] not in ("pass", "receive")
            or int(getattr(search_result, "depth", 0))
            < int(self.CONDITIONAL_RESPONSE_MIN_DEPTH)
        ):
            return None
        if selected_action != search_result.action:
            return None

        followup = None
        if selected_action[0] == "receive":
            followup = self._low_reentry_followup_piece(
                state,
                player,
                str(selected_action[1]),
            )
        plan = ConditionalResponsePlan(
            action=selected_action,
            followup_attack_piece=followup,
            baseline_action=baseline_action,
            source=str(source),
            depth=int(search_result.depth),
            agreement=float(search_result.agreement),
            information_confidence=float(search_result.information_confidence),
            margin=float(search_result.margin),
            cache_source=str(
                self.last_time_search_cache_source
                or (
                    "background"
                    if getattr(self, "_time_search_cancel_event", None) is not None
                    else "foreground"
                )
            ),
            cache_branch_kind=(
                self.last_time_search_cache_branch_kind
                or getattr(self, "_time_search_background_branch_kind", None)
            ),
            cache_branch_context=(
                self.last_time_search_cache_branch_context
                or getattr(self, "_time_search_background_branch_context", None)
            ),
            cached_compute_ms=float(
                self.last_time_search_cached_compute_ms
                or max(0.0, float(search_result.elapsed_seconds)) * 1000.0
            ),
        )
        actions_list = list(actions)
        if not self._conditional_response_plan_is_legal(
            state,
            player,
            actions_list,
            plan,
        ):
            return None
        key = self._conditional_response_key(
            state,
            player,
            actions_list,
            baseline_action,
        )
        self._conditional_response_dictionary.put(key, plan)
        self.last_conditional_response_key = key
        return plan

    def _commit_conditional_response_plan(
        self,
        state,
        plan: ConditionalResponsePlan,
    ) -> None:
        tracker = self._track.get(id(state))
        if tracker is None:
            return
        tracker["pending_conditional_response_attack_piece"] = (
            plan.followup_attack_piece
            if plan.action[0] == "receive"
            else None
        )

    def conditional_response_dictionary_snapshot(self) -> Dict[str, object]:
        return self._conditional_response_dictionary.snapshot()

    def clear_conditional_response_dictionary(self) -> None:
        self._conditional_response_dictionary.clear()

    def _record_conditional_response_followup(self, *, used: bool) -> None:
        self._conditional_response_dictionary.record_followup(used=used)


def merge_conditional_response_snapshots(
    snapshots: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    """Combine per-agent counters for the administrator dashboard."""
    additive = (
        "hits",
        "misses",
        "stores",
        "invalid",
        "evictions",
        "expired",
        "pass_hits",
        "receive_hits",
        "background_hits",
        "foreground_hits",
        "followup_hits",
        "followup_unavailable",
        "lookups",
        "size",
        "max_entries",
    )
    merged = {
        key: sum(int(snapshot.get(key, 0) or 0) for snapshot in snapshots)
        for key in additive
    }
    saved_ms = sum(
        float(snapshot.get("estimated_saved_ms", 0.0) or 0.0)
        for snapshot in snapshots
    )
    merged["estimated_saved_ms"] = round(saved_ms, 3)
    merged["estimated_saved_seconds"] = round(saved_ms / 1000.0, 4)
    merged["hit_rate"] = round(
        merged["hits"] / merged["lookups"]
        if merged["lookups"]
        else 0.0,
        5,
    )
    merged["dictionary_instances"] = len(snapshots)
    return merged


def conditional_response_runtime_snapshot(
    live_snapshots: Sequence[Dict[str, object]] = (),
) -> Dict[str, object]:
    """Return process-wide totals plus the currently active dictionary size."""
    live = merge_conditional_response_snapshots(live_snapshots)
    with _RUNTIME_LOCK:
        counters = dict(_RUNTIME_COUNTERS)
        saved_ms = float(_RUNTIME_ESTIMATED_SAVED_MS)
    hits = int(counters["hits"])
    lookups = hits + int(counters["misses"]) + int(counters["invalid"])
    return {
        **counters,
        "lookups": lookups,
        "hit_rate": round(hits / lookups if lookups else 0.0, 5),
        "estimated_saved_ms": round(saved_ms, 3),
        "estimated_saved_seconds": round(saved_ms / 1000.0, 4),
        "size": int(live.get("size", 0) or 0),
        "max_entries": int(live.get("max_entries", 0) or 0),
        "dictionary_instances": len(live_snapshots),
    }

"""Stores reusable responses for publicly identical receive positions.
Each entry keeps pass/receive together with the planned follow-up attack, while
strict public-state keys and legality checks prevent stale plans from playing.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from goita_ai2.constants import ALL_SEATS
from goita_ai2.current_ai.search_cache import _digest_payload


Action = Tuple[str, Optional[str], Optional[str]]


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
        }

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

    def get(self, key: str) -> Optional[ConditionalResponsePlan]:
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            entry = self._entries.get(key)
            if entry is None:
                self._counters["misses"] += 1
                return None
            entry.last_accessed_at = now
            entry.hits += 1
            self._entries.move_to_end(key)
            self._counters["hits"] += 1
            return entry.plan

    def put(self, key: str, plan: ConditionalResponsePlan) -> None:
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            self._entries.pop(key, None)
            self._entries[key] = _ConditionalResponseEntry(plan, now, now)
            self._counters["stores"] += 1
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self._counters["evictions"] += 1

    def mark_invalid(self, key: str) -> None:
        with self._lock:
            self._entries.pop(key, None)
            self._counters["invalid"] += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            self._prune(time.monotonic())
            hits = int(self._counters["hits"])
            misses = int(self._counters["misses"])
            lookups = hits + misses
            return {
                **self._counters,
                "lookups": lookups,
                "hit_rate": round(hits / lookups if lookups else 0.0, 5),
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

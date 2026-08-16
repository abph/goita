"""Connects evaluated branched attack plans to the production decision path.
It applies safety gates and keeps proven/public tactics above the planner, while
bounded generation, evaluation, caching, and metrics limit per-turn overhead.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from goita_ai2.constants import ALL_SEATS, POINTS
from goita_ai2.current_ai.branched_attack_evaluator import (
    AttackRouteDangerSeverity,
    EvaluatedBranchedAttackPlan,
)
from goita_ai2.current_ai.branched_attack_lifecycle import (
    ActiveAttackPlanState,
    AttackPlanLifecycleStatus,
)
from goita_ai2.current_ai.branched_attack_plan import Action
from goita_ai2.current_ai.search_cache import _digest_payload


@dataclass
class _BranchedPlanCacheEntry:
    value: Tuple[EvaluatedBranchedAttackPlan, ...]
    created_at: float
    last_accessed_at: float
    hits: int = 0


class BranchedAttackPlanCache:
    """Small thread-safe LRU cache shared by rule-preview clones."""

    def __init__(self, *, max_entries: int, ttl_seconds: float) -> None:
        self.max_entries = max(1, int(max_entries))
        self.ttl_seconds = max(0.01, float(ttl_seconds))
        self._entries: "OrderedDict[str, _BranchedPlanCacheEntry]" = OrderedDict()
        self._lock = threading.RLock()
        self._counters = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
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

    def get(self, key: str) -> Optional[Tuple[EvaluatedBranchedAttackPlan, ...]]:
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
            return entry.value

    def put(
        self,
        key: str,
        value: Sequence[EvaluatedBranchedAttackPlan],
    ) -> None:
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            self._entries.pop(key, None)
            self._entries[key] = _BranchedPlanCacheEntry(
                value=tuple(value),
                created_at=now,
                last_accessed_at=now,
            )
            self._counters["stores"] += 1
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self._counters["evictions"] += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            now = time.monotonic()
            self._prune(now)
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


@dataclass(frozen=True)
class BranchedAttackProductionChoice:
    """The plan and action accepted by the production safety gate."""

    action: Action
    active: ActiveAttackPlanState
    continued: bool
    cache_hit: bool


class BranchedAttackRuntimeMixin:
    """Runs bounded plan selection and exposes diagnostics for production."""

    def _initialize_branched_attack_runtime(self) -> None:
        self._branched_attack_plan_cache = BranchedAttackPlanCache(
            max_entries=int(self.BRANCHED_ATTACK_CACHE_MAX_ENTRIES),
            ttl_seconds=float(self.BRANCHED_ATTACK_CACHE_TTL_SECONDS),
        )
        self.last_branched_attack_metrics: Dict[str, object] = {}

    def _branched_runtime_cache(self) -> BranchedAttackPlanCache:
        cache = getattr(self, "_branched_attack_plan_cache", None)
        if cache is None:
            self._initialize_branched_attack_runtime()
            cache = self._branched_attack_plan_cache
        return cache

    def _branched_attack_cache_key(
        self,
        state,
        player: str,
        actions: Sequence[Action],
    ) -> str:
        tr = self._track.get(id(state)) or {}
        public_models = tr.get("public_hand_models", {})
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
                "team_score": dict(state.team_score),
                "hand_sizes": {seat: len(state.hands[seat]) for seat in ALL_SEATS},
                "my_hand": sorted(state.hands[player]),
                "my_hidden": sorted(state.face_down_hidden[player]),
                "my_had_both_kings": bool(state.had_both_kings.get(player, False)),
            },
            "public": {
                "revision": int(tr.get("piece_inference_revision", 0)),
                "seen": tr.get("public_seen_counts", {}),
                "estimated_hands": tr.get("estimated_current_hands", {}),
                "count_caps": tr.get("current_piece_count_caps", {}),
                "my_attack_count": int(tr.get("my_attack_count", 0)),
                "my_attacks": tr.get("my_attack_history", ()),
                "ally_attacks": tr.get("ally_past_attacks", set()),
                "enemy_attacks": tr.get("enemy_past_attacks", set()),
                "shi_attack_mode": bool(tr.get("shi_attack_mode")),
                "ally_shi_signal": tr.get("ally_shi_signal"),
                "special_attack_plan": tr.get("special_attack_plan"),
                "ranks": {
                    seat: {
                        "rank": model.get("estimated_rank"),
                        "confidence": model.get("rank_confidence"),
                    }
                    for seat, model in public_models.items()
                },
            },
            "actions": sorted(tuple(action) for action in actions),
            "policy": {
                "template_limit": int(self.BRANCHED_ATTACK_MAX_TEMPLATE_PLANS),
                "generic_limit": int(self.BRANCHED_ATTACK_MAX_GENERIC_ROOTS),
                "total_limit": int(self.BRANCHED_ATTACK_MAX_TOTAL_PLANS),
                "evaluation_limit": int(self.BRANCHED_ATTACK_MAX_EVALUATED_PLANS),
                "max_seconds": float(self.BRANCHED_ATTACK_MAX_SECONDS),
                "risk": float(self.BRANCHED_ATTACK_GENERIC_MAX_FAILURE_RISK),
                "width": float(self.BRANCHED_ATTACK_GENERIC_MIN_RECEIVE_WIDTH),
            },
        }
        return _digest_payload(payload)

    def _rank_branched_generic_root_actions(
        self,
        state,
        player: str,
        actions: Sequence[Action],
    ) -> Tuple[Action, ...]:
        tr = self._track.get(id(state))
        attack_number = int(tr.get("my_attack_count", 0)) + 1 if tr else 1
        hand = tuple(sorted(state.hands[player]))

        def key(action: Action) -> tuple:
            action_type, block, attack = action
            score = self._planned_future_attack_value(
                state,
                player,
                str(attack),
                attack_number,
                tr.get("my_last_attack") if tr else None,
                hand,
            )
            if block is not None:
                score += self._planned_future_block_value(
                    state,
                    player,
                    block,
                    hand,
                )
            return (
                score,
                POINTS.get(str(attack), 0),
                -POINTS.get(str(block), 0) if block is not None else 0,
                action_type,
                block or "",
                attack or "",
            )

        roots = list(self._branched_root_attack_candidates(actions))
        roots.sort(key=key, reverse=True)
        return tuple(roots[:max(0, int(self.BRANCHED_ATTACK_MAX_GENERIC_ROOTS))])

    def _branched_attack_candidates(
        self,
        state,
        player: str,
        actions: Sequence[Action],
    ) -> Tuple[object, ...]:
        templates = self._generate_representative_attack_plans(
            state,
            player,
            actions,
            max_plans=int(self.BRANCHED_ATTACK_MAX_TEMPLATE_PLANS),
        )
        generic_actions = self._rank_branched_generic_root_actions(
            state,
            player,
            actions,
        )
        generic = self._generate_branched_attack_plans(
            state,
            player,
            generic_actions,
        )
        combined = list(templates) + list(generic)
        return tuple(combined[:max(0, int(self.BRANCHED_ATTACK_MAX_TOTAL_PLANS))])

    def _evaluate_branched_attack_candidates_bounded(
        self,
        state,
        player: str,
        plans: Sequence[object],
        *,
        started: float,
    ) -> Tuple[Tuple[EvaluatedBranchedAttackPlan, ...], bool]:
        deadline = started + max(0.001, float(self.BRANCHED_ATTACK_MAX_SECONDS))
        limit = max(1, int(self.BRANCHED_ATTACK_MAX_EVALUATED_PLANS))
        evaluated: List[EvaluatedBranchedAttackPlan] = []
        timed_out = False
        previous_deadline = getattr(
            self,
            "_branched_attack_evaluation_deadline",
            None,
        )
        self._branched_attack_evaluation_deadline = deadline
        try:
            for plan in plans[:limit]:
                if time.perf_counter() >= deadline:
                    timed_out = True
                    break
                evaluated.append(self._evaluate_branched_attack_plan(state, player, plan))
        finally:
            if previous_deadline is None:
                delattr(self, "_branched_attack_evaluation_deadline")
            else:
                self._branched_attack_evaluation_deadline = previous_deadline
        if len(evaluated) < min(len(plans), limit):
            timed_out = True
        return tuple(evaluated), timed_out

    @staticmethod
    def _branched_root_has_critical_danger(
        item: EvaluatedBranchedAttackPlan,
    ) -> bool:
        root_id = item.plan.root_node_id
        root_eval = next(
            (
                node
                for node in item.report.node_evaluations
                if node.node_id == root_id
            ),
            None,
        )
        return bool(root_eval and any(
            danger.severity >= AttackRouteDangerSeverity.CRITICAL
            for danger in root_eval.dangers
        ))

    def _branched_plan_is_production_eligible(
        self,
        item: EvaluatedBranchedAttackPlan,
    ) -> bool:
        evaluation = item.plan.evaluation
        if evaluation.guaranteed_win:
            return True
        if self._branched_root_has_critical_danger(item):
            return False
        if item.plan.source.startswith("representative:"):
            # Royal preservation is an advisory constraint, not a fixed attack
            # order. It may support a proven route but must not replace a more
            # specific local tactic by itself.
            if item.plan.source == "representative:royal_receive_width":
                return False
            return True
        return (
            evaluation.failure_risk
            <= float(self.BRANCHED_ATTACK_GENERIC_MAX_FAILURE_RISK)
            and evaluation.receive_width
            >= float(self.BRANCHED_ATTACK_GENERIC_MIN_RECEIVE_WIDTH)
            and evaluation.covered_public_responses > 0
        )

    @staticmethod
    def _choose_production_branched_plan(
        plans: Iterable[EvaluatedBranchedAttackPlan],
    ) -> Optional[EvaluatedBranchedAttackPlan]:
        candidates = list(plans)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: (
                item.plan.evaluation.preference_key(),
                1 if item.plan.source.startswith("representative:") else 0,
                item.plan.plan_id,
            ),
        )

    def _record_branched_attack_metrics(
        self,
        state,
        **values: object,
    ) -> None:
        self.last_branched_attack_metrics = dict(values)
        tr = self._track.get(id(state))
        if tr is not None:
            tr["last_branched_attack_metrics"] = dict(values)

    @staticmethod
    def _branched_inference_summary(
        selected: EvaluatedBranchedAttackPlan,
    ) -> Dict[str, object]:
        counts = Counter(
            item.support.level.label
            for item in selected.report.branch_inference
        )
        likely = [
            item
            for item in selected.report.branch_inference
            if int(item.support.level) >= 3 and item.route_continues
        ]
        summary = selected.report.inference_summary.as_dict()
        summary.update({
            "support_counts": {
                level: int(counts.get(level, 0))
                for level in ("certain", "likely", "possible", "low", "impossible")
            },
            "likely_continuations": len(likely),
        })
        return summary

    def _production_branched_attack_action(
        self,
        state,
        player: str,
        actions: Sequence[Action],
    ) -> Optional[BranchedAttackProductionChoice]:
        """Continue a plan or select a bounded replacement for this attack turn."""
        if (
            not bool(self.BRANCHED_ATTACK_ENABLED)
            or state.phase != "attack"
            or state.turn != player
        ):
            return None

        started = time.perf_counter()
        active = self._active_branched_attack_plan(state)
        continued_action = self._branched_planned_action(state, player, actions)
        tr = self._track.get(id(state)) or {}
        current_revision = int(tr.get("piece_inference_revision", 0))
        requires_revalidation = bool(
            continued_action is not None
            and active is not None
            and current_revision > int(active.installed_revision)
        )
        previous_plan_id = active.plan.plan_id if requires_revalidation and active else None
        previous_action = continued_action if requires_revalidation else None
        if (
            continued_action is not None
            and active is not None
            and not requires_revalidation
        ):
            self._record_branched_attack_metrics(
                state,
                cache_hit=False,
                continued=True,
                revalidated=False,
                generated=0,
                evaluated=0,
                truncated=False,
                timed_out=False,
                cache_ms=0.0,
                generation_ms=0.0,
                evaluation_ms=0.0,
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
                selected_plan_id=active.plan.plan_id,
                selected_source=active.plan.source,
                status=active.status.value,
            )
            return BranchedAttackProductionChoice(
                action=continued_action,
                active=active,
                continued=True,
                cache_hit=False,
            )

        if active is not None and active.status in (
            AttackPlanLifecycleStatus.OBSERVING,
            AttackPlanLifecycleStatus.WAITING,
        ):
            self._invalidate_branched_attack_plan(
                state,
                "owner_attack_turn_arrived_without_a_ready_plan_node",
            )

        key = self._branched_attack_cache_key(state, player, actions)
        cache = self._branched_runtime_cache()
        cache_started = time.perf_counter()
        evaluated = cache.get(key) if self.BRANCHED_ATTACK_CACHE_ENABLED else None
        cache_ms = (time.perf_counter() - cache_started) * 1000.0
        cache_hit = evaluated is not None
        generated_count = 0
        timed_out = False
        generation_ms = 0.0
        evaluation_ms = 0.0
        if evaluated is None:
            generation_started = time.perf_counter()
            plans = self._branched_attack_candidates(state, player, actions)
            generation_ms = (time.perf_counter() - generation_started) * 1000.0
            generated_count = len(plans)
            evaluation_started = time.perf_counter()
            evaluated, timed_out = self._evaluate_branched_attack_candidates_bounded(
                state,
                player,
                plans,
                started=started,
            )
            evaluation_ms = (time.perf_counter() - evaluation_started) * 1000.0
            if self.BRANCHED_ATTACK_CACHE_ENABLED:
                cache.put(key, evaluated)
        truncated = bool(not cache_hit and len(evaluated) < generated_count)

        eligible = [
            item
            for item in evaluated
            if self._branched_plan_is_production_eligible(item)
            and item.plan.node(item.plan.root_node_id).action in actions
        ]
        selected = self._choose_production_branched_plan(eligible)
        if selected is None:
            if (
                requires_revalidation
                and active is not None
                and previous_action is not None
                and timed_out
            ):
                active.observed_revision = current_revision
                active.reason = "inference_revalidation_deferred_by_budget"
                self._branched_plan_tracker_summary(state, active)
                self._record_branched_attack_metrics(
                    state,
                    cache_hit=cache_hit,
                    continued=True,
                    revalidated=False,
                    revalidation_deferred=True,
                    previous_plan_id=previous_plan_id,
                    inference_revision=current_revision,
                    generated=generated_count,
                    evaluated=len(evaluated),
                    truncated=truncated,
                    eligible=0,
                    timed_out=True,
                    cache_ms=round(cache_ms, 3),
                    generation_ms=round(generation_ms, 3),
                    evaluation_ms=round(evaluation_ms, 3),
                    elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
                    selected_plan_id=active.plan.plan_id,
                    selected_source=active.plan.source,
                    status=active.status.value,
                )
                return BranchedAttackProductionChoice(
                    action=previous_action,
                    active=active,
                    continued=True,
                    cache_hit=cache_hit,
                )
            if requires_revalidation:
                self._invalidate_branched_attack_plan(
                    state,
                    "current_hand_inference_no_longer_supports_plan",
                )
            self._record_branched_attack_metrics(
                state,
                cache_hit=cache_hit,
                continued=False,
                revalidated=requires_revalidation,
                previous_plan_id=previous_plan_id,
                inference_revision=current_revision,
                generated=generated_count,
                evaluated=len(evaluated),
                truncated=truncated,
                eligible=0,
                timed_out=timed_out,
                cache_ms=round(cache_ms, 3),
                generation_ms=round(generation_ms, 3),
                evaluation_ms=round(evaluation_ms, 3),
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
                selected_plan_id=None,
                selected_source=None,
                status="fallback",
            )
            return None

        selected_action = selected.plan.node(selected.plan.root_node_id).action
        if (
            requires_revalidation
            and active is not None
            and previous_action is not None
            and selected_action == previous_action
        ):
            refreshed = self._refresh_branched_attack_plan(
                state,
                active,
                selected.plan,
                reason="revalidated_with_current_hand_inference",
            )
            self._record_branched_attack_metrics(
                state,
                cache_hit=cache_hit,
                continued=True,
                revalidated=True,
                revalidation_deferred=False,
                previous_plan_id=previous_plan_id,
                inference_revision=current_revision,
                generated=generated_count,
                evaluated=len(evaluated),
                truncated=truncated,
                eligible=len(eligible),
                timed_out=timed_out,
                cache_ms=round(cache_ms, 3),
                generation_ms=round(generation_ms, 3),
                evaluation_ms=round(evaluation_ms, 3),
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
                selected_plan_id=refreshed.plan.plan_id,
                selected_source=refreshed.plan.source,
                guaranteed_win=refreshed.plan.evaluation.guaranteed_win,
                minimum_score=refreshed.plan.evaluation.minimum_score,
                maximum_score=refreshed.plan.evaluation.maximum_score,
                expected_score=refreshed.plan.evaluation.expected_score,
                probability_coverage=(
                    refreshed.plan.evaluation.probability_coverage
                ),
                probability_failure_risk=(
                    refreshed.plan.evaluation.probability_failure_risk
                ),
                failure_risk=refreshed.plan.evaluation.failure_risk,
                inference_summary=self._branched_inference_summary(selected),
                status=refreshed.status.value,
            )
            return BranchedAttackProductionChoice(
                action=previous_action,
                active=refreshed,
                continued=True,
                cache_hit=cache_hit,
            )

        if requires_revalidation:
            self._invalidate_branched_attack_plan(
                state,
                "current_hand_inference_prefers_a_different_plan",
            )

        installed = self._install_branched_attack_plan(state, player, selected.plan)
        action = installed.current_node.action
        if action is None:
            return None
        self._record_branched_attack_metrics(
            state,
            cache_hit=cache_hit,
            continued=False,
            revalidated=requires_revalidation,
            previous_plan_id=previous_plan_id,
            inference_revision=current_revision,
            generated=generated_count,
            evaluated=len(evaluated),
            truncated=truncated,
            eligible=len(eligible),
            timed_out=timed_out,
            cache_ms=round(cache_ms, 3),
            generation_ms=round(generation_ms, 3),
            evaluation_ms=round(evaluation_ms, 3),
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
            selected_plan_id=installed.plan.plan_id,
            selected_source=installed.plan.source,
            guaranteed_win=installed.plan.evaluation.guaranteed_win,
            minimum_score=installed.plan.evaluation.minimum_score,
            maximum_score=installed.plan.evaluation.maximum_score,
            expected_score=installed.plan.evaluation.expected_score,
            probability_coverage=installed.plan.evaluation.probability_coverage,
            probability_failure_risk=(
                installed.plan.evaluation.probability_failure_risk
            ),
            failure_risk=installed.plan.evaluation.failure_risk,
            inference_summary=self._branched_inference_summary(selected),
            status=installed.status.value,
        )
        return BranchedAttackProductionChoice(
            action=action,
            active=installed,
            continued=False,
            cache_hit=cache_hit,
        )

    def _commit_branched_attack_choice(
        self,
        state,
        choice: BranchedAttackProductionChoice,
    ) -> None:
        tr = self._track.get(id(state))
        if tr is None:
            return
        tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
        tr["pending_weak_hand_shi_signal"] = False
        tr["pending_ally_force_king_attack_piece"] = None
        tr["pending_inferred_endgame_attack"] = None
        if (
            tr.get("kg_plan_active")
            and tr["my_attack_count"] == 2
            and choice.action[2] in ("8", "9")
            and tr.get("kg_second") is None
        ):
            tr["kg_second"] = choice.action[2]
        if tr.get("kg_plan_active") and tr["my_attack_count"] >= 3:
            tr["kg_plan_active"] = False

    def branched_attack_cache_snapshot(self) -> Dict[str, object]:
        return self._branched_runtime_cache().snapshot()

    def clear_branched_attack_cache(self) -> None:
        self._branched_runtime_cache().clear()


__all__ = [
    "BranchedAttackPlanCache",
    "BranchedAttackProductionChoice",
    "BranchedAttackRuntimeMixin",
]

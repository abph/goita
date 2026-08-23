"""Adjusts ordinary-search work from position complexity and measured speed.

The configured search limits remain hard ceilings and stable cache policy.
Only the effective limits for one search are reduced, then measured so later
turns can stay responsive without weakening protected tactical rules.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class SearchBudgetPlan:
    context: str
    reason: str
    configured_seconds: float
    effective_seconds: float
    configured_samples: int
    effective_samples: int
    configured_depth: int
    effective_depth: int
    configured_nodes: int
    effective_nodes: int
    complexity: float
    observations: int

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key in ("configured_seconds", "effective_seconds", "complexity"):
            payload[key] = round(float(payload[key]), 4)
        return payload


class AdaptiveSearchBudgetController:
    """Learns inexpensive EWMA speed estimates shared by current-AI seats."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._contexts: Dict[str, Dict[str, float]] = {}
        self._totals: Dict[str, float] = {
            "observations": 0,
            "cache_hits_skipped": 0,
            "cancelled_skipped": 0,
            "plans": 0,
            "adjusted_plans": 0,
            "overruns": 0,
        }
        self._recent_plans: List[Dict[str, Any]] = []

    def __deepcopy__(self, memo: Dict[int, Any]):
        memo[id(self)] = self
        return self

    @staticmethod
    def _bucket() -> Dict[str, float]:
        return {
            "observations": 0,
            "ewma_sample_ms_per_sample": 0.0,
            "ewma_search_ms": 0.0,
            "ewma_compute_ms": 0.0,
            "ewma_nodes_per_second": 0.0,
            "overruns": 0,
        }

    @staticmethod
    def _ewma(previous: float, current: float, alpha: float) -> float:
        if previous <= 0.0:
            return max(0.0, current)
        return previous * (1.0 - alpha) + max(0.0, current) * alpha

    @staticmethod
    def _context(state, player: str, legal_action_count: int) -> str:
        hand_size = len(state.hands.get(player, []))
        if hand_size >= 7:
            stage = "opening"
        elif hand_size >= 4:
            stage = "middle"
        else:
            stage = "endgame"
        if legal_action_count <= 2:
            action_band = "actions_2"
        elif legal_action_count <= 5:
            action_band = "actions_3_5"
        else:
            action_band = "actions_6_plus"
        return f"{stage}|{state.phase}|{action_band}"

    @staticmethod
    def _complexity(state, player: str, legal_action_count: int) -> float:
        if legal_action_count <= 2:
            action_factor = 0.55
        elif legal_action_count <= 3:
            action_factor = 0.7
        elif legal_action_count <= 5:
            action_factor = 0.85
        else:
            action_factor = 1.0

        hand_size = len(state.hands.get(player, []))
        if hand_size >= 7:
            stage_factor = 0.78
        elif hand_size >= 5:
            stage_factor = 0.9
        else:
            stage_factor = 1.0
        phase_factor = 1.0 if state.phase == "receive" else 0.92
        return max(0.45, min(1.0, action_factor * stage_factor * phase_factor))

    @staticmethod
    def _odd_at_most(value: int, ceiling: int) -> int:
        result = max(1, min(int(value), int(ceiling)))
        if result % 2 == 0:
            result = max(1, result - 1)
        return result

    def plan(
        self,
        state,
        player: str,
        actions,
        *,
        enabled: bool,
        warmup_observations: int,
        minimum_seconds: float,
        minimum_samples: int,
        configured_seconds: float,
        configured_samples: int,
        configured_depth: int,
        configured_nodes: int,
    ) -> SearchBudgetPlan:
        legal_action_count = len(actions)
        context = self._context(state, player, legal_action_count)
        complexity = self._complexity(state, player, legal_action_count)
        configured_seconds = min(10.0, max(0.01, float(configured_seconds)))
        configured_samples = max(1, int(configured_samples))
        configured_depth = max(1, int(configured_depth))
        configured_nodes = max(1, int(configured_nodes))

        with self._lock:
            bucket = dict(self._contexts.get(context, self._bucket()))
            observations = int(bucket["observations"])

        if not enabled:
            reason = "adaptive_disabled"
        elif observations < max(0, int(warmup_observations)):
            reason = "warmup"
        else:
            reason = "measured_complexity"

        if reason != "measured_complexity":
            seconds = configured_seconds
            samples = configured_samples
            depth = configured_depth
            nodes = configured_nodes
        else:
            complexity_scale = 0.66 + 0.34 * complexity
            compute_ms = float(bucket["ewma_compute_ms"])
            configured_ms = configured_seconds * 1000.0
            pressure = 1.0
            if compute_ms > configured_ms * 0.95:
                pressure = max(0.55, configured_ms * 0.9 / max(1.0, compute_ms))

            seconds = configured_seconds * complexity_scale * pressure
            seconds = max(min(configured_seconds, float(minimum_seconds)), seconds)
            seconds = min(configured_seconds, seconds)

            sample_scale = max(0.35, complexity)
            samples = max(1, int(round(configured_samples * sample_scale)))
            sample_cost = float(bucket["ewma_sample_ms_per_sample"])
            if sample_cost > 0.0:
                sample_time_cap = max(20.0, seconds * 1000.0 * 0.18)
                samples = min(samples, max(1, int(sample_time_cap / sample_cost)))
            samples = max(min(configured_samples, int(minimum_samples)), samples)
            samples = min(configured_samples, samples)

            depth = configured_depth
            hand_size = len(state.hands.get(player, []))
            if hand_size >= 7:
                depth = min(depth, 7)
            if legal_action_count >= 6 and hand_size > 3:
                depth -= 2
            if pressure < 0.75 and hand_size > 3:
                depth -= 2
            depth = self._odd_at_most(depth, configured_depth)

            nodes_per_second = float(bucket["ewma_nodes_per_second"])
            if nodes_per_second > 0.0:
                nodes = int(nodes_per_second * seconds * 0.92)
                nodes = max(min(configured_nodes, 500), nodes)
                nodes = min(configured_nodes, nodes)
            else:
                nodes = configured_nodes

        plan = SearchBudgetPlan(
            context=context,
            reason=reason,
            configured_seconds=configured_seconds,
            effective_seconds=seconds,
            configured_samples=configured_samples,
            effective_samples=samples,
            configured_depth=configured_depth,
            effective_depth=depth,
            configured_nodes=configured_nodes,
            effective_nodes=nodes,
            complexity=complexity,
            observations=observations,
        )
        with self._lock:
            self._totals["plans"] += 1
            if plan.as_dict() != SearchBudgetPlan(
                context=context,
                reason=reason,
                configured_seconds=configured_seconds,
                effective_seconds=configured_seconds,
                configured_samples=configured_samples,
                effective_samples=configured_samples,
                configured_depth=configured_depth,
                effective_depth=configured_depth,
                configured_nodes=configured_nodes,
                effective_nodes=configured_nodes,
                complexity=complexity,
                observations=observations,
            ).as_dict():
                self._totals["adjusted_plans"] += 1
            self._recent_plans.append(plan.as_dict())
            del self._recent_plans[:-32]
        return plan

    def observe(
        self,
        plan: SearchBudgetPlan,
        *,
        sample_ms: float,
        search_ms: float,
        samples: int,
        nodes: int,
        cache_hit: bool,
        cancelled: bool,
        alpha: float,
    ) -> None:
        with self._lock:
            if cache_hit:
                self._totals["cache_hits_skipped"] += 1
                return
            if cancelled:
                self._totals["cancelled_skipped"] += 1
                return
            sample_ms = max(0.0, float(sample_ms))
            search_ms = max(0.0, float(search_ms))
            if sample_ms <= 0.0 and search_ms <= 0.0:
                return
            bucket = self._contexts.setdefault(plan.context, self._bucket())
            alpha = max(0.05, min(1.0, float(alpha)))
            sample_cost = sample_ms / max(1, int(samples))
            nodes_per_second = max(0, int(nodes)) / max(0.001, search_ms / 1000.0)
            bucket["observations"] += 1
            bucket["ewma_sample_ms_per_sample"] = self._ewma(
                bucket["ewma_sample_ms_per_sample"], sample_cost, alpha
            )
            bucket["ewma_search_ms"] = self._ewma(
                bucket["ewma_search_ms"], search_ms, alpha
            )
            bucket["ewma_compute_ms"] = self._ewma(
                bucket["ewma_compute_ms"], sample_ms + search_ms, alpha
            )
            if nodes > 0 and search_ms > 0.0:
                bucket["ewma_nodes_per_second"] = self._ewma(
                    bucket["ewma_nodes_per_second"], nodes_per_second, alpha
                )
            self._totals["observations"] += 1
            if sample_ms + search_ms > plan.configured_seconds * 1000.0 * 1.05:
                bucket["overruns"] += 1
                self._totals["overruns"] += 1

    def reset(self) -> None:
        with self._lock:
            self._contexts.clear()
            for key in self._totals:
                self._totals[key] = 0
            self._recent_plans.clear()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "totals": {key: int(value) for key, value in self._totals.items()},
                "contexts": {
                    context: {
                        key: (
                            int(value)
                            if key in ("observations", "overruns")
                            else round(float(value), 3)
                        )
                        for key, value in values.items()
                    }
                    for context, values in sorted(self._contexts.items())
                },
                "recent_plans": [dict(plan) for plan in self._recent_plans],
            }


_SEARCH_BUDGET_CONTROLLER = AdaptiveSearchBudgetController()


def time_search_budget_snapshot() -> Dict[str, Any]:
    return _SEARCH_BUDGET_CONTROLLER.snapshot()


def reset_time_search_budget_model() -> None:
    _SEARCH_BUDGET_CONTROLLER.reset()


class SearchBudgetMixin:
    """Builds and records one effective budget around each timed search."""

    def _initialize_time_search_budget(self) -> None:
        self._time_search_effective_budget: Optional[Dict[str, Any]] = None
        self.last_time_search_budget: Optional[Dict[str, Any]] = None
        self._time_search_profile = "default"

    def _prepare_time_search_budget(
        self,
        state,
        player: str,
        actions,
        *,
        configured_seconds: Optional[float] = None,
        configured_samples: Optional[int] = None,
        configured_depth: Optional[int] = None,
        configured_nodes: Optional[int] = None,
        adaptive_enabled: Optional[bool] = None,
    ) -> SearchBudgetPlan:
        plan = _SEARCH_BUDGET_CONTROLLER.plan(
            state,
            player,
            actions,
            enabled=(
                bool(getattr(self, "TIME_SEARCH_ADAPTIVE_BUDGET_ENABLED", True))
                if adaptive_enabled is None
                else bool(adaptive_enabled)
            ),
            warmup_observations=int(
                getattr(self, "TIME_SEARCH_ADAPTIVE_BUDGET_WARMUP", 4)
            ),
            minimum_seconds=float(
                getattr(self, "TIME_SEARCH_ADAPTIVE_MIN_SECONDS", 0.15)
            ),
            minimum_samples=int(
                getattr(self, "TIME_SEARCH_ADAPTIVE_MIN_SAMPLES", 8)
            ),
            configured_seconds=float(
                self.TIME_SEARCH_MAX_SECONDS
                if configured_seconds is None
                else configured_seconds
            ),
            configured_samples=int(
                self.TIME_SEARCH_SAMPLE_COUNT
                if configured_samples is None
                else configured_samples
            ),
            configured_depth=int(
                self.TIME_SEARCH_MAX_DEPTH
                if configured_depth is None
                else configured_depth
            ),
            configured_nodes=int(
                self.TIME_SEARCH_MAX_NODES
                if configured_nodes is None
                else configured_nodes
            ),
        )
        self._time_search_effective_budget = plan.as_dict()
        self.last_time_search_budget = plan.as_dict()
        return plan

    def _finish_time_search_budget(self, plan: SearchBudgetPlan, result) -> None:
        active = self._active_performance_metrics or {}
        _SEARCH_BUDGET_CONTROLLER.observe(
            plan,
            sample_ms=float(active.get("sample_generation", 0.0)) * 1000.0,
            search_ms=float(active.get("search", 0.0)) * 1000.0,
            samples=(
                int(result.samples)
                if result is not None
                else int(plan.effective_samples)
            ),
            nodes=int(result.nodes) if result is not None else 0,
            cache_hit=bool(getattr(self, "last_time_search_cache_hit", False)),
            cancelled=bool(
                getattr(self, "_time_search_cancel_event", None) is not None
                and self._time_search_cancel_event.is_set()
            ),
            alpha=float(getattr(self, "TIME_SEARCH_ADAPTIVE_EWMA_ALPHA", 0.25)),
        )
        self._time_search_effective_budget = None

    def _effective_time_search_setting(self, key: str, configured: Any) -> Any:
        budget = self._time_search_effective_budget or {}
        return budget.get(key, configured)

    def time_search_budget_snapshot(self) -> Dict[str, Any]:
        return time_search_budget_snapshot()

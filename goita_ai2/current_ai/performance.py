"""Measures where the developing AI spends its thinking time.

It records inference, rule selection, hidden-hand sampling, and tree search
without changing move scores or decisions. Both per-turn and cumulative data
are kept so later optimization work has a stable baseline.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator, Optional


PERFORMANCE_STAGES = (
    "inference",
    "rule_based",
    "cache",
    "sample_generation",
    "search",
)


class PerformanceMetricsMixin:
    """Collects lightweight monotonic-clock measurements for AI stages."""

    def _initialize_performance_metrics(self) -> None:
        self.performance_totals: Dict[str, Dict[str, float]] = {
            stage: {"calls": 0, "total_seconds": 0.0, "max_seconds": 0.0}
            for stage in (*PERFORMANCE_STAGES, "total")
        }
        self.last_performance_metrics: Dict[str, float] = {}
        self._active_performance_metrics: Optional[Dict[str, float]] = None
        self._active_precomputed_inference_seconds = 0.0
        self._pending_inference_seconds = 0.0

    def _record_performance_timing(self, stage: str, elapsed_seconds: float) -> None:
        elapsed = max(0.0, float(elapsed_seconds))
        totals = self.performance_totals.setdefault(
            stage,
            {"calls": 0, "total_seconds": 0.0, "max_seconds": 0.0},
        )
        totals["calls"] += 1
        totals["total_seconds"] += elapsed
        totals["max_seconds"] = max(totals["max_seconds"], elapsed)

        if stage == "inference" and self._active_performance_metrics is None:
            self._pending_inference_seconds += elapsed
        elif self._active_performance_metrics is not None:
            self._active_performance_metrics[stage] = (
                self._active_performance_metrics.get(stage, 0.0) + elapsed
            )

    @contextmanager
    def _measure_performance(self, stage: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self._record_performance_timing(stage, time.perf_counter() - started)

    def _begin_performance_decision(self) -> float:
        precomputed_inference = self._pending_inference_seconds
        self._active_performance_metrics = {
            stage: 0.0 for stage in PERFORMANCE_STAGES
        }
        self._active_performance_metrics["inference"] = precomputed_inference
        self._active_precomputed_inference_seconds = precomputed_inference
        self._pending_inference_seconds = 0.0
        return time.perf_counter()

    def _finish_performance_decision(self, started: float) -> None:
        elapsed = max(0.0, time.perf_counter() - started)
        active = self._active_performance_metrics or {
            stage: 0.0 for stage in PERFORMANCE_STAGES
        }
        precomputed_inference = self._active_precomputed_inference_seconds
        measured_during_decision = (
            sum(float(active.get(stage, 0.0)) for stage in PERFORMANCE_STAGES)
            - precomputed_inference
        )
        total_work = elapsed + precomputed_inference
        self.last_performance_metrics = {
            "total_ms": total_work * 1000.0,
            "rule_based_ms": float(active.get("rule_based", 0.0)) * 1000.0,
            "inference_ms": float(active.get("inference", 0.0)) * 1000.0,
            "cache_ms": float(active.get("cache", 0.0)) * 1000.0,
            "sample_generation_ms": float(active.get("sample_generation", 0.0)) * 1000.0,
            "search_ms": float(active.get("search", 0.0)) * 1000.0,
            "other_ms": max(0.0, elapsed - measured_during_decision) * 1000.0,
        }
        self._active_performance_metrics = None
        self._active_precomputed_inference_seconds = 0.0
        self._record_performance_timing("total", total_work)
        from goita_ai2.current_ai.telemetry import record_ai_search_decision

        record_ai_search_decision(self, self.last_performance_metrics)

    def performance_metrics_snapshot(self) -> Dict[str, object]:
        totals: Dict[str, Dict[str, float]] = {}
        for stage, item in self.performance_totals.items():
            calls = int(item.get("calls", 0))
            total_seconds = float(item.get("total_seconds", 0.0))
            totals[stage] = {
                "calls": calls,
                "total_ms": round(total_seconds * 1000.0, 3),
                "average_ms": round(total_seconds * 1000.0 / max(1, calls), 3),
                "max_ms": round(float(item.get("max_seconds", 0.0)) * 1000.0, 3),
            }
        return {
            "last_decision": {
                key: round(value, 3)
                for key, value in self.last_performance_metrics.items()
            },
            "last_search_budget": dict(self.last_time_search_budget or {}),
            "last_prediction_cache": {
                "hit": bool(self.last_prediction_cache_hit),
                "samples": int(self.last_prediction_cache_samples),
            },
            "totals": totals,
        }

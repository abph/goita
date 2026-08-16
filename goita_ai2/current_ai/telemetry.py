"""Aggregates timed-search effectiveness across rooms and AI seats.

The counters distinguish foreground decisions from speculative work, including
whether a cached result was produced by background search. Optional JSONL
checkpoints make the process-level measurements available for long-term review.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


def _nonnegative_int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return max(0, int(default))


class AiSearchTelemetry:
    """Thread-safe process counters for search latency and cache usefulness."""

    def __init__(
        self,
        *,
        checkpoint_path: Optional[str] = None,
        checkpoint_decisions: int = 100,
    ) -> None:
        self._lock = threading.RLock()
        self._checkpoint_lock = threading.Lock()
        self._checkpoint_path = str(checkpoint_path or "").strip()
        self._checkpoint_decisions = max(0, int(checkpoint_decisions))
        self._started_at = time.time()
        self._last_checkpoint_decisions = 0
        self._counters: Dict[str, float] = {}
        self.reset()
        self._restore_latest_checkpoint()

    def reset(self) -> None:
        with self._lock:
            self._started_at = time.time()
            self._last_checkpoint_decisions = 0
            self._counters = {
                "foreground_decisions": 0,
                "speculative_decisions": 0,
                "search_requests": 0,
                "cache_hits": 0,
                "background_cache_hits": 0,
                "background_cache_hits_all_pass": 0,
                "background_cache_hits_sampled_pass": 0,
                "background_cache_hits_sampled_receive": 0,
                "background_cache_hits_sampled_attack": 0,
                "background_cache_hits_sampled_attack_after_block": 0,
                "background_cache_hits_current_turn": 0,
                "background_cache_hits_other": 0,
                "foreground_cache_hits": 0,
                "unknown_cache_hits": 0,
                "search_computations": 0,
                "foreground_total_ms": 0.0,
                "foreground_cache_ms": 0.0,
                "foreground_sample_ms": 0.0,
                "foreground_search_ms": 0.0,
                "foreground_max_ms": 0.0,
                "speculative_total_ms": 0.0,
                "speculative_compute_ms": 0.0,
                "estimated_saved_ms": 0.0,
                "cached_compute_ms": 0.0,
                "background_prefetch_calls": 0,
                "background_branch_candidates": 0,
                "background_branches_scheduled": 0,
                "background_branches_throttled": 0,
                "background_action_matches": 0,
                "background_path_mismatches": 0,
                "background_path_completions": 0,
                "background_cache_ready": 0,
                "background_cancelled": 0,
                "background_skipped": 0,
                "background_errors": 0,
            }

    def _restore_latest_checkpoint(self) -> bool:
        if not self._checkpoint_path:
            return False
        path = Path(self._checkpoint_path)
        if not path.is_file():
            return False
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - 65_536), os.SEEK_SET)
                lines = handle.read().decode("utf-8", errors="ignore").splitlines()
        except OSError:
            return False
        for line in reversed(lines):
            try:
                metrics = json.loads(line).get("metrics", {})
            except (AttributeError, json.JSONDecodeError):
                continue
            if not isinstance(metrics, dict):
                continue
            with self._lock:
                for key in self._counters:
                    try:
                        self._counters[key] = max(0.0, float(metrics.get(key, 0.0)))
                    except (TypeError, ValueError):
                        self._counters[key] = 0.0
                self._last_checkpoint_decisions = int(
                    self._counters["foreground_decisions"]
                )
            return True
        return False

    def record_decision(
        self,
        metrics: Dict[str, float],
        *,
        speculative: bool,
        search_requested: bool,
        cache_hit: bool,
        cache_source: Optional[str],
        cached_compute_ms: float,
        cache_branch_kind: Optional[str] = None,
    ) -> None:
        total_ms = max(0.0, float(metrics.get("total_ms", 0.0)))
        cache_ms = max(0.0, float(metrics.get("cache_ms", 0.0)))
        sample_ms = max(0.0, float(metrics.get("sample_generation_ms", 0.0)))
        search_ms = max(0.0, float(metrics.get("search_ms", 0.0)))
        cached_ms = max(0.0, float(cached_compute_ms))

        should_checkpoint = False
        with self._lock:
            if speculative:
                self._counters["speculative_decisions"] += 1
                self._counters["speculative_total_ms"] += total_ms
                self._counters["speculative_compute_ms"] += sample_ms + search_ms
                return

            self._counters["foreground_decisions"] += 1
            self._counters["foreground_total_ms"] += total_ms
            self._counters["foreground_cache_ms"] += cache_ms
            self._counters["foreground_sample_ms"] += sample_ms
            self._counters["foreground_search_ms"] += search_ms
            self._counters["foreground_max_ms"] = max(
                self._counters["foreground_max_ms"], total_ms
            )

            if search_requested:
                self._counters["search_requests"] += 1
            if sample_ms + search_ms > 0.0:
                self._counters["search_computations"] += 1
            if cache_hit:
                self._counters["cache_hits"] += 1
                source_key = {
                    "background": "background_cache_hits",
                    "foreground": "foreground_cache_hits",
                }.get(str(cache_source or ""), "unknown_cache_hits")
                self._counters[source_key] += 1
                if source_key == "background_cache_hits":
                    branch_key = {
                        "all_pass": "background_cache_hits_all_pass",
                        "sampled:pass": "background_cache_hits_sampled_pass",
                        "sampled:receive": "background_cache_hits_sampled_receive",
                        "sampled:attack": "background_cache_hits_sampled_attack",
                        "sampled:attack_after_block": (
                            "background_cache_hits_sampled_attack_after_block"
                        ),
                        "current_turn": "background_cache_hits_current_turn",
                    }.get(
                        str(cache_branch_kind or ""),
                        "background_cache_hits_other",
                    )
                    self._counters[branch_key] += 1
                self._counters["cached_compute_ms"] += cached_ms
                self._counters["estimated_saved_ms"] += max(0.0, cached_ms - cache_ms)

            decisions = int(self._counters["foreground_decisions"])
            should_checkpoint = bool(
                self._checkpoint_path
                and self._checkpoint_decisions > 0
                and decisions - self._last_checkpoint_decisions
                >= self._checkpoint_decisions
            )
            if should_checkpoint:
                self._last_checkpoint_decisions = decisions

        if should_checkpoint:
            self.checkpoint("decision_interval")

    def snapshot(self, *, include_runtime: bool = True) -> Dict[str, Any]:
        with self._lock:
            values = dict(self._counters)
            started_at = self._started_at

        decisions = int(values["foreground_decisions"])
        requests = int(values["search_requests"])
        cache_hits = int(values["cache_hits"])
        background_hits = int(values["background_cache_hits"])
        snapshot: Dict[str, Any] = {
            **{
                key: int(value) if key.endswith("decisions") or key.endswith("requests")
                or key.endswith("hits") or key == "search_computations"
                or key.startswith("background_")
                else round(float(value), 3)
                for key, value in values.items()
            },
            "cache_hit_rate": round(cache_hits / requests if requests else 0.0, 5),
            "background_hit_rate": round(
                background_hits / requests if requests else 0.0, 5
            ),
            "background_use_rate": round(
                background_hits / float(values["background_cache_ready"])
                if values["background_cache_ready"]
                else 0.0,
                5,
            ),
            "background_path_action_match_rate": round(
                float(values["background_action_matches"])
                / (
                    float(values["background_action_matches"])
                    + float(values["background_path_mismatches"])
                )
                if (
                    values["background_action_matches"]
                    + values["background_path_mismatches"]
                )
                else 0.0,
                5,
            ),
            "background_path_completion_rate": round(
                float(values["background_path_completions"])
                / float(values["background_branches_scheduled"])
                if values["background_branches_scheduled"]
                else 0.0,
                5,
            ),
            "speculative_compute_return": round(
                float(values["estimated_saved_ms"])
                / float(values["speculative_compute_ms"])
                if values["speculative_compute_ms"]
                else 0.0,
                5,
            ),
            "average_foreground_ms": round(
                float(values["foreground_total_ms"]) / decisions if decisions else 0.0,
                3,
            ),
            "uptime_seconds": round(max(0.0, time.time() - started_at), 3),
            "checkpoint_enabled": bool(self._checkpoint_path),
        }
        if include_runtime:
            from goita_ai2.current_ai.background_search import (
                background_search_runtime_snapshot,
            )

            snapshot["background_runtime"] = background_search_runtime_snapshot()
        return snapshot

    def checkpoint(self, reason: str) -> bool:
        if not self._checkpoint_path:
            return False
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reason": str(reason or "manual"),
            "metrics": self.snapshot(),
        }
        path = Path(self._checkpoint_path)
        try:
            with self._checkpoint_lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
                    handle.write("\n")
            return True
        except OSError:
            return False

    def record_background_prefetch(
        self,
        *,
        candidates: int,
        scheduled: int,
        throttled: int,
    ) -> None:
        with self._lock:
            self._counters["background_prefetch_calls"] += 1
            self._counters["background_branch_candidates"] += max(0, int(candidates))
            self._counters["background_branches_scheduled"] += max(0, int(scheduled))
            self._counters["background_branches_throttled"] += max(0, int(throttled))

    def record_background_paths(
        self,
        *,
        matches: int,
        mismatches: int,
        completions: int,
    ) -> None:
        with self._lock:
            self._counters["background_action_matches"] += max(0, int(matches))
            self._counters["background_path_mismatches"] += max(0, int(mismatches))
            self._counters["background_path_completions"] += max(0, int(completions))

    def record_background_finish(self, status: str) -> None:
        key = {
            "cache_ready": "background_cache_ready",
            "cancelled": "background_cancelled",
            "skipped": "background_skipped",
            "errors": "background_errors",
        }.get(str(status or ""))
        if key is None:
            return
        with self._lock:
            self._counters[key] += 1


_AI_SEARCH_TELEMETRY = AiSearchTelemetry(
    checkpoint_path=os.getenv("GOITA_AI_TELEMETRY_PATH", ""),
    checkpoint_decisions=_nonnegative_int_env(
        "GOITA_AI_TELEMETRY_CHECKPOINT_DECISIONS", 100
    ),
)


def record_ai_search_decision(agent: Any, metrics: Dict[str, float]) -> None:
    speculative = bool(getattr(agent, "_time_search_cancel_event", None))
    cache_hit = bool(getattr(agent, "last_time_search_cache_hit", False))
    cache_source = getattr(agent, "last_time_search_cache_source", None)
    cache_branch_kind = getattr(agent, "last_time_search_cache_branch_kind", None)
    cache_branch_context = getattr(
        agent, "last_time_search_cache_branch_context", None
    )
    _AI_SEARCH_TELEMETRY.record_decision(
        metrics,
        speculative=speculative,
        search_requested=bool(getattr(agent, "last_time_search_cache_key", None)),
        cache_hit=cache_hit,
        cache_source=cache_source,
        cached_compute_ms=float(
            getattr(agent, "last_time_search_cached_compute_ms", 0.0) or 0.0
        ),
        cache_branch_kind=cache_branch_kind,
    )
    if not speculative and cache_hit and cache_source == "background":
        recorder = getattr(agent, "record_background_cache_hit", None)
        if callable(recorder):
            recorder(
                cache_branch_kind,
                cache_branch_context,
                cached_compute_ms=float(
                    getattr(agent, "last_time_search_cached_compute_ms", 0.0) or 0.0
                ),
            )


def record_ai_background_prefetch(
    *,
    candidates: int,
    scheduled: int,
    throttled: int,
) -> None:
    _AI_SEARCH_TELEMETRY.record_background_prefetch(
        candidates=candidates,
        scheduled=scheduled,
        throttled=throttled,
    )


def record_ai_background_paths(
    *,
    matches: int,
    mismatches: int,
    completions: int,
) -> None:
    _AI_SEARCH_TELEMETRY.record_background_paths(
        matches=matches,
        mismatches=mismatches,
        completions=completions,
    )


def record_ai_background_finish(status: str) -> None:
    _AI_SEARCH_TELEMETRY.record_background_finish(status)


def ai_search_telemetry_snapshot() -> Dict[str, Any]:
    return _AI_SEARCH_TELEMETRY.snapshot()


def checkpoint_ai_search_telemetry(reason: str) -> bool:
    return _AI_SEARCH_TELEMETRY.checkpoint(reason)


def reset_ai_search_telemetry() -> None:
    """Reset helper used by deterministic diagnostics and tests."""
    _AI_SEARCH_TELEMETRY.reset()

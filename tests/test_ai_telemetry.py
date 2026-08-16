from __future__ import annotations

import json
import tempfile
from pathlib import Path

from goita_ai2.current_ai.telemetry import (
    AiSearchTelemetry,
    ai_search_telemetry_snapshot,
    reset_ai_search_telemetry,
)
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


HANDS = {
    "A": list("11123457"),
    "B": list("11122344"),
    "C": list("11244556"),
    "D": list("11356789"),
}


def _apply_and_notify(agent, state, player, action) -> None:
    action_type, block, attack = action
    if action_type == "pass":
        state.apply_pass(player)
    elif action_type == "receive":
        state.apply_receive(player, block)
    elif action_type == "attack":
        state.apply_attack(player, attack)
    else:
        state.apply_attack_after_block(player, block, attack)
    agent.on_public_action(state, player, action)


def test_background_cache_hit_is_counted_as_saved_foreground_work() -> None:
    reset_ai_search_telemetry()
    state = GoitaState(HANDS, dealer="A")
    agent = RuleBasedAgent()
    agent.bind_player("B")
    agent.TIME_SEARCH_MAX_SECONDS = 0.2
    agent.TIME_SEARCH_SAMPLE_COUNT = 2
    agent.TIME_SEARCH_MAX_DEPTH = 1
    agent.TIME_SEARCH_MAX_NODES = 500
    agent._ensure_trackers(state)
    _apply_and_notify(agent, state, "A", ("attack_after_block", "1", "2"))

    assert agent.prefetch_next_turn(state)
    assert agent.wait_for_background_search()
    chosen = agent.select_action(state, "B", state.legal_actions("B"))

    assert chosen in state.legal_actions("B")
    assert agent.last_time_search_cache_hit is True
    assert agent.last_time_search_cache_source == "background"
    assert agent.last_time_search_cached_compute_ms >= 0.0
    snapshot = ai_search_telemetry_snapshot()
    assert snapshot["foreground_decisions"] == 1
    assert snapshot["speculative_decisions"] >= 1
    assert snapshot["search_requests"] == 1
    assert snapshot["cache_hits"] == 1
    assert snapshot["background_cache_hits"] == 1
    assert snapshot["background_cache_hits_current_turn"] == 1
    assert snapshot["estimated_saved_ms"] >= 0.0


def test_jsonl_checkpoint_contains_aggregate_metrics() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "telemetry.jsonl"
        telemetry = AiSearchTelemetry(
            checkpoint_path=str(output),
            checkpoint_decisions=0,
        )
        telemetry.record_decision(
            {
                "total_ms": 12.0,
                "cache_ms": 1.0,
                "sample_generation_ms": 0.0,
                "search_ms": 0.0,
            },
            speculative=False,
            search_requested=True,
            cache_hit=True,
            cache_source="background",
            cached_compute_ms=20.0,
            cache_branch_kind="all_pass",
        )
        telemetry.record_background_prefetch(
            candidates=2,
            scheduled=2,
            throttled=0,
        )
        telemetry.record_background_paths(matches=2, mismatches=1, completions=1)
        telemetry.record_background_finish("cache_ready")

        assert telemetry.checkpoint("test")
        payload = json.loads(output.read_text(encoding="utf-8").strip())
        assert payload["reason"] == "test"
        assert payload["metrics"]["background_cache_hits"] == 1
        assert payload["metrics"]["background_cache_hits_all_pass"] == 1
        assert payload["metrics"]["estimated_saved_ms"] == 19.0
        assert payload["metrics"]["background_prefetch_calls"] == 1
        assert payload["metrics"]["background_path_action_match_rate"] == 0.66667
        assert payload["metrics"]["background_path_completion_rate"] == 0.5
        assert payload["metrics"]["speculative_compute_return"] == 0.0
        assert "background_runtime" in payload["metrics"]

        restored = AiSearchTelemetry(
            checkpoint_path=str(output),
            checkpoint_decisions=0,
        ).snapshot(include_runtime=False)
        assert restored["foreground_decisions"] == 1
        assert restored["background_cache_hits"] == 1
        assert restored["estimated_saved_ms"] == 19.0


if __name__ == "__main__":
    test_background_cache_hit_is_counted_as_saved_foreground_work()
    test_jsonl_checkpoint_contains_aggregate_metrics()
    print("AI_TELEMETRY_TEST_OK")

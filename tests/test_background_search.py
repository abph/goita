from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path

from goita_ai2.current_ai.background_search import (
    BackgroundSearchValueModel,
    BackgroundSearchRuntime,
    background_search_runtime_snapshot,
    reset_background_search_value_model,
)
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


HANDS = {
    "A": list("11123457"),
    "B": list("11122344"),
    "C": list("11244556"),
    "D": list("11356789"),
}
ALTERNATE_HANDS = {
    "A": list("11123457"),
    "B": list("11122346"),
    "C": list("11244556"),
    "D": list("11345789"),
}


def _fast_agent(seat: str) -> RuleBasedAgent:
    agent = RuleBasedAgent()
    agent.bind_player(seat)
    agent.TIME_SEARCH_MAX_SECONDS = 0.2
    agent.TIME_SEARCH_SAMPLE_COUNT = 2
    agent.TIME_SEARCH_MAX_DEPTH = 1
    agent.TIME_SEARCH_MAX_NODES = 500
    return agent


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


def test_background_search_reuses_exact_current_turn_result() -> None:
    reset_background_search_value_model()
    state = GoitaState(HANDS, dealer="A")
    agent = _fast_agent("B")
    agent._ensure_trackers(state)
    _apply_and_notify(agent, state, "A", ("attack_after_block", "1", "2"))

    assert agent.prefetch_next_turn(state)
    assert agent.wait_for_background_search()
    assert agent.background_search_snapshot()["last_status"] == "cache_ready"

    chosen = agent.select_action(state, "B", state.legal_actions("B"))

    assert chosen in state.legal_actions("B")
    assert agent.last_time_search_cache_hit is True
    assert agent.last_time_search_cache_branch_kind == "current_turn"
    assert agent.last_time_search_cache_branch_context == (
        "current_turn|early|distance_0"
    )
    snapshot = agent.background_search_snapshot()
    assert snapshot["background_cache_hits"] == 1
    assert snapshot["outcomes_by_kind"]["current_turn"]["cache_hits"] == 1
    events = snapshot["diagnostic_events"]
    scheduled = next(event for event in events if event["event"] == "branch_scheduled")
    assert scheduled["reason"] == "protected_branch"
    assert scheduled["branch_kind"] == "current_turn"
    assert scheduled["search_seconds"] == 0.2
    assert scheduled["search_max_depth"] == 1
    assert any(
        event["event"] == "branch_finished"
        and event["reason"] == "exact_result_cached"
        for event in events
    )
    cache_hit = next(
        event for event in events if event["event"] == "background_cache_hit"
    )
    assert cache_hit["reason"] == "foreground_reused_exact_result"
    assert cache_hit["branch_context"] == "current_turn|early|distance_0"


def test_adaptive_value_model_suppresses_and_periodically_reprobes() -> None:
    model = BackgroundSearchValueModel()
    kind = "sampled:attack"
    for _ in range(4):
        model.record_scheduled(kind)

    decisions = [
        model.should_admit(
            kind,
            enabled=True,
            min_scheduled=4,
            min_hit_rate=0.1,
            probe_interval=3,
        )
        for _ in range(3)
    ]
    assert decisions == [False, False, True]
    snapshot = model.snapshot()["by_kind"][kind]
    assert snapshot["suppressed"] == 3
    assert snapshot["probes"] == 1

    model.record_cache_hit(kind)
    assert model.should_admit(
        kind,
        enabled=True,
        min_scheduled=4,
        min_hit_rate=0.1,
        probe_interval=3,
    )


def test_adaptive_value_model_explains_admission_decisions() -> None:
    model = BackgroundSearchValueModel()
    kind = "sampled:attack"
    for _ in range(4):
        model.record_scheduled(kind)

    admitted, details = model.admission_decision(
        kind,
        enabled=True,
        min_scheduled=4,
        min_hit_rate=0.1,
        probe_interval=2,
    )
    assert admitted is False
    assert details["reason"] == "low_hit_rate"
    assert details["scope"] == "kind"
    assert details["scheduled"] == 4
    assert details["cache_hits"] == 0

    admitted, details = model.admission_decision(
        kind,
        enabled=True,
        min_scheduled=4,
        min_hit_rate=0.1,
        probe_interval=2,
    )
    assert admitted is True
    assert details["reason"] == "periodic_probe"


def test_adaptive_value_model_prefers_mature_context_over_global_average() -> None:
    model = BackgroundSearchValueModel()
    kind = "sampled:attack"
    context = "sampled:attack|endgame|distance_1"
    for _ in range(4):
        model.record_scheduled(kind, context)

    assert not model.should_admit(
        kind,
        context=context,
        enabled=True,
        min_scheduled=16,
        context_min_scheduled=4,
        min_hit_rate=0.1,
        probe_interval=8,
    )
    model.record_cache_hit(kind, context)
    assert model.should_admit(
        kind,
        context=context,
        enabled=True,
        min_scheduled=16,
        context_min_scheduled=4,
        min_hit_rate=0.1,
        probe_interval=8,
    )
    snapshot = model.snapshot()
    assert snapshot["by_context"][context]["hit_rate"] == 0.25


def test_adaptive_value_model_persists_and_restores_atomically() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "adaptive-value.json"
        kind = "sampled:attack"
        context = "sampled:attack|early|distance_1"
        model = BackgroundSearchValueModel(
            checkpoint_path=str(output),
            checkpoint_operations=2,
        )
        model.record_scheduled(kind, context)
        assert not output.exists()
        model.record_cache_hit(kind, context)
        assert output.is_file()

        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 3
        assert payload["reason"] == "operation_interval"
        assert payload["decay"]["learning_operations"] == 1
        assert not list(output.parent.glob("*.tmp"))

        restored = BackgroundSearchValueModel(
            checkpoint_path=str(output),
            checkpoint_operations=2,
        )
        snapshot = restored.snapshot()
        assert snapshot["persistence"]["loaded"] is True
        assert snapshot["by_kind"][kind]["scheduled"] == 1
        assert snapshot["by_kind"][kind]["cache_hits"] == 1
        assert snapshot["by_context"][context]["hit_rate"] == 1.0


def test_adaptive_value_model_decays_old_evidence() -> None:
    model = BackgroundSearchValueModel(
        decay_half_life_operations=4,
        decay_interval_operations=1,
    )
    kind = "sampled:attack"
    other_kind = "sampled:receive"
    for _ in range(4):
        model.record_scheduled(kind)
        model.record_cache_hit(kind)

    before = model.snapshot()["by_kind"][kind]
    for _ in range(4):
        model.record_scheduled(other_kind)
    after = model.snapshot()["by_kind"][kind]

    assert after["scheduled"] == 4
    assert after["cache_hits"] == 4
    assert after["effective_scheduled"] < before["effective_scheduled"]
    assert after["hit_rate"] == 1.0

    admitted, details = model.admission_decision(
        kind,
        enabled=True,
        min_scheduled=4,
        min_hit_rate=0.75,
        probe_interval=8,
    )
    assert admitted is True
    assert details["reason"] == "observation_window"

    model.record_scheduled(kind)
    latest = model.snapshot()["by_kind"][kind]
    assert latest["lifetime_hit_rate"] == 0.8
    assert latest["hit_rate"] < latest["lifetime_hit_rate"]
    admitted, details = model.admission_decision(
        kind,
        enabled=True,
        min_scheduled=1,
        min_hit_rate=0.75,
        probe_interval=8,
    )
    assert admitted is False
    assert details["hit_rate"] == latest["hit_rate"]
    assert details["lifetime_hit_rate"] == 0.8
    assert model.snapshot()["decay"]["events"] >= 1


def test_adaptive_value_model_migrates_legacy_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "adaptive-value.json"
        output.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "revision": 7,
                    "by_kind": {
                        "sampled:attack": {
                            "scheduled": 4,
                            "cache_hits": 1,
                            "suppressed": 2,
                            "probes": 0,
                        }
                    },
                    "by_context": {},
                }
            ),
            encoding="utf-8",
        )

        model = BackgroundSearchValueModel(checkpoint_path=str(output))
        snapshot = model.snapshot()

        assert snapshot["persistence"]["loaded"] is True
        assert snapshot["by_kind"]["sampled:attack"]["hit_rate"] == 0.25
        assert snapshot["by_kind"]["sampled:attack"][
            "effective_scheduled"
        ] == 4.0
        assert snapshot["decay"]["learning_operations"] == 0


def test_adaptive_value_model_restores_decayed_evidence() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "adaptive-value.json"
        kind = "sampled:attack"
        other_kind = "sampled:receive"
        model = BackgroundSearchValueModel(
            checkpoint_path=str(output),
            checkpoint_operations=100,
            decay_half_life_operations=4,
            decay_interval_operations=1,
        )
        for _ in range(4):
            model.record_scheduled(kind)
            model.record_cache_hit(kind)
        for _ in range(2):
            model.record_scheduled(other_kind)
        assert model.checkpoint("decay_restore_test")
        before = model.snapshot()

        restored = BackgroundSearchValueModel(
            checkpoint_path=str(output),
            checkpoint_operations=100,
            decay_half_life_operations=4,
            decay_interval_operations=1,
        )
        after = restored.snapshot()

        assert after["decay"]["learning_operations"] == before["decay"][
            "learning_operations"
        ]
        assert after["decay"]["events"] == before["decay"]["events"]
        assert after["by_kind"][kind]["effective_scheduled"] == before[
            "by_kind"
        ][kind]["effective_scheduled"]
        restored.record_scheduled(other_kind)
        assert restored.snapshot()["by_kind"][kind][
            "effective_scheduled"
        ] < after["by_kind"][kind]["effective_scheduled"]


def test_adaptive_value_model_bounds_learning_tables() -> None:
    model = BackgroundSearchValueModel(
        max_kind_entries=2,
        max_context_entries=2,
    )
    model.record_scheduled("old", "old|early|distance_1")
    model.record_scheduled("strong", "strong|early|distance_1")
    model.record_cache_hit("strong", "strong|early|distance_1")
    model.record_scheduled("strong", "strong|early|distance_1")
    model.record_scheduled("new", "new|early|distance_1")

    snapshot = model.snapshot()
    assert set(snapshot["by_kind"]) == {"strong", "new"}
    assert set(snapshot["by_context"]) == {
        "strong|early|distance_1",
        "new|early|distance_1",
    }
    assert snapshot["capacity"]["kind_entries"] == 2
    assert snapshot["capacity"]["context_entries"] == 2
    assert snapshot["capacity"]["kind_evictions"] == 1
    assert snapshot["capacity"]["context_evictions"] == 1

    model.record_cache_hit("old", "old|early|distance_1")
    snapshot = model.snapshot()
    assert "old" not in snapshot["by_kind"]
    assert "old|early|distance_1" not in snapshot["by_context"]
    assert snapshot["capacity"]["orphan_cache_hits"] == 1


def test_adaptive_value_model_admission_does_not_create_empty_entries() -> None:
    model = BackgroundSearchValueModel(
        max_kind_entries=1,
        max_context_entries=1,
    )
    for index in range(5):
        admitted, details = model.admission_decision(
            f"kind-{index}",
            context=f"context-{index}",
            enabled=True,
            min_scheduled=4,
            min_hit_rate=0.1,
            probe_interval=8,
        )
        assert admitted is True
        assert details["reason"] == "observation_window"

    snapshot = model.snapshot()
    assert snapshot["by_kind"] == {}
    assert snapshot["by_context"] == {}


def test_adaptive_value_model_keeps_stronger_entries_over_one_off_key() -> None:
    model = BackgroundSearchValueModel(
        max_kind_entries=2,
        max_context_entries=2,
    )
    for _ in range(2):
        model.record_scheduled("strong-a", "strong-context-a")
        model.record_scheduled("strong-b", "strong-context-b")

    model.record_scheduled("one-off", "one-off-context")
    snapshot = model.snapshot()

    assert set(snapshot["by_kind"]) == {"strong-a", "strong-b"}
    assert set(snapshot["by_context"]) == {
        "strong-context-a",
        "strong-context-b",
    }
    assert snapshot["capacity"]["kind_evictions"] == 1
    assert snapshot["capacity"]["context_evictions"] == 1


def test_adaptive_value_model_prunes_oversized_checkpoint_on_restore() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "adaptive-value.json"
        bucket = {
            "scheduled": 1,
            "cache_hits": 0,
            "suppressed": 0,
            "probes": 0,
            "effective_scheduled": 1.0,
            "effective_cache_hits": 0.0,
        }
        output.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "revision": 9,
                    "decay": {
                        "learning_operations": 3,
                        "last_decay_operation": 0,
                    },
                    "by_kind": {
                        "kind-a": dict(bucket),
                        "kind-b": {**bucket, "effective_scheduled": 3.0},
                        "kind-c": {**bucket, "effective_scheduled": 2.0},
                    },
                    "by_context": {
                        "context-a": dict(bucket),
                        "context-b": {**bucket, "effective_scheduled": 3.0},
                        "context-c": {**bucket, "effective_scheduled": 2.0},
                    },
                }
            ),
            encoding="utf-8",
        )

        model = BackgroundSearchValueModel(
            checkpoint_path=str(output),
            max_kind_entries=2,
            max_context_entries=2,
        )
        snapshot = model.snapshot()

        assert set(snapshot["by_kind"]) == {"kind-b", "kind-c"}
        assert set(snapshot["by_context"]) == {"context-b", "context-c"}
        assert snapshot["capacity"]["restore_pruned_kind"] == 1
        assert snapshot["capacity"]["restore_pruned_context"] == 1
        assert snapshot["persistence"]["revision"] == 10


def test_adaptive_value_model_rotates_checkpoint_generations() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "adaptive-value.json"
        model = BackgroundSearchValueModel(
            checkpoint_path=str(output),
            checkpoint_operations=100,
            checkpoint_generations=2,
        )
        for index in range(1, 5):
            model.record_scheduled("sampled:attack")
            assert model.checkpoint(f"generation-{index}")

        primary = json.loads(output.read_text(encoding="utf-8"))
        first = json.loads(
            Path(f"{output}.bak.1").read_text(encoding="utf-8")
        )
        second = json.loads(
            Path(f"{output}.bak.2").read_text(encoding="utf-8")
        )

        assert primary["by_kind"]["sampled:attack"]["scheduled"] == 4
        assert first["by_kind"]["sampled:attack"]["scheduled"] == 3
        assert second["by_kind"]["sampled:attack"]["scheduled"] == 2
        assert not Path(f"{output}.bak.3").exists()
        assert primary["generation_limit"] == 2
        assert model.snapshot()["persistence"]["generation_limit"] == 2
        assert not list(output.parent.glob("*.tmp"))


def test_adaptive_value_model_recovers_from_previous_generation() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "adaptive-value.json"
        model = BackgroundSearchValueModel(
            checkpoint_path=str(output),
            checkpoint_operations=100,
            checkpoint_generations=2,
        )
        model.record_scheduled("sampled:attack")
        assert model.checkpoint("first")
        model.record_scheduled("sampled:attack")
        assert model.checkpoint("second")
        output.write_text("{broken", encoding="utf-8")

        recovered = BackgroundSearchValueModel(
            checkpoint_path=str(output),
            checkpoint_operations=100,
            checkpoint_generations=2,
        )
        snapshot = recovered.snapshot()

        assert snapshot["persistence"]["loaded"] is True
        assert snapshot["persistence"]["loaded_generation"] == 1
        assert snapshot["persistence"]["recovery_used"] is True
        assert snapshot["persistence"]["recovery_count"] == 1
        assert snapshot["persistence"]["restore_errors"]
        assert snapshot["persistence"]["last_error"] == ""
        assert snapshot["by_kind"]["sampled:attack"]["scheduled"] == 1

        assert recovered.checkpoint("heal_primary")
        healed = BackgroundSearchValueModel(
            checkpoint_path=str(output),
            checkpoint_generations=2,
        ).snapshot()
        assert healed["persistence"]["loaded_generation"] == 0
        assert healed["persistence"]["recovery_used"] is False
        assert healed["by_kind"]["sampled:attack"]["scheduled"] == 1


def test_adaptive_value_model_skips_multiple_corrupt_generations() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "adaptive-value.json"
        model = BackgroundSearchValueModel(
            checkpoint_path=str(output),
            checkpoint_generations=3,
        )
        for index in range(1, 4):
            model.record_scheduled("sampled:attack")
            assert model.checkpoint(f"generation-{index}")
        output.write_text("{broken-primary", encoding="utf-8")
        Path(f"{output}.bak.1").write_text(
            "{broken-first",
            encoding="utf-8",
        )

        recovered = BackgroundSearchValueModel(
            checkpoint_path=str(output),
            checkpoint_generations=3,
        ).snapshot()

        assert recovered["persistence"]["loaded_generation"] == 2
        assert len(recovered["persistence"]["restore_errors"]) == 2
        assert recovered["by_kind"]["sampled:attack"]["scheduled"] == 1


def test_adaptive_value_model_ignores_corrupt_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "adaptive-value.json"
        output.write_text("{broken", encoding="utf-8")

        model = BackgroundSearchValueModel(
            checkpoint_path=str(output),
            checkpoint_operations=2,
        )
        snapshot = model.snapshot()
        assert snapshot["persistence"]["loaded"] is False
        assert snapshot["persistence"]["last_error"]
        assert snapshot["by_kind"] == {}


def test_projected_public_passes_match_the_real_tracker_and_state() -> None:
    state = GoitaState(HANDS, dealer="A")
    agent = _fast_agent("D")
    agent._ensure_trackers(state)
    _apply_and_notify(agent, state, "A", ("attack_after_block", "1", "3"))

    assert agent.prefetch_next_turn(state)
    assert agent.background_search_snapshot()["last_pass_count"] == 2
    assert agent.wait_for_background_search()
    _apply_and_notify(agent, state, "B", ("pass", None, None))
    assert agent.retain_background_search_for_action(("pass", None, None))
    _apply_and_notify(agent, state, "C", ("pass", None, None))
    assert agent.retain_background_search_for_action(("pass", None, None))

    chosen = agent.select_action(state, "D", state.legal_actions("D"))

    assert chosen in state.legal_actions("D")
    assert agent.last_time_search_cache_hit is True
    assert agent.last_time_search_cache_branch_kind == "all_pass"


def test_background_search_can_be_cancelled() -> None:
    state = GoitaState(HANDS, dealer="A")
    agent = _fast_agent("D")
    agent.TIME_SEARCH_MAX_SECONDS = 1.0
    agent.TIME_SEARCH_SAMPLE_COUNT = 40
    agent.TIME_SEARCH_MAX_DEPTH = 7
    agent._ensure_trackers(state)
    _apply_and_notify(agent, state, "A", ("attack_after_block", "1", "3"))

    assert agent.prefetch_next_turn(state)
    agent.cancel_background_search()
    assert agent.wait_for_background_search()
    assert agent.background_search_snapshot()["last_status"] == "cancelled"


def test_foreground_and_background_share_one_inflight_search() -> None:
    state = GoitaState(HANDS, dealer="A")
    agent = _fast_agent("B")
    agent.TIME_SEARCH_MAX_SECONDS = 0.3
    agent.TIME_SEARCH_SAMPLE_COUNT = 8
    agent.TIME_SEARCH_MAX_DEPTH = 5
    agent._ensure_trackers(state)
    _apply_and_notify(agent, state, "A", ("attack_after_block", "1", "2"))

    assert agent.prefetch_next_turn(state)
    chosen = agent.select_action(state, "B", state.legal_actions("B"))
    assert agent.wait_for_background_search()

    assert chosen in state.legal_actions("B")
    cache = agent.time_search_cache_snapshot()
    assert cache["stores"] == 1
    assert cache["hits"] >= 1
    assert background_search_runtime_snapshot()["foreground_active"] == 0


def _prepare_b_attack_turn(hands):
    state = GoitaState(hands, dealer="A")
    agent = _fast_agent("C")
    agent._ensure_trackers(state)
    _apply_and_notify(agent, state, "A", ("attack_after_block", "1", "2"))
    _apply_and_notify(agent, state, "B", ("receive", "2", None))
    return state, agent


def _prepare_b_hidden_attack_turn():
    state = GoitaState(HANDS, dealer="B")
    agent = _fast_agent("C")
    agent.TIME_SEARCH_BACKGROUND_ALLOWED_SAMPLED_ACTIONS = (
        "pass",
        "receive",
        "attack",
        "attack_after_block",
    )
    agent._ensure_trackers(state)
    _apply_and_notify(agent, state, "B", ("attack_after_block", "1", "3"))
    _apply_and_notify(agent, state, "C", ("pass", None, None))
    _apply_and_notify(agent, state, "D", ("pass", None, None))
    _apply_and_notify(agent, state, "A", ("pass", None, None))
    return state, agent


def test_inferred_non_pass_branch_reuses_cache_when_attack_matches() -> None:
    state, agent = _prepare_b_attack_turn(HANDS)

    assert agent.prefetch_next_turn(state)
    paths = agent.background_search_snapshot()["last_paths"]
    predicted = next(
        tuple(path[0])
        for path in paths
        if path and tuple(path[0]) in state.legal_actions("B")
    )
    _apply_and_notify(agent, state, "B", predicted)
    agent.retain_background_search_for_action(predicted)
    assert agent.wait_for_background_search()

    chosen = agent.select_action(state, "C", state.legal_actions("C"))

    assert chosen in state.legal_actions("C")
    assert agent.last_time_search_cache_hit is True
    assert agent.last_time_search_cache_source == "background"
    snapshot = agent.background_search_snapshot()
    assert snapshot["projected_non_pass"] >= 1


def test_sampled_action_allowlist_can_disable_non_pass_projection() -> None:
    state, agent = _prepare_b_attack_turn(HANDS)
    agent.TIME_SEARCH_BACKGROUND_ALLOWED_SAMPLED_ACTIONS = tuple()

    assert agent._background_search_projections(state) == []


def test_projected_paths_do_not_depend_on_real_opponent_hands() -> None:
    first_state, first_agent = _prepare_b_attack_turn(HANDS)
    second_state, second_agent = _prepare_b_attack_turn(ALTERNATE_HANDS)

    first_paths = [
        branch.path
        for branch in first_agent._background_search_projections(first_state)
    ]
    second_paths = [
        branch.path
        for branch in second_agent._background_search_projections(second_state)
    ]

    assert first_paths == second_paths
    assert any(
        action[0] != "pass"
        for path in first_paths
        for action in path
    )


def test_hidden_block_projection_matches_only_the_public_attack() -> None:
    state, agent = _prepare_b_hidden_attack_turn()

    assert agent.prefetch_next_turn(state)
    paths = agent.background_search_snapshot()["last_paths"]
    predicted = next(
        tuple(path[0])
        for path in paths
        if path
        and path[0][0] == "attack_after_block"
        and any(
            action[0] == "attack_after_block" and action[2] == path[0][2]
            for action in state.legal_actions("B")
        )
    )
    assert predicted[1] is None
    actual = next(
        action
        for action in reversed(state.legal_actions("B"))
        if action[0] == "attack_after_block" and action[2] == predicted[2]
    )
    _apply_and_notify(agent, state, "B", actual)

    assert agent.retain_background_search_for_action(actual)
    assert agent.wait_for_background_search()
    chosen = agent.select_action(state, "C", state.legal_actions("C"))

    assert chosen in state.legal_actions("C")
    assert agent.last_time_search_cache_hit is True
    assert agent.last_time_search_cache_source == "background"


def test_opponent_hidden_block_never_changes_tracker_or_cache_key() -> None:
    first_state, first_agent = _prepare_b_hidden_attack_turn()
    second_state, second_agent = _prepare_b_hidden_attack_turn()
    first_action = ("attack_after_block", "2", "1")
    second_action = ("attack_after_block", "4", "1")

    assert first_action in first_state.legal_actions("B")
    assert second_action in second_state.legal_actions("B")
    _apply_and_notify(first_agent, first_state, "B", first_action)
    _apply_and_notify(second_agent, second_state, "B", second_action)

    first_tracker = first_agent._track[id(first_state)]
    second_tracker = second_agent._track[id(second_state)]
    assert first_tracker == second_tracker
    first_actions = first_state.legal_actions("C")
    second_actions = second_state.legal_actions("C")
    first_key = first_agent._timed_search_cache_key(
        first_state,
        "C",
        first_tracker,
        first_actions,
        first_actions[0],
    )
    second_key = second_agent._timed_search_cache_key(
        second_state,
        "C",
        second_tracker,
        second_actions,
        second_actions[0],
    )
    assert first_key.digest == second_key.digest


def test_runtime_bounds_pending_work_and_prioritizes_foreground() -> None:
    runtime = BackgroundSearchRuntime(max_workers=1, max_pending=2)
    gate = threading.Event()
    try:
        first = runtime.submit(gate.wait)
        assert first is not None
        deadline = time.monotonic() + 1.0
        while runtime.snapshot()["active"] == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert runtime.snapshot()["active"] == 1

        queued, reason = runtime.submit_with_reason(gate.wait)
        assert queued is not None
        assert reason == "accepted"
        rejected, reason = runtime.submit_with_reason(gate.wait)
        assert rejected is None
        assert reason == "pending_full"

        runtime.foreground_started()
        rejected, reason = runtime.submit_with_reason(gate.wait)
        assert rejected is None
        assert reason == "foreground_busy"
        runtime.foreground_finished()
        assert queued.cancelled()

        gate.set()
        first.result(timeout=1.0)
        snapshot = runtime.snapshot()
        assert snapshot["pending"] == 0
        assert snapshot["active"] == 0
        assert snapshot["rejected_full"] == 1
        assert snapshot["rejected_busy"] == 1
        assert snapshot["cancelled_queued"] == 1
        assert snapshot["max_pending_seen"] == 2
        assert snapshot["max_active_seen"] == 1
    finally:
        gate.set()
        runtime.shutdown()


if __name__ == "__main__":
    test_background_search_reuses_exact_current_turn_result()
    test_adaptive_value_model_suppresses_and_periodically_reprobes()
    test_adaptive_value_model_explains_admission_decisions()
    test_adaptive_value_model_prefers_mature_context_over_global_average()
    test_adaptive_value_model_persists_and_restores_atomically()
    test_adaptive_value_model_decays_old_evidence()
    test_adaptive_value_model_migrates_legacy_checkpoint()
    test_adaptive_value_model_restores_decayed_evidence()
    test_adaptive_value_model_bounds_learning_tables()
    test_adaptive_value_model_admission_does_not_create_empty_entries()
    test_adaptive_value_model_keeps_stronger_entries_over_one_off_key()
    test_adaptive_value_model_prunes_oversized_checkpoint_on_restore()
    test_adaptive_value_model_rotates_checkpoint_generations()
    test_adaptive_value_model_recovers_from_previous_generation()
    test_adaptive_value_model_skips_multiple_corrupt_generations()
    test_adaptive_value_model_ignores_corrupt_checkpoint()
    test_projected_public_passes_match_the_real_tracker_and_state()
    test_background_search_can_be_cancelled()
    test_foreground_and_background_share_one_inflight_search()
    test_inferred_non_pass_branch_reuses_cache_when_attack_matches()
    test_sampled_action_allowlist_can_disable_non_pass_projection()
    test_projected_paths_do_not_depend_on_real_opponent_hands()
    test_hidden_block_projection_matches_only_the_public_attack()
    test_opponent_hidden_block_never_changes_tracker_or_cache_key()
    test_runtime_bounds_pending_work_and_prioritizes_foreground()
    print("BACKGROUND_SEARCH_TEST_OK")

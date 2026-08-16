from __future__ import annotations

import copy
import threading
import time

from goita_ai2.current_ai.search_cache import SearchPositionKey, TimedSearchCache
from goita_ai2.current_ai.timed_search import TimedSearchResult
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


HANDS_ONE = {
    "A": list("11123457"),
    "B": list("11122334"),
    "C": list("11244556"),
    "D": list("11356789"),
}
HANDS_TWO = {
    "A": list("11123457"),
    "B": list("11134455"),
    "C": list("11223346"),
    "D": list("11256789"),
}


def _key(agent: RuleBasedAgent, state: GoitaState) -> SearchPositionKey:
    agent._ensure_trackers(state)
    actions = state.legal_actions("A")
    return agent._timed_search_cache_key(
        state,
        "A",
        agent._track[id(state)],
        actions,
        actions[0],
    )


def _result(depth: int) -> TimedSearchResult:
    return TimedSearchResult(
        action=("attack_after_block", "1", "2"),
        depth=depth,
        samples=4,
        nodes=100,
        elapsed_seconds=0.01,
        value=10.0,
        margin=2.0,
        agreement=0.75,
        decisive=False,
    )


def test_position_key_never_depends_on_opponents_real_hands() -> None:
    first_state = GoitaState(HANDS_ONE, dealer="A")
    second_state = GoitaState(HANDS_TWO, dealer="A")
    first_agent = RuleBasedAgent()
    second_agent = RuleBasedAgent()
    first_agent.bind_player("A")
    second_agent.bind_player("A")

    assert _key(first_agent, first_state).digest == _key(second_agent, second_state).digest

    second_state.hands["A"][-1] = "6"
    assert _key(first_agent, first_state).digest != _key(second_agent, second_state).digest


def test_position_key_changes_with_public_inference_and_policy() -> None:
    state = GoitaState(HANDS_ONE, dealer="A")
    agent = RuleBasedAgent()
    agent.bind_player("A")
    original = _key(agent, state)

    tracker = agent._track[id(state)]
    tracker["estimated_current_hands"]["B"]["5"]["expected"] += 0.25
    changed_inference = agent._timed_search_cache_key(
        state,
        "A",
        tracker,
        state.legal_actions("A"),
        state.legal_actions("A")[0],
    )
    assert original.digest != changed_inference.digest

    tracker["estimated_current_hands"]["B"]["5"]["expected"] -= 0.25
    agent.TIME_SEARCH_BRANCH_BEAM += 1
    assert original.digest != _key(agent, state).digest

    agent.TIME_SEARCH_BRANCH_BEAM -= 1
    agent.TIME_SEARCH_ADAPTIVE_BUDGET_ENABLED = False
    assert original.digest != _key(agent, state).digest


def test_cache_enforces_lru_capacity_ttl_and_quality() -> None:
    now = [10.0]
    cache = TimedSearchCache(
        max_entries=2,
        ttl_seconds=5.0,
        clock=lambda: now[0],
    )
    keys = [SearchPositionKey(str(index), "A", "attack", 8, 2) for index in range(3)]

    assert cache.get(keys[0]) is None
    assert cache.put(keys[0], _result(5))
    assert cache.get(keys[0]).depth == 5
    assert not cache.put(keys[0], _result(3))
    assert cache.get(keys[0]).depth == 5
    assert cache.put(keys[1], _result(5))
    assert cache.put(keys[2], _result(7))
    assert cache.get(keys[0]) is None

    now[0] += 6.0
    assert cache.get(keys[2]) is None
    snapshot = cache.snapshot()
    assert snapshot["evictions"] == 1
    assert snapshot["expired"] >= 1
    assert snapshot["size"] == 0


def test_rule_preview_shares_cache_without_copying_lock() -> None:
    agent = RuleBasedAgent()
    preview = copy.deepcopy(agent)
    assert preview._time_search_cache is agent._time_search_cache


def test_cache_singleflight_publishes_one_result_to_waiters() -> None:
    cache = TimedSearchCache(max_entries=2, ttl_seconds=5.0)
    key = SearchPositionKey("shared", "A", "attack", 8, 2)
    owner, event = cache.claim_compute(key)
    assert owner is True

    def complete() -> None:
        time.sleep(0.02)
        cache.finish_compute(key, _result(5))

    thread = threading.Thread(target=complete)
    thread.start()
    second_owner, second_event = cache.claim_compute(key)
    completed, result = cache.wait_for_compute(key, second_event, 1.0)
    thread.join()

    assert second_owner is False
    assert completed is True
    assert result is not None and result.depth == 5
    snapshot = cache.snapshot()
    assert snapshot["stores"] == 1
    assert snapshot["inflight_hits"] == 1
    assert snapshot["inflight"] == 0


def test_cache_records_result_source_and_compute_time() -> None:
    cache = TimedSearchCache(max_entries=2, ttl_seconds=30.0)
    key = SearchPositionKey("source", "A", "attack", 8, 2)
    result = _result(3)

    cache.finish_compute(
        key,
        result,
        source="background",
        compute_seconds=0.125,
        branch_kind="all_pass",
        branch_context="all_pass|early|distance_2",
    )

    assert cache.get(key) is result
    metadata = cache.entry_metadata(key)
    assert metadata is not None
    assert metadata["source"] == "background"
    assert metadata["compute_seconds"] == 0.125
    assert metadata["branch_kind"] == "all_pass"
    assert metadata["branch_context"] == "all_pass|early|distance_2"
    assert metadata["hits"] == 1


def test_timed_search_reuses_exact_cached_result() -> None:
    state = GoitaState(HANDS_ONE, dealer="A")
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent.TIME_SEARCH_MAX_SECONDS = 0.2
    agent.TIME_SEARCH_SAMPLE_COUNT = 2
    agent.TIME_SEARCH_MAX_DEPTH = 1
    agent.TIME_SEARCH_MAX_NODES = 200
    agent._ensure_trackers(state)
    actions = state.legal_actions("A")
    baseline = actions[0]

    first = agent._time_limited_search_action(state, "A", actions, baseline)
    first_stats = agent.time_search_cache_snapshot()
    second = agent._time_limited_search_action(state, "A", actions, baseline)
    second_stats = agent.time_search_cache_snapshot()

    assert first is not None
    assert second == first
    assert first_stats["stores"] == 1
    assert second_stats["hits"] == 1
    assert agent.last_time_search_cache_hit is True


if __name__ == "__main__":
    test_position_key_never_depends_on_opponents_real_hands()
    test_position_key_changes_with_public_inference_and_policy()
    test_cache_enforces_lru_capacity_ttl_and_quality()
    test_rule_preview_shares_cache_without_copying_lock()
    test_cache_singleflight_publishes_one_result_to_waiters()
    test_cache_records_result_source_and_compute_time()
    test_timed_search_reuses_exact_cached_result()
    print("SEARCH_CACHE_TEST_OK")

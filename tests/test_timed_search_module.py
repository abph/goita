from __future__ import annotations

import copy
import time

from goita_ai2.current_ai.prediction_cache import clear_prediction_sample_cache
from goita_ai2.current_ai.search_budget import reset_time_search_budget_model
from goita_ai2.current_ai.timed_search import TimedSearchMixin
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


def _initial_state(*, permuted: bool = False) -> GoitaState:
    other_hands = {
        "B": list("11122334"),
        "C": list("11244556"),
        "D": list("11356789"),
    }
    if permuted:
        other_hands = {
            "B": list("11356789"),
            "C": list("11122334"),
            "D": list("11244556"),
        }
    return GoitaState(
        hands={"A": list("11123457"), **other_hands},
        dealer="A",
    )


def _sample_key(states) -> tuple:
    return tuple(
        tuple(
            (seat, tuple(sorted(state.hands[seat])))
            for seat in "BCD"
        )
        for state in states
    )


def test_rule_based_agent_uses_timed_search_mixin() -> None:
    assert issubclass(RuleBasedAgent, TimedSearchMixin)


def test_hidden_hand_sampling_does_not_read_actual_opponent_hands() -> None:
    first_state = _initial_state()
    second_state = _initial_state(permuted=True)
    first_agent = RuleBasedAgent()
    second_agent = RuleBasedAgent()
    first_agent.bind_player("A")
    second_agent.bind_player("A")
    first_agent._ensure_trackers(first_state)
    second_agent._ensure_trackers(second_state)

    first_samples = first_agent._timed_search_sample_states(
        first_state,
        "A",
        first_agent._track[id(first_state)],
        6,
    )
    second_samples = second_agent._timed_search_sample_states(
        second_state,
        "A",
        second_agent._track[id(second_state)],
        6,
    )

    assert len(first_samples) == 6
    assert len(second_samples) == 6
    assert _sample_key(first_samples) == _sample_key(second_samples)


def test_time_limited_search_returns_a_completed_legal_result() -> None:
    state = _initial_state()
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent.TIME_SEARCH_MAX_SECONDS = 0.4
    agent.TIME_SEARCH_SAMPLE_COUNT = 4
    agent.TIME_SEARCH_MAX_DEPTH = 5
    agent.TIME_SEARCH_MAX_NODES = 20_000
    agent._ensure_trackers(state)
    actions = state.legal_actions("A")
    baseline = agent._select_rule_based_action(state, "A", actions)

    result = agent._time_limited_search_action(
        state,
        "A",
        actions,
        baseline,
    )

    assert result is not None
    assert result.action in actions
    assert result.depth in (1, 3, 5)
    assert result.samples == 4
    assert result.nodes > 0
    assert result.elapsed_seconds <= 0.8


def test_select_action_records_search_without_changing_public_state() -> None:
    state = _initial_state()
    hands_before = {seat: list(hand) for seat, hand in state.hands.items()}
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent.TIME_SEARCH_MAX_SECONDS = 0.4
    agent.TIME_SEARCH_SAMPLE_COUNT = 4
    agent.TIME_SEARCH_MAX_DEPTH = 5
    agent.TIME_SEARCH_MAX_NODES = 20_000

    action = agent.select_action(state, "A", state.legal_actions("A"))
    search = agent._track[id(state)].get("last_time_limited_search")

    assert action in state.legal_actions("A")
    assert search is not None
    assert search["depth"] in (1, 3, 5)
    assert state.hands == hands_before


def test_enemy_second_attack_wait_requires_a_robust_inferred_win() -> None:
    state = _initial_state()
    state.phase = "receive"
    state.turn = "A"
    state.attacker = "B"
    state.current_attack = "5"

    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent._ensure_trackers(state)
    tracker = agent._track[id(state)]
    tracker["enemy_attack_counts"]["B"] = 2

    common = dict(
        baseline_action=("receive", "5", None),
        best_action=("pass", None, None),
        completed_depth=7,
        agreement=0.875,
        information_enabled=True,
        information_confidence=0.75,
        best_minimum=110000.0,
        baseline_minimum=105000.0,
        margin=5000.0,
    )
    assert agent._timed_search_enemy_third_attack_wait_is_safe(
        state,
        "A",
        tracker,
        **common,
    )

    unsafe = dict(common)
    unsafe["best_minimum"] = 1200.0
    assert not agent._timed_search_enemy_third_attack_wait_is_safe(
        state,
        "A",
        tracker,
        **unsafe,
    )

    tracker["enemy_attack_counts"]["B"] = 1
    assert not agent._timed_search_enemy_third_attack_wait_is_safe(
        state,
        "A",
        tracker,
        **common,
    )


def test_weak_first_receive_search_accepts_only_a_clear_robust_result() -> None:
    agent = RuleBasedAgent()
    agent._time_search_profile = "weak_first_receive"
    common = dict(
        baseline_action=("pass", None, None),
        best_action=("receive", "5", None),
        completed_depth=9,
        agreement=0.43,
        margin=38000.0,
        information_enabled=True,
        information_confidence=0.55,
    )

    assert agent._timed_search_weak_first_receive_is_decisive(**common)

    narrow = dict(common)
    narrow["margin"] = 1000.0
    assert not agent._timed_search_weak_first_receive_is_decisive(**narrow)

    uncertain = dict(common)
    uncertain["information_confidence"] = 0.30
    assert not agent._timed_search_weak_first_receive_is_decisive(**uncertain)


def test_zero_shi_stop_signal_uses_context_limited_search_thresholds() -> None:
    agent = RuleBasedAgent()
    agent._time_search_profile = "weak_first_receive"
    result = dict(
        baseline_action=("pass", None, None),
        best_action=("receive", "4", None),
        completed_depth=7,
        agreement=0.582,
        margin=87.32,
        information_enabled=True,
        information_confidence=0.593,
    )

    assert not agent._timed_search_weak_first_receive_is_decisive(**result)
    assert agent._timed_search_weak_first_receive_is_decisive(
        **result,
        zero_shi_stop_signal=True,
    )


def test_zero_shi_stop_signal_context_matches_enemy_reply_to_ally_shi() -> None:
    state = GoitaState(
        hands={
            "A": list("56117431"),
            "B": list("41141319"),
            "C": list("45337562"),
            "D": list("58121221"),
        },
        dealer="A",
    )
    agent = RuleBasedAgent()
    agent.bind_player("C")
    agent._ensure_trackers(state)

    actions = (
        ("A", ("attack_after_block", "3", "1")),
        ("B", ("receive", "1", None)),
        ("B", ("attack", None, "4")),
    )
    for player, action in actions:
        action_type, block, attack = action
        if action_type == "receive":
            state.apply_receive(player, block)
        elif action_type == "attack":
            state.apply_attack(player, attack)
        else:
            state.apply_attack_after_block(player, block, attack)
        agent.on_public_action(state, player, action)

    assert agent._timed_search_zero_shi_stop_signal_context(
        state,
        "C",
        agent._track[id(state)],
        baseline_action=("pass", None, None),
        best_action=("receive", "4", None),
    )


def test_weak_rank_search_receives_gold_to_continue_ally_shi_attack() -> None:
    clear_prediction_sample_cache()
    reset_time_search_budget_model()
    state = GoitaState(
        hands={
            "A": list("19731141"),
            "B": list("52315436"),
            "C": list("51711262"),
            "D": list("51124843"),
        },
        dealer="A",
    )
    agents = {player: RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)
        agent._ensure_trackers(state)

    def apply_public(action_player: str, action) -> None:
        action_type, block, attack = action
        if action_type == "pass":
            state.apply_pass(action_player)
        elif action_type == "receive":
            state.apply_receive(action_player, block)
        elif action_type == "attack":
            state.apply_attack(action_player, attack)
        else:
            state.apply_attack_after_block(action_player, block, attack)
        for agent in agents.values():
            agent.on_public_action(state, action_player, action)

    opening = (
        ("A", ("attack_after_block", "1", "1")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("receive", "1", None)),
        ("D", ("attack", None, "4")),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("attack_after_block", "3", "4")),
        ("A", ("receive", "4", None)),
        ("A", ("attack", None, "1")),
        ("B", ("receive", "1", None)),
        ("B", ("attack", None, "5")),
    )
    for action_player, action in opening:
        apply_public(action_player, action)

    c_agent = agents["C"]
    receive = c_agent.select_action(state, "C", state.legal_actions("C"))
    search = c_agent._track[id(state)]["last_time_limited_search"]

    assert receive == ("receive", "5", None)
    assert c_agent.last_decision_reason == "time_search"
    assert c_agent.last_score_fallback_detail.startswith("weak_first_receive_")
    assert search["decisive"] is True
    assert search["budget"]["effective_seconds"] == 5.0

    apply_public("C", receive)
    attack = c_agent.select_action(state, "C", state.legal_actions("C"))

    assert attack == ("attack", None, "1")


def test_one_second_search_keeps_ally_gold_pass_after_broader_sampling() -> None:
    clear_prediction_sample_cache()
    reset_time_search_budget_model()
    state = GoitaState(
        hands={
            "A": list("24115126"),
            "B": list("41167234"),
            "C": list("89513451"),
            "D": list("31135723"),
        },
        dealer="B",
    )
    agents = {player: RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)
        agent._ensure_trackers(state)

    def apply_public(player: str, action) -> None:
        action_type, block, attack = action
        if action_type == "pass":
            state.apply_pass(player)
        elif action_type == "receive":
            state.apply_receive(player, block)
        elif action_type == "attack":
            state.apply_attack(player, attack)
        else:
            state.apply_attack_after_block(player, block, attack)
        for agent in agents.values():
            agent.on_public_action(state, player, action)

    opening = (
        ("B", ("attack_after_block", "1", "4")),
        ("C", ("receive", "4", None)),
        ("C", ("attack", None, "5")),
        ("D", ("pass", None, None)),
    )
    for player, action in opening:
        apply_public(player, action)

    a_agent = agents["A"]
    legal = state.legal_actions("A")
    preview = copy.deepcopy(a_agent)
    baseline = preview._select_rule_based_action(state, "A", legal)
    chosen = a_agent.select_action(state, "A", legal)
    search = a_agent._track[id(state)]["last_time_limited_search"]

    assert baseline == ("pass", None, None)
    assert chosen == ("pass", None, None)
    assert search["samples"] == 80
    assert search["decisive"] is False
    assert a_agent.last_score_fallback_detail == "pass_base"


def test_receive_branch_compares_followup_attacks_through_the_final_score() -> None:
    state = GoitaState(
        hands={
            "A": list("1145"),
            "B": list("9281"),
            "C": list("3366"),
            "D": list("5577"),
        },
        dealer="A",
    )
    state.phase = "receive"
    state.turn = "B"
    state.attacker = "A"
    state.current_attack = "4"

    agent = RuleBasedAgent()
    agent.bind_player("B")
    agent._ensure_trackers(state)
    after_receive = agent._timed_search_apply(
        state,
        "B",
        ("receive", "8", None),
    )
    baseline_scores = dict(state.team_score)
    deadline = time.perf_counter() + 2.0
    stats = {"nodes": 0, "max_nodes": 100_000}
    followup_values = {}
    for action in after_receive.legal_actions("B"):
        child = agent._timed_search_apply(after_receive, "B", action)
        followup_values[action] = agent._timed_search_minimax(
            child,
            "B",
            baseline_scores,
            8,
            -float("inf"),
            float("inf"),
            deadline,
            stats,
            {},
        )

    lance = ("attack", None, "2")
    assert max(followup_values, key=followup_values.get) == lance
    assert followup_values[lance] >= 125_000.0
    assert followup_values[lance] > followup_values[("attack", None, "9")]


if __name__ == "__main__":
    test_rule_based_agent_uses_timed_search_mixin()
    test_hidden_hand_sampling_does_not_read_actual_opponent_hands()
    test_time_limited_search_returns_a_completed_legal_result()
    test_select_action_records_search_without_changing_public_state()
    test_enemy_second_attack_wait_requires_a_robust_inferred_win()
    test_one_second_search_keeps_ally_gold_pass_after_broader_sampling()
    test_receive_branch_compares_followup_attacks_through_the_final_score()
    print("TIMED_SEARCH_MODULE_TEST_OK")

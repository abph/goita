"""Regression tests for guaranteed endgame routes that maximize score.

The search must keep a proven win ahead of normal rank/pass policy, then
compare every proven continuation by minimum, expected, and maximum score.
"""

from __future__ import annotations

from typing import Dict, Tuple

from goita_ai2.current_ai.endgame import ForcedWinResult, ForcedWinStatus
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState
from goita_ai2.instruction_case_audit import load_cases, reconstruct_case

Action = Tuple[str, str | None, str | None]


def _apply_public_action(
    state: GoitaState,
    agents: Dict[str, RuleBasedAgent],
    player: str,
    action: Action,
) -> None:
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


def _silver_receive_position() -> tuple[
    GoitaState,
    Dict[str, RuleBasedAgent],
]:
    state = GoitaState(
        hands={
            "A": list("51165328"),
            "B": list("45123124"),
            "C": list("23415116"),
            "D": list("14731719"),
        },
        dealer="C",
    )
    agents = {player: RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)
        agent.TIME_SEARCH_ENABLED = False
        agent._ensure_trackers(state)

    actions = (
        ("C", ("attack_after_block", "3", "1")),
        ("D", ("receive", "1", None)),
        ("D", ("attack", None, "7")),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("attack_after_block", "3", "7")),
        ("A", ("receive", "8", None)),
        ("A", ("attack", None, "5")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("attack_after_block", "3", "6")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("receive", "9", None)),
        ("D", ("attack", None, "1")),
        ("A", ("receive", "1", None)),
        ("A", ("attack", None, "5")),
        ("B", ("receive", "5", None)),
        ("B", ("attack", None, "4")),
    )
    for player, action in actions:
        _apply_public_action(state, agents, player, action)
    return state, agents


def test_proven_high_score_receive_outranks_first_attack_pass_policy() -> None:
    state, agents = _silver_receive_position()
    c_agent = agents["C"]
    receive = ("receive", "4", None)

    result = c_agent._forced_win_result_after_receive_action(
        state,
        "C",
        receive,
    )
    chosen = c_agent.select_action(state, "C", state.legal_actions("C"))

    assert result.status == ForcedWinStatus.PROVEN
    assert result.minimum_score == 20.0
    assert chosen == receive
    assert c_agent.last_score_fallback_detail == "receive_tsume_after"


def test_proven_route_continues_with_a_publicly_unreceivable_attack() -> None:
    state, agents = _silver_receive_position()
    c_agent = agents["C"]
    receive = ("receive", "4", None)
    _apply_public_action(state, agents, "C", receive)

    attack = c_agent.select_action(state, "C", state.legal_actions("C"))

    assert attack in (("attack", None, "5"), ("attack", None, "6"))
    result = c_agent._forced_win_result_after_attack_action(state, "C", attack)
    assert result.status == ForcedWinStatus.PROVEN
    assert result.minimum_score == 20.0
    assert c_agent.last_decision_reason == "tsume"
    assert c_agent.last_score_fallback_detail == "high_score_20"


def test_receive_routes_compare_minimum_expected_then_maximum_score() -> None:
    state = GoitaState(
        hands={
            "A": list("11111111"),
            "B": list("11111111"),
            "C": list("38111111"),
            "D": list("11111111"),
        },
        dealer="A",
    )
    state.phase = "receive"
    state.turn = "C"
    state.attacker = "B"
    state.current_attack = "3"

    agent = RuleBasedAgent()
    agent.bind_player("C")
    agent._ensure_trackers(state)
    receive_results = {
        ("receive", "3", None): ForcedWinResult(
            ForcedWinStatus.PROVEN,
            30.0,
            30.0,
            30.0,
        ),
        ("receive", "8", None): ForcedWinResult(
            ForcedWinStatus.PROVEN,
            30.0,
            40.0,
            50.0,
        ),
    }
    agent._forced_win_result_after_receive_action = (
        lambda _state, _player, action: receive_results[action]
    )
    agent._win_after_receive_bonus = lambda _state, _player, _action: 0.0

    chosen = agent._guaranteed_finish_receive_action(
        state,
        "C",
        state.legal_actions("C"),
    )

    assert chosen == ("receive", "8", None)


def test_kifu008_shi_or_kyosha_block_both_preserve_silver_thirty_points() -> None:
    case = next(case for case in load_cases() if case["id"] == "KIFU-008")
    agent = RuleBasedAgent(name="kifu-008-score-comparison")
    agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    state = reconstruct_case(case, agent)

    shi_block = ("attack_after_block", "1", "5")
    kyosha_block = ("attack_after_block", "2", "5")
    silver_block = ("attack_after_block", "4", "5")

    shi_result = agent._forced_win_result_after_attack_action(state, "A", shi_block)
    kyosha_result = agent._forced_win_result_after_attack_action(state, "A", kyosha_block)
    silver_result = agent._forced_win_result_after_attack_action(state, "A", silver_block)

    assert shi_result.status == ForcedWinStatus.PROVEN
    assert kyosha_result.status == ForcedWinStatus.PROVEN
    assert shi_result.minimum_score == 30.0
    assert kyosha_result.minimum_score == 30.0
    assert silver_result.minimum_score == 20.0

    chosen = agent.select_action(state, "A", state.legal_actions("A"))
    assert chosen in (shi_block, kyosha_block)


if __name__ == "__main__":
    test_proven_high_score_receive_outranks_first_attack_pass_policy()
    test_proven_route_continues_with_a_publicly_unreceivable_attack()
    test_receive_routes_compare_minimum_expected_then_maximum_score()
    test_kifu008_shi_or_kyosha_block_both_preserve_silver_thirty_points()
    print("GUARANTEED_HIGH_SCORE_SEARCH_TEST_OK")

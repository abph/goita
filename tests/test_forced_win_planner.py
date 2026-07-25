from __future__ import annotations

from goita_ai2.constants import PIECE_TOTALS
from goita_ai2.current_ai.forced_win_planner import (
    ForcedWinPlannerMixin,
    ForcedWinScoreMode,
    ForcedWinTiming,
)
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


def _state_with_a_hand(hand: str) -> GoitaState:
    remaining = [
        piece
        for piece, total in PIECE_TOTALS.items()
        for _ in range(total)
    ]
    for piece in hand:
        remaining.remove(piece)
    return GoitaState(
        hands={
            "A": list(hand),
            "B": remaining[:8],
            "C": remaining[8:16],
            "D": remaining[16:24],
        },
        dealer="A",
    )


def _apply_public_action(state, agents, player: str, action) -> None:
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


def test_rule_based_agent_uses_forced_win_planner_mixin() -> None:
    assert issubclass(RuleBasedAgent, ForcedWinPlannerMixin)


def test_initial_fixed_route_is_normalized_as_category_one() -> None:
    state = _state_with_a_hand("11122228")
    agent = RuleBasedAgent()
    agent.bind_player("A")

    chosen = agent.select_action(state, "A", state.legal_actions("A"))
    plan = agent._track[id(state)]["active_forced_win_plan"]

    assert chosen[0] == "attack_after_block"
    assert chosen[2] == "2"
    assert plan["category"] == "initial_fixed"
    assert plan["timing"] == ForcedWinTiming.INITIAL.value
    assert plan["score_mode"] == ForcedWinScoreMode.FIXED.value
    assert plan["minimum_score"] == 50.0
    assert plan["maximum_score"] == 50.0


def test_initial_upside_route_is_normalized_as_category_two() -> None:
    state = _state_with_a_hand("11127789")
    agent = RuleBasedAgent()
    agent.bind_player("A")

    chosen = agent.select_action(state, "A", state.legal_actions("A"))
    plan = agent._track[id(state)]["active_forced_win_plan"]

    assert chosen == ("attack_after_block", "1", "2")
    assert plan["category"] == "initial_high_score_branch"
    assert plan["timing"] == ForcedWinTiming.INITIAL.value
    assert plan["score_mode"] == ForcedWinScoreMode.HIGH_SCORE_BRANCH.value
    assert plan["minimum_score"] == 50.0
    assert plan["maximum_score"] == 100.0


def test_triggered_routes_cover_branching_and_fixed_categories() -> None:
    state = GoitaState(
        hands={
            "A": list("12244557"),
            "B": list("11133456"),
            "C": list("11123567"),
            "D": list("11123489"),
        },
        dealer="B",
    )
    agents = {player: RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)

    opening = (
        ("B", ("attack_after_block", "1", "3")),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("attack_after_block", "1", "3")),
        ("C", ("receive", "3", None)),
        ("C", ("attack", None, "1")),
        ("D", ("receive", "1", None)),
    )
    for player, action in opening:
        _apply_public_action(state, agents, player, action)

    d_agent = agents["D"]
    first = d_agent.select_action(state, "D", state.legal_actions("D"))
    branch_plan = d_agent._track[id(state)]["active_forced_win_plan"]

    assert first == ("attack", None, "1")
    assert branch_plan["category"] == "conditional_high_score_branch"
    assert branch_plan["minimum_score"] == 30.0
    assert branch_plan["expected_score"] > 30.0
    assert branch_plan["maximum_score"] == 50.0

    continuation = (
        ("D", first),
        ("A", ("receive", "1", None)),
        ("A", ("attack", None, "2")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("receive", "2", None)),
    )
    for player, action in continuation:
        _apply_public_action(state, agents, player, action)

    second = d_agent.select_action(state, "D", state.legal_actions("D"))
    fixed_plan = d_agent._track[id(state)]["active_forced_win_plan"]

    assert second == ("attack", None, "3")
    assert fixed_plan["category"] == "conditional_fixed"
    assert fixed_plan["minimum_score"] == 50.0
    assert fixed_plan["maximum_score"] == 50.0


if __name__ == "__main__":
    test_rule_based_agent_uses_forced_win_planner_mixin()
    test_initial_fixed_route_is_normalized_as_category_one()
    test_initial_upside_route_is_normalized_as_category_two()
    test_triggered_routes_cover_branching_and_fixed_categories()
    print("FORCED_WIN_PLANNER_TEST_OK")

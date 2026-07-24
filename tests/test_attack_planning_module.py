from __future__ import annotations

from goita_ai2.current_ai.attack_planning import AttackPlanningMixin
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


def _apply_public_action(agent, state, player, action) -> None:
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


def _horse_kakari_after_angle_state() -> tuple[RuleBasedAgent, GoitaState]:
    state = GoitaState(
        hands={
            "A": list("91311314"),
            "B": list("12518112"),
            "C": list("64723346"),
            "D": list("12575145"),
        },
        dealer="B",
    )
    agent = RuleBasedAgent()
    agent.bind_player("C")

    actions = (
        ("B", ("attack_after_block", "1", "2")),
        ("C", ("pass", None, None)),
        ("D", ("receive", "2", None)),
        ("D", ("attack", None, "5")),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("attack_after_block", "1", "5")),
        ("A", ("receive", "9", None)),
        ("A", ("attack", None, "3")),
        ("B", ("pass", None, None)),
        ("C", ("receive", "3", None)),
        ("C", ("attack", None, "6")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
    )
    for player, action in actions:
        _apply_public_action(agent, state, player, action)

    agent._track[id(state)]["my_attack_count"] = 1
    return agent, state


def test_rule_based_agent_uses_attack_planning_mixin() -> None:
    assert issubclass(RuleBasedAgent, AttackPlanningMixin)


def test_full_attack_plan_keeps_angle_for_third_attack() -> None:
    agent, state = _horse_kakari_after_angle_state()

    chosen = agent.select_action(state, "C", state.legal_actions("C"))
    plan = agent._future_attack_plan_for_action(state, "C", chosen)

    assert chosen[2] == "3"
    assert chosen[1] != "6"
    assert plan is not None
    assert plan["attacks"][0] == "6"


def test_eight_card_shallow_planner_records_complete_route() -> None:
    state = GoitaState(
        hands={
            "A": list("11123459"),
            "B": list("11122334"),
            "C": list("11124567"),
            "D": list("13455678"),
        },
        dealer="A",
    )
    agent = RuleBasedAgent()
    agent.bind_player("A")

    chosen = agent.select_action(state, "A", state.legal_actions("A"))
    plan = agent._track[id(state)]["shallow_eight_card_plan"]

    assert plan is not None
    assert tuple(plan["steps"][0]) == (chosen[1], chosen[2])
    assert len(plan["steps"]) == 4
    assert len(plan["attacks"]) == 4
    assert plan["finish_score"] == 50.0
    assert agent.last_score_fallback_detail == "attack_shallow_eight_card_plan_50"

    used = []
    for block, attack in plan["steps"]:
        used.extend((block, attack))
    assert sorted(used) == sorted(state.hands["A"])


def test_replanned_third_attack_uses_reserved_angle() -> None:
    agent, state = _horse_kakari_after_angle_state()
    chosen = agent.select_action(state, "C", state.legal_actions("C"))
    _apply_public_action(agent, state, "C", chosen)
    _apply_public_action(agent, state, "D", ("pass", None, None))
    _apply_public_action(agent, state, "A", ("pass", None, None))
    _apply_public_action(agent, state, "B", ("pass", None, None))

    third_attack = agent.select_action(state, "C", state.legal_actions("C"))

    assert third_attack[2] == "6"


if __name__ == "__main__":
    test_rule_based_agent_uses_attack_planning_mixin()
    test_full_attack_plan_keeps_angle_for_third_attack()
    test_eight_card_shallow_planner_records_complete_route()
    test_replanned_third_attack_uses_reserved_angle()
    print("ATTACK_PLANNING_MODULE_TEST_OK")

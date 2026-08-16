from __future__ import annotations

from goita_ai2.constants import PIECE_TOTALS
from goita_ai2.current_ai.branched_attack_lifecycle import (
    AttackPlanLifecycleStatus,
    BranchedAttackLifecycleMixin,
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


def _plan_for_action(plans, action):
    return next(plan for plan in plans if plan.node(plan.root_node_id).action == action)


def _apply(agent: RuleBasedAgent, state: GoitaState, player: str, action) -> None:
    action_type, block, attack = action
    if action_type == "pass":
        state.apply_pass(player)
    elif action_type == "receive":
        assert block is not None
        state.apply_receive(player, block)
    elif action_type == "attack":
        assert attack is not None
        state.apply_attack(player, attack)
    else:
        assert block is not None and attack is not None
        state.apply_attack_after_block(player, block, attack)
    agent.on_public_action(state, player, action)


def _installed_plan(state: GoitaState, action):
    agent = RuleBasedAgent()
    agent.bind_player("A")
    plans = agent._generate_branched_attack_plans(
        state,
        "A",
        state.legal_actions("A"),
    )
    plan = _plan_for_action(plans, action)
    active = agent._install_branched_attack_plan(state, "A", plan)
    return agent, active


def test_rule_based_agent_uses_branched_attack_lifecycle_mixin() -> None:
    assert issubclass(RuleBasedAgent, BranchedAttackLifecycleMixin)


def test_plan_is_persisted_and_advances_after_a_full_lap() -> None:
    state = _state_with_a_hand("11122357")
    action = ("attack_after_block", "1", "3")
    agent, active = _installed_plan(state, action)

    _apply(agent, state, "A", action)
    assert active.status == AttackPlanLifecycleStatus.OBSERVING
    for passer in ("B", "C", "D"):
        _apply(agent, state, passer, ("pass", None, None))

    assert active.status == AttackPlanLifecycleStatus.READY
    assert active.current_node.attack_number == 2
    assert active.current_node.action in state.legal_actions("A")
    assert agent._track[id(state)]["active_branched_attack_plan"]["status"] == "ready"


def test_plan_waits_through_unrelated_play_until_owner_receives() -> None:
    state = GoitaState(
        hands={
            "A": list("11122357"),
            "B": list("23445678"),
            "C": list("11112234"),
            "D": list("11112569"),
        },
        dealer="A",
    )
    action = ("attack_after_block", "1", "3")
    agent, active = _installed_plan(state, action)

    _apply(agent, state, "A", action)
    _apply(agent, state, "B", ("receive", "3", None))
    assert active.status == AttackPlanLifecycleStatus.WAITING

    _apply(agent, state, "B", ("attack", None, "2"))
    _apply(agent, state, "C", ("pass", None, None))
    _apply(agent, state, "D", ("pass", None, None))
    assert active.status == AttackPlanLifecycleStatus.WAITING

    _apply(agent, state, "A", ("receive", "2", None))
    assert active.status == AttackPlanLifecycleStatus.READY
    assert active.current_node.attack_number == 2
    assert active.current_node.action is not None
    assert active.current_node.action[0] == "attack"


def test_owner_action_mismatch_requests_a_replan() -> None:
    state = _state_with_a_hand("11122357")
    planned = ("attack_after_block", "1", "3")
    agent, active = _installed_plan(state, planned)
    different = next(action for action in state.legal_actions("A") if action != planned)

    _apply(agent, state, "A", different)

    assert active.status == AttackPlanLifecycleStatus.REPLAN_REQUIRED
    assert active.reason == "owner_selected_a_different_action"


def test_rebuild_replaces_an_invalidated_plan_with_a_fresh_public_revision() -> None:
    state = _state_with_a_hand("11122357")
    planned = ("attack_after_block", "1", "3")
    agent, old = _installed_plan(state, planned)
    agent._invalidate_branched_attack_plan(state, "test_public_assumption_changed")

    replacement = agent._rebuild_branched_attack_plan(
        state,
        "A",
        state.legal_actions("A"),
    )

    assert replacement is not None
    assert replacement is not old
    assert replacement.status == AttackPlanLifecycleStatus.READY
    assert agent._track[id(state)]["branched_attack_plan_history"][-1]["reason"] == "replaced_by_new_plan"


if __name__ == "__main__":
    test_rule_based_agent_uses_branched_attack_lifecycle_mixin()
    test_plan_is_persisted_and_advances_after_a_full_lap()
    test_plan_waits_through_unrelated_play_until_owner_receives()
    test_owner_action_mismatch_requests_a_replan()
    test_rebuild_replaces_an_invalidated_plan_with_a_fresh_public_revision()
    print("BRANCHED_ATTACK_LIFECYCLE_TEST_OK")

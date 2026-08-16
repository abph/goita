from __future__ import annotations

from goita_ai2.constants import PIECE_TOTALS
from goita_ai2.current_ai.branched_attack_generator import (
    BranchedAttackGeneratorMixin,
)
from goita_ai2.current_ai.branched_attack_plan import (
    PlanActorScope,
    PublicPlanEventKind,
)
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


def _state_with_a_hand(hand: str, *, reverse_opponents: bool = False) -> GoitaState:
    remaining = [
        piece
        for piece, total in PIECE_TOTALS.items()
        for _ in range(total)
    ]
    for piece in hand:
        remaining.remove(piece)
    if reverse_opponents:
        remaining.reverse()
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


def test_rule_based_agent_uses_branched_attack_generator_mixin() -> None:
    assert issubclass(RuleBasedAgent, BranchedAttackGeneratorMixin)


def test_stage_two_generates_one_plan_for_every_legal_root_attack() -> None:
    state = _state_with_a_hand("11122357")
    agent = RuleBasedAgent()
    agent.bind_player("A")
    actions = state.legal_actions("A")

    plans = agent._generate_branched_attack_plans(state, "A", actions)

    assert len(plans) == len(agent._branched_root_attack_candidates(actions))
    assert {plan.node(plan.root_node_id).action for plan in plans} == set(actions)


def test_root_branches_cover_receive_lap_and_fallback_public_events() -> None:
    state = _state_with_a_hand("11122357")
    agent = RuleBasedAgent()
    agent.bind_player("A")
    plans = agent._generate_branched_attack_plans(
        state,
        "A",
        state.legal_actions("A"),
    )
    plan = _plan_for_action(plans, ("attack_after_block", "1", "3"))
    root = plan.node(plan.root_node_id)
    conditions = {
        (branch.condition.kind, branch.condition.actor_scope)
        for branch in root.branches
    }

    assert (PublicPlanEventKind.SAME_PIECE_RECEIVE, PlanActorScope.ENEMY) in conditions
    assert (PublicPlanEventKind.PASS, PlanActorScope.ENEMY) in conditions
    assert (PublicPlanEventKind.PASS, PlanActorScope.ALLY) in conditions
    assert (PublicPlanEventKind.LAP_COMPLETED, PlanActorScope.ANY) in conditions
    assert root.branches[-1].condition.kind == PublicPlanEventKind.ALWAYS


def test_pass_branch_keeps_waiting_for_the_remaining_public_responses() -> None:
    state = _state_with_a_hand("11122357")
    agent = RuleBasedAgent()
    agent.bind_player("A")
    plans = agent._generate_branched_attack_plans(
        state,
        "A",
        state.legal_actions("A"),
    )
    plan = _plan_for_action(plans, ("attack_after_block", "1", "3"))
    root = plan.node(plan.root_node_id)
    enemy_pass = next(
        branch
        for branch in root.branches
        if branch.condition.kind == PublicPlanEventKind.PASS
        and branch.condition.actor_scope == PlanActorScope.ENEMY
    )
    checkpoint = plan.node(enemy_pass.target_node_id)
    repeated_enemy_pass = next(
        branch
        for branch in checkpoint.branches
        if branch.condition.kind == PublicPlanEventKind.PASS
        and branch.condition.actor_scope == PlanActorScope.ENEMY
    )

    assert checkpoint.checkpoint
    assert repeated_enemy_pass.target_node_id == checkpoint.node_id
    assert any(
        branch.condition.kind == PublicPlanEventKind.LAP_COMPLETED
        for branch in checkpoint.branches
    )


def test_public_count_bounds_remove_an_impossible_same_piece_branch() -> None:
    state = _state_with_a_hand("11122357")
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent._ensure_trackers(state)
    tracker = agent._track[id(state)]
    for seat in ("B", "C", "D"):
        tracker["estimated_current_hands"][seat]["3"]["max"] = 0

    responses = agent._branched_public_response_branches(state, "A", "3", 1)

    assert not any(
        response.condition.kind == PublicPlanEventKind.SAME_PIECE_RECEIVE
        for response in responses
    )
    assert any(
        response.condition.kind == PublicPlanEventKind.LAP_COMPLETED
        for response in responses
    )


def test_lap_route_contains_attacks_two_and_three() -> None:
    state = _state_with_a_hand("11122357")
    agent = RuleBasedAgent()
    agent.bind_player("A")
    plans = agent._generate_branched_attack_plans(
        state,
        "A",
        state.legal_actions("A"),
    )
    plan = _plan_for_action(plans, ("attack_after_block", "1", "3"))
    root = plan.node(plan.root_node_id)
    lap_branch = next(
        branch
        for branch in root.branches
        if branch.condition.kind == PublicPlanEventKind.LAP_COMPLETED
    )
    second = plan.node(lap_branch.target_node_id)
    second_lap = next(
        branch
        for branch in second.branches
        if branch.condition.kind == PublicPlanEventKind.LAP_COMPLETED
    )
    third = plan.node(second_lap.target_node_id)

    assert second.attack_number == 2
    assert second.action is not None and second.action[0] == "attack_after_block"
    assert third.attack_number == 3
    assert third.action is not None and third.action[0] == "attack_after_block"


def test_received_route_waits_for_public_self_receive_before_attack_two() -> None:
    state = _state_with_a_hand("11122357")
    agent = RuleBasedAgent()
    agent.bind_player("A")
    plans = agent._generate_branched_attack_plans(
        state,
        "A",
        state.legal_actions("A"),
    )
    plan = _plan_for_action(plans, ("attack_after_block", "1", "3"))
    root = plan.node(plan.root_node_id)
    receive_branch = next(
        branch
        for branch in root.branches
        if branch.condition.kind == PublicPlanEventKind.SAME_PIECE_RECEIVE
        and branch.condition.actor_scope == PlanActorScope.ENEMY
    )
    checkpoint = plan.node(receive_branch.target_node_id)

    assert checkpoint.checkpoint
    assert checkpoint.action is None
    assert any(
        branch.condition.actor_scope == PlanActorScope.SELF
        for branch in checkpoint.branches[:-1]
    )
    assert checkpoint.branches[-1].condition.kind == PublicPlanEventKind.ALWAYS


def test_generated_plans_do_not_change_when_hidden_opponent_deal_changes() -> None:
    first_state = _state_with_a_hand("11122357")
    second_state = _state_with_a_hand("11122357", reverse_opponents=True)
    first_agent = RuleBasedAgent()
    second_agent = RuleBasedAgent()
    first_agent.bind_player("A")
    second_agent.bind_player("A")

    first = first_agent._generate_branched_attack_plans(
        first_state,
        "A",
        first_state.legal_actions("A"),
    )
    second = second_agent._generate_branched_attack_plans(
        second_state,
        "A",
        second_state.legal_actions("A"),
    )

    assert [plan.as_dict() for plan in first] == [plan.as_dict() for plan in second]


if __name__ == "__main__":
    test_rule_based_agent_uses_branched_attack_generator_mixin()
    test_stage_two_generates_one_plan_for_every_legal_root_attack()
    test_root_branches_cover_receive_lap_and_fallback_public_events()
    test_pass_branch_keeps_waiting_for_the_remaining_public_responses()
    test_public_count_bounds_remove_an_impossible_same_piece_branch()
    test_lap_route_contains_attacks_two_and_three()
    test_received_route_waits_for_public_self_receive_before_attack_two()
    test_generated_plans_do_not_change_when_hidden_opponent_deal_changes()
    print("BRANCHED_ATTACK_GENERATOR_TEST_OK")

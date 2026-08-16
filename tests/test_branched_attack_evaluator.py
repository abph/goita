from __future__ import annotations

from goita_ai2.constants import PIECE_TOTALS
from goita_ai2.current_ai.branched_attack_evaluator import (
    AttackRouteDangerSeverity,
    BranchedAttackEvaluatorMixin,
)
from goita_ai2.current_ai.branched_attack_plan import (
    AttackPlanEvaluation,
    PublicPlanEventKind,
)
from goita_ai2.current_ai.endgame import ForcedWinStatus
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


def _generated_plan(agent, state, action):
    plans = agent._generate_branched_attack_plans(
        state,
        "A",
        state.legal_actions("A"),
    )
    return next(plan for plan in plans if plan.node(plan.root_node_id).action == action)


def test_rule_based_agent_uses_branched_attack_evaluator_mixin() -> None:
    assert issubclass(RuleBasedAgent, BranchedAttackEvaluatorMixin)


def test_initial_fixed_route_receives_guaranteed_minimum_score() -> None:
    state = _state_with_a_hand("11122228")
    agent = RuleBasedAgent()
    agent.bind_player("A")
    plan = _generated_plan(agent, state, ("attack_after_block", "1", "2"))

    evaluated = agent._evaluate_branched_attack_plan(state, "A", plan)

    assert evaluated.report.root_proof_status == ForcedWinStatus.PROVEN
    assert evaluated.plan.evaluation.guaranteed_win
    assert evaluated.plan.evaluation.minimum_score == 50.0
    assert evaluated.plan.evaluation.maximum_score >= 50.0
    assert evaluated.plan.evaluation.public_certainty == 1.0
    assert evaluated.report.inference_summary.uncertainty_penalty >= 0.0


def test_initial_upside_route_keeps_fifty_minimum_and_hundred_maximum() -> None:
    state = _state_with_a_hand("11127789")
    agent = RuleBasedAgent()
    agent.bind_player("A")
    plan = _generated_plan(agent, state, ("attack_after_block", "1", "2"))

    evaluated = agent._evaluate_branched_attack_plan(state, "A", plan)

    assert evaluated.plan.evaluation.guaranteed_win
    assert evaluated.plan.evaluation.minimum_score == 50.0
    assert evaluated.plan.evaluation.maximum_score == 100.0
    assert evaluated.report.proof_scope == "root action with public-event replanning"


def test_receive_width_counts_every_piece_type_a_royal_can_cover() -> None:
    agent = RuleBasedAgent()

    assert agent._branched_receive_width(("1", "2", "8")) == 9.0
    assert agent._branched_receive_width(("1", "2", "3")) == 3.0
    assert agent._branched_receive_width(("8",)) == 7.0


def test_hiding_a_royal_before_the_finish_is_a_critical_danger() -> None:
    state = _state_with_a_hand("11122379")
    agent = RuleBasedAgent()
    agent.bind_player("A")
    plan = _generated_plan(agent, state, ("attack_after_block", "9", "3"))

    evaluated = agent._evaluate_branched_attack_plan(state, "A", plan)
    root = next(
        item
        for item in evaluated.report.node_evaluations
        if item.node_id == plan.root_node_id
    )
    royal_danger = next(
        danger for danger in root.dangers if danger.code == "royal_hidden_before_finish"
    )

    assert royal_danger.severity == AttackRouteDangerSeverity.CRITICAL
    assert evaluated.plan.evaluation.failure_risk > 0.0


def test_unexpected_public_response_is_reported_as_replan_danger() -> None:
    state = _state_with_a_hand("11122357")
    agent = RuleBasedAgent()
    agent.bind_player("A")
    plan = _generated_plan(agent, state, ("attack_after_block", "1", "3"))

    evaluated = agent._evaluate_branched_attack_plan(state, "A", plan)

    assert any(
        danger.code == "unexpected_response_replan"
        for danger in evaluated.report.dangerous_branches
    )
    assert all(
        branch.condition.kind != PublicPlanEventKind.ALWAYS
        or plan.node(branch.target_node_id).terminal
        for node in plan.nodes
        for branch in node.branches
    )


def test_batch_evaluation_returns_updated_plans_without_selecting_an_action() -> None:
    state = _state_with_a_hand("11122357")
    agent = RuleBasedAgent()
    agent.bind_player("A")
    plans = agent._generate_branched_attack_plans(
        state,
        "A",
        state.legal_actions("A"),
    )[:3]

    evaluated = agent._evaluate_branched_attack_plans(state, "A", plans)

    assert len(evaluated) == 3
    assert [item.plan.plan_id for item in evaluated] == [plan.plan_id for plan in plans]
    assert all(item.report.node_evaluations for item in evaluated)
    assert all(item.report.inference_summary.supported_branches >= 0 for item in evaluated)
    assert all("inference_summary" in item.report.as_dict() for item in evaluated)


def test_probability_summary_exposes_expected_score_and_failure_risk() -> None:
    state = _state_with_a_hand("11122357")
    agent = RuleBasedAgent()
    agent.bind_player("A")
    plan = _generated_plan(agent, state, ("attack_after_block", "1", "3"))

    evaluated = agent._evaluate_branched_attack_plan(state, "A", plan)
    summary = evaluated.report.inference_summary

    assert 0.0 <= summary.probability_coverage <= 1.0
    assert summary.expected_minimum_score <= summary.expected_maximum_score
    assert 0.0 <= summary.probability_failure_risk <= 1.0
    assert evaluated.plan.evaluation.expected_score >= 0.0
    assert (
        evaluated.plan.evaluation.probability_failure_risk
        == summary.probability_failure_risk
    )


def test_guaranteed_route_always_outranks_a_higher_expected_unproven_route() -> None:
    guaranteed = AttackPlanEvaluation(
        guaranteed_win=True,
        minimum_score=20.0,
        maximum_score=20.0,
        expected_score=20.0,
        failure_risk=0.0,
    )
    speculative = AttackPlanEvaluation(
        guaranteed_win=False,
        minimum_score=0.0,
        maximum_score=100.0,
        expected_score=90.0,
        failure_risk=0.01,
    )

    assert guaranteed.preference_key() > speculative.preference_key()


def test_minimum_guaranteed_score_precedes_expected_score() -> None:
    higher_minimum = AttackPlanEvaluation(
        guaranteed_win=True,
        minimum_score=40.0,
        maximum_score=40.0,
        expected_score=40.0,
        failure_risk=0.0,
    )
    higher_expectation = AttackPlanEvaluation(
        guaranteed_win=True,
        minimum_score=30.0,
        maximum_score=100.0,
        expected_score=80.0,
        failure_risk=0.0,
    )

    assert higher_minimum.preference_key() > higher_expectation.preference_key()


if __name__ == "__main__":
    test_rule_based_agent_uses_branched_attack_evaluator_mixin()
    test_initial_fixed_route_receives_guaranteed_minimum_score()
    test_initial_upside_route_keeps_fifty_minimum_and_hundred_maximum()
    test_receive_width_counts_every_piece_type_a_royal_can_cover()
    test_hiding_a_royal_before_the_finish_is_a_critical_danger()
    test_unexpected_public_response_is_reported_as_replan_danger()
    test_batch_evaluation_returns_updated_plans_without_selecting_an_action()
    test_probability_summary_exposes_expected_score_and_failure_risk()
    test_guaranteed_route_always_outranks_a_higher_expected_unproven_route()
    test_minimum_guaranteed_score_precedes_expected_score()
    print("BRANCHED_ATTACK_EVALUATOR_TEST_OK")

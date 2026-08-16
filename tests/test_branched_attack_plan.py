from __future__ import annotations

from dataclasses import fields

from goita_ai2.current_ai.branched_attack_plan import (
    ATTACK_DECISION_AUDIT,
    AttackDecisionPriority,
    AttackPlanBranch,
    AttackPlanEvaluation,
    AttackPlanNode,
    BranchedAttackPlan,
    PlanActorScope,
    PublicBranchCondition,
    PublicPlanEvent,
    PublicPlanEventKind,
    choose_preferred_attack_plan,
)


def _sample_plan(
    plan_id: str = "kyosha_branch",
    evaluation: AttackPlanEvaluation | None = None,
) -> BranchedAttackPlan:
    root = AttackPlanNode(
        node_id="attack_1",
        action=("attack", None, "2"),
        attack_number=1,
        purpose="probe with kyosha",
        reserved_pieces=("7", "7", "9"),
        branches=(
            AttackPlanBranch(
                PublicBranchCondition(
                    PublicPlanEventKind.ROYAL_RECEIVE,
                    actor_scope=PlanActorScope.ENEMY,
                    attack_number=1,
                ),
                "high_score_route",
                "enemy spent a royal",
            ),
            AttackPlanBranch(
                PublicBranchCondition(PublicPlanEventKind.LAP_COMPLETED),
                "safe_route",
                "the kyosha returned",
            ),
            AttackPlanBranch(
                PublicBranchCondition(PublicPlanEventKind.ALWAYS),
                "replan",
                "unexpected public response",
            ),
        ),
    )
    return BranchedAttackPlan(
        plan_id=plan_id,
        root_node_id="attack_1",
        nodes=(
            root,
            AttackPlanNode(
                node_id="high_score_route",
                action=("attack", None, "7"),
                attack_number=2,
                purpose="start the big pair after a royal was spent",
            ),
            AttackPlanNode(
                node_id="safe_route",
                action=("attack_after_block", "1", "7"),
                attack_number=2,
                purpose="continue after one full lap",
            ),
            AttackPlanNode(
                node_id="replan",
                action=None,
                attack_number=1,
                terminal=True,
                purpose="discard the plan and rebuild it",
            ),
        ),
        evaluation=evaluation or AttackPlanEvaluation(
            guaranteed_win=True,
            minimum_score=50.0,
            maximum_score=100.0,
            covered_public_responses=3,
            public_certainty=1.0,
            receive_width=4.0,
            preserved_royals=1,
            failure_risk=0.0,
            route_length=4,
        ),
        source="initial_hand_pattern",
        public_revision=2,
        assumptions=("only public royal usage may select the high-score branch",),
    )


def test_priority_audit_places_branched_plan_between_tactics_and_sequences() -> None:
    priorities = {layer.name: layer.priority for layer in ATTACK_DECISION_AUDIT}

    assert priorities["public_tactic"] > priorities["branched_attack_plan"]
    assert priorities["branched_attack_plan"] > priorities["fixed_attack_sequence"]
    assert priorities["immediate_win"] == AttackDecisionPriority.IMMEDIATE_WIN


def test_branch_conditions_use_only_public_event_fields() -> None:
    event_fields = {field.name for field in fields(PublicPlanEvent)}

    assert "hand" not in event_fields
    assert "hidden_hand" not in event_fields
    assert event_fields == {
        "kind",
        "actor_scope",
        "piece",
        "current_attack",
        "attack_number",
        "reached",
        "round_finished",
    }


def test_public_conditions_match_same_piece_and_royal_receives() -> None:
    same_piece = PublicBranchCondition(
        PublicPlanEventKind.SAME_PIECE_RECEIVE,
        actor_scope=PlanActorScope.ENEMY,
        piece="4",
    )
    royal = PublicBranchCondition(
        PublicPlanEventKind.ROYAL_RECEIVE,
        actor_scope=PlanActorScope.ENEMY,
    )

    assert same_piece.matches(PublicPlanEvent(
        PublicPlanEventKind.RECEIVE,
        actor_scope=PlanActorScope.ENEMY,
        piece="4",
        current_attack="4",
    ))
    assert not same_piece.matches(PublicPlanEvent(
        PublicPlanEventKind.RECEIVE,
        actor_scope=PlanActorScope.ENEMY,
        piece="4",
        current_attack="3",
    ))
    assert royal.matches(PublicPlanEvent(
        PublicPlanEventKind.RECEIVE,
        actor_scope=PlanActorScope.ENEMY,
        piece="9",
        current_attack="6",
    ))


def test_plan_advances_to_the_first_matching_public_branch() -> None:
    plan = _sample_plan()
    event = PublicPlanEvent(
        PublicPlanEventKind.RECEIVE,
        actor_scope=PlanActorScope.ENEMY,
        piece="8",
        current_attack="2",
        attack_number=1,
    )

    next_node = plan.advance("attack_1", event)

    assert next_node is not None
    assert next_node.node_id == "high_score_route"
    assert next_node.action == ("attack", None, "7")
    assert plan.as_dict()["priority"] == int(AttackDecisionPriority.BRANCHED_ATTACK_PLAN)


def test_evaluation_prefers_guaranteed_minimum_score_before_upside() -> None:
    guaranteed_30 = _sample_plan(
        "guaranteed_30",
        AttackPlanEvaluation(
            guaranteed_win=True,
            minimum_score=30.0,
            maximum_score=30.0,
            covered_public_responses=2,
            public_certainty=1.0,
            receive_width=2.0,
            failure_risk=0.0,
            route_length=3,
        ),
    )
    risky_100 = _sample_plan(
        "risky_100",
        AttackPlanEvaluation(
            guaranteed_win=False,
            minimum_score=0.0,
            maximum_score=100.0,
            covered_public_responses=8,
            public_certainty=0.8,
            receive_width=7.0,
            failure_risk=0.2,
            route_length=3,
        ),
    )
    guaranteed_50 = _sample_plan(
        "guaranteed_50",
        AttackPlanEvaluation(
            guaranteed_win=True,
            minimum_score=50.0,
            maximum_score=50.0,
            covered_public_responses=1,
            public_certainty=1.0,
            receive_width=1.0,
            failure_risk=0.0,
            route_length=4,
        ),
    )

    assert choose_preferred_attack_plan((risky_100, guaranteed_30)) is guaranteed_30
    assert choose_preferred_attack_plan((guaranteed_30, guaranteed_50)) is guaranteed_50


def test_evaluation_uses_upside_then_coverage_for_equal_minimum_score() -> None:
    fixed = _sample_plan(
        "fixed",
        AttackPlanEvaluation(
            guaranteed_win=True,
            minimum_score=30.0,
            maximum_score=30.0,
            covered_public_responses=4,
            public_certainty=1.0,
            receive_width=5.0,
            failure_risk=0.0,
        ),
    )
    upside = _sample_plan(
        "upside",
        AttackPlanEvaluation(
            guaranteed_win=True,
            minimum_score=30.0,
            maximum_score=50.0,
            covered_public_responses=2,
            public_certainty=1.0,
            receive_width=2.0,
            failure_risk=0.0,
        ),
    )

    assert choose_preferred_attack_plan((fixed, upside)) is upside


if __name__ == "__main__":
    test_priority_audit_places_branched_plan_between_tactics_and_sequences()
    test_branch_conditions_use_only_public_event_fields()
    test_public_conditions_match_same_piece_and_royal_receives()
    test_plan_advances_to_the_first_matching_public_branch()
    test_evaluation_prefers_guaranteed_minimum_score_before_upside()
    test_evaluation_uses_upside_then_coverage_for_equal_minimum_score()
    print("BRANCHED_ATTACK_PLAN_TEST_OK")

from __future__ import annotations

from goita_ai2.constants import PIECE_TOTALS
from goita_ai2.current_ai.branched_attack_inference import (
    BranchSupportLevel,
    BranchedAttackInferenceCache,
    BranchedAttackInferenceMixin,
)
from goita_ai2.current_ai.branched_attack_plan import (
    AttackPlanBranch,
    AttackPlanNode,
    PlanActorScope,
    PublicBranchCondition,
    PublicPlanEventKind,
)
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


def _state_with_a_hand(hand: str, *, reverse_hidden: bool = False) -> GoitaState:
    remaining = [
        piece
        for piece, total in PIECE_TOTALS.items()
        for _ in range(total)
    ]
    for piece in hand:
        remaining.remove(piece)
    if reverse_hidden:
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


def _agent_and_state(hand: str = "11122357"):
    state = _state_with_a_hand(hand)
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent.TIME_SEARCH_ENABLED = False
    agent._ensure_trackers(state)
    return agent, state


def _root_node() -> AttackPlanNode:
    return AttackPlanNode(
        node_id="root",
        action=("attack_after_block", "1", "3"),
        attack_number=1,
        reserved_pieces=tuple("112257"),
    )


def _enemy_receive_branch() -> AttackPlanBranch:
    return AttackPlanBranch(
        condition=PublicBranchCondition(
            PublicPlanEventKind.SAME_PIECE_RECEIVE,
            actor_scope=PlanActorScope.ENEMY,
            piece="3",
            attack_number=1,
        ),
        target_node_id="target",
        label="enemy_same_piece_receive_3",
    )


def _set_estimate(
    tracker: dict,
    seat: str,
    piece: str,
    *,
    minimum: int,
    maximum: int,
    expected: float,
    map_count: int,
    confidence: float,
) -> None:
    tracker["estimated_current_hands"][seat][piece].update({
        "min": minimum,
        "max": maximum,
        "expected": expected,
        "map_count": map_count,
        "confidence": confidence,
        "source": "test_public_inference",
    })


def _enemy_support(agent, state):
    return agent._branched_branch_support(
        state,
        "A",
        _root_node(),
        _enemy_receive_branch(),
    )


def test_rule_based_agent_uses_branched_attack_inference_mixin() -> None:
    assert issubclass(RuleBasedAgent, BranchedAttackInferenceMixin)


def test_min_max_and_current_map_assign_all_five_support_levels() -> None:
    agent, state = _agent_and_state()
    tracker = agent._track[id(state)]
    _set_estimate(
        tracker, "D", "3",
        minimum=0, maximum=0, expected=0.0, map_count=0, confidence=1.0,
    )

    _set_estimate(
        tracker, "B", "3",
        minimum=1, maximum=1, expected=1.0, map_count=1, confidence=1.0,
    )
    assert _enemy_support(agent, state).level == BranchSupportLevel.CERTAIN

    _set_estimate(
        tracker, "B", "3",
        minimum=0, maximum=1, expected=0.9, map_count=1, confidence=0.8,
    )
    assert _enemy_support(agent, state).level == BranchSupportLevel.LIKELY

    _set_estimate(
        tracker, "B", "3",
        minimum=0, maximum=1, expected=0.4, map_count=0, confidence=0.5,
    )
    assert _enemy_support(agent, state).level == BranchSupportLevel.POSSIBLE

    _set_estimate(
        tracker, "B", "3",
        minimum=0, maximum=1, expected=0.1, map_count=0, confidence=0.8,
    )
    assert _enemy_support(agent, state).level == BranchSupportLevel.LOW

    _set_estimate(
        tracker, "B", "3",
        minimum=0, maximum=0, expected=0.0, map_count=0, confidence=1.0,
    )
    assert _enemy_support(agent, state).level == BranchSupportLevel.IMPOSSIBLE


def test_active_first_attack_strategy_raises_current_piece_support() -> None:
    agent, state = _agent_and_state()
    tracker = agent._track[id(state)]
    _set_estimate(
        tracker, "B", "3",
        minimum=0, maximum=1, expected=0.4, map_count=1, confidence=0.5,
    )
    _set_estimate(
        tracker, "D", "3",
        minimum=0, maximum=0, expected=0.0, map_count=0, confidence=1.0,
    )
    tracker["public_hand_models"]["B"].update({
        "first_attack": "3",
        "inferred_attack_strategy_active": True,
        "strategy_broken": False,
    })

    support = _enemy_support(agent, state)

    assert support.level == BranchSupportLevel.LIKELY
    assert "B:active_first_attack_strategy" in support.evidence


def test_repeated_late_enemy_passes_lower_current_piece_support() -> None:
    agent, state = _agent_and_state()
    tracker = agent._track[id(state)]
    _set_estimate(
        tracker, "B", "3",
        minimum=0, maximum=1, expected=0.9, map_count=1, confidence=0.8,
    )
    _set_estimate(
        tracker, "D", "3",
        minimum=0, maximum=0, expected=0.0, map_count=0, confidence=1.0,
    )
    tracker["piece_pass_evidence"]["B"]["3"] = [
        {"relation": "enemy", "attack_no": 2},
        {"relation": "enemy", "attack_no": 3},
    ]

    support = _enemy_support(agent, state)

    assert support.level == BranchSupportLevel.POSSIBLE
    assert "strong_enemy_passes:2" in support.evidence


def test_no_direct_or_royal_receiver_makes_enemy_pass_certain() -> None:
    agent, state = _agent_and_state()
    tracker = agent._track[id(state)]
    for seat in ("B", "D"):
        for piece in ("3", "8", "9"):
            _set_estimate(
                tracker, seat, piece,
                minimum=0, maximum=0, expected=0.0,
                map_count=0, confidence=1.0,
            )
    pass_branch = AttackPlanBranch(
        condition=PublicBranchCondition(
            PublicPlanEventKind.PASS,
            actor_scope=PlanActorScope.ENEMY,
            attack_number=1,
        ),
        target_node_id="target",
        label="enemy_pass",
    )

    support = agent._branched_branch_support(
        state,
        "A",
        _root_node(),
        pass_branch,
    )

    assert support.level == BranchSupportLevel.CERTAIN
    assert "receive_support:impossible" in support.evidence


def test_evaluation_report_excludes_an_impossible_receive_branch() -> None:
    agent, state = _agent_and_state()
    plans = agent._generate_branched_attack_plans(
        state,
        "A",
        state.legal_actions("A"),
    )
    plan = next(
        item
        for item in plans
        if item.node(item.root_node_id).action == ("attack_after_block", "1", "3")
    )
    tracker = agent._track[id(state)]
    for seat in ("B", "D"):
        _set_estimate(
            tracker, seat, "3",
            minimum=0, maximum=0, expected=0.0,
            map_count=0, confidence=1.0,
        )

    evaluated = agent._evaluate_branched_attack_plan(state, "A", plan)
    branch = next(
        item
        for item in evaluated.report.branch_inference
        if item.support.node_id == plan.root_node_id
        and item.support.branch_label == "enemy_same_piece_receive_3"
    )

    assert branch.support.level == BranchSupportLevel.IMPOSSIBLE
    assert branch.outcome_kind == "wait_or_replan"
    assert branch.route_continues
    assert branch.maximum_score >= branch.minimum_score
    assert branch.as_dict()["support"]["level"] == "impossible"


def test_branch_evaluation_is_independent_of_hidden_opponent_deal() -> None:
    first_state = _state_with_a_hand("11122357")
    second_state = _state_with_a_hand("11122357", reverse_hidden=True)
    first_agent = RuleBasedAgent()
    second_agent = RuleBasedAgent()
    first_agent.bind_player("A")
    second_agent.bind_player("A")
    first_plans = first_agent._generate_branched_attack_plans(
        first_state,
        "A",
        first_state.legal_actions("A"),
    )
    second_plans = second_agent._generate_branched_attack_plans(
        second_state,
        "A",
        second_state.legal_actions("A"),
    )
    first_plan = next(
        item for item in first_plans
        if item.node(item.root_node_id).action == ("attack_after_block", "1", "3")
    )
    second_plan = next(
        item for item in second_plans
        if item.node(item.root_node_id).action == ("attack_after_block", "1", "3")
    )

    first = first_agent._evaluate_branched_attack_plan(
        first_state,
        "A",
        first_plan,
    )
    second = second_agent._evaluate_branched_attack_plan(
        second_state,
        "A",
        second_plan,
    )

    assert first.as_dict() == second.as_dict()


def test_cache_reuses_support_when_only_an_unrelated_ally_changes() -> None:
    agent, state = _agent_and_state()
    tracker = agent._track[id(state)]
    agent.clear_branched_attack_inference_cache()

    first = _enemy_support(agent, state)
    tracker["estimated_current_hands"]["C"]["3"]["expected"] = 0.125
    revisions = tracker.setdefault("piece_inference_player_revisions", {})
    revisions["C"] = int(revisions.get("C", 0)) + 1
    second = _enemy_support(agent, state)

    snapshot = agent.branched_attack_inference_cache_snapshot()
    assert first == second
    assert snapshot["hits"] == 1
    assert snapshot["misses"] == 1


def test_cache_misses_when_a_relevant_enemy_estimate_changes() -> None:
    agent, state = _agent_and_state()
    tracker = agent._track[id(state)]
    agent.clear_branched_attack_inference_cache()

    _enemy_support(agent, state)
    tracker["estimated_current_hands"]["B"]["3"]["expected"] = 0.125
    revisions = tracker.setdefault("piece_inference_player_revisions", {})
    revisions["B"] = int(revisions.get("B", 0)) + 1
    _enemy_support(agent, state)

    snapshot = agent.branched_attack_inference_cache_snapshot()
    assert snapshot["hits"] == 0
    assert snapshot["misses"] == 2


def test_inference_cache_is_bounded() -> None:
    agent, state = _agent_and_state()
    agent._branched_attack_inference_cache = BranchedAttackInferenceCache(2)
    branch = _enemy_receive_branch()
    for index in range(3):
        node = AttackPlanNode(
            node_id=f"root_{index}",
            action=("attack_after_block", "1", "3"),
            attack_number=1,
            reserved_pieces=tuple("112257"),
        )
        agent._branched_branch_support(state, "A", node, branch)

    snapshot = agent.branched_attack_inference_cache_snapshot()
    assert snapshot["size"] == 2
    assert snapshot["evictions"] == 1


def test_public_response_branches_receive_probabilities_from_joint_deals() -> None:
    agent, state = _agent_and_state()
    agent.BRANCHED_ATTACK_PROBABILISTIC_SAMPLE_COUNT = 96
    node = _root_node()
    receive = agent._branched_branch_support(
        state,
        "A",
        node,
        _enemy_receive_branch(),
    )
    pass_branch = AttackPlanBranch(
        condition=PublicBranchCondition(
            PublicPlanEventKind.PASS,
            actor_scope=PlanActorScope.ENEMY,
            attack_number=1,
        ),
        target_node_id="target",
        label="enemy_pass",
    )
    passed = agent._branched_branch_support(state, "A", node, pass_branch)
    lap_branch = AttackPlanBranch(
        condition=PublicBranchCondition(
            PublicPlanEventKind.LAP_COMPLETED,
            attack_number=1,
        ),
        target_node_id="target",
        label="full_lap",
    )
    lap = agent._branched_branch_support(state, "A", node, lap_branch)

    for support in (receive, passed, lap):
        assert support.event_probability is not None
        assert 0.0 <= support.event_probability <= 1.0
        assert 0.0 <= support.probability_confidence <= 1.0
        assert support.probability_source in (
            "joint_action_weighted_posterior",
            "public_hard_bound",
        )
    assert lap.event_probability <= passed.event_probability


def test_hard_piece_bounds_remain_zero_or_one_above_the_posterior() -> None:
    agent, state = _agent_and_state()
    tracker = agent._track[id(state)]
    for seat in ("B", "D"):
        _set_estimate(
            tracker,
            seat,
            "3",
            minimum=0,
            maximum=0,
            expected=0.0,
            map_count=0,
            confidence=1.0,
        )
    impossible = _enemy_support(agent, state)
    assert impossible.holding_probability == 0.0
    assert impossible.event_probability == 0.0
    assert impossible.probability_source == "public_hard_bound"

    _set_estimate(
        tracker,
        "B",
        "3",
        minimum=1,
        maximum=1,
        expected=1.0,
        map_count=1,
        confidence=1.0,
    )
    certain = _enemy_support(agent, state)
    assert certain.holding_probability == 1.0
    assert certain.event_probability is not None
    assert 0.0 < certain.event_probability <= 1.0


def test_probability_branch_evaluation_does_not_read_the_live_hidden_deal() -> None:
    first_state = _state_with_a_hand("11122357")
    second_state = _state_with_a_hand("11122357", reverse_hidden=True)
    first_agent = RuleBasedAgent()
    second_agent = RuleBasedAgent()
    first_agent.bind_player("A")
    second_agent.bind_player("A")
    first_agent.BRANCHED_ATTACK_PROBABILISTIC_SAMPLE_COUNT = 96
    second_agent.BRANCHED_ATTACK_PROBABILISTIC_SAMPLE_COUNT = 96

    first = first_agent._branched_branch_support(
        first_state,
        "A",
        _root_node(),
        _enemy_receive_branch(),
    )
    second = second_agent._branched_branch_support(
        second_state,
        "A",
        _root_node(),
        _enemy_receive_branch(),
    )

    assert first.as_dict() == second.as_dict()


if __name__ == "__main__":
    test_rule_based_agent_uses_branched_attack_inference_mixin()
    test_min_max_and_current_map_assign_all_five_support_levels()
    test_active_first_attack_strategy_raises_current_piece_support()
    test_repeated_late_enemy_passes_lower_current_piece_support()
    test_no_direct_or_royal_receiver_makes_enemy_pass_certain()
    test_evaluation_report_excludes_an_impossible_receive_branch()
    test_branch_evaluation_is_independent_of_hidden_opponent_deal()
    test_cache_reuses_support_when_only_an_unrelated_ally_changes()
    test_cache_misses_when_a_relevant_enemy_estimate_changes()
    test_inference_cache_is_bounded()
    test_public_response_branches_receive_probabilities_from_joint_deals()
    test_hard_piece_bounds_remain_zero_or_one_above_the_posterior()
    test_probability_branch_evaluation_does_not_read_the_live_hidden_deal()
    print("BRANCHED_ATTACK_INFERENCE_TEST_OK")

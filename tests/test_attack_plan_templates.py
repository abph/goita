from __future__ import annotations

from goita_ai2.current_ai.attack_plan_templates import (
    AttackPlanTemplateFamily,
    AttackPlanTemplateMixin,
)
from goita_ai2.current_ai.branched_attack_plan import PublicPlanEventKind
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


def _state(hand: str) -> GoitaState:
    filler = list("11112234")
    return GoitaState(
        hands={"A": list(hand), "B": filler, "C": filler, "D": filler},
        dealer="A",
    )


def _lap_target(plan, node):
    branch = next(
        item
        for item in node.branches
        if item.condition.kind == PublicPlanEventKind.LAP_COMPLETED
    )
    return plan.node(branch.target_node_id)


def _plan_from_source(plans, source: str):
    return next(plan for plan in plans if plan.source == source)


def test_rule_based_agent_uses_representative_attack_template_mixin() -> None:
    assert issubclass(RuleBasedAgent, AttackPlanTemplateMixin)


def test_two_kyosha_single_big_becomes_big_kyosha_kyosha_template() -> None:
    state = _state("11122347")
    agent = RuleBasedAgent()
    agent.bind_player("A")

    templates = agent._representative_attack_templates(state, "A")
    template = next(item for item in templates if item.template_id == "two_kyosha_single_big")

    assert template.family == AttackPlanTemplateFamily.KYOSHA_BIG
    assert template.attack_sequence == ("7", "2", "2")


def test_two_kyosha_middle_pair_with_royal_keeps_kyosha_then_pair() -> None:
    state = _state("11224459")
    agent = RuleBasedAgent()
    agent.bind_player("A")

    templates = agent._representative_attack_templates(state, "A")
    template = next(
        item
        for item in templates
        if item.template_id == "two_kyosha_middle_pair_royal"
    )

    assert template.family == AttackPlanTemplateFamily.MIDDLE_PAIR
    assert template.attack_sequence == ("2", "4")


def test_four_shi_template_generates_three_consecutive_shi_attacks() -> None:
    state = _state("11112357")
    agent = RuleBasedAgent()
    agent.bind_player("A")
    plans = agent._generate_representative_attack_plans(
        state,
        "A",
        state.legal_actions("A"),
    )
    plan = _plan_from_source(plans, "representative:four_shi_pressure")
    root = plan.node(plan.root_node_id)
    second = _lap_target(plan, root)
    third = _lap_target(plan, second)

    assert root.action == ("attack_after_block", "1", "1")
    assert second.action is not None and second.action[2] == "1"
    assert third.action is not None and third.action[2] == "1"


def test_fourth_middle_template_reserves_the_piece_for_attack_three() -> None:
    state = _state("11122357")
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent._ensure_trackers(state)
    agent._track[id(state)]["public_seen_counts"]["5"] = 3

    templates = agent._representative_attack_templates(state, "A")
    template = next(item for item in templates if item.template_id == "fourth_middle_finisher_5")
    assert template.attack_sequence[-1] == "5"

    plans = agent._generate_representative_attack_plans(
        state,
        "A",
        state.legal_actions("A"),
    )
    plan = _plan_from_source(plans, "representative:fourth_middle_finisher_5")
    root = plan.node(plan.root_node_id)
    second = _lap_target(plan, root)
    third = _lap_target(plan, second)

    assert root.action is not None and root.action[1] != "5"
    assert second.action is not None and second.action[1] != "5"
    assert third.action is not None and third.action[2] == "5"


def test_royal_preservation_template_never_hides_the_royal_early() -> None:
    state = _state("11233459")
    agent = RuleBasedAgent()
    agent.bind_player("A")
    plans = agent._generate_representative_attack_plans(
        state,
        "A",
        state.legal_actions("A"),
    )
    royal_plans = [
        plan for plan in plans
        if plan.source == "representative:royal_receive_width"
    ]

    assert royal_plans
    for plan in royal_plans:
        root = plan.node(plan.root_node_id)
        assert root.action is not None and root.action[1] != "9"
        second = _lap_target(plan, root)
        assert second.action is not None and second.action[1] != "9"


if __name__ == "__main__":
    test_rule_based_agent_uses_representative_attack_template_mixin()
    test_two_kyosha_single_big_becomes_big_kyosha_kyosha_template()
    test_two_kyosha_middle_pair_with_royal_keeps_kyosha_then_pair()
    test_four_shi_template_generates_three_consecutive_shi_attacks()
    test_fourth_middle_template_reserves_the_piece_for_attack_three()
    test_royal_preservation_template_never_hides_the_royal_early()
    print("ATTACK_PLAN_TEMPLATES_TEST_OK")

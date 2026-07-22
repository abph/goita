from __future__ import annotations

from goita_ai2.current_ai.forced_plans import ForcedPlansMixin
from goita_ai2.rule_based import RuleBasedAgent


def test_rule_based_agent_uses_forced_plans_mixin() -> None:
    assert issubclass(RuleBasedAgent, ForcedPlansMixin)


def test_forced_plan_methods_are_owned_by_mixin() -> None:
    for method_name in (
        "_plan_perfect_game",
        "_plan_perfect_game_after_first_receive",
        "_plan_any_win_after_first_receive",
        "_plan_win_after_two_receives",
    ):
        assert method_name in ForcedPlansMixin.__dict__
        assert method_name not in RuleBasedAgent.__dict__


if __name__ == "__main__":
    test_rule_based_agent_uses_forced_plans_mixin()
    test_forced_plan_methods_are_owned_by_mixin()
    print("FORCED_PLANS_MODULE_TEST_OK")

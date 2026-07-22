from __future__ import annotations

from goita_ai2.current_ai.attack_strategy import AttackStrategyMixin
from goita_ai2.rule_based import RuleBasedAgent


def test_rule_based_agent_uses_attack_strategy_mixin() -> None:
    assert issubclass(RuleBasedAgent, AttackStrategyMixin)


def test_attack_strategy_methods_are_owned_by_mixin() -> None:
    for method_name in (
        "_dealer_opening_plan_adjustment",
        "_shi_attack_score_adjustment",
        "_fuse_strategy_hidden_block_adjustment",
        "_score_attack_phase",
    ):
        assert method_name in AttackStrategyMixin.__dict__
        assert method_name not in RuleBasedAgent.__dict__


if __name__ == "__main__":
    test_rule_based_agent_uses_attack_strategy_mixin()
    test_attack_strategy_methods_are_owned_by_mixin()
    print("ATTACK_STRATEGY_MODULE_TEST_OK")

from __future__ import annotations

from goita_ai2.current_ai.decision import DecisionMixin
from goita_ai2.rule_based import RuleBasedAgent


def test_rule_based_agent_uses_decision_mixin() -> None:
    assert issubclass(RuleBasedAgent, DecisionMixin)


def test_decision_methods_are_owned_by_mixin() -> None:
    for method_name in (
        "_set_decision_reason",
        "_set_score_fallback_detail",
        "_classify_score_fallback",
        "select_action",
    ):
        assert method_name in DecisionMixin.__dict__
        assert method_name not in RuleBasedAgent.__dict__


if __name__ == "__main__":
    test_rule_based_agent_uses_decision_mixin()
    test_decision_methods_are_owned_by_mixin()
    print("DECISION_MODULE_TEST_OK")

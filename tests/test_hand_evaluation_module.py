from __future__ import annotations

from goita_ai2.current_ai.hand_evaluation import HandEvaluationMixin
from goita_ai2.rule_based import RuleBasedAgent


def test_rule_based_agent_uses_hand_evaluation_mixin() -> None:
    assert issubclass(RuleBasedAgent, HandEvaluationMixin)


def test_hand_evaluation_methods_are_owned_by_mixin() -> None:
    for method_name in (
        "_classify_hand_axes",
        "_classify_attack_type",
        "_special_attack_sequence_plan",
        "_load_relative_hand_rank_table",
    ):
        assert method_name in HandEvaluationMixin.__dict__
        assert method_name not in RuleBasedAgent.__dict__


if __name__ == "__main__":
    test_rule_based_agent_uses_hand_evaluation_mixin()
    test_hand_evaluation_methods_are_owned_by_mixin()
    print("HAND_EVALUATION_MODULE_TEST_OK")

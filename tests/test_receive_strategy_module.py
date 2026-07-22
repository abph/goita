from __future__ import annotations

from goita_ai2.current_ai.receive_strategy import ReceiveStrategyMixin
from goita_ai2.rule_based import RuleBasedAgent


def test_rule_based_agent_uses_receive_strategy_mixin() -> None:
    assert issubclass(RuleBasedAgent, ReceiveStrategyMixin)


def test_receive_strategy_methods_are_owned_by_mixin() -> None:
    for method_name in (
        "_enemy_first_same_piece_rank_policy_action",
        "_guaranteed_finish_receive_action",
        "_king_gyoku_opening_keep_receive_width_action",
        "_score_receive_phase",
    ):
        assert method_name in ReceiveStrategyMixin.__dict__
        assert method_name not in RuleBasedAgent.__dict__


if __name__ == "__main__":
    test_rule_based_agent_uses_receive_strategy_mixin()
    test_receive_strategy_methods_are_owned_by_mixin()
    print("RECEIVE_STRATEGY_MODULE_TEST_OK")

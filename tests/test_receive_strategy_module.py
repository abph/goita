from __future__ import annotations

from goita_ai2.current_ai.receive_strategy import ReceiveStrategyMixin
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


def test_rule_based_agent_uses_receive_strategy_mixin() -> None:
    assert issubclass(RuleBasedAgent, ReceiveStrategyMixin)


def test_receive_strategy_methods_are_owned_by_mixin() -> None:
    for method_name in (
        "_enemy_first_same_piece_rank_policy_action",
        "_enemy_second_attack_royal_reserve_pass_action",
        "_guaranteed_finish_receive_action",
        "_king_gyoku_opening_keep_receive_width_action",
        "_score_receive_phase",
    ):
        assert method_name in ReceiveStrategyMixin.__dict__
        assert method_name not in RuleBasedAgent.__dict__


def test_enemy_second_big_attack_is_passed_with_royal_reserve() -> None:
    state = GoitaState(
        hands={
            "A": list("11224458"),
            "B": list("11135569"),
            "C": list("12334457"),
            "D": list("11112367"),
        },
        dealer="B",
    )
    agents = {player: RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)

    actions = (
        ("B", ("attack_after_block", "1", "5")),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("receive", "5", None)),
        ("A", ("attack", None, "2")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("attack_after_block", "1", "4")),
        ("B", ("receive", "9", None)),
        ("B", ("attack", None, "6")),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
    )

    for player, action in actions:
        action_type, block, attack = action
        if action_type == "pass":
            state.apply_pass(player)
        elif action_type == "receive":
            state.apply_receive(player, block)
        elif action_type == "attack":
            state.apply_attack(player, attack)
        else:
            state.apply_attack_after_block(player, block, attack)
        for agent in agents.values():
            agent.on_public_action(state, player, action)

    a_agent = agents["A"]
    chosen = a_agent.select_action(state, "A", state.legal_actions("A"))

    assert chosen == ("pass", None, None)
    assert a_agent.last_decision_reason == "score_fallback"
    assert a_agent.last_score_fallback_detail == "pass_royal_reserve_wait_ally_kakari"


if __name__ == "__main__":
    test_rule_based_agent_uses_receive_strategy_mixin()
    test_receive_strategy_methods_are_owned_by_mixin()
    test_enemy_second_big_attack_is_passed_with_royal_reserve()
    print("RECEIVE_STRATEGY_MODULE_TEST_OK")

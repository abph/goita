from __future__ import annotations

from goita_ai2.current_ai.attack_strategy import AttackStrategyMixin
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


def test_rule_based_agent_uses_attack_strategy_mixin() -> None:
    assert issubclass(RuleBasedAgent, AttackStrategyMixin)


def test_attack_strategy_methods_are_owned_by_mixin() -> None:
    for method_name in (
        "_dealer_opening_plan_adjustment",
        "_four_shi_after_big_receive_first_attack_bonus",
        "_fourth_middle_early_attack_delay_penalty",
        "_fourth_middle_third_attack_bonus",
        "_second_kyosha_single_shi_block_adjustment",
        "_shi_attack_score_adjustment",
        "_fuse_strategy_hidden_block_adjustment",
        "_score_attack_phase",
    ):
        assert method_name in AttackStrategyMixin.__dict__
        assert method_name not in RuleBasedAgent.__dict__


def test_four_shi_is_preferred_after_receiving_big_piece() -> None:
    state = GoitaState(
        hands={
            "A": list("42126581"),
            "B": list("15111426"),
            "C": list("42171513"),
            "D": list("43139375"),
        },
        dealer="D",
    )
    agents = {player: RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)

    actions = (
        ("D", ("attack_after_block", "1", "3")),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("receive", "3", None)),
        ("C", ("attack", None, "1")),
        ("D", ("pass", None, None)),
        ("A", ("receive", "1", None)),
        ("A", ("attack", None, "2")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("attack_after_block", "4", "6")),
        ("B", ("receive", "6", None)),
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

    chosen = agents["B"].select_action(state, "B", state.legal_actions("B"))

    assert chosen == ("attack", None, "1")
    assert agents["B"].last_score_fallback_detail == "attack_four_shi_over_single_middle"


def test_fourth_kyosha_is_reserved_after_receiving_gold() -> None:
    state = GoitaState(
        hands={
            "A": list("32675114"),
            "B": list("15814311"),
            "C": list("26117423"),
            "D": list("19254513"),
        },
        dealer="B",
    )
    agents = {player: RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)

    actions = (
        ("B", ("attack_after_block", "1", "1")),
        ("C", ("receive", "1", None)),
        ("C", ("attack", None, "2")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("attack_after_block", "3", "2")),
        ("D", ("receive", "2", None)),
        ("D", ("attack", None, "5")),
        ("A", ("receive", "5", None)),
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
    assert a_agent._is_fourth_middle_attack(state, "A", "2")
    chosen = a_agent.select_action(state, "A", state.legal_actions("A"))

    assert chosen[0] == "attack"
    assert chosen[2] != "2"
    assert a_agent.last_decision_reason != "kakari"

    tracker = a_agent._track[id(state)]
    tracker["my_attack_count"] = 2
    third_attack = a_agent.select_action(state, "A", state.legal_actions("A"))

    assert third_attack == ("attack", None, "2")
    assert a_agent.last_score_fallback_detail == "attack_fourth_middle_third"


def test_remaining_silver_pair_continues_three_silver_attack() -> None:
    state = GoitaState(
        hands={
            "A": list("51512181"),
            "B": list("13411276"),
            "C": list("44237934"),
            "D": list("11635215"),
        },
        dealer="D",
    )
    agents = {player: RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)
        agent._ensure_trackers(state)

    opening_actions = (
        ("D", ("attack_after_block", "1", "5")),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("attack_after_block", "1", "5")),
        ("A", ("receive", "5", None)),
        ("A", ("attack", None, "2")),
        ("B", ("pass", None, None)),
    )

    for action_player, action in opening_actions:
        action_type, block, attack = action
        if action_type == "pass":
            state.apply_pass(action_player)
        elif action_type == "receive":
            state.apply_receive(action_player, block)
        elif action_type == "attack":
            state.apply_attack(action_player, attack)
        else:
            state.apply_attack_after_block(action_player, block, attack)
        for agent in agents.values():
            agent.on_public_action(state, action_player, action)

    c_agent = agents["C"]
    first_receive = c_agent.select_action(state, "C", state.legal_actions("C"))
    state.apply_receive("C", first_receive[1])
    for agent in agents.values():
        agent.on_public_action(state, "C", first_receive)
    first_attack = c_agent.select_action(state, "C", state.legal_actions("C"))
    state.apply_attack("C", first_attack[2])
    for agent in agents.values():
        agent.on_public_action(state, "C", first_attack)

    later_actions = (
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("receive", "4", None)),
        ("B", ("attack", None, "6")),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("attack_after_block", "1", "7")),
    )

    for action_player, action in later_actions:
        action_type, block, attack = action
        if action_type == "pass":
            state.apply_pass(action_player)
        elif action_type == "receive":
            state.apply_receive(action_player, block)
        elif action_type == "attack":
            state.apply_attack(action_player, attack)
        else:
            state.apply_attack_after_block(action_player, block, attack)
        for agent in agents.values():
            agent.on_public_action(state, action_player, action)

    second_receive = c_agent.select_action(state, "C", state.legal_actions("C"))
    state.apply_receive("C", second_receive[1])
    for agent in agents.values():
        agent.on_public_action(state, "C", second_receive)

    assert state.hands["C"].count("4") == 2
    assert not c_agent._is_fourth_middle_attack(state, "C", "4")
    chosen = c_agent.select_action(state, "C", state.legal_actions("C"))

    assert first_attack == ("attack", None, "4")
    assert chosen == ("attack", None, "4")


def test_second_kyosha_keeps_single_shi_and_blocks_low_middle() -> None:
    state = GoitaState(
        hands={
            "A": list("63921422"),
            "B": list("53741811"),
            "C": list("16411335"),
            "D": list("51125417"),
        },
        dealer="D",
    )
    agents = {player: RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)
        agent._ensure_trackers(state)

    actions = (
        ("D", ("attack_after_block", "1", "5")),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("receive", "5", None)),
        ("C", ("attack", None, "3")),
        ("D", ("pass", None, None)),
        ("A", ("receive", "3", None)),
        ("A", ("attack", None, "2")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
    )

    for action_player, action in actions:
        action_type, block, attack = action
        if action_type == "pass":
            state.apply_pass(action_player)
        elif action_type == "receive":
            state.apply_receive(action_player, block)
        elif action_type == "attack":
            state.apply_attack(action_player, attack)
        else:
            state.apply_attack_after_block(action_player, block, attack)
        for agent in agents.values():
            agent.on_public_action(state, action_player, action)

    a_agent = agents["A"]
    a_agent._track[id(state)]["my_attack_count"] = 1
    chosen = a_agent.select_action(state, "A", state.legal_actions("A"))

    assert chosen == ("attack_after_block", "4", "2")
    assert a_agent.last_score_fallback_detail == "block_low_middle_keep_single_shi"


def test_third_kyosha_compares_shi_royal_and_big_royal_waits() -> None:
    state = GoitaState(
        hands={
            "A": list("63921422"),
            "B": list("53741811"),
            "C": list("16411335"),
            "D": list("51125417"),
        },
        dealer="D",
    )
    agents = {player: RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)
        agent._ensure_trackers(state)

    actions = (
        ("D", ("attack_after_block", "1", "5")),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("receive", "5", None)),
        ("C", ("attack", None, "3")),
        ("D", ("pass", None, None)),
        ("A", ("receive", "3", None)),
        ("A", ("attack", None, "2")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("attack_after_block", "4", "2")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
    )

    for action_player, action in actions:
        action_type, block, attack = action
        if action_type == "pass":
            state.apply_pass(action_player)
        elif action_type == "receive":
            state.apply_receive(action_player, block)
        elif action_type == "attack":
            state.apply_attack(action_player, attack)
        else:
            state.apply_attack_after_block(action_player, block, attack)
        for agent in agents.values():
            agent.on_public_action(state, action_player, action)

    a_agent = agents["A"]
    tracker = a_agent._track[id(state)]
    tracker["my_attack_count"] = 2
    chosen = a_agent.select_action(state, "A", state.legal_actions("A"))

    assert chosen == ("attack_after_block", "6", "2")
    assert sorted(
        a_agent._remaining_hand_after_attack_action(state, "A", chosen[1], chosen[2])
    ) == ["1", "9"]

    for enemy in ("B", "D"):
        tracker["estimated_current_hands"][enemy]["1"] = {
            "min": 0,
            "max": 0,
            "expected": 0.0,
            "map_count": 0,
        }
    tracker["my_attack_count"] = 2
    low_shi_pressure_choice = a_agent.select_action(
        state,
        "A",
        state.legal_actions("A"),
    )

    assert low_shi_pressure_choice == ("attack_after_block", "6", "2")
    assert sorted(
        a_agent._remaining_hand_after_attack_action(
            state,
            "A",
            low_shi_pressure_choice[1],
            low_shi_pressure_choice[2],
        )
    ) == ["1", "9"]


if __name__ == "__main__":
    test_rule_based_agent_uses_attack_strategy_mixin()
    test_attack_strategy_methods_are_owned_by_mixin()
    test_four_shi_is_preferred_after_receiving_big_piece()
    test_fourth_kyosha_is_reserved_after_receiving_gold()
    test_remaining_silver_pair_continues_three_silver_attack()
    test_second_kyosha_keeps_single_shi_and_blocks_low_middle()
    test_third_kyosha_compares_shi_royal_and_big_royal_waits()
    print("ATTACK_STRATEGY_MODULE_TEST_OK")

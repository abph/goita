from __future__ import annotations

from collections import Counter

from goita_ai2.current_ai.receive_strategy import ReceiveStrategyMixin
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


def test_rule_based_agent_uses_receive_strategy_mixin() -> None:
    assert issubclass(RuleBasedAgent, ReceiveStrategyMixin)


def test_receive_strategy_methods_are_owned_by_mixin() -> None:
    for method_name in (
        "_enemy_first_same_piece_rank_policy_action",
        "_enemy_second_attack_royal_reserve_pass_action",
        "_ally_kyosha_continuation_pass_action",
        "_full_receive_cover_royal_wait_pass_action",
        "_guaranteed_finish_receive_action",
        "_king_gyoku_opening_keep_receive_width_action",
        "_no_shi_royal_endgame_commit_action",
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


def test_receive_ally_second_attack_to_play_fourth_silver_third() -> None:
    state = GoitaState(
        hands={
            "A": list("14442215"),
            "B": list("15138163"),
            "C": list("67325531"),
            "D": list("71411291"),
        },
        dealer="D",
    )
    agents = {player: RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)
        agent._ensure_trackers(state)

    actions = (
        ("D", ("attack_after_block", "1", "1")),
        ("A", ("receive", "1", None)),
        ("A", ("attack", None, "4")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("receive", "4", None)),
        ("D", ("attack", None, "1")),
        ("A", ("receive", "1", None)),
        ("A", ("attack", None, "4")),
        ("B", ("receive", "8", None)),
        ("B", ("attack", None, "3")),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("attack_after_block", "1", "6")),
        ("C", ("receive", "6", None)),
        ("C", ("attack", None, "5")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("receive", "5", None)),
        ("B", ("attack", None, "3")),
        ("C", ("receive", "3", None)),
        ("C", ("attack", None, "5")),
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
    receive = a_agent.select_action(state, "A", state.legal_actions("A"))

    assert tracker["public_seen_counts"]["4"] == 3
    assert state.hands["A"].count("4") == 1
    assert receive == ("receive", "5", None)
    assert a_agent.last_decision_reason == "score_fallback"
    assert a_agent.last_score_fallback_detail == "ally_force_king_receive"

    state.apply_receive("A", "5")
    for agent in agents.values():
        agent.on_public_action(state, "A", receive)
    attack = a_agent.select_action(state, "A", state.legal_actions("A"))

    assert attack == ("attack", None, "4")
    assert a_agent.last_decision_reason == "score_fallback"
    assert a_agent.last_score_fallback_detail == "attack_force_enemy_king"


def test_weak_next_player_receives_dealer_silver_and_signals_with_shi() -> None:
    state = GoitaState(
        hands={
            "A": list("14542215"),
            "B": list("14138173"),
            "C": list("67325531"),
            "D": list("61411291"),
        },
        dealer="A",
    )
    agents = {player: RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)
        agent._ensure_trackers(state)

    opening = ("attack_after_block", "1", "4")
    state.apply_attack_after_block("A", "1", "4")
    for agent in agents.values():
        agent.on_public_action(state, "A", opening)

    b_agent = agents["B"]
    axes = b_agent._initial_hand_axes_for_state(state, "B")
    receive = b_agent.select_action(state, "B", state.legal_actions("B"))

    assert str(axes["absolute_rank"]) in ("D", "E", "F", "X")
    assert b_agent._effective_receive_type(Counter(state.hands["B"])) == 3
    assert receive == ("receive", "4", None)
    assert b_agent.last_decision_reason == "score_fallback"
    assert b_agent.last_score_fallback_detail.endswith(
        "_weak_shi_signal_receive"
    )

    state.apply_receive("B", "4")
    for agent in agents.values():
        agent.on_public_action(state, "B", receive)
    attack = b_agent.select_action(state, "B", state.legal_actions("B"))

    assert attack == ("attack", None, "1")
    assert b_agent.last_decision_reason == "score_fallback"
    assert b_agent.last_score_fallback_detail == "attack_weak_hand_shi_signal"


def test_full_receive_cover_waits_for_enemy_third_attack_and_fifty_points() -> None:
    state = GoitaState(
        hands={
            "A": list("41755141"),
            "B": list("31621113"),
            "C": list("33291855"),
            "D": list("47412216"),
        },
        dealer="B",
    )
    agents = {player: RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)
        agent._ensure_trackers(state)

    actions = (
        ("B", ("attack_after_block", "1", "3")),
        ("C", ("receive", "3", None)),
        ("C", ("attack", None, "5")),
        ("D", ("pass", None, None)),
        ("A", ("receive", "5", None)),
        ("A", ("attack", None, "5")),
        ("B", ("pass", None, None)),
        ("C", ("receive", "5", None)),
        ("C", ("attack", None, "3")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("receive", "3", None)),
        ("B", ("attack", None, "6")),
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

    c_agent = agents["C"]
    tracker = c_agent._track[id(state)]
    tracker["my_attack_count"] = 2

    chosen = c_agent.select_action(state, "C", state.legal_actions("C"))

    assert state.hands["C"] == list("2918")
    assert chosen == ("pass", None, None)
    assert c_agent.last_decision_reason == "score_fallback"
    assert (
        c_agent.last_score_fallback_detail
        == "pass_full_receive_cover_royal_wait_high_score"
    )

    followup_actions = (
        ("C", chosen),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("attack_after_block", "1", "1")),
    )
    for action_player, action in followup_actions:
        action_type, block, attack = action
        if action_type == "pass":
            state.apply_pass(action_player)
        else:
            state.apply_attack_after_block(action_player, block, attack)
        for agent in agents.values():
            agent.on_public_action(state, action_player, action)

    receive_shi = c_agent.select_action(state, "C", state.legal_actions("C"))
    assert receive_shi == ("receive", "1", None)
    state.apply_receive("C", "1")
    for agent in agents.values():
        agent.on_public_action(state, "C", receive_shi)

    attack_gyoku = c_agent.select_action(state, "C", state.legal_actions("C"))
    assert attack_gyoku == ("attack", None, "8")
    state.apply_attack("C", "8")
    for agent in agents.values():
        agent.on_public_action(state, "C", attack_gyoku)

    for passer in ("D", "A", "B"):
        pass_action = ("pass", None, None)
        state.apply_pass(passer)
        for agent in agents.values():
            agent.on_public_action(state, passer, pass_action)

    finish = c_agent.select_action(state, "C", state.legal_actions("C"))
    assert finish == ("attack_after_block", "2", "9")
    state.apply_attack_after_block("C", "2", "9")
    assert state.finished is True
    assert state.winner == "C"
    assert state.team_score["AC"] == 50


def test_no_shi_endgame_uses_royal_instead_of_passing_enemy_first_attack() -> None:
    state = GoitaState(
        hands={
            "A": list("21328655"),
            "B": list("51244313"),
            "C": list("13114452"),
            "D": list("61911717"),
        },
        dealer="C",
    )
    agents = {player: RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)
        agent._ensure_trackers(state)

    actions = (
        ("C", ("attack_after_block", "1", "4")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("receive", "4", None)),
        ("B", ("attack", None, "3")),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("receive", "3", None)),
        ("A", ("attack", None, "2")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("attack_after_block", "1", "5")),
        ("B", ("receive", "5", None)),
        ("B", ("attack", None, "3")),
        ("C", ("receive", "3", None)),
        ("C", ("attack", None, "2")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("receive", "2", None)),
        ("B", ("attack", None, "4")),
        ("C", ("receive", "4", None)),
        ("C", ("attack", None, "5")),
        ("D", ("receive", "9", None)),
        ("D", ("attack", None, "7")),
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
    receive = a_agent.select_action(state, "A", state.legal_actions("A"))

    assert sorted(state.hands["A"]) == ["2", "5", "6", "8"]
    assert receive == ("receive", "8", None)
    assert a_agent.last_decision_reason == "score_fallback"
    assert (
        a_agent.last_score_fallback_detail
        == "receive_no_shi_royal_endgame_commit"
    )

    state.apply_receive("A", "8")
    for agent in agents.values():
        agent.on_public_action(state, "A", receive)
    attack = a_agent.select_action(state, "A", state.legal_actions("A"))

    assert attack == ("attack", None, "5")
    state.apply_attack("A", "5")
    for agent in agents.values():
        agent.on_public_action(state, "A", attack)

    for passer in ("B", "C", "D"):
        pass_action = ("pass", None, None)
        assert pass_action in state.legal_actions(passer)
        state.apply_pass(passer)
        for agent in agents.values():
            agent.on_public_action(state, passer, pass_action)

    finish = a_agent.select_action(state, "A", state.legal_actions("A"))
    assert finish == ("attack_after_block", "2", "6")
    state.apply_attack_after_block("A", "2", "6")
    assert state.finished is True
    assert state.winner == "A"
    assert state.team_score["AC"] == 40


def test_passes_ally_first_kyosha_to_preserve_likely_three_kyosha_route() -> None:
    state = GoitaState(
        hands={
            "A": list("34448661"),
            "B": list("11217715"),
            "C": list("11133345"),
            "D": list("11252295"),
        },
        dealer="C",
    )
    agents = {player: RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)
        agent._ensure_trackers(state)

    actions = (
        ("C", ("attack_after_block", "1", "3")),
        ("D", ("pass", None, None)),
        ("A", ("receive", "3", None)),
        ("A", ("attack", None, "6")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("attack_after_block", "6", "4")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("receive", "9", None)),
        ("D", ("attack", None, "2")),
        ("A", ("pass", None, None)),
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

    b_agent = agents["B"]
    chosen = b_agent.select_action(state, "B", state.legal_actions("B"))
    tracker = b_agent._track[id(state)]
    ally = tracker["ally"]
    remaining_min, remaining_max = b_agent._estimate_remaining_range(
        tracker,
        ally,
        "2",
    )

    assert state.hands["B"].count("2") == 1
    assert state.hands["B"].count("7") == 2
    assert remaining_max >= 1
    assert chosen == ("pass", None, None)
    assert b_agent.last_decision_reason == "score_fallback"
    assert (
        b_agent.last_score_fallback_detail
        == "pass_ally_kyosha_continuation"
    )

    continuation = (
        ("B", chosen),
        ("C", ("pass", None, None)),
        ("D", ("attack_after_block", "1", "2")),
        ("A", ("pass", None, None)),
    )
    for action_player, action in continuation:
        action_type, block, attack = action
        if action_type == "pass":
            state.apply_pass(action_player)
        else:
            state.apply_attack_after_block(action_player, block, attack)
        for agent in agents.values():
            agent.on_public_action(state, action_player, action)

    second_kyosha = b_agent.select_action(state, "B", state.legal_actions("B"))
    assert second_kyosha == ("pass", None, None)
    assert (
        b_agent.last_score_fallback_detail
        == "pass_ally_kyosha_continuation"
    )

    continuation = (
        ("B", second_kyosha),
        ("C", ("pass", None, None)),
        ("D", ("attack_after_block", "1", "2")),
        ("A", ("pass", None, None)),
    )
    for action_player, action in continuation:
        action_type, block, attack = action
        if action_type == "pass":
            state.apply_pass(action_player)
        else:
            state.apply_attack_after_block(action_player, block, attack)
        for agent in agents.values():
            agent.on_public_action(state, action_player, action)

    third_kyosha = b_agent.select_action(state, "B", state.legal_actions("B"))
    assert third_kyosha == ("pass", None, None)
    assert (
        b_agent.last_score_fallback_detail
        == "pass_ally_kyosha_continuation"
    )

    for passer in ("B", "C"):
        pass_action = ("pass", None, None)
        state.apply_pass(passer)
        for agent in agents.values():
            agent.on_public_action(state, passer, pass_action)

    state.apply_attack_after_block("D", "5", "5")
    assert state.finished is True
    assert state.winner == "D"


if __name__ == "__main__":
    test_rule_based_agent_uses_receive_strategy_mixin()
    test_receive_strategy_methods_are_owned_by_mixin()
    test_enemy_second_big_attack_is_passed_with_royal_reserve()
    test_receive_ally_second_attack_to_play_fourth_silver_third()
    test_weak_next_player_receives_dealer_silver_and_signals_with_shi()
    test_full_receive_cover_waits_for_enemy_third_attack_and_fifty_points()
    test_no_shi_endgame_uses_royal_instead_of_passing_enemy_first_attack()
    test_passes_ally_first_kyosha_to_preserve_likely_three_kyosha_route()
    print("RECEIVE_STRATEGY_MODULE_TEST_OK")

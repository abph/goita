from __future__ import annotations

from goita_ai2.current_ai.endgame import EndgameMixin
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


def test_rule_based_agent_uses_endgame_mixin() -> None:
    assert issubclass(RuleBasedAgent, EndgameMixin)


def test_endgame_methods_are_owned_by_mixin() -> None:
    for method_name in (
        "_max_tsume_score",
        "_high_score_tsume_action",
        "_reach_avoidance_conditional_tsume_action",
        "_inferred_ally_shi_sashikomi_finish_action",
        "_inferred_endgame_team_result_action",
        "_give_way_to_ally_guaranteed_win_action",
    ):
        assert method_name in EndgameMixin.__dict__
        assert method_name not in RuleBasedAgent.__dict__


def test_reach_avoidance_conditional_tsume_prefers_hisha_over_kin() -> None:
    state = GoitaState(
        hands={
            "A": list("69615751"),
            "B": list("33543712"),
            "C": list("18215144"),
            "D": list("12411231"),
        },
        dealer="B",
    )
    agents = {player: RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)

    actions = (
        ("B", ("attack_after_block", "1", "3")),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("attack_after_block", "7", "3")),
        ("C", ("receive", "8", None)),
        ("C", ("attack", None, "4")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("receive", "4", None)),
        ("B", ("attack", None, "3")),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("receive", "9", None)),
        ("A", ("attack", None, "6")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("attack_after_block", "1", "6")),
        ("B", ("pass", None, None)),
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
    tracker = a_agent._track[id(state)]
    kin_risk = a_agent._estimated_receive_risk_for_player(tracker, "B", "5")
    hisha_risk = a_agent._estimated_receive_risk_for_player(tracker, "B", "7")
    chosen = a_agent.select_action(state, "A", state.legal_actions("A"))

    assert kin_risk is not None and hisha_risk is not None
    assert hisha_risk < kin_risk
    assert chosen == ("attack_after_block", "5", "7")
    assert a_agent.last_decision_reason == "conditional_tsume"
    assert a_agent.last_score_fallback_detail == "reach_avoid_next_B_piece_7_risk_11"


def test_third_attack_inserts_shi_for_inferred_ally_finish() -> None:
    state = GoitaState(
        hands={
            "A": list("36257431"),
            "B": list("54351128"),
            "C": list("12171451"),
            "D": list("92614311"),
        },
        dealer="D",
    )
    agents = {player: RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)
        agent._ensure_trackers(state)

    actions = (
        ("D", ("attack_after_block", "3", "1")),
        ("A", ("receive", "1", None)),
        ("A", ("attack", None, "3")),
        ("B", ("receive", "3", None)),
        ("B", ("attack", None, "5")),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("receive", "5", None)),
        ("A", ("attack", None, "3")),
        ("B", ("receive", "8", None)),
        ("B", ("attack", None, "5")),
        ("C", ("receive", "5", None)),
        ("C", ("attack", None, "7")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("attack_after_block", "1", "2")),
        ("D", ("receive", "2", None)),
        ("D", ("attack", None, "1")),
        ("A", ("pass", None, None)),
        ("B", ("receive", "1", None)),
        ("B", ("attack", None, "2")),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("receive", "2", None)),
        ("A", ("attack", None, "7")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("receive", "9", None)),
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

    d_agent = agents["D"]
    tracker = d_agent._track[id(state)]
    tracker["my_attack_count"] = 2
    inferred = tracker["joint_hand_inference"]["map_current_counts"]
    chosen = d_agent.select_action(state, "D", state.legal_actions("D"))

    assert inferred["A"]["1"] == 0
    assert inferred["B"]["1"] >= 1
    assert chosen == ("attack", None, "1")
    assert d_agent.last_decision_reason == "score_fallback"
    assert d_agent.last_score_fallback_detail == "attack_inferred_ally_shi_sashikomi_win"


def test_inferred_endgame_prefers_lower_scoring_enemy_finish() -> None:
    state = GoitaState(
        hands={
            "A": list("12517153"),
            "B": list("29715411"),
            "C": list("24161135"),
            "D": list("43183246"),
        },
        dealer="D",
    )
    agents = {player: RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)
        agent._ensure_trackers(state)

    actions = (
        ("D", ("attack_after_block", "1", "4")),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("receive", "4", None)),
        ("C", ("attack", None, "1")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("receive", "1", None)),
        ("B", ("attack", None, "4")),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("attack_after_block", "5", "7")),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("receive", "7", None)),
        ("A", ("attack", None, "5")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("attack_after_block", "3", "5")),
        ("B", ("receive", "9", None)),
        ("B", ("attack", None, "2")),
        ("C", ("receive", "2", None)),
        ("C", ("attack", None, "5")),
        ("D", ("receive", "8", None)),
        ("D", ("attack", None, "2")),
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
    legal = state.legal_actions("A")
    inferred_result = a_agent._inferred_endgame_team_result_action(state, "A", legal)
    chosen = a_agent.select_action(state, "A", legal)

    assert inferred_result is not None
    assert inferred_result[0] == ("receive", "2", None)
    assert inferred_result[1] == "B"
    assert inferred_result[2] < 40
    assert chosen == ("receive", "2", None)
    assert a_agent.last_decision_reason == "inferred_endgame"
    assert a_agent.last_score_fallback_detail.startswith("inferred_endgame_min_loss_B_")


def test_inferred_endgame_prefers_ally_finish_over_enemy_finish() -> None:
    state = GoitaState(
        hands={
            "A": list("5111"),
            "B": list("11"),
            "C": list("57"),
            "D": list("34"),
        },
        dealer="D",
    )
    state.phase = "receive"
    state.turn = "A"
    state.attacker = "D"
    state.current_attack = "5"

    agent = RuleBasedAgent()
    agent.bind_player("A")
    zero_counts = {str(i): 0 for i in range(1, 10)}

    def counts(hand: str) -> dict[str, int]:
        result = dict(zero_counts)
        for piece in hand:
            result[piece] += 1
        return result

    agent._track[id(state)] = {
        "public_hand_models": {
            seat: {"attack_count": 2 if seat == "D" else 0}
            for seat in "ABCD"
        },
        "joint_hand_inference": {
            "feasible": True,
            "map_current_counts": {
                "B": counts("11"),
                "C": counts("57"),
                "D": counts("34"),
            },
            "map_hidden_counts": {
                "B": dict(zero_counts),
                "C": dict(zero_counts),
                "D": dict(zero_counts),
            },
            "map_original_counts": {
                "B": counts("11"),
                "C": counts("57"),
                "D": counts("34"),
            },
        },
    }

    result = agent._inferred_endgame_team_result_action(
        state,
        "A",
        state.legal_actions("A"),
    )

    assert result == (("pass", None, None), "C", 40)


def test_high_score_tsume_outranks_kakarigotae() -> None:
    state = GoitaState(
        hands={
            "A": list("11113448"),
            "B": list("13555679"),
            "C": list("11223445"),
            "D": list("11122367"),
        },
        dealer="D",
    )
    agents = {player: RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)
        agent._ensure_trackers(state)

    actions = (
        ("D", ("attack_after_block", "3", "7")),
        ("A", ("pass", None, None)),
        ("B", ("receive", "7", None)),
        ("B", ("attack", None, "5")),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("attack_after_block", "1", "5")),
        ("C", ("receive", "5", None)),
        ("C", ("attack", None, "2")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("attack_after_block", "1", "2")),
        ("D", ("receive", "2", None)),
        ("D", ("attack", None, "6")),
        ("A", ("receive", "8", None)),
        ("A", ("attack", None, "4")),
        ("B", ("receive", "9", None)),
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
    b_agent._track[id(state)]["my_attack_count"] = 2
    chosen = b_agent.select_action(state, "B", state.legal_actions("B"))

    assert chosen == ("attack", None, "5")
    assert b_agent.last_decision_reason == "tsume"
    assert b_agent.last_score_fallback_detail == "high_score_40"


if __name__ == "__main__":
    test_rule_based_agent_uses_endgame_mixin()
    test_endgame_methods_are_owned_by_mixin()
    test_reach_avoidance_conditional_tsume_prefers_hisha_over_kin()
    test_third_attack_inserts_shi_for_inferred_ally_finish()
    test_inferred_endgame_prefers_lower_scoring_enemy_finish()
    test_inferred_endgame_prefers_ally_finish_over_enemy_finish()
    test_high_score_tsume_outranks_kakarigotae()
    print("ENDGAME_MODULE_TEST_OK")

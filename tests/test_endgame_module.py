from __future__ import annotations

from goita_ai2.current_ai.endgame import EndgameMixin, ForcedWinStatus
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


def test_rule_based_agent_uses_endgame_mixin() -> None:
    assert issubclass(RuleBasedAgent, EndgameMixin)


def test_endgame_methods_are_owned_by_mixin() -> None:
    for method_name in (
        "_max_tsume_score",
        "_forced_win_result_after_attack_action",
        "_forced_win_result_after_receive_action",
        "_high_score_tsume_action",
        "_royal_bridge_finish_action",
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


def test_third_attack_uses_royal_bridge_for_thirty_point_finish() -> None:
    state = GoitaState(
        hands={
            "A": list("17934111"),
            "B": list("28163321"),
            "C": list("31647252"),
            "D": list("15445151"),
        },
        dealer="A",
    )
    agents = {player: RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)
        agent._ensure_trackers(state)

    actions = (
        ("A", ("attack_after_block", "1", "1")),
        ("B", ("receive", "1", None)),
        ("B", ("attack", None, "2")),
        ("C", ("receive", "2", None)),
        ("C", ("attack", None, "6")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("receive", "6", None)),
        ("B", ("attack", None, "3")),
        ("C", ("receive", "3", None)),
        ("C", ("attack", None, "7")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("receive", "8", None)),
        ("B", ("attack", None, "3")),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("receive", "3", None)),
        ("A", ("attack", None, "7")),
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
    a_agent._track[id(state)]["my_attack_count"] = 2
    legal = state.legal_actions("A")
    bridge = a_agent._royal_bridge_finish_action(state, "A", legal)
    chosen = a_agent.select_action(state, "A", legal)

    assert sorted(state.hands["A"]) == sorted("9411")
    assert bridge == (("attack_after_block", "1", "9"), 30.0)
    assert chosen == ("attack_after_block", "1", "9")
    assert a_agent.last_decision_reason == "tsume"
    assert a_agent.last_score_fallback_detail == "high_score_30"

    state.apply_attack_after_block("A", "1", "9")
    for agent in agents.values():
        agent.on_public_action(state, "A", chosen)
    for action_player in ("B", "C", "D"):
        passed = ("pass", None, None)
        state.apply_pass(action_player)
        for agent in agents.values():
            agent.on_public_action(state, action_player, passed)

    finish = a_agent.select_action(state, "A", state.legal_actions("A"))
    assert finish == ("attack_after_block", "1", "4")
    state.apply_attack_after_block("A", "1", "4")
    assert state.finished
    assert state.winner == "A"
    assert state.team_score["AC"] == 30


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


def test_inferred_endgame_carries_receive_followup_attack() -> None:
    state = GoitaState(
        hands={
            "A": list("29511431"),
            "B": list("22145814"),
            "C": list("31534577"),
            "D": list("63116211"),
        },
        dealer="A",
    )
    agents = {player: RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)
        agent._ensure_trackers(state)

    actions = (
        ("A", ("attack_after_block", "3", "1")),
        ("B", ("receive", "1", None)),
        ("B", ("attack", None, "2")),
        ("C", ("pass", None, None)),
        ("D", ("receive", "2", None)),
        ("D", ("attack", None, "6")),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("attack_after_block", "1", "6")),
        ("A", ("receive", "9", None)),
        ("A", ("attack", None, "1")),
        ("B", ("receive", "1", None)),
        ("B", ("attack", None, "4")),
        ("C", ("receive", "4", None)),
        ("C", ("attack", None, "7")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("attack_after_block", "3", "7")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("receive", "8", None)),
        ("B", ("attack", None, "4")),
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
    receive = a_agent.select_action(state, "A", state.legal_actions("A"))

    assert receive == ("receive", "4", None)
    assert a_agent.last_decision_reason == "inferred_endgame"
    assert a_agent.last_score_fallback_detail == "inferred_endgame_self_win_A_20"
    assert tracker["pending_inferred_endgame_attack"] == ("attack", None, "5")

    state.apply_receive("A", "4")
    for agent in agents.values():
        agent.on_public_action(state, "A", receive)
    attack = a_agent.select_action(state, "A", state.legal_actions("A"))

    assert attack == ("attack", None, "5")
    assert a_agent.last_decision_reason == "inferred_endgame"
    assert a_agent.last_score_fallback_detail == "inferred_endgame_followup_attack"
    assert tracker["pending_inferred_endgame_attack"] is None


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


def test_second_shi_attack_is_proven_at_thirty_over_safe_kyosha_ten() -> None:
    state = GoitaState(
        hands={
            "A": list("45143156"),
            "B": list("27124331"),
            "C": list("12821165"),
            "D": list("73911145"),
        },
        dealer="B",
    )
    agents = {player: RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)
        agent._ensure_trackers(state)

    actions = (
        ("B", ("attack_after_block", "3", "2")),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("attack_after_block", "1", "2")),
        ("C", ("receive", "2", None)),
        ("C", ("attack", None, "6")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
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
    shi_route = ("attack_after_block", "1", "1")
    kyosha_route = ("attack_after_block", "1", "2")
    shi_result = c_agent._forced_win_result_after_attack_action(state, "C", shi_route)
    kyosha_result = c_agent._forced_win_result_after_attack_action(state, "C", kyosha_route)
    chosen = c_agent.select_action(state, "C", state.legal_actions("C"))

    assert sorted(state.hands["C"]) == sorted("111258")
    assert shi_result.status == ForcedWinStatus.PROVEN
    assert shi_result.minimum_score == 30.0
    assert kyosha_result.status == ForcedWinStatus.PROVEN
    assert kyosha_result.minimum_score == 10.0
    assert chosen == shi_route
    assert c_agent.last_decision_reason == "tsume"
    assert c_agent.last_score_fallback_detail == "high_score_30"


def test_high_score_tsume_keeps_kaku_for_forty_point_finish() -> None:
    state = GoitaState(
        hands={
            "A": list("14115173"),
            "B": list("16423721"),
            "C": list("29154685"),
            "D": list("14251331"),
        },
        dealer="C",
    )
    agents = {player: RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)
        agent._ensure_trackers(state)

    actions = (
        ("C", ("attack_after_block", "1", "5")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
    )
    for action_player, action in actions:
        action_type, block, attack = action
        if action_type == "pass":
            state.apply_pass(action_player)
        else:
            state.apply_attack_after_block(action_player, block, attack)
        for agent in agents.values():
            agent.on_public_action(state, action_player, action)

    c_agent = agents["C"]
    legal = state.legal_actions("C")
    kaku_block = ("attack_after_block", "6", "8")
    gin_block = ("attack_after_block", "4", "8")
    kaku_result = c_agent._forced_win_result_after_attack_action(
        state,
        "C",
        kaku_block,
    )
    gin_result = c_agent._forced_win_result_after_attack_action(
        state,
        "C",
        gin_block,
    )
    chosen = c_agent.select_action(state, "C", legal)

    assert kaku_result.status == ForcedWinStatus.PROVEN
    assert kaku_result.minimum_score == 30.0
    assert gin_result.status == ForcedWinStatus.PROVEN
    assert gin_result.minimum_score == 40.0
    assert chosen == gin_block
    assert c_agent.last_decision_reason == "tsume"
    assert c_agent.last_score_fallback_detail == "high_score_40"


def test_early_forced_win_search_reports_unknown_instead_of_guessing() -> None:
    state = GoitaState(
        hands={
            "A": list("11112345"),
            "B": list("11112345"),
            "C": list("11112345"),
            "D": list("11112345"),
        },
        dealer="A",
    )
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent._ensure_trackers(state)

    result = agent._forced_win_result_after_attack_action(
        state,
        "A",
        ("attack_after_block", "1", "2"),
    )

    assert result.status == ForcedWinStatus.UNKNOWN
    assert result.minimum_score is None


def test_receive_then_safe_kyosha_is_proven_at_thirty() -> None:
    state = GoitaState(
        hands={
            "A": list("1111"),
            "B": list("1111"),
            "C": list("1285"),
            "D": list("1111"),
        },
        dealer="A",
    )
    state.phase = "receive"
    state.turn = "C"
    state.attacker = "D"
    state.current_attack = "5"

    agent = RuleBasedAgent()
    agent.bind_player("C")
    agent._ensure_trackers(state)
    tracker = agent._track[id(state)]
    tracker["public_seen_counts"]["2"] = 3
    tracker["public_seen_counts"]["5"] = 1

    result = agent._forced_win_result_after_receive_action(
        state,
        "C",
        ("receive", "8", None),
    )

    assert result.status == ForcedWinStatus.PROVEN
    assert result.minimum_score == 30.0


def test_royal_cannot_receive_shi_in_forced_win_search() -> None:
    state = GoitaState(
        hands={
            "A": list("1111"),
            "B": list("1111"),
            "C": list("1285"),
            "D": list("1111"),
        },
        dealer="A",
    )
    state.phase = "receive"
    state.turn = "C"
    state.attacker = "D"
    state.current_attack = "1"

    agent = RuleBasedAgent()
    agent.bind_player("C")
    agent._ensure_trackers(state)
    result = agent._forced_win_result_after_receive_action(
        state,
        "C",
        ("receive", "8", None),
    )

    assert result.status == ForcedWinStatus.COUNTEREXAMPLE
    assert result.minimum_score is None


def test_safe_third_attack_keeps_silver_for_thirty_point_finish() -> None:
    state = GoitaState(
        hands={
            "A": list("85217194"),
            "B": list("33627151"),
            "C": list("31461155"),
            "D": list("41234112"),
        },
        dealer="C",
    )
    agents = {player: RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)
        agent._ensure_trackers(state)

    actions = (
        ("C", ("attack_after_block", "1", "5")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("receive", "5", None)),
        ("B", ("attack", None, "3")),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("attack_after_block", "1", "3")),
        ("C", ("receive", "3", None)),
        ("C", ("attack", None, "5")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("attack_after_block", "1", "6")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("receive", "6", None)),
        ("B", ("attack", None, "7")),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("receive", "7", None)),
        ("A", ("attack", None, "8")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("attack_after_block", "1", "9")),
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
    third_attack = a_agent.select_action(state, "A", state.legal_actions("A"))

    assert sorted(state.hands["A"]) == ["1", "2", "4", "5"]
    assert third_attack == ("attack_after_block", "2", "5")
    assert a_agent.last_decision_reason == "tsume"
    assert a_agent.last_score_fallback_detail == "high_score_30"
    state.apply_attack_after_block("A", third_attack[1], third_attack[2])
    for agent in agents.values():
        agent.on_public_action(state, "A", third_attack)

    for passer in ("B", "C", "D"):
        pass_action = ("pass", None, None)
        assert pass_action in state.legal_actions(passer)
        state.apply_pass(passer)
        for agent in agents.values():
            agent.on_public_action(state, passer, pass_action)

    finish = a_agent.select_action(state, "A", state.legal_actions("A"))
    assert finish == ("attack_after_block", "1", "4")
    state.apply_attack_after_block("A", "1", "4")
    assert state.finished is True
    assert state.winner == "A"
    assert state.team_score["AC"] == 30


def test_global_endgame_planner_keeps_king_for_fifty_point_finish() -> None:
    state = GoitaState(
        hands={
            "A": list("32171647"),
            "B": list("41543221"),
            "C": list("55235191"),
            "D": list("83411161"),
        },
        dealer="C",
    )
    agents = {player: RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)
        agent._ensure_trackers(state)

    actions = (
        ("C", ("attack_after_block", "1", "5")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("receive", "5", None)),
        ("B", ("attack", None, "4")),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("receive", "4", None)),
        ("A", ("attack", None, "7")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("attack_after_block", "1", "6")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("receive", "6", None)),
        ("D", ("attack", None, "4")),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("attack_after_block", "3", "1")),
        ("A", ("receive", "1", None)),
        ("A", ("attack", None, "7")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("receive", "8", None)),
        ("D", ("attack", None, "1")),
        ("A", ("pass", None, None)),
        ("B", ("receive", "1", None)),
        ("B", ("attack", None, "2")),
        ("C", ("receive", "2", None)),
        ("C", ("attack", None, "5")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
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
    third_attack = c_agent.select_action(state, "C", state.legal_actions("C"))

    assert sorted(state.hands["C"]) == ["1", "3", "5", "9"]
    assert third_attack[0] == "attack_after_block"
    assert third_attack[2] == "5"
    assert c_agent.last_decision_reason == "tsume"
    assert c_agent.last_score_fallback_detail == "high_score_50"
    state.apply_attack_after_block("C", third_attack[1], third_attack[2])
    for agent in agents.values():
        agent.on_public_action(state, "C", third_attack)

    assert "9" in state.hands["C"]
    for passer in ("D", "A", "B"):
        pass_action = ("pass", None, None)
        assert pass_action in state.legal_actions(passer)
        state.apply_pass(passer)
        for agent in agents.values():
            agent.on_public_action(state, passer, pass_action)

    finish = c_agent.select_action(state, "C", state.legal_actions("C"))
    assert finish[0] == "attack_after_block"
    assert finish[2] == "9"
    state.apply_attack_after_block("C", finish[1], finish[2])
    assert state.finished is True
    assert state.winner == "C"
    assert state.team_score["AC"] == 50


def test_seven_card_conditional_plan_prefers_shi_for_fifty_point_branches() -> None:
    state = GoitaState(
        hands={
            "A": list("12244557"),
            "B": list("11133456"),
            "C": list("11123567"),
            "D": list("11123489"),
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
        ("B", ("attack_after_block", "1", "3")),
        ("C", ("receive", "3", None)),
        ("C", ("attack", None, "1")),
        ("D", ("receive", "1", None)),
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
    legal = state.legal_actions("D")
    shi_action = ("attack", None, "1")
    royal_action = ("attack", None, "8")
    shi_result = d_agent._forced_win_result_after_attack_action(
        state,
        "D",
        shi_action,
    )
    royal_result = d_agent._forced_win_result_after_attack_action(
        state,
        "D",
        royal_action,
    )
    chosen = d_agent.select_action(state, "D", legal)
    plan = d_agent._track[id(state)]["last_forced_win_score_plan"]

    assert shi_result.status == ForcedWinStatus.PROVEN
    assert shi_result.minimum_score == 30.0
    assert shi_result.expected_score is not None
    assert shi_result.expected_score > 30.0
    assert shi_result.maximum_score == 50.0
    assert royal_result.status == ForcedWinStatus.PROVEN
    assert royal_result.minimum_score == 30.0
    assert chosen == shi_action
    assert plan["attack"] == "1"
    assert plan["minimum_score"] == 30.0
    assert plan["expected_score"] > 30.0
    assert plan["maximum_score"] == 50.0

    followup_actions = (
        ("D", chosen),
        ("A", ("receive", "1", None)),
        ("A", ("attack", None, "2")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("receive", "2", None)),
    )
    for action_player, action in followup_actions:
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

    followup_legal = state.legal_actions("D")
    followup_scores = {
        action: d_agent._forced_win_result_after_attack_action(state, "D", action)
        for action in followup_legal
    }
    horse_followup = d_agent.select_action(state, "D", followup_legal)
    assert horse_followup == ("attack", None, "3"), (
        horse_followup,
        d_agent.last_decision_reason,
        d_agent.last_score_fallback_detail,
        followup_scores,
    )


if __name__ == "__main__":
    test_rule_based_agent_uses_endgame_mixin()
    test_endgame_methods_are_owned_by_mixin()
    test_reach_avoidance_conditional_tsume_prefers_hisha_over_kin()
    test_third_attack_uses_royal_bridge_for_thirty_point_finish()
    test_third_attack_inserts_shi_for_inferred_ally_finish()
    test_inferred_endgame_prefers_lower_scoring_enemy_finish()
    test_inferred_endgame_prefers_ally_finish_over_enemy_finish()
    test_inferred_endgame_carries_receive_followup_attack()
    test_high_score_tsume_outranks_kakarigotae()
    test_second_shi_attack_is_proven_at_thirty_over_safe_kyosha_ten()
    test_high_score_tsume_keeps_kaku_for_forty_point_finish()
    test_early_forced_win_search_reports_unknown_instead_of_guessing()
    test_receive_then_safe_kyosha_is_proven_at_thirty()
    test_royal_cannot_receive_shi_in_forced_win_search()
    test_safe_third_attack_keeps_silver_for_thirty_point_finish()
    test_global_endgame_planner_keeps_king_for_fifty_point_finish()
    test_seven_card_conditional_plan_prefers_shi_for_fifty_point_branches()
    print("ENDGAME_MODULE_TEST_OK")

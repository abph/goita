from __future__ import annotations

from goita_ai2.current_ai.decision import DecisionMixin
from goita_ai2.experimental_ai2 import RuleBasedAgent as ExperimentalAI2RuleBasedAgent
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


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


def test_one_shi_receive_and_return_is_enough_to_signal_approval() -> None:
    state = GoitaState(
        hands={
            "A": list("32225454"),
            "B": list("76131511"),
            "C": list("11794431"),
            "D": list("11125863"),
        },
        dealer="D",
    )
    agents = {player: RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)
        agent._ensure_trackers(state)

    def apply(action_player: str, action) -> None:
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

    apply("D", ("attack_after_block", "3", "1"))
    apply("A", ("pass", None, None))

    b_agent = agents["B"]
    first_receive = b_agent.select_action(state, "B", state.legal_actions("B"))
    assert first_receive == ("receive", "1", None)
    assert b_agent.last_decision_reason == "shi_signal"
    apply("B", first_receive)

    first_return = b_agent.select_action(state, "B", state.legal_actions("B"))
    assert first_return == ("attack", None, "1")
    apply("B", first_return)
    assert b_agent._track[id(state)]["my_shi_approval_sent"] is True

    later_actions = (
        ("C", ("receive", "1", None)),
        ("C", ("attack", None, "4")),
        ("D", ("pass", None, None)),
        ("A", ("receive", "4", None)),
        ("A", ("attack", None, "4")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("attack_after_block", "5", "2")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("receive", "2", None)),
        ("D", ("attack", None, "1")),
        ("A", ("pass", None, None)),
    )
    for action_player, action in later_actions:
        apply(action_player, action)

    second_response = b_agent.select_action(state, "B", state.legal_actions("B"))

    assert state.hands["B"].count("1") == 2
    assert second_response == ("pass", None, None)
    assert b_agent.last_decision_reason == "shi_signal"
    assert (
        b_agent.last_score_fallback_detail
        == "ally_shi_approval_already_sent_pass"
    )


def test_passes_ally_shi_when_public_exhaustion_guarantees_ally_finish() -> None:
    state = GoitaState(
        hands={
            "A": list("11123478"),
            "B": list("11344569"),
            "C": list("11112355"),
            "D": list("12234567"),
        },
        dealer="C",
    )
    agents = {player: ExperimentalAI2RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)
        agent.TIME_SEARCH_BACKGROUND_ENABLED = False
        agent._ensure_trackers(state)

    def apply(action_player: str, action) -> None:
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

    prefix = (
        ("C", ("attack_after_block", "1", "1")),
        ("D", ("receive", "1", None)),
        ("D", ("attack", None, "7")),
        ("A", ("receive", "7", None)),
        ("A", ("attack", None, "1")),
        ("B", ("receive", "1", None)),
        ("B", ("attack", None, "4")),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("receive", "4", None)),
        ("A", ("attack", None, "1")),
        ("B", ("receive", "1", None)),
        ("B", ("attack", None, "6")),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("receive", "8", None)),
        ("A", ("attack", None, "1")),
        ("B", ("pass", None, None)),
    )
    for action_player, action in prefix:
        assert action in state.legal_actions(action_player)
        apply(action_player, action)

    c_agent = agents["C"]
    assert state.hands["A"] == ["2", "3"]
    assert c_agent._ally_current_attack_is_publicly_unstoppable(state, "C")

    choice = c_agent.select_action(state, "C", state.legal_actions("C"))

    assert choice == ("pass", None, None)
    assert c_agent.last_decision_reason == "score_fallback"
    assert (
        c_agent.last_score_fallback_detail
        == "pass_ally_guaranteed_win_no_self_finish"
    )
    assert state.hands["C"].count("1") == 2

    apply("C", choice)
    assert state.legal_actions("D") == [("pass", None, None)]
    apply("D", ("pass", None, None))
    finish = agents["A"].select_action(state, "A", state.legal_actions("A"))
    apply("A", finish)

    assert state.finished is True
    assert state.winner == "A"
    assert state.team_score == {"AC": 20, "BD": 0}


def test_four_shi_plan_keeps_last_pair_for_own_third_attack() -> None:
    state = GoitaState(
        hands={
            "A": list("11344567"),
            "B": list("22345689"),
            "C": list("11112355"),
            "D": list("22345678"),
        },
        dealer="C",
    )
    agents = {player: ExperimentalAI2RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)
        agent.TIME_SEARCH_BACKGROUND_ENABLED = False
        agent._ensure_trackers(state)

    def apply(action_player: str, action) -> None:
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

    apply("C", ("attack_after_block", "1", "1"))
    apply("D", ("pass", None, None))
    apply("A", ("receive", "1", None))
    apply("A", ("attack", None, "1"))
    apply("B", ("pass", None, None))

    choice = agents["C"].select_action(state, "C", state.legal_actions("C"))

    assert choice == ("pass", None, None)
    assert agents["C"].last_decision_reason == "shi_signal"
    assert (
        agents["C"].last_score_fallback_detail
        == "pass_ally_shi_preserve_third_attack"
    )


def test_three_shi_plan_does_not_spend_last_pair_acknowledging_ally() -> None:
    """Regression for round 2, C turn 18 of the 2026-10-02 review."""
    state = GoitaState(
        hands={
            "A": list("13115174"),
            "B": list("43827214"),
            "C": list("11461935"),
            "D": list("63215251"),
        },
        dealer="B",
    )
    agents = {player: ExperimentalAI2RuleBasedAgent() for player in "ABCD"}
    for player, agent in agents.items():
        agent.bind_player(player)
        agent.TIME_SEARCH_BACKGROUND_ENABLED = False
        agent._ensure_trackers(state)

    def apply(action_player: str, action) -> None:
        action_type, block, attack = action
        assert action in state.legal_actions(action_player)
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

    prefix = (
        ("B", ("attack_after_block", "4", "2")),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("attack_after_block", "1", "4")),
        ("C", ("receive", "4", None)),
        ("C", ("attack", None, "1")),
        ("D", ("receive", "1", None)),
        ("D", ("attack", None, "2")),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("attack_after_block", "3", "2")),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("attack_after_block", "6", "5")),
        ("A", ("receive", "5", None)),
        ("A", ("attack", None, "1")),
        ("B", ("pass", None, None)),
    )
    for action_player, action in prefix:
        apply(action_player, action)

    c_agent = agents["C"]
    assert c_agent._track[id(state)]["my_attack_count"] == 1
    assert state.hands["C"].count("1") == 2

    choice = c_agent.select_action(state, "C", state.legal_actions("C"))

    assert choice == ("pass", None, None)
    assert c_agent.last_decision_reason == "shi_signal"
    assert (
        c_agent.last_score_fallback_detail
        == "pass_ally_shi_preserve_third_attack"
    )
    assert c_agent.last_neural_shadow["recommended_action"] == [
        "pass",
        None,
        None,
    ]
    assert c_agent.last_neural_shadow["match"] is True


if __name__ == "__main__":
    test_rule_based_agent_uses_decision_mixin()
    test_decision_methods_are_owned_by_mixin()
    test_one_shi_receive_and_return_is_enough_to_signal_approval()
    print("DECISION_MODULE_TEST_OK")

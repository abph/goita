from __future__ import annotations

from goita_ai2.current_ai.decision import DecisionMixin
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


if __name__ == "__main__":
    test_rule_based_agent_uses_decision_mixin()
    test_decision_methods_are_owned_by_mixin()
    test_one_shi_receive_and_return_is_enough_to_signal_approval()
    print("DECISION_MODULE_TEST_OK")

from __future__ import annotations

from goita_ai2.current_ai.agent import RuleBasedAgent
from goita_ai2.state import GoitaState


def _apply_public(state, agent, player, action) -> None:
    action_type, block, attack = action
    if action_type == "pass":
        state.apply_pass(player)
    elif action_type == "receive":
        state.apply_receive(player, block)
    elif action_type == "attack":
        state.apply_attack(player, attack)
    else:
        state.apply_attack_after_block(player, block, attack)
    agent.on_public_action(state, player, action)


def _horse_insertion_state(*, one_royal: bool = False):
    hands = {
        "A": list("51731133"),
        "B": list("71215514"),
        "C": list("14412122"),
        "D": list("36119158"),
    }
    if one_royal:
        hands["A"].remove("5")
        hands["A"].append("8")
        hands["D"].remove("8")
        hands["D"].append("5")

    state = GoitaState(hands, dealer="A")
    agent = RuleBasedAgent()
    agent.bind_player("D")
    agent.TIME_SEARCH_ENABLED = False
    agent._ensure_trackers(state)
    opening = (
        ("A", ("attack_after_block", "1", "3")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
    )
    for player, action in opening:
        _apply_public(state, agent, player, action)
    return state, agent


def test_shi_insertion_compares_every_legal_followup() -> None:
    state, agent = _horse_insertion_state()

    analysis = agent._shi_insertion_plan_analysis(
        state,
        "D",
        state.legal_actions("D"),
    )

    assert analysis is not None
    assert analysis["downstream"] == "A"
    assert analysis["downstream_hidden_count"] == 1
    assert analysis["royal_count"] == 2
    assert {item["attack"] for item in analysis["followups"]} == {
        "1",
        "5",
        "6",
        "8",
        "9",
    }
    assert analysis["recommended"]["followup"] == "1"


def test_one_downstream_hidden_block_raises_shi_insertion_value() -> None:
    state, agent = _horse_insertion_state()

    without_hidden, without_components = agent._shi_insertion_followup_score(
        state,
        "D",
        "3",
        "1",
        downstream="A",
        downstream_hidden_count=0,
    )
    with_hidden, with_components = agent._shi_insertion_followup_score(
        state,
        "D",
        "3",
        "1",
        downstream="A",
        downstream_hidden_count=1,
    )

    assert with_hidden > without_hidden
    assert with_components["downstream_one_hidden_shi"] == 72.0
    assert (
        with_components["downstream_interception_risk"]
        > without_components["downstream_interception_risk"]
    )


def test_both_royals_are_better_than_one_royal() -> None:
    both_state, both_agent = _horse_insertion_state()
    one_state, one_agent = _horse_insertion_state(one_royal=True)

    both = both_agent._shi_insertion_plan_analysis(
        both_state,
        "D",
        both_state.legal_actions("D"),
    )
    one = one_agent._shi_insertion_plan_analysis(
        one_state,
        "D",
        one_state.legal_actions("D"),
    )

    assert both is not None and one is not None
    assert both["recommended"]["components"]["royal_safety"] == 105.0
    assert one["recommended"]["components"]["royal_safety"] == 48.0


def test_immediate_receive_keeps_the_planned_shi_followup() -> None:
    state, agent = _horse_insertion_state()

    receive = agent.select_action(state, "D", state.legal_actions("D"))
    assert receive == ("receive", "3", None)
    assert agent.last_decision_reason == "shi_insertion"
    assert agent.last_score_fallback_detail == "shi_insertion_immediate_1"

    _apply_public(state, agent, "D", receive)
    attack = agent.select_action(state, "D", state.legal_actions("D"))

    assert attack == ("attack", None, "1")
    assert agent.last_decision_reason == "shi_insertion"
    assert agent.last_score_fallback_detail == "shi_insertion_followup_1"


def test_delayed_plan_waits_only_one_cycle() -> None:
    state, agent = _horse_insertion_state()
    agent.SHI_INSERTION_EXTRA_BLOCK_VALUE = 200.0
    agent.SHI_INSERTION_WAIT_AFTER_ONE_HIDDEN_VALUE = 100.0
    agent.SHI_INSERTION_REPEAT_ATTACK_PENALTY = 0.0

    first = agent.select_action(state, "D", state.legal_actions("D"))
    assert first == ("pass", None, None)
    assert agent.last_score_fallback_detail == "shi_insertion_delayed_1"
    _apply_public(state, agent, "D", first)

    continuation = (
        ("A", ("attack_after_block", "5", "3")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
    )
    for player, action in continuation:
        _apply_public(state, agent, player, action)

    second = agent.select_action(state, "D", state.legal_actions("D"))
    analysis = agent._track[id(state)]["last_shi_insertion_analysis"]

    assert second == ("receive", "3", None)
    assert analysis["waited_once"] is True
    assert {route["timing"] for route in analysis["routes"]} == {"immediate"}


if __name__ == "__main__":
    test_shi_insertion_compares_every_legal_followup()
    test_one_downstream_hidden_block_raises_shi_insertion_value()
    test_both_royals_are_better_than_one_royal()
    test_immediate_receive_keeps_the_planned_shi_followup()
    test_delayed_plan_waits_only_one_cycle()
    print("SHI_INSERTION_STRATEGY_TEST_OK")

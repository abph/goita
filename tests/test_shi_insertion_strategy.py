from __future__ import annotations

import copy

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


def _review_c_turn7(rotation=0):
    seats = "ABCD"
    def seat(p):
        return seats[(seats.index(p) + rotation) % 4]
    state = GoitaState({seat(p): list(h) for p, h in {
        "A": "31251532", "B": "21611134", "C": "44158741", "D": "51162793",
    }.items()}, dealer=seat("A"))
    agent = RuleBasedAgent()
    player = seat("C")
    agent.bind_player(player)
    agent._ensure_trackers(state)
    agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    agent.TIME_SEARCH_CACHE_ENABLED = False
    agent.TIME_SEARCH_PREDICTION_CACHE_ENABLED = False
    agent.TIME_SEARCH_ADAPTIVE_BUDGET_ENABLED = False
    for p, action in (
        ("A", ("attack_after_block", "3", "2")),
        ("B", ("pass", None, None)), ("C", ("pass", None, None)),
        ("D", ("receive", "2", None)), ("D", ("attack", None, "7")),
        ("A", ("pass", None, None)), ("B", ("pass", None, None)),
    ):
        assert action in state.legal_actions(seat(p))
        _apply_public(state, agent, seat(p), action)
    return state, agent, player


def test_review_c_turn7_receives_before_lance_and_attacks_three_silver():
    for rotation in range(4):
        state, agent, player = _review_c_turn7(rotation)
        analysis = agent._shi_insertion_plan_analysis(state, player, state.legal_actions(player))
        assert analysis["recommended"]["followup"] == "4"
        assert analysis["recommended"]["timing"] == "immediate"
        assert analysis["uncovered_next_attack_weight"] > 0
        lance = next(r for r in analysis["next_attack_candidates"] if r["piece"] == "2")
        assert not lance["can_receive"] and lance["weight"] > 0
        assert any(r["root_action"][0] == "pass" for r in analysis["routes"])
        assert all("common_attack_evaluation" in r["components"] for r in analysis["followups"])
        receive = agent.select_action(state, player, state.legal_actions(player))
        assert receive == ("receive", "7", None)
        assert agent.last_score_fallback_detail == "shi_insertion_immediate_4_avoid_2"
        _apply_public(state, agent, player, receive)
        assert sorted(state.hands[player]) == list("1144458")
        attack = agent.select_action(state, player, state.legal_actions(player))
        assert attack == ("attack", None, "4")
        assert agent.last_score_fallback_detail == "shi_insertion_followup_4_avoid_2"


def test_wait_exposure_uses_own_coverage_and_not_real_opponent_hands():
    state, agent, player = _review_c_turn7()
    original = agent._shi_insertion_plan_analysis(state, player, state.legal_actions(player))
    tracker_before = copy.deepcopy(agent._track[id(state)])
    state.hands["A"], state.hands["D"] = state.hands["D"], state.hands["A"]
    changed = agent._shi_insertion_plan_analysis(state, player, state.legal_actions(player))
    assert changed == original
    assert agent._track[id(state)] == tracker_before
    # Coverage counterfactual: having a lance removes the uncovered next attack.
    state.hands[player].append("2")
    exposure, _ = agent._shi_insertion_wait_risk(state, player)
    assert exposure == 0


def test_enemy_shi_possession_reduces_both_pressure_and_ally_delivery():
    state, agent, player = _review_c_turn7()
    def score(enemy_probability):
        agent._shi_insertion_piece_probability = lambda s, p, seat, piece: (
            enemy_probability if seat == "D" else 1.0
        )
        return agent._shi_insertion_followup_score(
            state, player, "7", "1", downstream="D", downstream_hidden_count=0,
        )[1]
    blocked = score(1.0)
    open_route = score(0.0)
    assert blocked["ally_reach_probability"] == 0
    assert blocked["shi_pressure"] == 0
    assert open_route["ally_reach_probability"] > 0
    assert open_route["shi_pressure"] > 0


if __name__ == "__main__":
    test_shi_insertion_compares_every_legal_followup()
    test_one_downstream_hidden_block_raises_shi_insertion_value()
    test_both_royals_are_better_than_one_royal()
    test_immediate_receive_keeps_the_planned_shi_followup()
    test_delayed_plan_waits_only_one_cycle()
    test_review_c_turn7_receives_before_lance_and_attacks_three_silver()
    test_wait_exposure_uses_own_coverage_and_not_real_opponent_hands()
    test_enemy_shi_possession_reduces_both_pressure_and_ally_delivery()
    print("SHI_INSERTION_STRATEGY_TEST_OK")

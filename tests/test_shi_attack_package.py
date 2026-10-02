from __future__ import annotations

from goita_ai2.experimental_ai2 import RuleBasedAgent as ExperimentalAI2RuleBasedAgent
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


INITIAL_HANDS = {
    "A": list("13124135"),
    "B": list("78265131"),
    "C": list("15149161"),
    "D": list("24421735"),
}

PREFIX = (
    ("C", ("attack_after_block", "1", "1")),
    ("D", ("pass", None, None)),
    ("A", ("pass", None, None)),
    ("B", ("receive", "1", None)),
    ("B", ("attack", None, "7")),
    ("C", ("pass", None, None)),
    ("D", ("pass", None, None)),
    ("A", ("pass", None, None)),
    ("B", ("attack_after_block", "3", "6")),
    ("C", ("receive", "6", None)),
    ("C", ("attack", None, "1")),
    ("D", ("receive", "1", None)),
    ("D", ("attack", None, "4")),
    ("A", ("receive", "4", None)),
    ("A", ("attack", None, "1")),
    ("B", ("receive", "1", None)),
    ("B", ("attack", None, "2")),
    ("C", ("pass", None, None)),
    ("D", ("pass", None, None)),
    ("A", ("receive", "2", None)),
    ("A", ("attack", None, "1")),
    ("B", ("pass", None, None)),
    ("C", ("pass", None, None)),
    ("D", ("pass", None, None)),
)


def _review_state():
    state = GoitaState(
        hands={seat: list(hand) for seat, hand in INITIAL_HANDS.items()},
        dealer="C",
    )
    agents = {seat: ExperimentalAI2RuleBasedAgent() for seat in "ABCD"}
    for seat, agent in agents.items():
        agent.bind_player(seat)
        agent.TIME_SEARCH_BACKGROUND_ENABLED = False
        agent._ensure_trackers(state)

    for seat, action in PREFIX:
        assert action in state.legal_actions(seat)
        action_type, block, attack = action
        if action_type == "pass":
            state.apply_pass(seat)
        elif action_type == "receive":
            state.apply_receive(seat, block)
        elif action_type == "attack":
            state.apply_attack(seat, attack)
        else:
            state.apply_attack_after_block(seat, block, attack)
        for agent in agents.values():
            agent.on_public_action(state, seat, action)
    return state, agents


def test_final_shi_attack_uses_joint_distribution_when_enemy_team_is_exhausted() -> None:
    """Regression for round 5, A turn 19 of the 2026-10-02 review."""
    state, agents = _review_state()
    agent = agents["A"]

    chosen = agent.select_action(state, "A", state.legal_actions("A"))
    analysis = agent.last_shi_attack_package_analysis

    assert chosen == ("attack_after_block", "5", "1")
    assert agent.last_decision_reason == "shi_attack_package"
    assert agent.last_score_fallback_detail == (
        "shi_package_final_attack_enemy_map_0_expected_29_"
        "distribution_A3_B2_C4_D1"
    )
    assert analysis is not None
    assert analysis["initial_distribution"] == {
        "A": 3,
        "B": 2,
        "C": 4,
        "D": 1,
    }
    assert analysis["current_distribution"] == {
        "A": 1,
        "B": 0,
        "C": 1,
        "D": 0,
    }
    assert analysis["enemy_used_shi"] == 3
    assert analysis["purpose"] == "finish_third_attack"
    assert agent.last_neural_shadow["recommended_action"][2] == "1"


def test_final_shi_attack_is_not_forced_when_joint_map_has_enemy_shi() -> None:
    state, agents = _review_state()
    agent = agents["A"]
    tracker = agent._track[id(state)]
    tracker["joint_hand_inference"]["map_current_counts"]["B"]["1"] = 1

    result = agent._shi_attack_package_action(
        state,
        "A",
        state.legal_actions("A"),
        has_non_king_attack_option=True,
    )

    assert result is None
    assert agent.last_shi_attack_package_analysis["enemy_map_remaining"] == 1


def test_shi_attack_package_is_isolated_to_experimental_ai2() -> None:
    assert RuleBasedAgent().SHI_ATTACK_PACKAGE_ENABLED is False
    assert ExperimentalAI2RuleBasedAgent().SHI_ATTACK_PACKAGE_ENABLED is True

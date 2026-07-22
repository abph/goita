from __future__ import annotations

from typing import Dict, List, Tuple

from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


Action = Tuple[str, str | None, str | None]


def _apply_public_action(
    agent: RuleBasedAgent,
    state: GoitaState,
    player: str,
    action: Action,
) -> None:
    action_type, block, attack = action
    if action_type == "pass":
        state.apply_pass(player)
    elif action_type == "receive":
        assert block is not None
        state.apply_receive(player, block)
    elif action_type == "attack":
        assert attack is not None
        state.apply_attack(player, attack)
    elif action_type == "attack_after_block":
        assert block is not None and attack is not None
        state.apply_attack_after_block(player, block, attack)
    else:
        raise AssertionError(f"unexpected action: {action}")
    agent.on_public_action(state, player, action)


def _rank(agent: RuleBasedAgent, state: GoitaState, player: str) -> Dict[str, object]:
    return agent._public_hand_rank_estimate(agent._track[id(state)], player)


def test_rank_updates_after_every_attack_and_timed_pass() -> None:
    state = GoitaState(
        hands={
            "A": ["1"] * 8,
            "B": ["1", "1", "1", "1", "1", "4", "4", "4"],
            "C": ["1"] * 8,
            "D": ["1"] * 8,
        },
        dealer="B",
    )
    agent = RuleBasedAgent()
    agent.bind_player("A")

    _apply_public_action(agent, state, "B", ("attack_after_block", "1", "4"))
    assert _rank(agent, state, "B")["rank"] == "C"
    assert _rank(agent, state, "B")["reason"] == "first_attack_middle_repeat"

    _apply_public_action(agent, state, "C", ("pass", None, None))
    assert _rank(agent, state, "C")["reason"] == "pass_enemy_attack_1"
    _apply_public_action(agent, state, "D", ("pass", None, None))
    _apply_public_action(agent, state, "A", ("pass", None, None))

    _apply_public_action(agent, state, "B", ("attack_after_block", "1", "4"))
    assert _rank(agent, state, "B")["rank"] == "C"
    assert _rank(agent, state, "B")["reason"] == "repeat_attack_4"
    _apply_public_action(agent, state, "C", ("pass", None, None))
    assert _rank(agent, state, "C")["reason"] == "pass_enemy_attack_2"
    _apply_public_action(agent, state, "D", ("pass", None, None))
    _apply_public_action(agent, state, "A", ("pass", None, None))

    _apply_public_action(agent, state, "B", ("attack_after_block", "1", "4"))
    assert _rank(agent, state, "B")["rank"] == "B"
    _apply_public_action(agent, state, "C", ("pass", None, None))
    c_estimate = _rank(agent, state, "C")
    assert c_estimate["rank"] == "E"
    assert c_estimate["reason"] == "pass_enemy_attack_3"
    assert float(c_estimate["confidence"]) > 0.2


def test_receive_timing_is_recorded() -> None:
    state = GoitaState(
        hands={
            "A": ["1"] * 8,
            "B": ["1"] * 8,
            "C": ["1", "1", "1", "1", "1", "1", "1", "5"],
            "D": ["1", "1", "1", "1", "1", "1", "5", "8"],
        },
        dealer="C",
    )
    agent = RuleBasedAgent()
    agent.bind_player("A")

    _apply_public_action(agent, state, "C", ("attack_after_block", "1", "5"))
    _apply_public_action(agent, state, "D", ("receive", "5", None))
    d_estimate = _rank(agent, state, "D")
    assert d_estimate["reason"] == "receive_enemy_attack_1_same"
    assert float(d_estimate["score"]) > 5.0


def _first_attack_estimate(hidden_hand: List[str], block: str) -> Dict[str, object]:
    state = GoitaState(
        hands={
            "A": ["1"] * 8,
            "B": hidden_hand,
            "C": ["1"] * 8,
            "D": ["1"] * 8,
        },
        dealer="B",
    )
    agent = RuleBasedAgent()
    agent.bind_player("A")
    _apply_public_action(agent, state, "B", ("attack_after_block", block, "4"))
    return _rank(agent, state, "B")


def test_hidden_hand_does_not_change_public_rank_estimate() -> None:
    estimate_a = _first_attack_estimate(
        ["1", "1", "1", "1", "1", "4", "4", "4"],
        "1",
    )
    estimate_b = _first_attack_estimate(
        ["2", "3", "4", "4", "5", "6", "7", "8"],
        "2",
    )
    assert estimate_a == estimate_b


if __name__ == "__main__":
    test_rank_updates_after_every_attack_and_timed_pass()
    test_receive_timing_is_recorded()
    test_hidden_hand_does_not_change_public_rank_estimate()
    print("PUBLIC_HAND_RANK_INFERENCE_TEST_OK")

"""Checks attack sequences built from a middle-piece pair and one big piece.

These cases do not require two lances. They verify the pair/big-piece order and
the conditional shi-counter route that keeps the big piece for the final score.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Tuple

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
    else:
        assert block is not None and attack is not None
        state.apply_attack_after_block(player, block, attack)
    agent.on_public_action(state, player, action)


def _attack_sequence(hand: List[str]) -> tuple[List[str], Action]:
    filler = ["1"] * 8
    state = GoitaState(
        hands={"A": hand, "B": filler, "C": filler, "D": filler},
        dealer="A",
    )
    agent = RuleBasedAgent()
    agent.bind_player("A")
    attacks: List[str] = []

    first_action = agent.select_action(state, "A", state.legal_actions("A"))
    assert first_action[2] is not None
    if first_action[0] == "attack_after_block":
        assert first_action[1] is not None
        state.apply_attack_after_block("A", first_action[1], first_action[2])
    else:
        assert first_action[0] == "attack"
        state.apply_attack("A", first_action[2])
    agent.on_public_action(state, "A", first_action)
    attacks.append(first_action[2])

    while len(attacks) < 3:
        state.phase = "attack"
        state.turn = "A"
        state.attacker = "A"
        state.current_attack = None
        action = agent.select_action(state, "A", state.legal_actions("A"))
        assert action[2] is not None
        if action[0] == "attack_after_block":
            assert action[1] is not None
            state.apply_attack_after_block("A", action[1], action[2])
        else:
            assert action[0] == "attack"
            state.apply_attack("A", action[2])
        agent.on_public_action(state, "A", action)
        attacks.append(action[2])

    return attacks, first_action


def test_middle_pair_single_big_without_royal_uses_pair_pair_big() -> None:
    attacks, first_action = _attack_sequence(
        ["1", "1", "2", "3", "4", "5", "5", "7"]
    )
    assert attacks == ["5", "5", "7"]
    assert first_action[1] not in ("5", "7")


def test_middle_pair_single_big_with_one_royal_uses_pair_big_pair() -> None:
    attacks, first_action = _attack_sequence(
        ["1", "1", "2", "3", "5", "5", "7", "9"]
    )
    assert attacks == ["5", "7", "5"]
    assert first_action[1] not in ("5", "7")


def test_middle_pair_single_big_generalizes_to_silver_and_horse() -> None:
    agent = RuleBasedAgent()
    assert agent._middle_pair_single_big_attack_plan(
        Counter(["1", "1", "2", "3", "4", "4", "6", "9"])
    ) == ["4", "6", "4"]
    assert agent._middle_pair_single_big_attack_plan(
        Counter(["1", "1", "2", "3", "3", "4", "5", "7"])
    ) == ["3", "3", "7"]


def test_middle_pair_plan_does_not_override_higher_attack_types() -> None:
    agent = RuleBasedAgent()
    assert agent._middle_pair_single_big_attack_plan(
        Counter(["1", "2", "3", "5", "5", "5", "7", "9"])
    ) is None
    assert agent._middle_pair_single_big_attack_plan(
        Counter(["1", "2", "3", "4", "5", "5", "7", "7"])
    ) is None
    assert agent._middle_pair_single_big_attack_plan(
        Counter(["1", "2", "3", "4", "4", "5", "5", "7"])
    ) is None


def test_middle_pair_plan_keeps_big_piece_in_shi_counter_position() -> None:
    state = GoitaState(
        hands={
            "A": ["2", "1", "1", "4", "1", "2", "4", "2"],
            "B": ["4", "1", "5", "5", "5", "6", "7", "3"],
            "C": ["2", "1", "1", "4", "3", "9", "6", "1"],
            "D": ["1", "1", "7", "5", "8", "3", "3", "1"],
        },
        dealer="B",
    )
    agent = RuleBasedAgent()
    agent.bind_player("D")

    actions: List[tuple[str, Action]] = [
        ("B", ("attack_after_block", "1", "5")),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("attack_after_block", "3", "5")),
        ("C", ("receive", "9", None)),
        ("C", ("attack", None, "1")),
        ("D", ("receive", "1", None)),
        ("D", ("attack", None, "5")),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
    ]
    for player, action in actions:
        _apply_public_action(agent, state, player, action)

    assert agent._conditional_shi_royal_finish_score(
        state, "D", "attack_after_block", "1", "3"
    ) == 40.0
    assert agent._conditional_shi_royal_finish_score(
        state, "D", "attack_after_block", "7", "3"
    ) == 20.0

    state.king_block_used = 0
    assert agent._conditional_shi_royal_finish_score(
        state, "D", "attack_after_block", "1", "3"
    ) is None
    state.king_block_used = 1

    c_attacks = agent._track[id(state)]["public_hand_models"]["C"]["attacks"]
    c_shi_attacks = c_attacks["1"]
    c_attacks["1"] = 0
    assert agent._conditional_shi_royal_finish_score(
        state, "D", "attack_after_block", "1", "3"
    ) is None
    c_attacks["1"] = c_shi_attacks

    chosen = agent.select_action(state, "D", state.legal_actions("D"))
    assert chosen == ("attack_after_block", "1", "3")
    assert agent.last_score_fallback_detail == "attack_conditional_shi_royal_finish_40"

    _apply_public_action(agent, state, "D", chosen)
    _apply_public_action(agent, state, "A", ("pass", None, None))
    _apply_public_action(agent, state, "B", ("pass", None, None))
    _apply_public_action(agent, state, "C", ("receive", "3", None))
    _apply_public_action(agent, state, "C", ("attack", None, "1"))

    receive_shi = agent.select_action(state, "D", state.legal_actions("D"))
    assert receive_shi == ("receive", "1", None)
    _apply_public_action(agent, state, "D", receive_shi)

    attack_gyoku = agent.select_action(state, "D", state.legal_actions("D"))
    assert attack_gyoku == ("attack", None, "8")
    _apply_public_action(agent, state, "D", attack_gyoku)
    _apply_public_action(agent, state, "A", ("pass", None, None))
    _apply_public_action(agent, state, "B", ("pass", None, None))
    _apply_public_action(agent, state, "C", ("pass", None, None))

    finish = agent.select_action(state, "D", state.legal_actions("D"))
    assert finish == ("attack_after_block", "3", "7")


if __name__ == "__main__":
    test_middle_pair_single_big_without_royal_uses_pair_pair_big()
    test_middle_pair_single_big_with_one_royal_uses_pair_big_pair()
    test_middle_pair_single_big_generalizes_to_silver_and_horse()
    test_middle_pair_plan_does_not_override_higher_attack_types()
    test_middle_pair_plan_keeps_big_piece_in_shi_counter_position()
    print("MIDDLE_PAIR_ATTACK_PLAN_TEST_OK")

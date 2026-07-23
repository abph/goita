from __future__ import annotations

from collections import Counter
from typing import List, Tuple

from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


Action = Tuple[str, str | None, str | None]


def _state(hand: List[str]) -> GoitaState:
    filler = ["1"] * 8
    return GoitaState(
        hands={"A": hand, "B": filler, "C": filler, "D": filler},
        dealer="A",
    )


def _choose_and_apply(agent: RuleBasedAgent, state: GoitaState) -> Action:
    action = agent.select_action(state, "A", state.legal_actions("A"))
    action_type, block, attack = action
    if action_type == "attack_after_block":
        assert block is not None and attack is not None
        state.apply_attack_after_block("A", block, attack)
    elif action_type == "attack":
        assert attack is not None
        state.apply_attack("A", attack)
    else:
        raise AssertionError(f"unexpected action: {action}")
    agent.on_public_action(state, "A", action)
    return action


def _return_attack_to_a(state: GoitaState) -> None:
    state.phase = "attack"
    state.turn = "A"
    state.attacker = "A"
    state.current_attack = None


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


def _attack_sequence(hand: List[str]) -> tuple[List[str], Action]:
    state = _state(hand)
    agent = RuleBasedAgent()
    agent.bind_player("A")

    attacks: List[str] = []
    first_action = _choose_and_apply(agent, state)
    assert first_action[2] is not None
    attacks.append(first_action[2])

    for _ in range(2):
        _return_attack_to_a(state)
        action = _choose_and_apply(agent, state)
        assert action[2] is not None
        attacks.append(action[2])

    return attacks, first_action


def test_two_kyosha_single_big_without_royal_uses_big_then_kyosha() -> None:
    attacks, first_action = _attack_sequence(["1", "1", "1", "2", "2", "3", "4", "7"])
    assert attacks == ["7", "2", "2"]
    assert first_action[1] != "2"


def test_two_kyosha_two_singleton_bigs_without_royal_uses_one_big_first() -> None:
    hand = ["1", "2", "7", "6", "1", "3", "1", "2"]
    attacks, first_action = _attack_sequence(hand)

    assert attacks == ["7", "2", "2"]
    assert first_action[1] not in ("2", "6")


def test_two_kyosha_single_big_with_one_royal_uses_kyosha_big_kyosha() -> None:
    attacks, first_action = _attack_sequence(["1", "1", "2", "2", "3", "4", "7", "9"])
    assert attacks == ["2", "7", "2"]
    assert first_action[1] not in ("2", "7")


def test_kakarigotae_stays_above_two_kyosha_single_big_plan() -> None:
    state = _state(["1", "1", "1", "2", "2", "3", "4", "7"])
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent._ensure_trackers(state)
    agent._track[id(state)]["ally_first_attack"] = "3"
    agent._track[id(state)]["ally_past_attacks"].add("3")

    action = agent.select_action(state, "A", state.legal_actions("A"))
    assert action[2] == "3"
    assert agent.last_decision_reason == "kakari"


def test_big_piece_is_not_treated_as_kakarigotae() -> None:
    state = _state(["1", "1", "2", "2", "3", "4", "6", "7"])
    agent = RuleBasedAgent()
    agent.bind_player("A")
    agent._ensure_trackers(state)
    agent._track[id(state)]["ally_first_attack"] = "6"
    agent._track[id(state)]["ally_past_attacks"].add("6")

    action = agent.select_action(state, "A", state.legal_actions("A"))

    assert action[2] == "7"
    assert agent.last_decision_reason != "kakari"


def test_two_kyosha_gold_pair_uses_gold_then_kyosha_without_royal() -> None:
    attacks, first_action = _attack_sequence(["1", "1", "2", "2", "3", "5", "5", "6"])
    assert attacks == ["5", "2", "2"]
    assert first_action[1] != "2"


def test_two_kyosha_gold_pair_with_one_royal_uses_kyosha_then_gold() -> None:
    state = _state(["1", "2", "2", "3", "5", "5", "6", "9"])
    agent = RuleBasedAgent()
    agent.bind_player("A")

    first = _choose_and_apply(agent, state)
    _return_attack_to_a(state)
    second = _choose_and_apply(agent, state)

    assert first[2] == "2"
    assert second[2] == "5"


def test_two_kyosha_middle_pair_royal_uses_kyosha_then_silver_after_receive() -> None:
    state = GoitaState(
        hands={
            "A": list("14218245"),
            "B": list("53169151"),
            "C": list("43124357"),
            "D": list("32111761"),
        },
        dealer="B",
    )
    agent = RuleBasedAgent()
    agent.bind_player("A")

    _apply_public_action(agent, state, "B", ("attack_after_block", "1", "5"))
    _apply_public_action(agent, state, "C", ("pass", None, None))
    _apply_public_action(agent, state, "D", ("pass", None, None))
    _apply_public_action(agent, state, "A", ("receive", "5", None))

    first = _choose_and_apply(agent, state)
    for player in "BCD":
        _apply_public_action(agent, state, player, ("pass", None, None))
    second = _choose_and_apply(agent, state)

    assert agent._track[id(state)]["special_attack_plan"] == {
        "label": "two_kyosha_middle_pair_royal",
        "sequence": ["2", "4"],
    }
    assert first[2] == "2"
    assert second == ("attack_after_block", "1", "4")
    assert agent.last_score_fallback_detail == "attack_sequence_two_kyosha_middle_pair_royal"


def test_two_kyosha_middle_pair_royal_allows_horse_pair_and_big_piece() -> None:
    agent = RuleBasedAgent()

    assert agent._two_kyosha_middle_pair_royal_attack_plan(
        Counter(["1", "2", "2", "3", "3", "6", "7", "9"])
    ) == ["2", "3"]


def test_middle_pair_single_big_without_royal_uses_pair_pair_big() -> None:
    attacks, first_action = _attack_sequence(["1", "1", "2", "3", "4", "5", "5", "7"])
    assert attacks == ["5", "5", "7"]
    assert first_action[1] not in ("5", "7")


def test_middle_pair_single_big_with_one_royal_uses_pair_big_pair() -> None:
    attacks, first_action = _attack_sequence(["1", "1", "2", "3", "5", "5", "7", "9"])
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


def test_new_plans_do_not_override_higher_attack_types() -> None:
    agent = RuleBasedAgent()
    assert agent._two_kyosha_gold_pair_attack_plan(
        Counter(["1", "2", "2", "2", "5", "5", "6", "9"])
    ) is None
    assert agent._middle_pair_single_big_attack_plan(
        Counter(["1", "2", "3", "5", "5", "5", "7", "9"])
    ) is None
    assert agent._middle_pair_single_big_attack_plan(
        Counter(["1", "2", "3", "4", "5", "5", "7", "7"])
    ) is None
    assert agent._middle_pair_single_big_attack_plan(
        Counter(["1", "2", "3", "4", "4", "5", "5", "7"])
    ) is None


def test_plan_is_limited_to_the_requested_hand_shapes() -> None:
    agent = RuleBasedAgent()
    assert agent._two_kyosha_single_big_attack_plan(
        Counter(["1", "1", "2", "2", "3", "4", "6", "8"])
    ) == ["2", "6", "2"]
    assert agent._two_kyosha_single_big_attack_plan(
        Counter(["1", "2", "2", "2", "3", "4", "6", "8"])
    ) is None
    assert agent._two_kyosha_single_big_attack_plan(
        Counter(["1", "2", "2", "3", "3", "4", "6", "8"])
    ) is None
    assert agent._two_kyosha_single_big_attack_plan(
        Counter(["1", "2", "2", "3", "4", "6", "7", "8"])
    ) is None
    assert agent._two_kyosha_single_big_attack_plan(
        Counter(["2", "2", "3", "4", "5", "6", "8", "9"])
    ) is None


if __name__ == "__main__":
    test_two_kyosha_single_big_without_royal_uses_big_then_kyosha()
    test_two_kyosha_two_singleton_bigs_without_royal_uses_one_big_first()
    test_two_kyosha_single_big_with_one_royal_uses_kyosha_big_kyosha()
    test_kakarigotae_stays_above_two_kyosha_single_big_plan()
    test_big_piece_is_not_treated_as_kakarigotae()
    test_two_kyosha_gold_pair_uses_gold_then_kyosha_without_royal()
    test_two_kyosha_gold_pair_with_one_royal_uses_kyosha_then_gold()
    test_two_kyosha_middle_pair_royal_uses_kyosha_then_silver_after_receive()
    test_two_kyosha_middle_pair_royal_allows_horse_pair_and_big_piece()
    test_middle_pair_single_big_without_royal_uses_pair_pair_big()
    test_middle_pair_single_big_with_one_royal_uses_pair_big_pair()
    test_middle_pair_single_big_generalizes_to_silver_and_horse()
    test_middle_pair_plan_keeps_big_piece_in_shi_counter_position()
    test_new_plans_do_not_override_higher_attack_types()
    test_plan_is_limited_to_the_requested_hand_shapes()
    print("TWO_KYOSHA_BIG_PLAN_TEST_OK")

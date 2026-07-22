from __future__ import annotations

from typing import Dict, Tuple

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


def _tracker(agent: RuleBasedAgent, state: GoitaState) -> dict:
    agent._ensure_trackers(state)
    return agent._track[id(state)]


def test_initial_self_hand_constrains_all_other_hands() -> None:
    state = GoitaState(
        hands={
            "A": ["1", "1", "1", "1", "2", "3", "6", "8"],
            "B": ["1", "1", "2", "2", "3", "4", "5", "7"],
            "C": ["1", "1", "2", "3", "4", "5", "6", "9"],
            "D": ["1", "1", "3", "4", "4", "5", "5", "7"],
        },
        dealer="C",
    )
    agent = RuleBasedAgent()
    agent.bind_player("A")
    tr = _tracker(agent, state)

    for other in ("B", "C", "D"):
        assert tr["estimated_current_hands"][other]["8"]["max"] == 0
    bishop_expected = sum(
        float(tr["estimated_current_hands"][other]["6"]["expected"])
        for other in ("B", "C", "D")
    )
    assert abs(bishop_expected - 1.0) < 0.01
    assert tr["unknown_piece_pool"]["6"] == 1


def test_first_big_piece_attack_allows_single_or_pair() -> None:
    agent = RuleBasedAgent()
    assert agent._first_attack_count_range("6") == (1, 2)
    assert agent._first_attack_count_range("7") == (1, 2)


def test_visible_attack_removes_last_unknown_piece_from_all_hands() -> None:
    state = GoitaState(
        hands={
            "A": ["1", "1", "1", "1", "2", "3", "6", "8"],
            "B": ["1", "1", "2", "2", "3", "4", "5", "7"],
            "C": ["1", "1", "2", "3", "4", "5", "6", "9"],
            "D": ["1", "1", "3", "4", "4", "5", "5", "7"],
        },
        dealer="C",
    )
    agent = RuleBasedAgent()
    agent.bind_player("A")
    before_revision = int(_tracker(agent, state)["piece_inference_revision"])

    _apply_public_action(agent, state, "C", ("attack_after_block", "1", "6"))
    tr = _tracker(agent, state)
    assert tr["piece_inference_revision"] == before_revision + 1
    assert tr["last_piece_inference_reason"] == "C:attack:6"
    assert tr["unknown_piece_pool"]["6"] == 0
    for other in ("B", "C", "D"):
        assert tr["estimated_current_hands"][other]["6"]["max"] == 0
        assert agent._estimate_remaining_range(tr, other, "6") == (0, 0)


def test_visible_receive_and_attack_reconcile_the_public_pool() -> None:
    state = GoitaState(
        hands={
            "A": ["1", "1", "1", "1", "1", "1", "5", "5"],
            "B": ["1", "1", "1", "1", "1", "1", "1", "5"],
            "C": ["1", "1", "1", "1", "1", "1", "1", "5"],
            "D": ["1"] * 8,
        },
        dealer="B",
    )
    agent = RuleBasedAgent()
    agent.bind_player("A")

    _apply_public_action(agent, state, "B", ("attack_after_block", "1", "5"))
    _apply_public_action(agent, state, "C", ("receive", "5", None))
    tr = _tracker(agent, state)
    assert tr["unknown_piece_pool"]["5"] == 0
    assert tr["last_piece_inference_reason"] == "C:receive:5"
    for other in ("B", "C", "D"):
        assert tr["estimated_current_hands"][other]["5"]["max"] == 0


def _partner_rejection_state(
    first_attack: str,
    partner_return: str,
    *,
    original_attacker_switches: bool = True,
) -> Tuple[RuleBasedAgent, GoitaState]:
    state = GoitaState(
        hands={
            "A": [partner_return, "3", "1", "1", "1", "1", "1", "1"],
            "B": ["1", first_attack, first_attack, "3", "2", "1", "1", "1"],
            "C": ["1"] * 8,
            "D": [first_attack, partner_return, "1", "1", "1", "1", "1", "1"],
        },
        dealer="B",
    )
    agent = RuleBasedAgent()
    agent.bind_player("A")

    _apply_public_action(agent, state, "B", ("attack_after_block", "1", first_attack))
    _apply_public_action(agent, state, "C", ("pass", None, None))
    _apply_public_action(agent, state, "D", ("receive", first_attack, None))
    _apply_public_action(agent, state, "D", ("attack", None, partner_return))
    if not original_attacker_switches:
        return agent, state
    _apply_public_action(agent, state, "A", ("receive", partner_return, None))
    _apply_public_action(agent, state, "A", ("attack", None, "3"))
    _apply_public_action(agent, state, "B", ("receive", "3", None))
    _apply_public_action(agent, state, "B", ("attack", None, "2"))
    return agent, state


def test_partner_rejection_alone_does_not_break_first_strategy() -> None:
    agent, state = _partner_rejection_state(
        "5",
        "4",
        original_attacker_switches=False,
    )
    tr = _tracker(agent, state)
    model = tr["public_hand_models"]["B"]

    assert model["partner_first_strategy_reaction"]["status"] == "rejected"
    assert model["strategy_broken"] is False
    assert tr["other_first_attack_strategy_by_player"]["B"]["active"] is True


def test_partner_kakarigotae_rejection_then_switch_breaks_first_strategy() -> None:
    agent, state = _partner_rejection_state("5", "4")
    tr = _tracker(agent, state)
    model = tr["public_hand_models"]["B"]
    reaction = model["partner_first_strategy_reaction"]

    assert reaction["status"] == "rejected"
    assert reaction["reason"] == "kakarigotae_not_returned"
    assert model["strategy_broken"] is True
    assert model["strategy_broken_on_attack"] == "2"
    assert tr["other_first_attack_strategy_by_player"]["B"]["active"] is False
    assert tr["other_piece_count_estimates"]["B"]["5"]["source"] == "strategy_broken_after_partner_rejection"
    assert agent._opponent_first_attack_strategy_penalty(tr, "A", "5") == 0.0


def test_partner_shi_rejection_then_switch_breaks_first_strategy() -> None:
    agent, state = _partner_rejection_state("1", "4")
    model = _tracker(agent, state)["public_hand_models"]["B"]

    assert model["partner_first_strategy_reaction"]["reason"] == "shi_attack_rejected"
    assert model["strategy_broken"] is True


def test_returning_same_piece_keeps_first_strategy_active() -> None:
    agent, state = _partner_rejection_state("5", "5")
    tr = _tracker(agent, state)
    model = tr["public_hand_models"]["B"]

    assert model["partner_first_strategy_reaction"]["status"] == "accepted"
    assert model["strategy_broken"] is False
    assert tr["other_first_attack_strategy_by_player"]["B"]["active"] is True


def _shi_pass_state() -> Tuple[RuleBasedAgent, GoitaState]:
    state = GoitaState(
        hands={
            "A": ["2", "2", "3", "3", "4", "4", "5", "5"],
            "B": ["1", "1", "1", "1", "2", "3", "4", "5"],
            "C": ["1", "1", "2", "3", "4", "5", "6", "7"],
            "D": ["1", "1", "1", "1", "2", "3", "4", "5"],
        },
        dealer="B",
    )
    agent = RuleBasedAgent()
    agent.bind_player("A")
    _apply_public_action(agent, state, "B", ("attack_after_block", "2", "1"))
    _apply_public_action(agent, state, "C", ("pass", None, None))
    return agent, state


def test_passing_shi_caps_current_shi_count_at_one() -> None:
    agent, state = _shi_pass_state()
    tr = _tracker(agent, state)
    estimate = tr["estimated_current_hands"]["C"]["1"]

    assert estimate["min"] == 0
    assert estimate["max"] == 1
    assert estimate["source"] == "pass_shi_current_0_1"
    assert tr["public_hand_models"]["C"]["shi_pass_current_range"] == {"min": 0, "max": 1}


def test_using_shi_after_pass_reduces_current_cap_to_zero() -> None:
    agent, state = _shi_pass_state()
    _apply_public_action(agent, state, "D", ("pass", None, None))
    _apply_public_action(agent, state, "A", ("pass", None, None))
    _apply_public_action(agent, state, "B", ("attack_after_block", "3", "1"))
    _apply_public_action(agent, state, "C", ("receive", "1", None))
    tr = _tracker(agent, state)

    assert tr["current_piece_count_caps"]["C"]["1"]["active"] is True
    assert tr["estimated_current_hands"]["C"]["1"]["max"] == 0


def test_visible_contradiction_releases_shi_pass_cap() -> None:
    agent, state = _shi_pass_state()
    _apply_public_action(agent, state, "D", ("pass", None, None))
    _apply_public_action(agent, state, "A", ("pass", None, None))
    _apply_public_action(agent, state, "B", ("attack_after_block", "3", "1"))
    _apply_public_action(agent, state, "C", ("receive", "1", None))
    _apply_public_action(agent, state, "C", ("attack", None, "1"))
    tr = _tracker(agent, state)

    assert tr["current_piece_count_caps"]["C"]["1"]["active"] is False
    assert tr["other_piece_count_estimates"]["C"]["1"]["source"] == "observed_after_shi_pass_contradiction"


def _estimate_after_hidden_block(block: str) -> Dict[str, Dict[str, Dict[str, object]]]:
    c_hand = ["1", "2", "3", "4", "5", "6", "7", "9"]
    state = GoitaState(
        hands={
            "A": ["1", "1", "1", "1", "2", "3", "6", "8"],
            "B": ["1"] * 8,
            "C": c_hand,
            "D": ["1"] * 8,
        },
        dealer="C",
    )
    agent = RuleBasedAgent()
    agent.bind_player("A")
    _apply_public_action(agent, state, "C", ("attack_after_block", block, "7"))
    return agent._track[id(state)]["estimated_current_hands"]


def test_hidden_block_identity_is_not_used() -> None:
    estimate_a = _estimate_after_hidden_block("2")
    estimate_b = _estimate_after_hidden_block("5")
    assert estimate_a == estimate_b


def test_joint_inference_reconstructs_deck_consistent_public_sequence() -> None:
    state = GoitaState(
        hands={
            "A": ["1", "1", "2", "3", "4", "5", "6", "6"],
            "B": ["1", "1", "3", "3", "4", "5", "7", "9"],
            "C": ["1", "1", "1", "2", "4", "5", "5", "7"],
            "D": ["1", "1", "1", "2", "2", "3", "4", "8"],
        },
        dealer="D",
    )
    agent = RuleBasedAgent()
    agent.bind_player("A")

    actions = [
        ("D", ("attack_after_block", "1", "2")),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("receive", "2", None)),
        ("C", ("attack", None, "5")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("attack_after_block", "1", "5")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("receive", "5", None)),
        ("B", ("attack", None, "3")),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("receive", "3", None)),
        ("A", ("attack", None, "6")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("attack_after_block", "1", "6")),
        ("B", ("receive", "9", None)),
        ("B", ("attack", None, "7")),
        ("C", ("receive", "7", None)),
        ("C", ("attack", None, "4")),
    ]
    for player, action in actions:
        _apply_public_action(agent, state, player, action)

    joint = _tracker(agent, state)["joint_hand_inference"]
    assert joint["feasible"] is True
    expected_original = {
        "B": {"1": 2, "2": 0, "3": 2, "4": 1, "5": 1, "6": 0, "7": 1, "8": 0, "9": 1},
        "C": {"1": 3, "2": 1, "3": 0, "4": 1, "5": 2, "6": 0, "7": 1, "8": 0, "9": 0},
        "D": {"1": 3, "2": 2, "3": 1, "4": 1, "5": 0, "6": 0, "7": 0, "8": 1, "9": 0},
    }
    assert joint["map_original_counts"] == expected_original
    assert joint["map_current_counts"]["B"] == {
        "1": 2, "2": 0, "3": 1, "4": 1, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0,
    }
    assert joint["map_current_counts"]["C"] == {
        "1": 2, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0,
    }
    assert joint["map_current_counts"]["D"] == {
        "1": 2, "2": 1, "3": 1, "4": 1, "5": 0, "6": 0, "7": 0, "8": 1, "9": 0,
    }


if __name__ == "__main__":
    test_initial_self_hand_constrains_all_other_hands()
    test_first_big_piece_attack_allows_single_or_pair()
    test_visible_attack_removes_last_unknown_piece_from_all_hands()
    test_visible_receive_and_attack_reconcile_the_public_pool()
    test_partner_rejection_alone_does_not_break_first_strategy()
    test_partner_kakarigotae_rejection_then_switch_breaks_first_strategy()
    test_partner_shi_rejection_then_switch_breaks_first_strategy()
    test_returning_same_piece_keeps_first_strategy_active()
    test_passing_shi_caps_current_shi_count_at_one()
    test_using_shi_after_pass_reduces_current_cap_to_zero()
    test_visible_contradiction_releases_shi_pass_cap()
    test_hidden_block_identity_is_not_used()
    test_joint_inference_reconstructs_deck_consistent_public_sequence()
    print("PUBLIC_PIECE_INFERENCE_TEST_OK")

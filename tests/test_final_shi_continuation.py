from copy import deepcopy

import pytest

from goita_ai2.experimental_ai2 import RuleBasedAgent
from goita_ai2.intermediate_middle3 import RuleBasedAgent as SavedAgent
from goita_ai2.state import GoitaState


def replay(agent, *, hidden_shi_at_d=False, own_kyosha=False):
    # goita_kifu_score-oK, before C's third attack. A swap changes only an
    # enemy's hidden block and another enemy's concealed hand, not C's view.
    hands = dict(A="11123455", B="11223448", C="11135567", D="11234679")
    if hidden_shi_at_d:
        hands.update(B="12223448", D="11134679")
    if own_kyosha:
        hands.update(B="11233448", C="11125567")
    state = GoitaState({seat: list(hand) for seat, hand in hands.items()}, dealer="B")
    agent.bind_player("C")
    agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    agent._ensure_trackers(state)
    b_block = "2" if hidden_shi_at_d else "1"
    for seat, block, attack in (
        ("B", "1", "4"), ("C", "p", ""), ("D", "p", ""), ("A", "p", ""),
        ("B", b_block, "4"), ("C", "p", ""), ("D", "p", ""),
        ("A", "4", ""), ("A", "", "1"), ("B", "p", ""), ("C", "p", ""),
        ("D", "1", ""), ("D", "", "7"), ("A", "p", ""), ("B", "p", ""),
        ("C", "7", ""), ("C", "", "1"), ("D", "1", ""), ("D", "", "6"),
        ("A", "p", ""), ("B", "p", ""), ("C", "6", ""), ("C", "", "1"),
        ("D", "p", ""), ("A", "p", ""), ("B", "p", ""),
    ):
        if block == "p":
            action = ("pass", None, None)
        elif block and attack:
            action = ("attack_after_block", block, attack)
        elif block:
            action = ("receive", block, None)
        else:
            action = ("attack", None, attack)
        assert state.turn == seat and action in state.legal_actions(seat)
        if action[0] == "pass":
            state.apply_pass(seat)
        elif action[0] == "receive":
            state.apply_receive(seat, block)
        elif action[0] == "attack":
            state.apply_attack(seat, attack)
        else:
            state.apply_attack_after_block(seat, block, attack)
        agent.on_public_action(state, seat, action)
    return state


@pytest.mark.parametrize("risk,block,cover,points", [
    (False, "5", "35", 30),
    (True, "3", "5", 60),
])
def test_last_shi_stays_the_attack_and_style_decides_the_block(risk, block, cover, points):
    agent = RuleBasedAgent()
    assert agent.final_shi_risk is False
    agent.final_shi_risk = risk
    state = replay(agent)

    chosen = agent.select_action(state, "C", state.legal_actions("C"))

    assert chosen == ("attack_after_block", block, "1")
    assert agent.last_decision_reason == "shi_attack_package"
    assert agent.last_rule_search_authority == "strong"
    style = "risk" if risk else "safe"
    assert agent.last_score_fallback_detail == (
        f"shi_package_continuation_{style}_enemy_1_cover_{cover}_finish_{points}"
    )
    assert agent.last_attack_candidate_snapshot["chosen"]["block"] == block
    assert agent.last_neural_shadow["applied"] is False

    state.apply_attack_after_block("C", block, "1")
    for seat in "DAB":
        state.apply_pass(seat)
    last_block = "5" if risk else "3"
    state.apply_attack_after_block("C", last_block, "5")
    assert state.winner == "C" and state.team_score["AC"] == points


def test_concealed_opponent_cards_do_not_change_the_decision():
    choices = []
    for swap in (False, True):
        agent = RuleBasedAgent()
        state = replay(agent, hidden_shi_at_d=swap)
        assert state.hands["D"].count("1") == int(swap)
        choice = agent._shi_attack_package_action(
            state, "C", state.legal_actions("C"), has_non_king_attack_option=True,
        )
        assert choice is not None
        choices.append(deepcopy(choice))
    assert choices[0] == choices[1]


def test_extension_defers_when_immediate_enemy_finish_is_possible_or_last_shi_did_not_go_around():
    agent = RuleBasedAgent()
    state = replay(agent)

    def choose():
        return agent._shi_attack_package_action(
            state, "C", state.legal_actions("C"), has_non_king_attack_option=True,
        )
    assert choose() is not None
    # Enemy hand size is public. A ready enemy must be assessed by the
    # established endgame rules, even when its previous response was a pass.
    hand = state.hands["D"]
    state.hands["D"] = hand[:2]
    assert choose() is None
    state.hands["D"] = hand
    agent._neural_public_history_by_state_id[id(state)].pop()
    assert choose() is None


def test_extension_does_not_dismiss_a_legal_kyosha_attack_as_royal_receivable():
    agent = RuleBasedAgent()
    state = replay(agent, own_kyosha=True)
    actions = state.legal_actions("C")
    assert any(action[2] == "2" for action in actions)
    result = agent._shi_attack_package_action(
        state, "C", actions, has_non_king_attack_option=True,
    )
    assert agent.last_shi_attack_package_analysis["enemy_map_remaining"] == 1
    assert result is None


def test_saved_middle_three_keeps_its_original_decision():
    agent = SavedAgent()
    state = replay(agent)
    assert agent.select_action(state, "C", state.legal_actions("C")) == (
        "attack_after_block", "3", "5",
    )

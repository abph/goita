"""Public-information regression for the C turn 13 review, plus counterexamples."""

import copy

import pytest

from goita_ai2.current_ai.agent import RuleBasedAgent
from goita_ai2.state import GoitaState


HANDS = {"A": "13345679", "B": "11122456", "C": "11123458", "D": "15314172"}
PREFIX = (
    ("C", ("attack_after_block", "3", "1")),
    ("D", ("receive", "1", None)), ("D", ("attack", None, "7")),
    ("A", ("receive", "7", None)), ("A", ("attack", None, "3")),
    ("B", ("pass", None, None)), ("C", ("pass", None, None)),
    ("D", ("receive", "3", None)), ("D", ("attack", None, "4")),
    ("A", ("receive", "4", None)), ("A", ("attack", None, "3")),
    ("B", ("pass", None, None)), ("C", ("pass", None, None)),
    ("D", ("pass", None, None)), ("A", ("attack_after_block", "5", "6")),
    ("B", ("receive", "6", None)), ("B", ("attack", None, "4")),
)
PASS = ("pass", None, None)


def review_position(rotation=0, swap_royals=False):
    seats = "ABCD"
    seat = lambda p: seats[(seats.index(p) + rotation) % 4]
    piece = lambda p: ({"8": "9", "9": "8"}.get(p, p) if swap_royals else p)
    state = GoitaState({seat(p): [piece(v) for v in hand] for p, hand in HANDS.items()}, dealer=seat("C"))
    agent = RuleBasedAgent()
    player = seat("C")
    agent.bind_player(player)
    agent._ensure_trackers(state)
    agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    agent.TIME_SEARCH_CACHE_ENABLED = False
    agent.TIME_SEARCH_PREDICTION_CACHE_ENABLED = False
    agent.TIME_SEARCH_ADAPTIVE_BUDGET_ENABLED = False
    # Wall-clock speed must not change the candidate set in deterministic tests.
    agent.ALLY_REACH_HANDOFF_MAX_SECONDS = 10.0
    for source, recorded in PREFIX:
        source = seat(source)
        kind, block, attack = tuple(piece(p) for p in recorded)
        action = (kind, block, attack)
        assert action in state.legal_actions(source)
        if kind == "pass":
            state.apply_pass(source)
        elif kind == "receive":
            state.apply_receive(source, block)
        elif kind == "attack":
            state.apply_attack(source, attack)
        else:
            state.apply_attack_after_block(source, block, attack)
        agent.on_public_action(state, source, action)
    return state, agent, player


def test_review_turn13_passes_before_selecting_an_attack():
    state, agent, player = review_position()
    assert state.hands[player] == list("112458")
    assert agent.select_action(state, player, state.legal_actions(player)) == PASS
    detail = agent.last_score_fallback_detail
    assert detail.startswith("pass_ally_reach_royal_A_piece_4_chance_")
    assert agent._rule_search_authority(agent.last_decision_reason, detail) == "strong"
    comparison = agent.last_ally_reach_comparison
    assert comparison["adopted"] and not comparison["royal_known"]
    rows = comparison["comparisons"]
    gold = next(r for r in rows if r["receive"] == ("receive", "4", None) and r["attack"] == ("attack", None, "5"))
    lance = next(r for r in rows if r["receive"] == ("receive", "4", None) and r["attack"] == ("attack", None, "2"))
    assert gold["keeps_cover"] and not lance["keeps_cover"]
    assert rows[0]["ally_finish"] > gold["ally_finish"]
    assert rows[0]["interception"] > 0 and rows[0]["both_pass"] > 0
    assert rows[0]["enemy_finish"] == 0
    # The alternative royal receive + silver return is compared too.
    assert any(r["receive"] == ("receive", "8", None) and r["attack"] == ("attack", None, "4") for r in rows)


@pytest.mark.parametrize("swap_royals", (False, True))
def test_handoff_does_not_depend_on_which_royal_is_named_king(swap_royals):
    state, agent, player = review_position(0, swap_royals)
    assert agent._ally_reach_handoff_action(state, player, state.legal_actions(player)) == PASS


def test_actual_hidden_royal_location_cannot_change_the_comparison():
    state, agent, player = review_position()
    assert agent._ally_reach_handoff_action(state, player, state.legal_actions(player)) == PASS
    original = copy.deepcopy(agent.last_ally_reach_comparison)
    tracker_before = copy.deepcopy(agent._track[id(state)])
    # Same own hand, public history and opponent counts; king is really held by B.
    state.hands["A"][1], state.hands["B"][0] = state.hands["B"][0], state.hands["A"][1]
    assert agent._ally_reach_handoff_action(state, player, state.legal_actions(player)) == PASS
    assert agent.last_ally_reach_comparison == original
    assert agent._track[id(state)] == tracker_before


@pytest.mark.parametrize("unsafe_change", ("ally_not_ready", "no_lance", "silver_not_exhausted", "enemy_next_ready", "attacker_ready", "no_ally_royal"))
def test_does_not_probe_without_safety_conditions(unsafe_change, monkeypatch):
    state, agent, player = review_position()
    tr = agent._track[id(state)]
    if unsafe_change == "ally_not_ready":
        tr["ally_consumed_count"] = 4
    elif unsafe_change == "no_lance":
        state.hands[player].remove("2")
    elif unsafe_change == "silver_not_exhausted":
        tr["public_seen_counts"]["4"] -= 1
    elif unsafe_change == "enemy_next_ready":
        state.hands["D"] = state.hands["D"][:2]
    elif unsafe_change == "attacker_ready":
        state.hands["B"] = state.hands["B"][:2]
    else:
        monkeypatch.setattr(agent, "_estimated_piece_hold_risk", lambda *args: 0.0)
    assert agent._ally_reach_handoff_action(state, player, state.legal_actions(player)) is None
    assert agent.last_ally_reach_comparison is None


def test_no_comparison_if_sampling_deadline_expires():
    state, agent, player = review_position()
    agent.ALLY_REACH_HANDOFF_MAX_SECONDS = 0
    assert agent._ally_reach_handoff_action(state, player, state.legal_actions(player)) is None


def test_interception_and_all_pass_are_not_counted_as_partner_wins():
    _, agent, _ = review_position()
    hands = {"A": list("19"), "B": list("1255"), "C": list("112458"), "D": list("1125")}
    assert agent._ally_reach_offer_outcome(hands, "C", "B", "4")["ally_finish"] == 1
    hands["D"] = list("29")
    denied = agent._ally_reach_offer_outcome(hands, "C", "B", "4")
    assert denied["ally_finish"] == 0 and denied["enemy_finish"] == 1
    hands["D"], hands["A"], hands["B"] = list("1125"), list("11"), list("55")
    all_pass = agent._ally_reach_offer_outcome(hands, "C", "B", "4")
    assert all_pass["ally_finish"] == 0 and all_pass["both_pass"] == 1 and all_pass["enemy_finish"] == 1


def test_immediate_self_finish_has_priority_over_handoff(monkeypatch):
    state, agent, player = review_position()
    state.hands[player] = list("48")
    def unexpected(*args):
        pytest.fail("Immediate finish must be decided before the handoff tactic")
    monkeypatch.setattr(agent, "_ally_reach_handoff_action", unexpected)
    assert agent._select_rule_based_action(state, player, state.legal_actions(player))[0] == "receive"

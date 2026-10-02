from __future__ import annotations

from typing import Optional, Tuple

from goita_ai2.experimental_ai2 import RuleBasedAgent as ExperimentalAI2Agent
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


Action = Tuple[str, Optional[str], Optional[str]]


def _apply(
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


def _round3_c_turn13(
    *,
    experimental: bool = False,
) -> tuple[RuleBasedAgent, GoitaState]:
    """Reproduce the public history before 2026-10-02 round 3, C turn 13."""
    state = GoitaState(
        {
            "A": list("51115377"),
            "B": list("32193141"),
            "C": list("45611812"),
            "D": list("15232644"),
        },
        dealer="C",
    )
    agent = ExperimentalAI2Agent() if experimental else RuleBasedAgent()
    agent.bind_player("C")
    agent._ensure_trackers(state)
    if not experimental:
        agent.TIME_SEARCH_ENABLED = False
        agent.BRANCHED_ATTACK_ENABLED = False
    history = (
        ("C", ("attack_after_block", "4", "1")),
        ("D", ("pass", None, None)),
        ("A", ("receive", "1", None)),
        ("A", ("attack", None, "7")),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("attack_after_block", "5", "1")),
        ("B", ("receive", "1", None)),
        ("B", ("attack", None, "3")),
        ("C", ("pass", None, None)),
        ("D", ("pass", None, None)),
        ("A", ("receive", "3", None)),
        ("A", ("attack", None, "7")),
        ("B", ("receive", "9", None)),
        ("B", ("attack", None, "3")),
        ("C", ("receive", "8", None)),
    )
    for player, action in history:
        _apply(agent, state, player, action)
    return agent, state


def test_ally_second_attack_shi_claim_creates_two_attempt_relay() -> None:
    agent, state = _round3_c_turn13()
    tracker = agent._track[id(state)]

    claim = tracker["ally_shi_reserve_claim"]
    assert tracker["ally_attack_history"] == ["7", "1", "7"]
    assert tracker["ally_received_my_shi_count"] == 1
    assert claim == {
        "source": "received_my_shi_then_second_attack_shi",
        "initial_shi_claim": 3,
        "claim_attack_number": 2,
        "preceding_attack_piece": "7",
        "preceding_attack_was_big": True,
        "confidence": "strategy_signal",
        "public_shi_spent": 2,
        "likely_remaining_shi": 1,
        "ally_hand_size": 2,
        "status": "active",
    }

    chosen = agent.select_action(state, "C", state.legal_actions("C"))

    assert chosen == ("attack", None, "1")
    assert agent.last_decision_reason == "ally_shi_relay"
    assert agent.last_score_fallback_detail == (
        "attack_ally_reach_shi_relay_claim3_two_attempts"
    )
    planned = tracker["planned_attack_intent"]
    assert planned["kind"] == "relay_shi_to_ally"
    assert planned["target_team"] == "ally"
    assert planned["evidence"]["reserved_retry_shi"] == 1


def test_intercepted_relay_keeps_second_shi_until_it_reaches_ally() -> None:
    agent, state = _round3_c_turn13()
    tracker = agent._track[id(state)]
    first_shi = agent.select_action(state, "C", state.legal_actions("C"))
    _apply(agent, state, "C", first_shi)

    _apply(agent, state, "D", ("receive", "1", None))
    intent = tracker["attack_intent"]
    assert intent["status"] == "unresolved"
    assert intent["unresolved_reason"] == "intervening_enemy_spent_shi"
    assert intent["enemy_interception_count"] == 1

    continuation = (
        ("D", ("attack", None, "4")),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("pass", None, None)),
        ("D", ("attack_after_block", "5", "2")),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
    )
    for player, action in continuation:
        _apply(agent, state, player, action)

    receive_kyosha = agent.select_action(state, "C", state.legal_actions("C"))
    assert receive_kyosha == ("receive", "2", None)
    _apply(agent, state, "C", receive_kyosha)
    second_shi = agent.select_action(state, "C", state.legal_actions("C"))
    assert second_shi == ("attack", None, "1")
    assert agent.last_decision_reason == "inferred_endgame"
    assert agent.last_score_fallback_detail == (
        "inferred_endgame_followup_attack_ally_shi_relay_retry"
    )
    _apply(agent, state, "C", second_shi)
    _apply(agent, state, "D", ("pass", None, None))
    _apply(agent, state, "A", ("receive", "1", None))

    assert tracker["attack_intent"] is None
    resolved = tracker["attack_intent_history"][-1]
    assert resolved["status"] == "achieved"
    assert resolved["resolution_reason"] == "target_received"


def test_relay_plan_is_not_created_when_next_enemy_is_in_reach() -> None:
    agent, state = _round3_c_turn13()
    state.hands["D"] = state.hands["D"][:2]

    assert agent._ally_shi_relay_context(
        state,
        "C",
        state.legal_actions("C"),
    ) is None


def test_experimental_ai2_search_and_neural_keep_the_relay_choice() -> None:
    agent, state = _round3_c_turn13(experimental=True)

    chosen = agent.select_action(state, "C", state.legal_actions("C"))

    assert chosen == ("attack", None, "1")
    assert agent.last_decision_reason == "ally_shi_relay"
    assert agent.last_rule_search_authority == "strong"
    assert agent.last_neural_shadow["recommended_action"] == [
        "attack",
        None,
        "1",
    ]
    assert agent.last_neural_shadow["match"] is True
    search = agent._track[id(state)]["last_time_limited_search"]
    assert search["action"] == ("attack", None, "1")

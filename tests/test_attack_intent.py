from __future__ import annotations

from typing import Tuple

from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.state import GoitaState


Action = Tuple[str, str | None, str | None]
INITIAL_HANDS = {
    "A": ["4", "3", "3", "9", "5", "1", "2", "1"],
    "B": ["5", "4", "5", "1", "8", "1", "4", "1"],
    "C": ["1", "2", "6", "6", "1", "3", "1", "3"],
    "D": ["7", "2", "1", "4", "1", "7", "2", "5"],
}
PREFIX = (
    ("A", ("attack_after_block", "1", "3")),
    ("B", ("pass", None, None)),
    ("C", ("receive", "3", None)),
    ("C", ("attack", None, "3")),
    ("D", ("pass", None, None)),
    ("A", ("pass", None, None)),
    ("B", ("pass", None, None)),
    ("C", ("attack_after_block", "1", "6")),
    ("D", ("pass", None, None)),
    ("A", ("pass", None, None)),
    ("B", ("receive", "8", None)),
    ("B", ("attack", None, "4")),
    ("C", ("pass", None, None)),
)
AFTER_FIRST_ATTACK = (
    ("A", ("pass", None, None)),
    ("B", ("pass", None, None)),
    ("C", ("pass", None, None)),
)


def _apply(agent: RuleBasedAgent, state: GoitaState, player: str, action: Action) -> None:
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


def _reach_d_first_attack() -> tuple[RuleBasedAgent, GoitaState, dict]:
    state = GoitaState(hands=INITIAL_HANDS, dealer="A")
    agent = RuleBasedAgent()
    agent.bind_player("D")
    agent.TIME_SEARCH_ENABLED = False
    agent.BRANCHED_ATTACK_ENABLED = False

    # D が銀を受ける直前までを棋譜どおりに公開する。
    for player, action in PREFIX:
        _apply(agent, state, player, action)

    receive = agent.select_action(state, "D", state.legal_actions("D"))
    assert receive == ("receive", "4", None)
    _apply(agent, state, "D", receive)
    intent = agent._track[id(state)]["planned_attack_intent"]
    assert intent["kind"] == "force_enemy_royal"
    assert intent["attack_piece"] == "7"

    first_attack = agent.select_action(state, "D", state.legal_actions("D"))
    assert first_attack[2] == "7"
    _apply(agent, state, "D", first_attack)
    return agent, state, {}


def test_attack_intent_reconsiders_fixed_sequence_after_all_passes() -> None:
    agent, state, _report = _reach_d_first_attack()
    tracker = agent._track[id(state)]
    assert tracker["attack_intent"]["status"] == "created"
    assert tracker["attack_intent"]["continuation_count"] == 0

    for player, action in AFTER_FIRST_ATTACK:
        _apply(agent, state, player, action)

    assert tracker["attack_intent"]["status"] == "unresolved"
    chosen = agent.select_action(state, "D", state.legal_actions("D"))

    assert chosen == ("attack_after_block", "1", "7")
    assert agent.last_decision_reason == "attack_intent"
    assert agent.last_score_fallback_detail == (
        "attack_intent_force_enemy_royal_unresolved_continue_7"
    )
    comparison = tracker["last_attack_intent_comparison"]
    assert comparison["selected"] == "intent"
    assert comparison["baseline_action"] == ["attack_after_block", "1", "2"]
    assert comparison["intent_score"] > comparison["baseline_score"]
    assert {row["candidate_role"] for row in agent.last_attack_candidate_scores} == {
        "intent_continuation",
        "fixed_plan_baseline",
    }
    assert agent.last_attack_candidate_snapshot["attack_intent"]["selected"] == "intent"
    assert agent.last_attack_candidate_snapshot["chosen"]["candidate_role"] == (
        "intent_continuation"
    )
    assert {
        row["candidate_role"]
        for row in agent.last_attack_candidate_snapshot["alternatives"]
        if "candidate_role" in row
    } == {
        "fixed_plan_baseline",
    }


def test_attack_intent_is_archived_when_enemy_uses_royal() -> None:
    agent, state, _report = _reach_d_first_attack()
    _apply(agent, state, "A", ("receive", "9", None))

    tracker = agent._track[id(state)]
    assert tracker["attack_intent"] is None
    assert tracker["attack_intent_history"][-1]["status"] == "achieved"
    assert tracker["attack_intent_history"][-1]["resolution_reason"] == "target_received"


if __name__ == "__main__":
    test_attack_intent_reconsiders_fixed_sequence_after_all_passes()
    test_attack_intent_is_archived_when_enemy_uses_royal()
    print("ATTACK_INTENT_TEST_OK")

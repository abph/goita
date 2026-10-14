from __future__ import annotations

from goita_ai2.current_ai import RuleBasedAgent as CurrentRuleBasedAgent
from goita_ai2.experimental_ai2 import RuleBasedAgent as ExperimentalAI2RuleBasedAgent
from goita_ai2.instruction_case_audit import apply_action
from goita_ai2.state import GoitaState


REPORTED_HANDS_WITH_THREE_ENEMY_KYOSHA = {
    "A": list("15411135"),
    "B": list("91864216"),
    "C": list("73122215"),
    "D": list("17434315"),
}

REPORTED_HANDS_WITH_SPLIT_ENEMY_KYOSHA = {
    "A": list("15411135"),
    "B": list("91864216"),
    "C": list("73124145"),
    "D": list("17232315"),
}


def _agent(agent_class=ExperimentalAI2RuleBasedAgent):
    state = GoitaState(REPORTED_HANDS_WITH_THREE_ENEMY_KYOSHA, dealer="B")
    agent = agent_class()
    agent.bind_player("B")
    agent.TIME_LIMITED_SEARCH_ENABLED = False
    agent.TIME_SEARCH_ENABLED = False
    agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    if hasattr(agent, "NEURAL_TIEBREAK_ENABLED"):
        agent.NEURAL_TIEBREAK_ENABLED = False
    return state, agent


def _public(agent, state, player, action) -> None:
    assert action in state.legal_actions(player)
    apply_action(state, player, action)
    agent.on_public_action(state, player, action)


def test_strong_hand_opens_with_shi_block_and_single_kyosha() -> None:
    state, agent = _agent()

    chosen = agent.select_action(state, "B", state.legal_actions("B"))
    plan = agent._track[id(state)]["active_forced_win_plan"]

    assert agent._initial_hand_axes_for_state(state, "B")["absolute_rank"] == "S"
    assert chosen == ("attack_after_block", "1", "2")
    assert plan["trigger"] == "strong_hand_single_kyosha_exhaustion_route"
    assert plan["minimum_score"] == 50.0
    assert plan["maximum_score"] == 100.0
    assert agent.last_decision_reason == "tsume"
    assert agent.last_score_fallback_detail == (
        "high_score_strong_single_kyosha_exhaustion_safe_50_max_100"
    )


def test_three_enemy_kyosha_branch_still_finishes_for_fifty() -> None:
    state, agent = _agent()
    first = agent.select_action(state, "B", state.legal_actions("B"))
    _public(agent, state, "B", first)

    # C owns all three remaining kyosha and returns them for as long as the
    # public game permits. The selecting rule itself reads B's hand alone.
    branch = (
        ("C", ("receive", "2", None)),
        ("C", ("attack", None, "2")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("attack_after_block", "1", "2")),
        ("D", ("pass", None, None)),
        ("A", ("pass", None, None)),
        ("B", ("pass", None, None)),
        ("C", ("attack_after_block", "3", "7")),
        ("D", ("receive", "7", None)),
        ("D", ("attack", None, "4")),
        ("A", ("receive", "4", None)),
        ("A", ("attack", None, "5")),
    )
    for player, action in branch:
        _public(agent, state, player, action)

    receive = agent.select_action(state, "B", state.legal_actions("B"))
    assert receive in (("receive", "8", None), ("receive", "9", None))
    _public(agent, state, "B", receive)
    first_kaku = agent.select_action(state, "B", state.legal_actions("B"))
    assert first_kaku == ("attack", None, "6")
    _public(agent, state, "B", first_kaku)
    for player in ("C", "D", "A"):
        _public(agent, state, player, ("pass", None, None))

    second_kaku = agent.select_action(state, "B", state.legal_actions("B"))
    assert second_kaku[0] == "attack_after_block"
    assert second_kaku[2] == "6"
    _public(agent, state, "B", second_kaku)
    for player in ("C", "D", "A"):
        _public(agent, state, player, ("pass", None, None))

    finish = agent.select_action(state, "B", state.legal_actions("B"))
    assert finish[0] == "attack_after_block"
    assert finish[2] in ("8", "9")
    _public(agent, state, "B", finish)

    assert state.finished is True
    assert state.winner == "B"
    assert state.team_score["BD"] == 50


def test_opening_does_not_depend_on_hidden_enemy_kyosha_distribution() -> None:
    choices = []
    for hands in (
        REPORTED_HANDS_WITH_THREE_ENEMY_KYOSHA,
        REPORTED_HANDS_WITH_SPLIT_ENEMY_KYOSHA,
    ):
        state = GoitaState(hands, dealer="B")
        agent = ExperimentalAI2RuleBasedAgent()
        agent.bind_player("B")
        agent.TIME_LIMITED_SEARCH_ENABLED = False
        agent.TIME_SEARCH_ENABLED = False
        agent.TIME_SEARCH_BACKGROUND_ENABLED = False
        agent.NEURAL_TIEBREAK_ENABLED = False
        choices.append(agent.select_action(state, "B", state.legal_actions("B")))

    assert choices == [
        ("attack_after_block", "1", "2"),
        ("attack_after_block", "1", "2"),
    ]


def test_single_kyosha_route_requires_absolute_a_or_better(monkeypatch) -> None:
    state, agent = _agent()
    monkeypatch.setattr(
        agent,
        "_initial_hand_axes_for_state",
        lambda _state, _player: {"rank": "B", "absolute_rank": "B"},
    )

    plan = agent._initial_upside_forced_win_plan(
        state,
        "B",
        state.legal_actions("B"),
    )

    assert plan is None


def test_saved_current_ai_does_not_gain_the_new_route() -> None:
    state, agent = _agent(CurrentRuleBasedAgent)

    chosen = agent.select_action(state, "B", state.legal_actions("B"))

    assert chosen != ("attack_after_block", "1", "2")
    assert agent._track[id(state)]["active_forced_win_plan"]["trigger"] == (
        "initial_hand_fixed_route"
    )

"""Build repeat-attack plans with sufficient copies and wider receiving cover."""
from collections import Counter
from dataclasses import replace

import pytest

from goita_ai2.constants import PIECE_TOTALS
from goita_ai2.current_ai import RuleBasedAgent as CurrentAI
from goita_ai2.current_ai.branched_attack_plan import PublicPlanEventKind
from goita_ai2.experimental_ai2 import RuleBasedAgent
from goita_ai2.instruction_case_audit import apply_action
from goita_ai2.rule_based_intermediate_middle3 import RuleBasedAgent as Middle3
from goita_ai2.state import GoitaState


HANDS = {"A": list("14411176"), "B": list("18522191"),
         "C": list("32513343"), "D": list("26451517")}
HISTORY = (
    ("A", ("attack_after_block", "1", "1")),
    ("B", ("receive", "1", None)), ("B", ("attack", None, "2")),
    ("C", ("receive", "2", None)), ("C", ("attack", None, "3")),
    ("D", ("pass", None, None)), ("A", ("pass", None, None)),
    ("B", ("pass", None, None)),
)


def position(*, stop=8, offset=0, mapping=None, agent_class=RuleBasedAgent,
             run_decisions=False, enabled=True, timed=False):
    seats = {seat: "ABCD"[(index + offset) % 4] for index, seat in enumerate("ABCD")}
    mapping = mapping or {}
    hands = {seats[seat]: [mapping.get(piece, piece) for piece in hand]
             for seat, hand in HANDS.items()}
    assert Counter(piece for hand in hands.values() for piece in hand) == Counter(PIECE_TOTALS)
    state = GoitaState(hands, dealer=seats["A"])
    player = seats["C"]
    agent = agent_class()
    agent.bind_player(player)
    # Recreate the report's search-created response/attack plan first. The
    # target turn can then be checked with either search setting.
    agent.TIME_SEARCH_ENABLED = timed or run_decisions
    agent.TIME_SEARCH_MAX_SECONDS = 1
    agent.TIME_SEARCH_HARD_MAX_SECONDS = 20
    agent.TIME_SEARCH_SAMPLE_COUNT = 80
    agent.TIME_SEARCH_MAX_DEPTH = 11
    agent.TIME_SEARCH_MAX_NODES = 250000
    agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    agent.BRANCHED_ATTACK_CACHE_ENABLED = False
    agent.PLANNED_BLOCK_RESERVE_ENABLED = enabled
    agent._ensure_trackers(state)
    for actor, action in HISTORY[:stop]:
        actor = seats[actor]
        action = tuple(mapping.get(piece, piece) for piece in action)
        assert action in state.legal_actions(actor)
        if run_decisions and actor == player:
            assert agent.select_action(state, actor, state.legal_actions(actor)) == action
        apply_action(state, actor, action)
        agent.on_public_action(state, actor, action)
    agent.TIME_SEARCH_ENABLED = timed
    return state, agent, player


def candidates(state, agent, player, **kwargs):
    arguments = dict(previous_attack="3", attack_number=2, requires_block=True,
                     preferred_attack="3", limit=100)
    arguments.update(kwargs)
    return agent._branched_continuation_candidates(state, player, state.hands[player], **arguments)


def lap_node(plan, node):
    branch = next(branch for branch in node.branches
                  if branch.condition.kind == PublicPlanEventKind.LAP_COMPLETED)
    return plan.node(branch.target_node_id)


def test_correct_block_is_in_the_plan_before_first_horse_is_played():
    state, agent, player = position(stop=4)
    assert state.hands[player].count("3") == 4
    (plan,) = agent._generate_branched_attack_plans(state, player, [("attack", None, "3")])
    second = lap_node(plan, plan.node(plan.root_node_id))
    third = lap_node(plan, second)
    assert second.action == ("attack_after_block", "3", "3")
    assert Counter(second.reserved_pieces) == Counter("1345")  # after the second attack
    assert third.action[2] == "3"
    agent.PLANNED_BLOCK_RESERVE_ENABLED = False
    (legacy,) = agent._generate_branched_attack_plans(state, player, [("attack", None, "3")])
    legacy_third = lap_node(legacy, lap_node(legacy, legacy.node(legacy.root_node_id)))
    assert agent._pair_finish_score(list(third.reserved_pieces)) >= (
        agent._pair_finish_score(list(legacy_third.reserved_pieces)))


def test_promotion_occurs_before_the_beam_discards_the_alternative():
    state, agent, player = position()
    improved = candidates(state, agent, player)
    original = CurrentAI._branched_continuation_candidates(
        agent, state, player, state.hands[player], previous_attack="3", attack_number=2,
        requires_block=True, preferred_attack="3", limit=100)
    assert original[0].action == ("attack_after_block", "5", "3")
    assert improved[0].action == ("attack_after_block", "3", "3")
    assert candidates(state, agent, player, limit=1) == improved[:1]
    assert {item.action: item for item in improved} == {item.action: item for item in original}
    assert improved[0].finish_score == original[0].finish_score == 30
    assert agent._branched_receive_width(improved[0].remaining_hand) == 4
    assert agent._branched_receive_width(original[0].remaining_hand) == 3


@pytest.mark.parametrize("offset", range(4))
@pytest.mark.parametrize("mapping", [{}, {"3": "4", "4": "3"}, {"3": "5", "5": "3"}])
def test_plan_generalizes_to_other_seats_and_middle_pieces(offset, mapping):
    state, agent, player = position(stop=4, offset=offset, mapping=mapping)
    piece = mapping.get("3", "3")
    (plan,) = agent._generate_branched_attack_plans(state, player, [("attack", None, piece)])
    second = lap_node(plan, plan.node(plan.root_node_id))
    assert second.action == ("attack_after_block", piece, piece)
    assert lap_node(plan, second).action[2] == piece


@pytest.mark.parametrize("timed", [False, True])
def test_report_persisted_plan_and_execution_keep_other_receivers(timed):
    state, agent, player = position(run_decisions=True, timed=timed)
    active = agent._active_branched_attack_plan(state)
    assert active.current_node.action == ("attack_after_block", "3", "3")
    # Force the branch evaluator to exhaust its budget; the prebuilt block
    # must already be correct, even when full inference revalidation is deferred.
    agent._evaluate_branched_attack_candidates_bounded = lambda *args, **kwargs: ((), True)
    chosen = agent.select_action(state, player, state.legal_actions(player))
    assert chosen == ("attack_after_block", "3", "3")
    assert agent.last_branched_attack_metrics["continued"] is True
    assert agent.last_branched_attack_metrics["revalidation_deferred"] is True
    assert agent.last_score_fallback_detail == "plan_block_surplus_3_reserve_1_needed_1_width_4"
    evidence = agent.last_attack_candidate_snapshot["planned_block_reserve"]
    assert evidence["selected_action"] == chosen
    assert evidence["previous_action"] == ("attack_after_block", "5", "3")
    assert evidence["remaining_hand"] == list("1345")
    assert evidence["remaining_attack_copies"] == evidence["required_future_attacks"] == 1
    assert evidence["receive_width"] == 4 and evidence["previous_receive_width"] == 3
    assert evidence["projected_finish_score"] == evidence["previous_projected_finish_score"] == 30
    assert evidence["continuation_assumption"] == "attacks_return_after_a_full_lap"
    assert evidence["proven"] is False
    apply_action(state, player, chosen)
    agent.on_public_action(state, player, chosen)
    assert Counter(state.hands[player]) == Counter("1345")
    assert agent._track[id(state)]["my_attack_count"] == 2


def test_old_cached_gold_fold_is_invalidated_even_when_revalidation_times_out():
    state, agent, player = position(run_decisions=True, enabled=False)
    old = agent._active_branched_attack_plan(state)
    assert old.current_node.action == ("attack_after_block", "5", "3")
    agent.PLANNED_BLOCK_RESERVE_ENABLED = True
    agent._evaluate_branched_attack_candidates_bounded = lambda *args, **kwargs: ((), True)
    chosen = agent.select_action(state, player, state.legal_actions(player))
    assert chosen == ("attack_after_block", "3", "3")
    # Rule preview runs on a clone and transfers the chosen lifecycle back.
    assert agent._active_branched_attack_plan(state).reason == "planned_block_loses_receive_width"
    assert agent.last_branched_attack_metrics["continued"] is False
    assert agent.last_branched_attack_metrics["status"] == "fallback"
    assert agent.last_attack_candidate_snapshot["planned_block_reserve"]["selected_action"] == chosen


def test_hidden_opponent_cards_do_not_change_plan_or_review():
    state, agent, player = position()
    first = candidates(state, agent, player)
    evidence = agent._live_planned_block_review(state, player, first[0].action)
    other = [seat for seat in "ABCD" if seat != player]
    # Exchange hidden cards only; preserve all public actions and hand sizes.
    a, b = other[:2]
    state.hands[a][0], state.hands[b][0] = state.hands[b][0], state.hands[a][0]
    assert candidates(state, agent, player) == first
    assert agent._live_planned_block_review(state, player, first[0].action) == evidence


def test_inferred_monopoly_is_not_treated_as_physical_monopoly():
    state, agent, player = position()
    tracker = agent._track[id(state)]
    tracker["unknown_piece_pool"]["3"] = 1
    for seat in "ABD":
        tracker["estimated_current_hands"][seat]["3"]["max"] = 0
    improved = candidates(state, agent, player)
    original = CurrentAI._branched_continuation_candidates(
        agent, state, player, state.hands[player], previous_attack="3", attack_number=2,
        requires_block=True, preferred_attack="3", limit=100)
    assert improved == original
    assert agent._live_planned_block_review(state, player, improved[0].action) is None


def test_lost_finish_value_or_insufficient_attack_reserve_prevents_promotion():
    state, agent, player = position()
    rows = candidates(state, agent, player)
    better = next(row for row in rows if row.action == ("attack_after_block", "3", "3"))
    original = next(row for row in rows if row.action == ("attack_after_block", "5", "3"))
    assert agent._planned_block_dominates(state, player, better, original, 2)
    assert not agent._planned_block_dominates(state, player, better, replace(original, finish_score=60), 2)
    assert not agent._planned_block_dominates(state, player, better, original, 1)
    assert not agent._planned_block_dominates(
        state, player, replace(better, remaining_hand=tuple("1456")), original, 2)
    # A royal supplies broad receiving coverage; counting duplicates as extra
    # receiving types must not invent an improvement.
    assert not agent._planned_block_dominates(
        state, player, replace(better, remaining_hand=tuple("1338")),
        replace(original, remaining_hand=tuple("1348")), 2)


@pytest.mark.parametrize("restriction", ["future_preferred_attacks", "permanently_protected_pieces"])
def test_future_piece_reservations_are_enforced_before_reordering(restriction):
    state, agent, player = position()
    rows = candidates(state, agent, player, **{restriction: ("3", "3")})
    assert rows
    assert all(row.remaining_hand.count("3") >= 2 for row in rows)
    assert all(row.action != ("attack_after_block", "3", "3") for row in rows)


@pytest.mark.parametrize("copies", [1, 2, 3])
def test_unconfirmed_first_attack_only_releases_a_surplus_copy(copies):
    state, agent, player = position()
    tracker = agent._track[id(state)]
    assert tracker["my_attack_history"] == ["3"]
    state.hands[player] = list("145") + ["3"] * copies
    assert agent._unconfirmed_first_attack_piece_for_next_action(state, player) == (
        None if copies >= 3 else "3")


def test_proven_cached_plan_is_not_replaced_by_this_heuristic():
    state, agent, player = position(run_decisions=True, enabled=False)
    old = agent._active_branched_attack_plan(state)
    old.plan = replace(old.plan, evaluation=replace(old.plan.evaluation, guaranteed_win=True))
    agent.PLANNED_BLOCK_RESERVE_ENABLED = True
    action = agent._branched_planned_action(state, player, state.legal_actions(player))
    assert action == ("attack_after_block", "5", "3")
    assert old.reason != "planned_block_loses_receive_width"


@pytest.mark.parametrize("agent_class", [CurrentAI, Middle3])
def test_saved_profiles_keep_their_existing_planning(agent_class):
    state, agent, player = position(stop=4, agent_class=agent_class)
    (plan,) = agent._generate_branched_attack_plans(state, player, [("attack", None, "3")])
    assert lap_node(plan, plan.node(plan.root_node_id)).action == ("attack_after_block", "5", "3")
    assert not hasattr(agent, "_live_planned_block_review")

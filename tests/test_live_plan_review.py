"""Public-event plan updates stay bounded and cannot overrule proven decisions."""
import copy
from collections import Counter
from types import SimpleNamespace

import pytest

from goita_ai2.constants import PIECE_TOTALS
from goita_ai2.current_ai import RuleBasedAgent as CurrentAI
from goita_ai2.experimental_ai2 import RuleBasedAgent
from goita_ai2.intermediate_middle3 import RuleBasedAgent as SavedAI
from goita_ai2.state import GoitaState


def position(agent_class=RuleBasedAgent, dealer="A", swap=False):
    hands = {"A": list("11552279"), "B": list("11553468")}
    rest = Counter(PIECE_TOTALS)
    rest.subtract(p for h in hands.values() for p in h)
    cards = list(sorted(rest.elements()))
    hands.update(C=cards[:8], D=cards[8:])
    if swap:
        hands["C"], hands["D"] = hands["D"], hands["C"]
    assert Counter(p for h in hands.values() for p in h) == Counter(PIECE_TOTALS)
    state = GoitaState(hands, dealer=dealer)
    agent = agent_class()
    agent.bind_player("A")
    agent.TIME_SEARCH_ENABLED = agent.TIME_SEARCH_BACKGROUND_ENABLED = False
    agent._relative_hand_rank_table = {}
    agent._ensure_trackers(state)
    return state, agent


def play(state, agent, actor, action):
    assert action in state.legal_actions(actor)
    if action[0] == "pass": state.apply_pass(actor)
    elif action[0] == "receive": state.apply_receive(actor, action[1])
    elif action[0] == "attack": state.apply_attack(actor, action[2])
    else: state.apply_attack_after_block(actor, action[1], action[2])
    agent.on_public_action(state, actor, action)


def review(state, agent):
    return agent._track[id(state)]["round_strategy"]["live_review"]


def test_enemy_receive_attack_and_pass_each_review_the_plan_before_our_turn():
    state, agent = position()
    opening = copy.deepcopy(agent._track[id(state)]["round_strategy"]["opening_attacks"])
    play(state, agent, "A", ("attack_after_block", "2", "1"))
    play(state, agent, "B", ("receive", "1", None))
    received = copy.deepcopy(review(state, agent))
    assert received["revision"] == 2 and state.turn == "B"
    assert received["event"]["action"] == ["receive", "1", None]
    play(state, agent, "B", ("attack", None, "5"))
    attacked = copy.deepcopy(review(state, agent))
    assert attacked["revision"] == 3 and state.turn == "C"
    assert attacked["public_unseen"]["5"] == received["public_unseen"]["5"] - 1
    assert attacked["attack_values"]["5"] != received["attack_values"]["5"]
    assert attacked["attacks"] and attacked["proven"] is False
    play(state, agent, "C", ("pass", None, None))
    assert review(state, agent)["revision"] == 4
    assert agent._track[id(state)]["round_strategy"]["opening_attacks"] == opening


def test_no_rollout_proof_neural_or_whole_hand_search_is_started(monkeypatch):
    state, agent = position()
    def forbidden(*args, **kwargs):
        pytest.fail("public-event review must not start heavy thinking")
    for name in ("_time_limited_search_action", "_forced_win_result_after_receive_action",
                 "_forced_win_resolve_external_return", "_search_future_attack_plan",
                 "_build_branched_attack_plan", "_record_neural_shadow", "prefetch_next_turn"):
        monkeypatch.setattr(agent, name, forbidden)
    play(state, agent, "A", ("attack_after_block", "2", "1"))
    play(state, agent, "B", ("receive", "1", None))
    play(state, agent, "B", ("attack", None, "5"))
    for c in review(state, agent)["candidates"]:
        assert len(c["steps"]) <= 4 and not c["proven"]
    assert len(review(state, agent)["candidates"]) <= 3


def without_time(data):
    data = copy.deepcopy(data)
    data.pop("elapsed_ms", None)
    return data


def test_opponents_real_hands_do_not_affect_the_review():
    results = []
    for swap in (False, True):
        state, agent = position(swap=swap)
        for actor, action in [("A", ("attack_after_block", "2", "1")),
                              ("B", ("receive", "1", None)), ("B", ("attack", None, "5"))]:
            play(state, agent, actor, action)
        results.append(without_time(review(state, agent)))
    assert results[0] == results[1]


def test_opponents_hidden_block_is_not_a_review_input():
    results = []
    for block in ("1", "3"):
        state, agent = position(dealer="B")
        play(state, agent, "B", ("attack_after_block", block, "5"))
        result = without_time(review(state, agent))
        assert result["event"]["action"] == ["attack_after_block", None, "5"]
        results.append(result)
    assert results[0] == results[1]


def test_own_receive_rebases_on_remaining_hand_and_is_not_a_new_attack():
    state, agent = position(dealer="B")
    play(state, agent, "B", ("attack_after_block", "1", "5"))
    play(state, agent, "C", ("pass", None, None))
    play(state, agent, "D", ("pass", None, None))
    play(state, agent, "A", ("receive", "5", None))
    r = review(state, agent)
    assert r["hand"] == sorted(state.hands["A"])
    assert r["first_attack_number"] == 1
    for candidate in r["candidates"]:
        assert candidate["steps"][0]["block"] is None
        used = Counter(p for step in candidate["steps"] for p in step.values() if p is not None)
        assert used == Counter(state.hands["A"])
    chosen = r["attacks"][0]
    # Only the public observation, not this outline, counts the played attack.
    play(state, agent, "A", ("attack", None, chosen))
    assert review(state, agent)["first_attack_number"] == 2


@pytest.mark.parametrize("kind", ["exact", "branched"])
def test_proven_plan_is_not_replaced_by_a_light_outline(monkeypatch, kind):
    state, agent = position()
    tr = agent._track[id(state)]
    tr["round_strategy"]["live_review"] = {"attacks": ["5", "5", "9"], "revision": 1}
    if kind == "exact":
        protected = {"minimum_score": 30, "maximum_score": 50}
        tr["active_forced_win_plan"] = protected
    else:
        protected = SimpleNamespace(plan=SimpleNamespace(evaluation=SimpleNamespace(guaranteed_win=True)))
        monkeypatch.setattr(agent, "_active_branched_attack_plan", lambda _: protected)
    result = agent._review_live_attack_plan(state, "B", ("pass", None, None))
    assert result["protected_plan"] and result["status"] == "protected"
    assert result["attacks"] == ["5", "5", "9"]
    if kind == "exact": assert tr["active_forced_win_plan"] is protected


def test_updated_outline_reaches_normal_candidate_evaluation_only(monkeypatch):
    state, agent = position()
    agent._review_live_attack_plan(state, "B", ("pass", None, None))
    r = review(state, agent)
    captured = []
    monkeypatch.setattr(CurrentAI, "_build_branched_attack_plan", lambda self, *args, **kw: captured.append(kw))
    action = ("attack_after_block", "1", r["attacks"][0])
    agent._build_branched_attack_plan(state, "A", action, attack_number=1, revision=0,
                                     candidate_label="c1", preferred_attacks=("1", "1"))
    assert captured[-1]["preferred_attacks"] == tuple(r["attacks"])
    agent._build_branched_attack_plan(state, "A", action, attack_number=1, revision=0,
                                     candidate_label="t1", source="proven_template", preferred_attacks=("1", "1"))
    assert captured[-1]["preferred_attacks"] == ("1", "1")


def test_finished_round_clears_current_outline_without_storing_an_event_history():
    state, agent = position()
    for _ in range(30):
        r = agent._review_live_attack_plan(state, "B", ("pass", None, None))
        assert len(r["candidates"]) <= 3
    state.finished = True
    agent._review_live_attack_plan(state, "A", ("attack", None, "9"))
    assert review(state, agent)["attacks"] == []
    assert review(state, agent)["status"] == "finished"
    assert "history" not in review(state, agent)


def test_decision_log_uses_the_current_state_review_not_another_cached_state(monkeypatch):
    state, agent = position()
    agent._review_live_attack_plan(state, "B", ("pass", None, None))
    review(state, agent)["revision"] = 101
    other = copy.deepcopy(state)
    agent._ensure_trackers(other)
    agent._review_live_attack_plan(other, "C", ("pass", None, None))
    review(other, agent)["revision"] = 202
    monkeypatch.setattr(CurrentAI, "_finalize_attack_candidate_snapshot",
                        lambda self, actions, chosen: setattr(self, "last_attack_candidate_snapshot", {}))
    def select(state, player, actions):
        chosen = actions[0]
        agent._finalize_attack_candidate_snapshot(actions, chosen)
        return chosen
    monkeypatch.setattr(agent, "_select_action_with_neural_tiebreak", select)
    agent.select_action(state, "A", state.legal_actions("A"))
    assert agent.last_attack_candidate_snapshot["live_plan_review"]["revision"] == 101
    assert agent._live_plan_review_state_id is None


@pytest.mark.parametrize("agent_class", [CurrentAI, SavedAI])
def test_current_and_saved_ai_do_not_enable_this_feature(agent_class):
    state, agent = position(agent_class)
    play(state, agent, "A", ("attack_after_block", "2", "1"))
    assert "live_review" not in agent._track[id(state)]["round_strategy"]

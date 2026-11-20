"""Royal trial overrides heuristics, while retaining proofs and ordinary tie-breaks."""
import pytest

from goita_ai2.current_ai.agent import RuleBasedAgent as CurrentAgent
from goita_ai2.experimental_ai2 import RuleBasedAgent as AI2
from goita_ai2.instruction_case_audit import apply_action
from goita_ai2.state import GoitaState


PASS = ("pass", None, None)
ROYAL_RECEIVE = ("receive", "8", None)
SILVER_RECEIVE = ("receive", "4", None)


def position(receive=False):
    state = GoitaState({
        "A": list("11123457"), "B": list("11144578"),
        "C": list("12334456"), "D": list("11122369"),
    }, dealer="A" if receive else "B")
    if receive:
        apply_action(state, "A", ("attack_after_block", "1", "4"))
    agent = AI2()
    agent.bind_player("B")
    return state, agent


def install(monkeypatch, rule, neural, *, authority="ordinary", margin=0.01,
            reason="score_fallback", detail="attack_piece_value", record=False):
    class Model:
        def rank_actions(self, payload, actions):
            assert neural in actions
            return [(neural, 10.0)] + [(a, 10.0 - margin) for a in actions if a != neural]

    def select(self, state, player, actions):
        assert rule in actions
        self.last_rule_search_authority = authority
        self.last_decision_reason = reason
        self.last_score_fallback_detail = detail
        self.last_attack_candidate_scores = [
            {"action": a, "score": 1000.0 if a == rule else -1000.0} for a in actions]
        if record:
            self._record_generic_response_evidence(state, player, actions, rule, rule,
                followup_piece=None, source="test", depth=8, agreement=1.0,
                information_confidence=1.0, margin=100.0)
        self._commit_neural_tiebreak_action(state, player, rule)
        self._track[id(state)]["pending_late_receive_route"] = {"action": rule}
        return rule

    monkeypatch.setattr(AI2, "_shared_neural_model", Model())
    monkeypatch.setattr(AI2, "_shared_neural_error", None)
    monkeypatch.setattr(CurrentAgent, "select_action", select)


@pytest.mark.parametrize("rule,neural", [
    (ROYAL_RECEIVE, PASS), (PASS, ROYAL_RECEIVE),
    (ROYAL_RECEIVE, SILVER_RECEIVE), (SILVER_RECEIVE, PASS),
])
@pytest.mark.parametrize("authority", ["ordinary", "strong"])
def test_royal_receive_uses_global_choice_without_score_or_margin_limit(monkeypatch, rule, neural, authority):
    state, agent = position(receive=True)
    install(monkeypatch, rule, neural, authority=authority)
    assert agent.select_action(state, "B", state.legal_actions("B")) == neural
    snapshot = agent.last_neural_shadow
    assert snapshot["mode"] == "royal_priority"
    assert snapshot["applied"] and not snapshot["safety_locked"]
    assert snapshot["required_margin"] == 0.0
    assert snapshot["margin"] < 0.5
    assert snapshot["current_score_gap"] == 2000.0
    assert snapshot["rule_action"] == list(rule)
    assert agent.last_decision_reason == "neural_royal"
    assert agent.last_attack_candidate_snapshot["decision_reason"] == "neural_royal"
    assert agent.last_attack_candidate_snapshot["neural_royal_decision"]["rule_action"] == list(rule)
    assert agent._track[id(state)].get("pending_late_receive_route") is None


@pytest.mark.parametrize("rule,neural", [
    (("attack_after_block", "1", "4"), ("attack_after_block", "8", "4")),
    (("attack_after_block", "8", "4"), ("attack_after_block", "1", "4")),
])
def test_royal_block_override_commits_once(monkeypatch, rule, neural):
    state, agent = position()
    install(monkeypatch, rule, neural, authority="strong")
    assert agent.select_action(state, "B", state.legal_actions("B")) == neural
    assert agent._track[id(state)]["my_attack_count"] == 1
    assert agent.last_neural_shadow["royal_priority_context"] == "attack_or_block_use_or_reserve"


@pytest.mark.parametrize("reverse", [False, True])
def test_royal_attack_uses_global_first_candidate(monkeypatch, reverse):
    state = GoitaState({
        "A": list("11123457"), "B": list("11144589"),
        "C": list("12334456"), "D": list("11122367"),
    }, dealer="B")
    agent = AI2()
    agent.bind_player("B")
    rule, neural = ("attack_after_block", "1", "4"), ("attack_after_block", "1", "8")
    if reverse:
        rule, neural = neural, rule
    # A strategic royal opening is not rewritten by the ordinary fallback;
    # the trial must still be able to replace that strong heuristic.
    install(monkeypatch, rule, neural, authority="strong")
    assert agent.select_action(state, "B", state.legal_actions("B")) == neural
    assert agent.last_decision_reason == "neural_royal"
    assert agent._track[id(state)]["my_attack_count"] == 1


@pytest.mark.parametrize("reason,authority,detail", [
    ("win_now", "ordinary", "high_score_50"),
    ("score_fallback", "proven", "pass_full_receive_cover_test"),
])
def test_royal_trial_does_not_override_proven_decision(monkeypatch, reason, authority, detail):
    state, agent = position(receive=True)
    install(monkeypatch, PASS, ROYAL_RECEIVE, reason=reason, authority=authority, detail=detail)
    assert agent.select_action(state, "B", state.legal_actions("B")) == PASS
    assert agent.last_neural_shadow["royal_guard"] == "proven_decision"
    assert agent.last_neural_shadow["safety_locked"]
    assert not agent.last_neural_shadow["applied"]


def test_holding_royal_does_not_change_unrelated_nonroyal_attack(monkeypatch):
    state, agent = position()
    rule, neural = ("attack_after_block", "1", "4"), ("attack_after_block", "1", "5")
    install(monkeypatch, rule, neural, margin=10.0)
    assert agent.select_action(state, "B", state.legal_actions("B")) == rule
    assert agent.last_neural_shadow["mode"] == "tiebreak"
    assert agent.last_neural_shadow["current_choice_status"] == "clear"


def test_trial_can_be_disabled(monkeypatch):
    state, agent = position(receive=True)
    install(monkeypatch, PASS, ROYAL_RECEIVE, authority="strong")
    agent.NEURAL_ROYAL_PRIORITY_ENABLED = False
    assert agent.select_action(state, "B", state.legal_actions("B")) == PASS
    assert agent.last_neural_shadow["mode"] == "tiebreak"


def test_royal_trial_uses_global_choice_even_outside_close_subset(monkeypatch):
    state, agent = position(receive=True)
    install(monkeypatch, SILVER_RECEIVE, ROYAL_RECEIVE, margin=10.0)
    old_select = CurrentAgent.select_action

    def close_subset(self, current_state, player, actions):
        chosen = old_select(self, current_state, player, actions)
        self.last_attack_candidate_scores = [
            {"action": SILVER_RECEIVE, "score": 100.0},
            {"action": PASS, "score": 99.0},
            {"action": ROYAL_RECEIVE, "score": -1000.0},
        ]
        return chosen

    monkeypatch.setattr(CurrentAgent, "select_action", close_subset)
    assert agent.select_action(state, "B", state.legal_actions("B")) == ROYAL_RECEIVE
    assert agent.last_neural_shadow["baseline_choice_status"] == "close"
    assert agent.last_neural_shadow["current_score_gap"] == 1100.0


def test_model_failure_falls_back_to_current_ai(monkeypatch):
    state, agent = position(receive=True)
    install(monkeypatch, PASS, ROYAL_RECEIVE, authority="strong")
    monkeypatch.setattr(AI2, "_neural_model", lambda self: None)
    assert agent.select_action(state, "B", state.legal_actions("B")) == PASS
    assert not agent.last_neural_shadow["available"]


@pytest.mark.parametrize("neural,expected_records", [(PASS, 0), (ROYAL_RECEIVE, 1)])
def test_rejected_current_decision_is_not_saved_as_adopted_evidence(monkeypatch, neural, expected_records):
    from goita_ai2.current_ai.generic_response_pattern import GenericResponsePatternMixin

    state, agent = position(receive=True)
    install(monkeypatch, ROYAL_RECEIVE, neural, record=True)
    recorded = []
    monkeypatch.setattr(GenericResponsePatternMixin, "_record_generic_response_evidence",
                        lambda self, *args, **kwargs: recorded.append(args) or True)
    assert agent.select_action(state, "B", state.legal_actions("B")) == neural
    assert len(recorded) == expected_records
    assert getattr(agent, "_royal_neural_deferred_records", None) is None

from pathlib import Path


HTML = (Path(__file__).parents[1] / "frontend" / "index.html").read_text(encoding="utf-8")


def test_transient_legal_action_failure_does_not_become_an_empty_success():
    assert "if(!res.ok) return null;" in HTML
    assert "return Array.isArray(acts) ? acts : null;" in HTML
    assert "scheduleLegalActionsRetry();" in HTML
    assert "if(!state?.is_started || isSpectator() || state.finished || state.turn !== mySeat)" in HTML


def test_receive_followup_attacks_are_restored_from_visible_hand():
    assert "function localFollowupAttackActions(state)" in HTML
    assert 'state.phase !== "attack"' in HTML
    assert 'state.attacker !== mySeat' in HTML
    assert '.map(attack => ({ action_type: "attack", block: null, attack }))' in HTML
    assert "renderHands(state);\n  renderActions();\n  await fetchBeginnerRecommendation(state);" in HTML


def test_stale_refresh_cannot_overwrite_a_newer_turn_state():
    assert "let refreshRequestSequence = 0;" in HTML
    assert "const requestSequence = ++refreshRequestSequence;" in HTML
    assert HTML.count("if(requestSequence !== refreshRequestSequence) return;") >= 3

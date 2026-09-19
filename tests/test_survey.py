from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from backend import app as app_module
from backend.survey_api import _validate_answers
from backend.survey_api import create_survey_router
from backend.survey_store import SurveyStore, resolve_survey_path
from backend.member_store import MemberError


ROOT = Path(__file__).parents[1]
HTML = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
ADMIN = (ROOT / "frontend" / "admin.html").read_text(encoding="utf-8")
SCRIPT = (ROOT / "frontend" / "survey.js").read_text(encoding="utf-8")
WHISPER = (ROOT / "frontend" / "lobbyWhisper.js").read_text(encoding="utf-8")


def detailed_answers():
    return {
        "frequency": "weekly", "experience": "sometimes", "play_styles": ["ai", "research"],
        "feature_awareness": {key: "used" for key in (
            "library", "auto_save", "statistics", "deal_style", "balanced_deal", "hand_spec", "replay_practice",
        )},
        "membership_awareness": "registered", "ui_clarity": "clear", "confusing_areas": [],
        "ai_used": "yes", "ai_strength": "right", "ai_naturalness": "natural", "ai_concerns": [], "ai_other": "",
        "score_usage": "once", "score_enjoyment": "fun", "score_good": ["comparison"], "score_improvements": [],
        "future_features": ["kifu_analysis"], "problems": [], "free_text": "分析を使いたい",
    }


def test_survey_path_prefers_explicit_and_persistent(tmp_path):
    assert resolve_survey_path({"GOITA_SURVEY_DB_PATH": "/custom/survey.sqlite"}) == Path("/custom/survey.sqlite")
    assert resolve_survey_path({"GOITA_PERSISTENT_DATA_DIR": "/data"}) == Path("/data/goita-survey.sqlite3")
    assert resolve_survey_path({}, local_fallback=tmp_path / "survey.sqlite") == tmp_path / "survey.sqlite"


def test_store_upgrades_quick_to_detailed_and_does_not_downgrade(tmp_path):
    store = SurveyStore(tmp_path / "survey.sqlite")
    key = "response_key_1234567890"
    quick = {"satisfaction": "satisfied", "primary_use": "ai", "improvement": "kifu"}
    assert store.save(response_key=key, kind="quick", member_type="guest", device="mobile", language="ja", answers=quick)["kind"] == "quick"
    assert store.save(response_key=key, kind="detailed", member_type="free", device="desktop", language="ja", answers=detailed_answers())["kind"] == "detailed"
    assert store.save(response_key=key, kind="quick", member_type="guest", device="mobile", language="ja", answers=quick)["kind"] == "detailed"
    snapshot = store.snapshot()
    assert snapshot["summary"]["total"] == 1
    assert snapshot["summary"]["detailed"] == 1
    assert snapshot["summary"]["answer_counts"]["frequency"]["weekly"] == 1
    assert "free_text" not in snapshot["summary"]["answer_counts"]


def test_answer_validation_is_fixed_and_limits_future_choices():
    quick = _validate_answers("quick", {"satisfaction": "satisfied", "primary_use": "ai", "improvement": "none"})
    assert quick["improvement"] == "none"
    answers = detailed_answers()
    assert _validate_answers("detailed", answers)["feature_awareness"]["library"] == "used"
    answers["future_features"] = ["kifu_analysis", "ai_advice", "beginner", "stronger_ai"]
    with pytest.raises(HTTPException):
        _validate_answers("detailed", answers)


def test_public_submission_is_anonymous_and_origin_protected(tmp_path):
    class GuestMembers:
        def authenticate(self, _token):
            raise MemberError(401, "not logged in")

    store = SurveyStore(tmp_path / "survey.sqlite")
    app = FastAPI()
    app.include_router(create_survey_router(store, GuestMembers()))
    client = TestClient(app, base_url="https://testserver")
    payload = {
        "response_key": "anonymous_response_123456",
        "kind": "quick", "device": "mobile", "language": "ja",
        "answers": {"satisfaction": "satisfied", "primary_use": "ai", "improvement": "none"},
    }
    assert client.post("/api/survey/responses", json=payload).status_code == 403
    response = client.post("/api/survey/responses", json=payload, headers={"X-Goita-Member": "1", "Origin": "https://testserver"})
    assert response.json() == {"ok": True, "kind": "quick"}
    saved = store.snapshot()["responses"][0]
    assert saved["member_type"] == "guest"
    assert "member_id" not in saved


def test_survey_notice_mode_and_frontend_are_connected():
    room = next(iter(app_module.MAIN_ROOM_NAMES))
    normalized = app_module._normalize_public_room_ads({room: {"enabled": True, "mode": "survey"}})
    assert normalized[room]["mode"] == "survey"
    previous = app_module.PUBLIC_ROOM_AD_SETTINGS
    try:
        app_module.PUBLIC_ROOM_AD_SETTINGS = normalized
        payload = app_module._public_room_ad_public_payload(room)
        assert payload["enabled"] is True and payload["mode"] == "survey"
    finally:
        app_module.PUBLIC_ROOM_AD_SETTINGS = previous
    assert '<option value="survey">アンケート</option>' in ADMIN
    assert 'data-tab="survey"' in ADMIN
    assert 'id="surveyView"' in ADMIN
    assert 'window.goitaSurvey?.open?.()' in WHISPER
    assert '/api/survey/responses' in SCRIPT
    assert '/static/survey.js' in HTML

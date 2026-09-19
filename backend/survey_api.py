"""Public survey submission API with a fixed, privacy-limited answer schema."""

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from backend.member_api import MEMBER_COOKIE, require_member_origin
from backend.member_store import MemberError


class SurveySubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    response_key: str = Field(min_length=16, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    kind: Literal["quick", "detailed"]
    device: Literal["mobile", "desktop", "tablet", "unknown"] = "unknown"
    language: Literal["ja", "zh", "en", "other"] = "other"
    answers: dict[str, Any]


QUICK_VALUES = {
    "satisfaction": {"very_satisfied", "satisfied", "neutral", "dissatisfied", "very_dissatisfied"},
    "primary_use": {"public", "ai", "private", "research", "score_attack", "spectate", "new"},
    "improvement": {"ai", "matching", "usability", "research", "kifu", "score_attack", "stability", "beginner", "none", "other"},
}
DETAILED_SINGLE = {
    "frequency": {"daily", "weekly", "monthly", "few", "first"},
    "experience": {"first", "few", "sometimes", "regular", "tournament"},
    "ui_clarity": {"very_clear", "clear", "neutral", "unclear", "very_unclear"},
    "ai_used": {"yes", "no"},
    "ai_strength": {"too_strong", "strong", "right", "weak", "too_weak", "unknown", ""},
    "ai_naturalness": {"very_natural", "natural", "neutral", "unnatural", "very_unnatural", ""},
    "score_usage": {"regular", "sometimes", "once", "known_unused", "unknown"},
    "score_enjoyment": {"very_fun", "fun", "neutral", "not_fun", "very_not_fun", ""},
    "membership_awareness": {"understand", "name_only", "unknown", "registered"},
}
DETAILED_LIST = {
    "play_styles": {"public", "ai", "private", "research", "score_attack", "spectate", "kifu", "new"},
    "confusing_areas": {"top", "room_entry", "game", "settings", "kifu", "score_attack", "account", "mobile", "other"},
    "ai_concerns": {"pass", "receive", "attack", "repeats", "slow", "shallow", "none", "other"},
    "score_good": {"comparison", "same_hand", "ranking", "rewards", "solo", "history", "none", "other"},
    "score_improvements": {"rules", "difference", "ai", "tempo", "ranking", "rewards", "choose_record", "none", "other"},
    "future_features": {"kifu_analysis", "ai_advice", "beginner", "stronger_ai", "matching", "events", "score_content", "profile", "research", "mobile", "other"},
    "problems": {"matching", "room_entry", "settings", "game", "find_features", "account", "kifu", "connection", "mobile", "none", "other"},
}
FEATURE_KEYS = {"library", "auto_save", "statistics", "deal_style", "balanced_deal", "hand_spec", "replay_practice"}
FEATURE_VALUES = {"unknown", "known", "used"}


def _text(value: Any, maximum: int) -> str:
    return str(value or "").strip()[:maximum]


def _validate_answers(kind: str, raw: dict[str, Any]) -> dict[str, Any]:
    if kind == "quick":
        result = {}
        for key, allowed in QUICK_VALUES.items():
            value = str(raw.get(key, ""))
            if value not in allowed:
                raise HTTPException(400, "必須の質問に回答してください。")
            result[key] = value
        result["improvement_other"] = _text(raw.get("improvement_other"), 200)
        return result

    result = {}
    for key, allowed in DETAILED_SINGLE.items():
        value = str(raw.get(key, ""))
        if value not in allowed:
            raise HTTPException(400, "回答内容を確認してください。")
        result[key] = value
    for key, allowed in DETAILED_LIST.items():
        values = raw.get(key, [])
        if not isinstance(values, list) or len(values) > len(allowed):
            raise HTTPException(400, "回答内容を確認してください。")
        normalized = list(dict.fromkeys(str(value) for value in values))
        if any(value not in allowed for value in normalized):
            raise HTTPException(400, "回答内容を確認してください。")
        result[key] = normalized
    if not result["play_styles"]:
        raise HTTPException(400, "普段の遊び方を1つ以上選んでください。")
    if len(result["future_features"]) > 3:
        raise HTTPException(400, "今後ほしいものは3つまで選んでください。")
    features = raw.get("feature_awareness", {})
    if not isinstance(features, dict) or set(features) != FEATURE_KEYS:
        raise HTTPException(400, "各機能について回答してください。")
    if any(str(value) not in FEATURE_VALUES for value in features.values()):
        raise HTTPException(400, "各機能について回答してください。")
    result["feature_awareness"] = {key: str(features[key]) for key in sorted(FEATURE_KEYS)}
    for key, maximum in (("ai_other", 200), ("free_text", 2000)):
        result[key] = _text(raw.get(key), maximum)
    if result["ai_used"] == "yes" and (not result["ai_strength"] or not result["ai_naturalness"]):
        raise HTTPException(400, "AIについての質問に回答してください。")
    played_score = result["score_usage"] in {"regular", "sometimes", "once"}
    if played_score and not result["score_enjoyment"]:
        raise HTTPException(400, "スコアアタックについての質問に回答してください。")
    return result


def create_survey_router(store, members):
    router = APIRouter()

    @router.post("/api/survey/responses")
    def submit(request: Request, data: SurveySubmission):
        require_member_origin(request)
        member_type = "guest"
        try:
            member = members.authenticate(request.cookies.get(MEMBER_COOKIE, ""))
            member_type = "supporter" if member.get("paid_active") else "free"
        except MemberError:
            pass
        answers = _validate_answers(data.kind, data.answers)
        try:
            record = store.save(
                response_key=data.response_key,
                kind=data.kind,
                member_type=member_type,
                device=data.device,
                language=data.language,
                answers=answers,
            )
        except ValueError as error:
            raise HTTPException(400, "回答を保存できませんでした。") from error
        return {"ok": True, "kind": record["kind"]}

    return router

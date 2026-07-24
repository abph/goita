from __future__ import annotations

import copy
from pathlib import Path

from fastapi import HTTPException

import backend.app as app_module
from goita_ai2.rule_based import RuleBasedAgent as CurrentRuleBasedAgent
from goita_ai2.state import GoitaState


ROOT = Path(__file__).resolve().parents[1]


def test_private_room_recommendation_is_legal_and_non_mutating() -> None:
    game_id = "test-beginner-support"
    game = app_module._create_game_obj(dealer="A", ai_profile="beginner_upper")
    game["is_started"] = True
    game["human_seats"] = {"A": "client-a"}
    app_module.GAMES[game_id] = game

    try:
        state = game["state"]
        hands_before = copy.deepcopy(state.hands)
        legal_before = state.legal_actions("A")
        support_agent = game["beginner_support_agents"]["A"]
        reason_before = support_agent.last_decision_reason
        detail_before = support_agent.last_score_fallback_detail

        result = app_module.get_beginner_recommendation(
            game_id,
            player="A",
            client_id="client-a",
        )

        action = result["action"]
        action_tuple = (action["action_type"], action["block"], action["attack"])
        assert action_tuple in legal_before
        assert result["forced"] is (len(legal_before) == 1)
        assert result["explanation"]
        assert state.hands == hands_before
        assert support_agent.last_decision_reason == reason_before
        assert support_agent.last_score_fallback_detail == detail_before
        assert isinstance(support_agent, CurrentRuleBasedAgent)
    finally:
        app_module.GAMES.pop(game_id, None)


def test_beginner_support_is_not_available_in_main_room() -> None:
    try:
        app_module.get_beginner_recommendation(
            app_module.MAIN_GID,
            player="A",
            client_id="client-a",
        )
    except HTTPException as exc:
        assert exc.status_code == 403
    else:
        raise AssertionError("The main room must reject beginner recommendations.")


def test_forced_pass_is_reported_as_forced() -> None:
    game_id = "test-beginner-support-forced-pass"
    game = app_module._create_game_obj(dealer="A", ai_profile="beginner_upper")
    state = GoitaState(
        {
            "A": ["1", "3", "3", "4", "4", "5", "5", "6"],
            "B": ["1", "1", "2", "2", "4", "5", "6", "7"],
            "C": ["1", "2", "3", "4", "5", "6", "7", "8"],
            "D": ["1", "2", "3", "4", "5", "6", "7", "9"],
        },
        dealer="A",
    )
    state.apply_attack_after_block("A", "1", "3")
    game["state"] = state
    game["is_started"] = True
    game["human_seats"] = {"B": "client-b"}
    app_module.GAMES[game_id] = game

    try:
        result = app_module.get_beginner_recommendation(
            game_id,
            player="B",
            client_id="client-b",
        )

        assert result["forced"] is True
        assert result["action"] == {
            "action_type": "pass",
            "block": None,
            "attack": None,
        }
        assert result["explanation"] == "受けられる駒がないため、パスしてください。"
    finally:
        app_module.GAMES.pop(game_id, None)


def test_ally_attack_pass_uses_ally_guidance() -> None:
    state = GoitaState(
        {
            "A": ["1", "3", "3", "4", "4", "5", "5", "6"],
            "B": ["1", "1", "2", "2", "4", "5", "6", "7"],
            "C": ["1", "2", "3", "4", "5", "6", "7", "8"],
            "D": ["1", "2", "3", "4", "5", "6", "7", "9"],
        },
        dealer="A",
    )
    state.apply_attack_after_block("A", "1", "3")
    state.apply_pass("B")

    explanation = app_module._beginner_support_explanation(
        state,
        "C",
        ("pass", None, None),
        object(),
    )

    assert explanation == (
        "味方の駒は基本的にパスします。"
        "3香を持っている、しを持っていないなど、"
        "大きな理由がない限りはパスしましょう。"
    )


def test_enemy_attack_pass_names_the_saved_royal() -> None:
    state = GoitaState(
        {
            "A": ["1", "3", "3", "4", "4", "5", "6", "6"],
            "B": ["1", "1", "2", "2", "3", "4", "5", "8"],
            "C": ["1", "2", "3", "4", "5", "6", "7", "9"],
            "D": ["1", "2", "3", "4", "5", "6", "7", "7"],
        },
        dealer="A",
    )
    state.apply_attack_after_block("A", "1", "6")

    explanation = app_module._beginner_support_explanation(
        state,
        "B",
        ("pass", None, None),
        object(),
    )

    assert explanation == "王（玉）を温存するため、今回はパスがおすすめです。"


def test_frontend_contains_beginner_support_controls() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert 'id="checkEnableBeginnerSupport"' in html
    assert "fetchBeginnerRecommendation" in html
    assert "beginner-recommended" in html
    assert "/beginner_recommendation" in html
    assert 'heading.textContent = forcedPass ? "操作" : "おすすめ";' in html
    assert "受けられる駒がないため、パスしてください。" in html
    assert "受けられる駒があります。本当にパスしますか？" in html


if __name__ == "__main__":
    test_private_room_recommendation_is_legal_and_non_mutating()
    test_beginner_support_is_not_available_in_main_room()
    test_forced_pass_is_reported_as_forced()
    test_ally_attack_pass_uses_ally_guidance()
    test_enemy_attack_pass_names_the_saved_royal()
    test_frontend_contains_beginner_support_controls()
    print("BEGINNER_SUPPORT_TEST_OK")

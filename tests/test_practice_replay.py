from __future__ import annotations

import asyncio
import copy

import pytest
from fastapi import HTTPException

from backend import app as app_module


def _disable_finish_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "_auto_save_member_round", lambda *_: None)
    monkeypatch.setattr(app_module, "_save_trace_result", lambda *_: None)
    for name in (
        "checkpoint_ai_search_telemetry",
        "checkpoint_background_search_value_model",
        "checkpoint_generic_response_patterns",
    ):
        monkeypatch.setattr(app_module, name, lambda *_: None)


def test_practice_replay_reuses_hands_and_dealer_without_adding_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_finish_side_effects(monkeypatch)

    async def scenario() -> None:
        game_id = "practice-replay-test"
        client_id = "practice-host"
        game = app_module._create_game_obj(dealer="C")
        game["human_seats"] = {"A": client_id}
        game["ai_seats"] = ["B", "C", "D"]
        game["is_started"] = True
        game["round_count"] = 4
        game["total_team_score"] = {"AC": 40, "BD": 30}
        initial_hands = copy.deepcopy(game["init_hands"])
        game["state"].finished = True
        game["state"].winner = "A"
        app_module.GAMES[game_id] = game

        try:
            app_module._handle_round_finish(
                game,
                game["state"],
                ("attack", None, "2"),
                [],
            )
            original_score = copy.deepcopy(game["total_team_score"])
            original_log = copy.deepcopy(game["log"])
            original_kifu = copy.deepcopy(game["last_completed_kifu"])
            assert game["practice_replay_source"]["hands"] == initial_hands
            assert game["practice_replay_source"]["dealer"] == "C"
            normal_view = app_module.get_state(
                game_id, viewer="A", client_id=client_id
            )
            assert normal_view["practice_replay_available"] is True
            assert normal_view["practice_replay_active"] is False

            started = await app_module.practice_replay(
                game_id,
                app_module.PracticeReplayRequest(
                    requester="A",
                    client_id=client_id,
                    action="start",
                ),
            )
            assert started["practice_replay_active"] is True
            practice = app_module.GAMES[game_id]
            assert practice["dealer"] == "C"
            assert practice["init_hands"] == initial_hands
            assert practice["total_team_score"] == original_score
            assert practice["round_count"] == 4
            assert practice["is_started"] is True
            assert practice["state"].finished is False
            practice_view = app_module.get_state(
                game_id, viewer="A", client_id=client_id
            )
            assert practice_view["practice_replay_active"] is True
            assert practice_view["practice_replay_available"] is False

            practice["state"].finished = True
            practice["state"].winner = "B"
            app_module._handle_round_finish(
                practice,
                practice["state"],
                ("attack", None, "3"),
                [],
            )
            assert practice["total_team_score"] == original_score
            assert practice["last_completed_kifu"] == original_kifu
            assert practice["last_round_score"] > 0
            finished_practice_view = app_module.get_state(
                game_id, viewer="A", client_id=client_id
            )
            assert finished_practice_view["practice_replay_available"] is True

            retried = await app_module.practice_replay(
                game_id,
                app_module.PracticeReplayRequest(
                    requester="A",
                    client_id=client_id,
                    action="retry",
                ),
            )
            assert retried["practice_replay_active"] is True
            retry_game = app_module.GAMES[game_id]
            assert retry_game["init_hands"] == initial_hands
            assert retry_game["dealer"] == "C"
            assert retry_game["total_team_score"] == original_score

            retry_game["state"].finished = True
            retry_game["state"].winner = "D"
            app_module._handle_round_finish(
                retry_game,
                retry_game["state"],
                ("attack", None, "4"),
                [],
            )
            returned = await app_module.practice_replay(
                game_id,
                app_module.PracticeReplayRequest(
                    requester="A",
                    client_id=client_id,
                    action="return",
                ),
            )
            assert returned["practice_replay_active"] is False
            restored = app_module.GAMES[game_id]
            assert restored["state"].finished is True
            assert restored["state"].winner == "A"
            assert restored["total_team_score"] == original_score
            assert restored["round_count"] == 4
            assert restored["log"] == original_log
            assert restored["last_completed_kifu"] == original_kifu
            assert restored["practice_replay_source"]["hands"] == initial_hands
        finally:
            app_module._cancel_turn_timeout_task(game_id)
            app_module.GAMES.pop(game_id, None)
            app_module.GAME_TURN_LOCKS.pop(game_id, None)

    asyncio.run(scenario())


def test_practice_replay_requires_finished_round_and_host_owner() -> None:
    async def scenario() -> None:
        game_id = "practice-replay-guard-test"
        game = app_module._create_game_obj(dealer="A")
        game["human_seats"] = {"A": "owner"}
        game["is_started"] = True
        app_module.GAMES[game_id] = game
        try:
            with pytest.raises(HTTPException) as unfinished:
                await app_module.practice_replay(
                    game_id,
                    app_module.PracticeReplayRequest(
                        requester="A", client_id="owner", action="start"
                    ),
                )
            assert unfinished.value.status_code == 409

            game["state"].finished = True
            game["practice_replay_source"] = app_module._practice_replay_source(game)
            with pytest.raises(HTTPException) as not_owner:
                await app_module.practice_replay(
                    game_id,
                    app_module.PracticeReplayRequest(
                        requester="A", client_id="other", action="start"
                    ),
                )
            assert not_owner.value.status_code == 403
        finally:
            app_module._cancel_turn_timeout_task(game_id)
            app_module.GAMES.pop(game_id, None)
            app_module.GAME_TURN_LOCKS.pop(game_id, None)

    asyncio.run(scenario())


def test_practice_replay_ui_has_all_controls_and_state_flags() -> None:
    html = app_module.FRONTEND_DIR.joinpath("index.html").read_text(encoding="utf-8")
    script = app_module.FRONTEND_DIR.joinpath("trace-results.js").read_text(encoding="utf-8")
    assert 'id="btnPracticeReplay"' in html
    assert 'id="btnPracticeReturn"' in html
    assert 'fetch(`${API}/games/${gid}/practice_replay`' in html
    assert "updatePracticeReplayButtons(state, isHost, autoNextRoundPending)" in html
    assert "state.practice_replay_available === true" in script
    assert "active ? 'retry' : 'start'" in script

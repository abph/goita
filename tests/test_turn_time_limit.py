from __future__ import annotations

import asyncio
import time

from backend import app as app_module


def test_turn_time_limit_defaults_to_none() -> None:
    game = app_module._create_game_obj(dealer="A")
    assert game["turn_time_limit_seconds"] == 0
    assert game["next_turn_time_limit_seconds"] == 0
    assert game["turn_deadline_at"] is None


def test_turn_time_limit_setting_applies_now_or_next_round() -> None:
    async def scenario() -> None:
        game_id = "test-turn-time-setting"
        client_id = "host-client"
        game = app_module._create_game_obj(dealer="A")
        game["human_seats"] = {"A": client_id}
        app_module.GAMES[game_id] = game
        try:
            saved = await app_module.update_turn_time_limit(
                game_id,
                app_module.TurnTimeLimitUpdateRequest(
                    requester="A",
                    client_id=client_id,
                    seconds=30,
                ),
            )
            assert saved["applies_next_round"] is False
            assert game["turn_time_limit_seconds"] == 30
            assert game["next_turn_time_limit_seconds"] == 30
            assert game["turn_deadline_at"] is None

            await app_module.start_game(game_id, requester="A", client_id=client_id)
            assert game["turn_deadline_at"] is not None
            original_deadline = game["turn_deadline_at"]

            queued = await app_module.update_turn_time_limit(
                game_id,
                app_module.TurnTimeLimitUpdateRequest(
                    requester="A",
                    client_id=client_id,
                    seconds=60,
                ),
            )
            assert queued["applies_next_round"] is True
            assert game["turn_time_limit_seconds"] == 30
            assert game["next_turn_time_limit_seconds"] == 60
            assert game["turn_deadline_at"] == original_deadline

            await app_module.reset_game(
                game_id,
                dealer="A",
                requester="A",
                client_id=client_id,
            )
            next_game = app_module.GAMES[game_id]
            assert next_game["turn_time_limit_seconds"] == 60
            assert next_game["next_turn_time_limit_seconds"] == 60
            assert next_game["turn_deadline_at"] is None
        finally:
            app_module._cancel_turn_timeout_task(game_id)
            app_module.GAMES.pop(game_id, None)
            app_module.GAME_TURN_LOCKS.pop(game_id, None)

    asyncio.run(scenario())


def test_timeout_automatically_passes_when_pass_is_legal() -> None:
    async def scenario() -> None:
        game_id = "test-turn-time-pass"
        game = app_module._create_game_obj(dealer="A")
        game["is_started"] = True
        game["turn_time_limit_seconds"] = 30
        game["next_turn_time_limit_seconds"] = 30
        app_module.GAMES[game_id] = game
        try:
            opening_player = game["state"].turn
            opening_result = app_module._apply_agent_turn(game, opening_player)
            assert opening_result["status"] == "ok"

            timed_out_player = game["state"].turn
            assert any(
                action[0] == "pass"
                for action in game["state"].legal_actions(timed_out_player)
            )
            deadline = time.time() - 0.01
            token = 77
            game["turn_timer_token"] = token
            game["turn_started_at"] = deadline - 30
            game["turn_deadline_at"] = deadline

            await app_module._turn_timeout_worker(
                game_id,
                token,
                deadline,
                timed_out_player,
            )

            assert game["log"][-1].startswith(f"{timed_out_player}: pass")
            assert game["log"][-1].endswith("[TIMEOUT]")
        finally:
            app_module._cancel_turn_timeout_task(game_id)
            app_module.GAMES.pop(game_id, None)
            app_module.GAME_TURN_LOCKS.pop(game_id, None)

    asyncio.run(scenario())


def test_frontend_contains_turn_timer_setting_and_countdown() -> None:
    html = app_module.FRONTEND_DIR.joinpath("index.html").read_text(encoding="utf-8")
    board_3d = app_module.FRONTEND_DIR.joinpath("board3d.js").read_text(encoding="utf-8")
    board_pixel = app_module.FRONTEND_DIR.joinpath("boardPixel.js").read_text(encoding="utf-8")

    assert 'id="turnTimeLimitSelect"' in html
    assert '<option value="0" selected>なし</option>' in html
    assert 'id="turnTimeLimitStatus"' in html
    assert "function syncTurnCountdown(state = latestState)" in html
    assert 'className = "turn-countdown"' in html
    assert "turnTimeLabel" in board_3d
    assert "turnTimeLabel" in board_pixel


if __name__ == "__main__":
    test_turn_time_limit_defaults_to_none()
    test_turn_time_limit_setting_applies_now_or_next_round()
    test_timeout_automatically_passes_when_pass_is_legal()
    test_frontend_contains_turn_timer_setting_and_countdown()
    print("TURN_TIME_LIMIT_TEST_OK")

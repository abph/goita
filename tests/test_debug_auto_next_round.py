from __future__ import annotations

import asyncio

from backend import app as app_module


def test_debug_auto_next_round_defaults_to_off() -> None:
    game = app_module._create_game_obj(dealer="A")
    assert game["debug_auto_next_round"] is False
    assert game["debug_auto_new_game"] is False


def test_debug_room_advances_after_delay_and_stops_at_match_end() -> None:
    async def scenario() -> None:
        game_id = app_module.DEBUG_GID
        client_id = "debug-host"
        previous_game = app_module.GAMES.get(game_id)
        previous_delay = app_module.DEBUG_AUTO_NEXT_ROUND_DELAY_SECONDS
        game = app_module._create_game_obj(dealer="A")
        game["is_debug_room"] = True
        game["is_started"] = True
        game["human_seats"] = {"A": client_id}
        game["total_team_score"] = {"AC": 40, "BD": 30}
        game["round_count"] = 3
        game["current_round_finished"] = True
        game["state"].finished = True
        game["state"].winner = "C"
        app_module.GAMES[game_id] = game
        app_module.DEBUG_AUTO_NEXT_ROUND_DELAY_SECONDS = 0.01

        try:
            saved = await app_module.update_debug_auto_next_round(
                game_id,
                app_module.DebugAutoNextRoundRequest(
                    requester="A",
                    client_id=client_id,
                    enabled=True,
                ),
            )
            assert saved["debug_auto_next_round"] is True
            await asyncio.sleep(0.05)

            next_game = app_module.GAMES[game_id]
            assert next_game is not game
            assert next_game["is_started"] is True
            assert next_game["dealer"] == "C"
            assert next_game["round_count"] == 4
            assert next_game["total_team_score"] == {"AC": 40, "BD": 30}
            assert next_game["debug_auto_next_round"] is True

            next_game["state"].finished = True
            next_game["state"].winner = "A"
            next_game["match_finished"] = True
            app_module._schedule_debug_auto_next_round(game_id)
            assert game_id not in app_module.DEBUG_AUTO_NEXT_ROUND_TASKS

            next_game["total_team_score"] = {"AC": 150, "BD": 30}
            saved = await app_module.update_debug_auto_next_round(
                game_id,
                app_module.DebugAutoNextRoundRequest(
                    requester="A",
                    client_id=client_id,
                    enabled=True,
                    auto_new_game=True,
                ),
            )
            assert saved["debug_auto_new_game"] is True
            await asyncio.sleep(0.05)

            restarted_game = app_module.GAMES[game_id]
            assert restarted_game is not next_game
            assert restarted_game["is_started"] is True
            assert restarted_game["dealer"] == "A"
            assert restarted_game["round_count"] == 1
            assert restarted_game["total_team_score"] == {"AC": 0, "BD": 0}
            assert restarted_game["debug_auto_next_round"] is True
            assert restarted_game["debug_auto_new_game"] is True
        finally:
            app_module._cancel_debug_auto_next_round_task(game_id)
            app_module.DEBUG_AUTO_NEXT_ROUND_DELAY_SECONDS = previous_delay
            if previous_game is None:
                app_module.GAMES.pop(game_id, None)
            else:
                app_module.GAMES[game_id] = previous_game
            app_module.GAME_TURN_LOCKS.pop(game_id, None)

    asyncio.run(scenario())


def test_frontend_exposes_debug_only_auto_next_round_setting() -> None:
    html = app_module.FRONTEND_DIR.joinpath("index.html").read_text(
        encoding="utf-8"
    )
    assert 'id="debugAutoNextRoundDetails"' in html
    assert 'id="checkDebugAutoNextRound"' in html
    assert 'id="checkDebugAutoNewGame"' in html
    assert "function saveDebugAutoNextRoundSetting()" in html
    assert "targetGid === DEBUG_GID" in html
    assert "3秒後に新規ゲームを開始します" in html
    assert "isGameStarted = state.is_started === true;" in html
    assert (
        "if(state.is_started === true && !state.finished && !isProcessingCpu)"
        in html
    )


if __name__ == "__main__":
    test_debug_auto_next_round_defaults_to_off()
    test_debug_room_advances_after_delay_and_stops_at_match_end()
    test_frontend_exposes_debug_only_auto_next_round_setting()
    print("DEBUG_AUTO_NEXT_ROUND_TEST_OK")

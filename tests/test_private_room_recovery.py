import asyncio

import pytest
from fastapi import HTTPException

from backend import app as app_module


def test_private_host_reset_clears_all_seats_and_preserves_room_settings() -> None:
    async def scenario() -> None:
        game_id = app_module.PRIVATE_A_GID
        previous = app_module.GAMES.get(game_id)
        game = app_module._create_game_obj(dealer="C", ai_profile="intermediate_middle2")
        game.update({
            "human_seats": {"A": "host-client", "B": "stale-client"},
            "ai_seats": ["C", "D"],
            "player_names": {"A": "ホスト", "B": "残った席", "C": "", "D": ""},
            "player_tags": {"A": "teacher", "B": "beginner", "C": "", "D": ""},
            "password": "room-pass",
            "owner_name": "研究部屋",
            "show_log": True,
            "total_team_score": {"AC": 120, "BD": 90},
            "is_started": True,
        })
        app_module.GAMES[game_id] = game
        try:
            result = await app_module.reset_private_room_all_seats(
                game_id,
                requester="A",
                client_id="host-client",
            )
            reset = app_module.GAMES[game_id]
            assert result["human_seats"] == []
            assert result["ai_seats"] == []
            assert reset["human_seats"] == {}
            assert reset["ai_seats"] == []
            assert reset["player_names"] == {seat: "" for seat in app_module.ALL_SEATS}
            assert reset["player_tags"] == {seat: "" for seat in app_module.ALL_SEATS}
            assert reset["total_team_score"] == {"AC": 0, "BD": 0}
            assert reset["is_started"] is False
            assert reset["password"] == "room-pass"
            assert reset["owner_name"] == "研究部屋"
            assert reset["show_log"] is True
            assert reset["ai_profile"] == "intermediate_middle2"
        finally:
            app_module._cancel_turn_timeout_task(game_id)
            if previous is None:
                app_module.GAMES.pop(game_id, None)
            else:
                app_module.GAMES[game_id] = previous
            app_module.GAME_TURN_LOCKS.pop(game_id, None)

    asyncio.run(scenario())


def test_reset_all_seats_requires_private_room_host() -> None:
    async def scenario() -> None:
        game_id = app_module.PRIVATE_A_GID
        previous = app_module.GAMES.get(game_id)
        game = app_module._create_game_obj()
        game["human_seats"] = {"A": "host-client"}
        app_module.GAMES[game_id] = game
        try:
            with pytest.raises(HTTPException) as wrong_client:
                await app_module.reset_private_room_all_seats(
                    game_id,
                    requester="A",
                    client_id="other-client",
                )
            assert wrong_client.value.status_code == 403

            with pytest.raises(HTTPException) as public_room:
                await app_module.reset_private_room_all_seats(
                    app_module.MAIN_GID,
                    requester="A",
                    client_id="host-client",
                )
            assert public_room.value.status_code == 403
        finally:
            app_module._cancel_turn_timeout_task(game_id)
            if previous is None:
                app_module.GAMES.pop(game_id, None)
            else:
                app_module.GAMES[game_id] = previous
            app_module.GAME_TURN_LOCKS.pop(game_id, None)

    asyncio.run(scenario())


def test_disconnected_seat_cleanup_keeps_connected_owner() -> None:
    async def scenario() -> None:
        game_id = app_module.PRIVATE_A_GID
        client_id = "recovery-client"
        key = (game_id, client_id)
        previous_game = app_module.GAMES.get(game_id)
        previous_connections = app_module.manager.client_connections.get(key)
        game = app_module._create_game_obj()
        game["human_seats"] = {"B": client_id}
        game["player_names"]["B"] = "切断者"
        app_module.GAMES[game_id] = game
        try:
            app_module.manager.client_connections[key] = {object()}
            assert await app_module._release_disconnected_client_seats(game_id, client_id) is False
            assert game["human_seats"] == {"B": client_id}

            app_module.manager.client_connections.pop(key, None)
            assert await app_module._release_disconnected_client_seats(game_id, client_id) is True
            assert game["human_seats"] == {}
            assert game["player_names"]["B"] == ""
        finally:
            if previous_connections is None:
                app_module.manager.client_connections.pop(key, None)
            else:
                app_module.manager.client_connections[key] = previous_connections
            if previous_game is None:
                app_module.GAMES.pop(game_id, None)
            else:
                app_module.GAMES[game_id] = previous_game

    asyncio.run(scenario())


def test_heartbeat_timeout_is_ninety_seconds() -> None:
    assert app_module.CLIENT_HEARTBEAT_TIMEOUT_SECONDS == 90


def test_unused_room_cleanup_releases_game_state_and_preserves_settings() -> None:
    async def scenario() -> None:
        game_id = app_module.PRIVATE_A_GID
        previous_game = app_module.GAMES.get(game_id)
        previous_connections = app_module.manager.active_connections.get(game_id)
        old_game = app_module._create_game_obj(
            dealer="C",
            ai_profile="intermediate_middle2",
            deal_mode="frequent",
        )
        old_agents = old_game["agents"]
        old_game.update({
            "human_seats": {"A": "departed-client"},
            "ai_seats": ["B", "C", "D"],
            "password": "room-pass",
            "admin_password_hash": "",
            "owner_name": "研究部屋",
            "show_log": True,
            "room_background_image": "/static/test.png",
            "total_team_score": {"AC": 120, "BD": 90},
            "is_started": True,
            "empty_room_cleanup_armed": True,
            "empty_room_since": 0.0,
        })
        app_module.GAMES[game_id] = old_game
        app_module.manager.active_connections.pop(game_id, None)
        try:
            await app_module._sweep_empty_rooms()
            reset = app_module.GAMES[game_id]
            assert reset is not old_game
            assert reset["agents"] is not old_agents
            assert reset["human_seats"] == {}
            assert reset["ai_seats"] == ["B", "C", "D"]
            assert reset["total_team_score"] == {"AC": 0, "BD": 0}
            assert reset["is_started"] is False
            assert reset["password"] == "room-pass"
            assert reset["owner_name"] == "研究部屋"
            assert reset["show_log"] is True
            assert reset["ai_profile"] == "intermediate_middle2"
            assert reset["deal_mode"] == "frequent"
            assert reset["empty_room_cleanup_armed"] is False
        finally:
            app_module._cancel_turn_timeout_task(game_id)
            if previous_game is None:
                app_module.GAMES.pop(game_id, None)
            else:
                app_module.GAMES[game_id] = previous_game
            if previous_connections is None:
                app_module.manager.active_connections.pop(game_id, None)
            else:
                app_module.manager.active_connections[game_id] = previous_connections
            app_module.GAME_TURN_LOCKS.pop(game_id, None)

    asyncio.run(scenario())


def test_unused_room_cleanup_waits_while_someone_is_connected() -> None:
    async def scenario() -> None:
        game_id = app_module.PRIVATE_A_GID
        previous_game = app_module.GAMES.get(game_id)
        previous_connections = app_module.manager.active_connections.get(game_id)
        game = app_module._create_game_obj()
        game["empty_room_cleanup_armed"] = True
        game["empty_room_since"] = 0.0
        app_module.GAMES[game_id] = game
        app_module.manager.active_connections[game_id] = [object()]
        try:
            await app_module._sweep_empty_rooms()
            assert app_module.GAMES[game_id] is game
            assert "empty_room_since" not in game
            assert game["empty_room_cleanup_armed"] is True
        finally:
            if previous_game is None:
                app_module.GAMES.pop(game_id, None)
            else:
                app_module.GAMES[game_id] = previous_game
            if previous_connections is None:
                app_module.manager.active_connections.pop(game_id, None)
            else:
                app_module.manager.active_connections[game_id] = previous_connections
            app_module.GAME_TURN_LOCKS.pop(game_id, None)

    asyncio.run(scenario())

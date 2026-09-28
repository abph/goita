from __future__ import annotations

import asyncio
import time

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from backend import app as app_module


def _same_origin_request(path: str = "/") -> Request:
    return Request({
        "type": "http",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": [(b"x-goita-member", b"1")],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    })


def test_private_room_seat_vacate_request_can_be_acknowledged() -> None:
    async def scenario() -> None:
        game_id = app_module.PRIVATE_A_GID
        requester_id = "spectator-client"
        target_id = "seat-b-client"
        connection_key = (game_id, requester_id)
        previous_game = app_module.GAMES.get(game_id)
        previous_connections = app_module.manager.client_connections.get(connection_key)
        game = app_module._create_game_obj()
        game["human_seats"] = {"B": target_id}
        app_module.GAMES[game_id] = game
        app_module.manager.client_connections[connection_key] = {object()}
        try:
            result = await app_module.request_seat_vacate(
                game_id,
                app_module.SeatVacateRequest(client_id=requester_id, seat="B"),
                _same_origin_request(f"/games/{game_id}/seat_vacate/request"),
            )
            assert result["ok"] is True
            assert game["seat_vacate_requests"]["B"]["target_client_id"] == target_id
            assert game["human_seats"]["B"] == target_id
            requester_view = app_module._state_public_view(
                game["state"],
                game_id=game_id,
                viewer="W",
                game_obj=game,
                client_id=requester_id,
            )
            target_view = app_module._state_public_view(
                game["state"],
                game_id=game_id,
                viewer="B",
                game_obj=game,
                client_id=target_id,
            )
            assert requester_view["seat_vacate_requests"]["B"]["is_target"] is False
            assert target_view["seat_vacate_requests"]["B"]["is_target"] is True
            assert "requester_client_id" not in target_view["seat_vacate_requests"]["B"]

            assert app_module._acknowledge_seat_vacate_requests(game_id, target_id) is True
            assert "B" not in game["seat_vacate_requests"]
            assert game["seat_vacate_request_cooldowns"]["B"] > time.time()
            assert game["human_seats"]["B"] == target_id
            await asyncio.sleep(0)
        finally:
            app_module._cancel_all_seat_vacate_requests(game_id, game)
            if previous_connections is None:
                app_module.manager.client_connections.pop(connection_key, None)
            else:
                app_module.manager.client_connections[connection_key] = previous_connections
            if previous_game is None:
                app_module.GAMES.pop(game_id, None)
            else:
                app_module.GAMES[game_id] = previous_game

    asyncio.run(scenario())


def test_expired_seat_vacate_request_removes_only_unchanged_occupant() -> None:
    async def scenario() -> None:
        game_id = app_module.PRIVATE_A_GID
        target_id = "stale-seat-client"
        previous_game = app_module.GAMES.get(game_id)
        game = app_module._create_game_obj()
        game["human_seats"] = {"B": target_id}
        game["player_names"]["B"] = "切断者"
        game["player_tags"]["B"] = "ごいた初心者"
        token = "test-token"
        game["seat_vacate_requests"]["B"] = {
            "token": token,
            "target_client_id": target_id,
            "requester_client_id": "requester-client",
            "expires_at": time.time() - 1,
        }
        app_module.GAMES[game_id] = game
        try:
            await app_module._expire_seat_vacate_request(game_id, "B", token)
            assert "B" not in game["human_seats"]
            assert game["player_names"]["B"] == ""
            assert game["player_tags"]["B"] == ""
            assert "B" not in game["seat_vacate_requests"]
            assert game["chat_messages"][-1]["message"] == "B席は応答がなかったため空席になりました。"
        finally:
            app_module._cancel_all_seat_vacate_requests(game_id, game)
            if previous_game is None:
                app_module.GAMES.pop(game_id, None)
            else:
                app_module.GAMES[game_id] = previous_game

    asyncio.run(scenario())


def test_seat_vacate_request_rejects_public_room_and_own_seat() -> None:
    async def scenario() -> None:
        with pytest.raises(HTTPException) as public_error:
            await app_module.request_seat_vacate(
                app_module.MAIN_GID,
                app_module.SeatVacateRequest(client_id="requester", seat="B"),
                _same_origin_request("/games/main/seat_vacate/request"),
            )
        assert public_error.value.status_code == 403

        game_id = app_module.PRIVATE_A_GID
        client_id = "seat-owner"
        connection_key = (game_id, client_id)
        previous_game = app_module.GAMES.get(game_id)
        previous_connections = app_module.manager.client_connections.get(connection_key)
        game = app_module._create_game_obj()
        game["human_seats"] = {"B": client_id}
        app_module.GAMES[game_id] = game
        app_module.manager.client_connections[connection_key] = {object()}
        try:
            with pytest.raises(HTTPException) as own_seat_error:
                await app_module.request_seat_vacate(
                    game_id,
                    app_module.SeatVacateRequest(client_id=client_id, seat="B"),
                    _same_origin_request(f"/games/{game_id}/seat_vacate/request"),
                )
            assert own_seat_error.value.status_code == 409
        finally:
            if previous_connections is None:
                app_module.manager.client_connections.pop(connection_key, None)
            else:
                app_module.manager.client_connections[connection_key] = previous_connections
            if previous_game is None:
                app_module.GAMES.pop(game_id, None)
            else:
                app_module.GAMES[game_id] = previous_game

    asyncio.run(scenario())


def test_frontend_exposes_private_seat_vacate_flow() -> None:
    html = app_module.FRONTEND_DIR.joinpath("index.html").read_text(encoding="utf-8")
    assert 'id="seatModeVacateRequestBtn"' in html
    assert 'id="seatVacateRequestModal"' in html
    assert 'onclick="respondSeatVacateRequest()"' in html
    assert "/seat_vacate/request" in html
    assert "/seat_vacate/respond" in html
    assert "syncSeatVacateRequests(state);" in html
    assert "occupiedByOther && !PRIVATE_ROOM_IDS.has(gid)" in html

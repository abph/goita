from __future__ import annotations

import asyncio

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


def test_private_room_participant_can_vacate_another_human_seat_immediately() -> None:
    async def scenario() -> None:
        game_id = app_module.PRIVATE_A_GID
        requester_id = "spectator-client"
        target_id = "seat-b-client"
        connection_key = (game_id, requester_id)
        previous_game = app_module.GAMES.get(game_id)
        previous_connections = app_module.manager.client_connections.get(connection_key)
        game = app_module._create_game_obj()
        game["human_seats"] = {"B": target_id}
        game["player_names"]["B"] = "残った人"
        game["player_tags"]["B"] = "ごいた初心者"
        app_module.GAMES[game_id] = game
        app_module.manager.client_connections[connection_key] = {object()}
        try:
            result = await app_module.vacate_private_room_seat(
                game_id,
                app_module.SeatVacateRequest(client_id=requester_id, seat="B"),
                _same_origin_request(f"/games/{game_id}/seat_vacate"),
            )
            assert result == {"ok": True, "vacated_seat": "B"}
            assert "B" not in game["human_seats"]
            assert game["player_names"]["B"] == ""
            assert game["player_tags"]["B"] == ""
            assert game["chat_messages"][-1]["message"] == "B席が空席になりました。"
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


def test_immediate_seat_vacate_is_private_only_and_cannot_target_own_seat() -> None:
    async def scenario() -> None:
        with pytest.raises(HTTPException) as public_error:
            await app_module.vacate_private_room_seat(
                app_module.MAIN_GID,
                app_module.SeatVacateRequest(client_id="requester", seat="B"),
                _same_origin_request("/games/main/seat_vacate"),
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
                await app_module.vacate_private_room_seat(
                    game_id,
                    app_module.SeatVacateRequest(client_id=client_id, seat="B"),
                    _same_origin_request(f"/games/{game_id}/seat_vacate"),
                )
            assert own_seat_error.value.status_code == 409
            assert game["human_seats"] == {"B": client_id}
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


def test_frontend_exposes_immediate_private_seat_vacate() -> None:
    html = app_module.FRONTEND_DIR.joinpath("index.html").read_text(encoding="utf-8")
    assert 'id="seatModeVacateBtn"' in html
    assert 'onclick="vacateOccupiedSeat()"' in html
    assert "/seat_vacate`" in html
    assert "occupiedByOther && !PRIVATE_ROOM_IDS.has(gid)" in html
    assert "btn.classList.toggle('occupied-seat', occupiedByOther);" in html
    assert ".seat-btn.occupied-seat" in html
    assert "seatVacateRequestModal" not in html
    assert "退席確認中" not in html

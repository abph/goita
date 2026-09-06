import asyncio

import pytest
from fastapi import HTTPException

from backend import app as game_app


@pytest.fixture
def game(monkeypatch, tmp_path):
    room_id = tmp_path.name
    game = game_app._create_game_obj(dealer="A")
    game.update(is_started=True, human_seats={"A": "owner"})
    monkeypatch.setitem(game_app.GAMES, room_id, game)
    monkeypatch.setattr(game_app, "_schedule_ai_background_search", lambda *_: None)
    return room_id, game


def request(room_id, game, token=True):
    action = game_app.get_legal_actions(room_id, "A", "owner")[0]
    data = dict(player="A", client_id="owner", action=action)
    if token:
        data["expected_action_token"] = game_app.get_state(room_id, "A", "owner")["action_token"]
    return game_app.StepRequest(**data)


def test_confirmed_step_returns_fresh_state_and_rejects_duplicate(game):
    room_id, state = game
    command = request(room_id, state)
    response = asyncio.run(game_app.step(room_id, command))
    assert response["state"]["action_token"] != command.expected_action_token
    assert response["state"]["board_public"] == state["board"]
    with pytest.raises(HTTPException) as error:
        asyncio.run(game_app.step(room_id, command))
    assert error.value.status_code == 409
    assert len(state["kifu_moves"]) == 1


def test_concurrent_retries_apply_only_once(game, monkeypatch):
    room_id, state = game
    command = request(room_id, state)
    async def broadcast(*_): await asyncio.sleep(0.001)
    monkeypatch.setattr(game_app.manager, "broadcast_update", broadcast)
    async def send():
        return await asyncio.gather(game_app.step(room_id, command), game_app.step(room_id, command), return_exceptions=True)
    results = asyncio.run(send())
    assert sum(isinstance(result, dict) for result in results) == 1
    assert [result.status_code for result in results if isinstance(result, HTTPException)] == [409]
    assert len(state["kifu_moves"]) == 1


def test_reset_invalidates_old_command_and_token_never_grants_seat_ownership(game, monkeypatch):
    room_id, state = game
    command = request(room_id, state)
    replacement = game_app._create_game_obj(dealer="A")
    replacement.update(is_started=True, human_seats={"A": "owner"})
    monkeypatch.setitem(game_app.GAMES, room_id, replacement)
    with pytest.raises(HTTPException) as error:
        asyncio.run(game_app.step(room_id, command))
    assert error.value.status_code == 409
    new_command = request(room_id, replacement)
    new_command.client_id = "intruder"
    with pytest.raises(HTTPException) as error:
        asyncio.run(game_app.step(room_id, new_command))
    assert error.value.status_code == 403
    assert replacement["kifu_moves"] == []


def test_legacy_clients_can_submit_without_token(game):
    room_id, state = game
    assert asyncio.run(game_app.step(room_id, request(room_id, state, token=False)))["ok"] is True


def test_update_notifications_correlate_with_snapshot_without_exposing_hands(game, monkeypatch):
    room_id, state = game
    sent = []
    async def capture(channel, payload): sent.append(payload)
    monkeypatch.setattr(game_app.manager, "_broadcast_payload", capture)
    asyncio.run(game_app.manager.broadcast_update(room_id))
    snapshot = game_app.get_state(room_id, "A", "owner")
    assert sent[0] == {"type": "update", "action_token": snapshot["action_token"], "update_version": snapshot["update_version"]}
    asyncio.run(game_app.manager.broadcast_update(room_id))
    assert sent[-1]["update_version"] > snapshot["update_version"]

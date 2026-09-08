import asyncio
import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend import app as game_app
from backend.score_rooms import IDLE_SECONDS, is_score_room
from test_trace_results import payload, trace_client


@pytest.fixture(autouse=True)
def remove_rooms():
    yield
    for room in list(game_app.GAMES):
        if is_score_room(room):
            game_app.GAMES.pop(room)
            game_app.GAME_TURN_LOCKS.pop(room, None)


def enter(client, client_id="owner"):
    response = client.post("/api/score-attack/enter", json={"client_id": client_id, "name": "挑戦者"})
    assert response.status_code == 200, response.text
    return response.json()["game_id"]


def begin(client, room):
    response = client.post(f"/games/{room}/trace_random_start", json={"client_id": "owner"})
    assert response.status_code == 200, response.text
    return response.json()["state"]["trace_attempt_id"]


def finish(client, room):
    for _ in range(200):
        game = game_app.GAMES[room]
        state = game["state"]
        if state.finished:
            return
        if state.turn == "A":
            action = game_app._debug_trace_action(game, state, "A", state.legal_actions("A"))
            assert action is not None
            response = client.post(f"/games/{room}/step", json={"client_id": "owner", "player": "A",
                "action": {"action_type": action[0], "block": action[1], "attack": action[2]}})
        else:
            response = client.post(f"/games/{room}/cpu_step")
        assert response.status_code == 200, response.text
    pytest.fail("round did not finish")


def test_entry_reconnect_before_first_attempt_and_owner_access(trace_client):
    client = trace_client
    room = enter(client)
    assert enter(client) == room
    assert client.get(f"/games/{room}/trace_results/history").json()["records"] == []
    state = client.get(f"/games/{room}/state?client_id=owner").json()
    assert state["owned_human_seats"] == ["A"]
    assert state["ai_seats"] == ["B", "C", "D"]
    assert state["is_score_attack_room"]
    listing = client.get("/games/list").json()
    assert room not in str(listing)
    other = TestClient(game_app.app, headers={"X-Goita-Member": "1"})
    assert enter(other) != room
    for endpoint in ["state?client_id=owner", "legal_actions?player=A&client_id=owner", "kifu", "trace_results/history"]:
        assert other.get(f"/games/{room}/{endpoint}").status_code == 403
    assert other.post(f"/games/{room}/cpu_step").status_code == 403
    assert client.get(f"/games/{room}/state", headers={"Origin": "https://evil.test"}).status_code == 403
    assert client.get(f"/games/{room}/state", headers={"X-Goita-Member": ""}).status_code == 403
    with pytest.raises(WebSocketDisconnect):
        with other.websocket_connect(f"/ws/{room}?client_id=owner", headers={"origin": "http://testserver"}):
            pytest.fail("another guest entered the private socket")
    with client.websocket_connect(f"/ws/{room}?client_id=owner", headers={"origin": "http://testserver"}) as socket:
        assert game_app.manager.has_client_connection(room, "owner")
    assert game_app.GAMES[room]["human_seats"] == {"A": "owner"}
    assert not any(key[0] == room for key in game_app.manager.disconnect_tasks)


def test_fixed_conditions_and_end_to_end_result_retry_reset(trace_client):
    client = trace_client
    room = enter(client)
    attempt = begin(client, room)
    for endpoint in ["claim?seat=B", "release?seat=A", "set_ai?seat=B", "reveal_hand?target=B", "toggle_reveal_hands",
                     "reset", "reset_config", "start", "auto_step?player=A", "turn_time_limit", "deal_mode",
                     "trace_start", "trace_same_start", "update_settings", "verify_admin"]:
        response = client.post(f"/games/{room}/{endpoint}", json={"requester": "A", "client_id": "owner"})
        assert response.status_code == 403, (endpoint, response.text)
    assert client.get(f"/games/{room}/beginner_recommendation?player=A&client_id=owner").status_code == 403
    assert client.get(f"/games/{room}/voice/config?seat=A&client_id=owner").status_code == 403
    assert client.get(f"/games/{room}/kifu").status_code == 409
    assert client.post(f"/games/{room}/score_reset", json={"client_id": "owner"}).status_code == 409
    assert client.post(f"/games/{room}/trace_random_start", json={"client_id": "owner"}).status_code == 409
    state = client.get(f"/games/{room}/state?client_id=owner&viewer=B&reveal_hands=1").json()
    assert all(isinstance(state["hands"][seat], dict) for seat in ("B", "C", "D"))
    assert "score_owner" not in state and "trace_payload" not in state
    finish(client, room)
    path = f"/games/{room}/trace_results/{attempt}"
    assert client.get(path).json()["improvement"] == 0
    assert client.get(path + "/original").status_code == 200
    other = TestClient(game_app.app, headers={"X-Goita-Member": "1"})
    enter(other)
    assert other.get(path + "/original").status_code == 403
    # The separate member-library save route must also enforce ownership.
    with pytest.raises(game_app.HTTPException) as error:
        cookie = "; ".join(f"{key}={value}" for key, value in other.cookies.items())
        request = game_app.Request({"type": "http", "headers": [(b"cookie", cookie.encode())],
                                   "scheme": "http", "server": ("testserver",80), "path":"/"})
        game_app._member_kifu_snapshot(request, room, True)
    assert error.value.status_code in {401,403}
    assert client.post(path + "/retry", json={"client_id": "owner"}).status_code == 403
    begin(client, room)
    finish(client, room)
    assert client.get(f"/games/{room}/trace_results/history").json()["total"] == 2
    assert client.post(f"/games/{room}/score_reset", json={"client_id": "owner"}).status_code == 200
    state = client.get(f"/games/{room}/state?client_id=owner").json()
    assert not state["is_started"] and state["owned_human_seats"] == ["A"]
    assert state["ai_seats"] == ["B", "C", "D"]
    assert enter(client) == room
    assert client.get(path).status_code == 200


def test_idle_cleanup_ignores_polling_and_ai_preserves_completed_records(trace_client):
    client = trace_client
    room = enter(client)
    attempt = begin(client, room)
    finish(client, room)
    game = game_app.GAMES[room]
    old = time.monotonic() - IDLE_SECONDS + 10
    game["score_last_active"] = old
    assert client.get(f"/games/{room}/state?client_id=owner").status_code == 200
    assert client.post(f"/games/{room}/cpu_step").status_code == 200
    assert game["score_last_active"] == old
    assert client.post(f"/games/{room}/score_activity").status_code == 200
    assert game["score_last_active"] > old
    game["score_last_active"] = time.monotonic() - IDLE_SECONDS - 1
    asyncio.run(game_app._sweep_score_rooms())
    assert room not in game_app.GAMES and room not in game_app.GAME_TURN_LOCKS
    assert client.get(f"/games/{room}/state").status_code == 410
    new_room = enter(client)
    assert room != new_room
    assert client.get(f"/games/{new_room}/trace_results/{attempt}").status_code == 200
    begin(client, new_room)
    game_app.GAMES[new_room]["score_last_active"] -= IDLE_SECONDS + 1
    assert client.post(f"/games/{new_room}/score_activity").status_code == 410
    assert new_room not in game_app.GAMES


def test_member_room_reused_across_sessions_and_guest_rankings_shared(trace_client, monkeypatch):
    client = trace_client
    guest_room = enter(client)
    attempt = begin(client, guest_room)
    finish(client, guest_room)
    member = TestClient(game_app.app, headers={"X-Goita-Member":"1"})
    monkeypatch.setattr(game_app.MEMBER_STORE, "authenticate", lambda _: {"member_id":"one"})
    member.cookies.set(game_app.MEMBER_COOKIE, "member-session")
    room = enter(member)
    other_device = TestClient(game_app.app, headers={"X-Goita-Member":"1"})
    other_device.cookies.set(game_app.MEMBER_COOKIE, "other-session")
    assert enter(other_device, "new-device") == room
    assert game_app.GAMES[room]["human_seats"] == {"A":"new-device"}
    assert enter(member) == room
    member_attempt = begin(member, room)
    finish(member, room)
    result = member.get(f"/games/{room}/trace_results/{member_attempt}").json()
    assert result["total"] == 2
    assert client.get(f"/games/{guest_room}/trace_results/{attempt}").json()["total"] == 2


def test_personal_chat_and_ai_are_available_only_to_owner(trace_client, monkeypatch):
    client = trace_client
    room = enter(client)
    body = {"client_id": "owner", "seat": "A", "message": "テスト"}
    response = client.post(f"/games/{room}/chat", json=body)
    assert response.status_code == 200
    assert any(item["message"] == "テスト" for item in response.json()["chat_messages"])
    async def answer(*args):
        return "テスト回答"
    monkeypatch.setattr(game_app, "_resolve_chat_ai_answer", answer)
    response = client.post(f"/games/{room}/chat/ask_ai", json=body)
    assert response.status_code == 200 and response.json()["answer"] == "テスト回答"
    other = TestClient(game_app.app, headers={"X-Goita-Member": "1"})
    enter(other)
    for endpoint in ("chat", "chat/ask_ai"):
        assert other.post(f"/games/{room}/{endpoint}", json=body).status_code == 403

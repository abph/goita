from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.app as app_module
from backend.member_api import MEMBER_COOKIE
from backend.member_store import MemberStore


@pytest.fixture
def stamp_env(monkeypatch, tmp_path):
    store = MemberStore(tmp_path / "members.sqlite3")
    store.clock = lambda: datetime(2026, 9, 4, 12, tzinfo=timezone.utc).timestamp()
    monkeypatch.setattr(app_module, "MEMBER_STORE", store)
    monkeypatch.setattr(app_module, "GAMES", {
        "main": {"chat_messages": []}, app_module.PRIVATE_A_GID: {"chat_messages": []},
        app_module.DEBUG_GID: {"chat_messages": []},
    })
    monkeypatch.setattr(app_module, "_ensure_main_game", lambda _: None)
    monkeypatch.setattr(app_module, "_chat_messages_for_game", lambda _, game: game["chat_messages"])
    monkeypatch.setattr(app_module, "LAST_CHAT_TIMESTAMP", 0)
    monkeypatch.setattr(app_module, "PUBLIC_CHAT_MESSAGES", [])
    monkeypatch.setattr(app_module, "LOBBY_HERE_CHAT_MESSAGES", [])
    monkeypatch.setattr(app_module, "EVERYONE_CHAT_MESSAGES", [])

    async def no_broadcast(_):
        pass

    monkeypatch.setattr(app_module.manager, "broadcast_update", no_broadcast)
    app = FastAPI()
    app.post("/games/{game_id}/chat")(app_module.post_chat_message)
    app.post("/lobby/chat")(app_module.post_lobby_chat_message)
    with TestClient(app, base_url="https://testserver", headers={
        "Origin": "https://testserver", "X-Goita-Member": "1",
    }) as client:
        yield client, store


def send(client, stamp="happy", room="main", **kwargs):
    url = "/lobby/chat" if room == "lobby" else f"/games/{room}/chat"
    return client.post(url, json={"message": "", "stamp_id": stamp}, **kwargs)


def login_paid(client, store, temporary=False):
    issued = store.create("tester")
    _, token, _ = store.login("tester", issued["temporary_password"])
    if not temporary:
        _, token, _ = store.change_password(token, issued["temporary_password"], "abcd1234")
    client.cookies.set(MEMBER_COOKIE, token)
    return token


def login_operator(client, store):
    issued = store.create("operator", paid_enabled=False, is_operator=True)
    _, token, _ = store.login("operator", issued["temporary_password"])
    _, token, _ = store.change_password(token, issued["temporary_password"], "abcd1234")
    client.cookies.set(MEMBER_COOKIE, token)


def test_chat_stamp_keeps_its_chat_entry_and_only_animates_at_its_table(stamp_env, monkeypatch):
    client, _ = stamp_env
    game = app_module.GAMES["main"]
    game.update(is_started=True, human_seats={"C": "owner"}, member_kifu_round_id="first")
    broadcasts = []

    async def send_event(channel, payload):
        broadcasts.append((channel, payload))

    monkeypatch.setattr(app_module.manager, "_broadcast_payload", send_event)
    url = "/games/main/chat"
    stamp = {"seat": "C", "client_id": "owner", "stamp_id": "nice", "message": ""}
    response = client.post(url, json={**stamp, "client_id": "other"}).json()
    assert response["chat_messages"][-1]["seat"] == "W"
    assert not response["board_stamps"]
    response = client.post(url, json=stamp)
    assert response.status_code == 200
    data = response.json()
    event, = data["board_stamps"]
    assert event == {"seat": "C", "stamp_id": "nice", "id": data["chat_messages"][-1]["ts"]}
    assert game["chat_messages"][-1]["stamp_id"] == "nice"
    assert broadcasts == [("main", {"type": "board_stamp", "board_stamp": event})]
    assert app_module._live_board_stamps(game) == [event]
    assert not app_module._live_board_stamps(app_module.GAMES[app_module.PRIVATE_A_GID])

    response = client.post(url, json={**stamp, "message": "@everyone"}).json()
    assert response["board_stamps"][0]["stamp_id"] == "nice"
    assert len(broadcasts) == 2 and all(channel == "main" for channel, _ in broadcasts)
    assert app_module.EVERYONE_CHAT_MESSAGES[-1]["stamp_id"] == "nice"
    assert not app_module._live_board_stamps(app_module.GAMES[app_module.PRIVATE_A_GID])

    game["member_kifu_round_id"] = "second"
    assert not app_module._live_board_stamps(game)
    game["board_stamps"]["C"] = (event, app_module.time.monotonic() - 1, "second")
    assert not app_module._live_board_stamps(game)
    game["is_started"] = False
    assert not client.post(url, json=stamp).json()["board_stamps"]


def test_all_twelve_chat_stamps_share_board_animation(stamp_env, monkeypatch):
    client, store = stamp_env
    room = app_module.PRIVATE_A_GID
    game = app_module.GAMES[room]
    game.update(is_started=True, human_seats={"A": "owner"})

    async def no_broadcast(*_args):
        pass

    monkeypatch.setattr(app_module.manager, "_broadcast_payload", no_broadcast)
    url = f"/games/{room}/chat"
    message = {"seat": "A", "client_id": "owner", "message": ""}
    assert len(app_module.CHAT_STAMPS) == 12
    assert client.post(url, json={**message, "stamp_id": "weekly_champion"}).status_code == 401
    login_operator(client, store)
    for stamp_id in app_module.CHAT_STAMPS:
        result = client.post(url, json={**message, "stamp_id": stamp_id}).json()
        assert result["chat_messages"][-1]["stamp_id"] == stamp_id
        assert result["board_stamps"][0]["stamp_id"] == stamp_id
    assert len(app_module._live_board_stamps(game)) == 1



@pytest.mark.parametrize("room", ["main", "lobby"])
def test_public_extra_stamps_require_membership_but_free_stamps_do_not(stamp_env, room):
    client, store = stamp_env
    free_stamps = {"greeting", "thanks", "thinking", "nice", "sorry", "surprised", "next"}
    paid_stamps = {"happy", "leave_it", "got_me", "goita_fun"}
    assert app_module.PUBLIC_CHAT_STAMP_IDS == free_stamps
    for stamp in free_stamps:
        assert send(client, stamp, room).status_code == 200
    for stamp in paid_stamps:
        assert send(client, stamp, room).status_code == 401
    client.cookies.set(MEMBER_COOKIE, "forged-session")
    assert send(client, room=room).status_code == 401
    login_paid(client, store)
    store.update("tester", enabled=True, paid_enabled=False, paid_until=None)
    for stamp in free_stamps:
        assert send(client, stamp, room).status_code == 200
    for stamp in paid_stamps:
        assert send(client, stamp, room).status_code == 403
    store.update("tester", enabled=True, paid_enabled=True, paid_until=None)
    for stamp in set(app_module.CHAT_STAMPS) - app_module.REWARD_CHAT_STAMP_IDS:
        response = send(client, stamp, room)
        assert response.status_code == 200
        item = response.json()["chat_messages"][-1]
        assert item["stamp_id"] == stamp
        assert "member_id" not in item


@pytest.mark.parametrize("reason", ["temporary", "unpaid", "expired", "suspended", "deleted", "reset", "logout"])
@pytest.mark.parametrize("room", ["main", "lobby"])
def test_extra_stamps_recheck_live_permissions(stamp_env, reason, room):
    client, store = stamp_env
    token = login_paid(client, store, temporary=reason == "temporary")
    if reason in ("unpaid", "expired", "suspended"):
        store.update("tester", enabled=reason != "suspended", paid_enabled=reason != "unpaid",
                     paid_until="2026-09-03" if reason == "expired" else None)
    elif reason == "deleted":
        store.delete("tester")
    elif reason == "reset":
        store.reset_password("tester")
    elif reason == "logout":
        store.logout(token)
    assert send(client, room=room).status_code in (401, 403)
    assert app_module.GAMES["main"]["chat_messages"] == []
    assert app_module.PUBLIC_CHAT_MESSAGES == []
    assert send(client, "nice", room).status_code == 200


def test_private_and_debug_stamps_remain_available_without_membership(stamp_env):
    client, _ = stamp_env
    for room in (app_module.PRIVATE_A_GID, app_module.DEBUG_GID):
        assert send(client, room=room).status_code == 200
    assert client.post("/lobby/chat", json={"message": "", "stamp_id": "happy"}).status_code == 401


def test_weekly_champion_stamp_requires_a_first_place_award_and_stays_unlocked(stamp_env):
    client, store = stamp_env
    login_paid(client, store)
    assert send(client, "weekly_champion", "main").status_code == 403
    store.grant_weekly_score_awards("2026-08-24", [{"member_id": "tester", "rank": 1}])
    store.update("tester", enabled=True, paid_enabled=False, paid_until=None)
    response = send(client, "weekly_champion", "main")
    assert response.status_code == 200
    assert response.json()["chat_messages"][-1]["stamp_id"] == "weekly_champion"


@pytest.mark.parametrize("room", ["main", "lobby"])
def test_operator_account_can_use_every_stamp_without_paid_or_reward_access(stamp_env, room):
    client, store = stamp_env
    login_operator(client, store)
    for stamp in app_module.CHAT_STAMPS:
        response = send(client, stamp, room)
        assert response.status_code == 200
        assert response.json()["chat_messages"][-1]["stamp_id"] == stamp


@pytest.mark.parametrize("room", ["main", "lobby"])
def test_paid_stamp_requires_same_origin_header(stamp_env, room):
    client, store = stamp_env
    login_paid(client, store)
    for headers in ({"Origin": "https://evil.example"}, {"X-Goita-Member": ""}, {"Sec-Fetch-Site": "cross-site"}):
        assert send(client, room=room, headers=headers).status_code == 403
    assert app_module.GAMES["main"]["chat_messages"] == []
    assert app_module.PUBLIC_CHAT_MESSAGES == []

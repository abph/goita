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


def send(client, stamp="sorry", room="main", **kwargs):
    return client.post(f"/games/{room}/chat", json={"message": "", "stamp_id": stamp}, **kwargs)


def login_paid(client, store, temporary=False):
    issued = store.create("tester")
    _, token, _ = store.login("tester", issued["temporary_password"])
    if not temporary:
        _, token, _ = store.change_password(token, issued["temporary_password"], "abcd1234")
    client.cookies.set(MEMBER_COOKIE, token)
    return token


def test_public_extra_stamps_require_membership_but_free_stamps_do_not(stamp_env):
    client, store = stamp_env
    for stamp in app_module.PUBLIC_CHAT_STAMP_IDS:
        assert send(client, stamp).status_code == 200
    assert send(client).status_code == 401
    client.cookies.set(MEMBER_COOKIE, "forged-session")
    assert send(client).status_code == 401
    login_paid(client, store)
    for stamp in app_module.CHAT_STAMPS:
        response = send(client, stamp)
        assert response.status_code == 200
        item = response.json()["chat_messages"][-1]
        assert item["stamp_id"] == stamp
        assert "member_id" not in item


@pytest.mark.parametrize("reason", ["temporary", "unpaid", "expired", "suspended", "deleted", "reset", "logout"])
def test_extra_stamps_recheck_live_permissions(stamp_env, reason):
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
    assert send(client).status_code in (401, 403)
    assert app_module.GAMES["main"]["chat_messages"] == []
    assert send(client, "nice").status_code == 200


def test_private_and_debug_stamps_remain_available_without_membership(stamp_env):
    client, _ = stamp_env
    for room in (app_module.PRIVATE_A_GID, app_module.DEBUG_GID):
        assert send(client, room=room).status_code == 200
    assert client.post("/lobby/chat", json={"message": "", "stamp_id": "sorry"}).status_code == 403


def test_paid_stamp_requires_same_origin_header(stamp_env):
    client, store = stamp_env
    login_paid(client, store)
    for headers in ({"Origin": "https://evil.example"}, {"X-Goita-Member": ""}, {"Sec-Fetch-Site": "cross-site"}):
        assert send(client, headers=headers).status_code == 403
    assert app_module.GAMES["main"]["chat_messages"] == []

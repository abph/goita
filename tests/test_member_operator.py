import hashlib
import sqlite3

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.datastructures import URL

from backend import app as app_module
from backend.member_api import MEMBER_COOKIE, create_member_router
from backend.member_store import MemberStore
from backend.analytics_store import AnalyticsStore


@pytest.fixture
def store(tmp_path):
    return MemberStore(tmp_path / "members.sqlite3")


def session(store, operator=True):
    issued = store.create("operator" if operator else "normal", is_operator=operator)
    _, token, _ = store.login(issued["member"]["member_id"], issued["temporary_password"])
    return token


def test_legacy_database_migration_preserves_member_and_session(tmp_path):
    database = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database) as db:
        db.executescript("""
            CREATE TABLE members (member_id TEXT PRIMARY KEY, password_hash TEXT NOT NULL,
                must_change_password INTEGER NOT NULL, temporary_expires_at REAL,
                enabled INTEGER NOT NULL DEFAULT 1, paid_enabled INTEGER NOT NULL DEFAULT 1,
                paid_until TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL);
            CREATE TABLE member_sessions (token_hash TEXT PRIMARY KEY,
                member_id TEXT NOT NULL REFERENCES members(member_id),
                created_at REAL NOT NULL, expires_at REAL NOT NULL);
            INSERT INTO members VALUES ('legacy', 'existing-hash', 0, NULL, 1, 1, NULL, 1, 1);
        """)
        db.execute("INSERT INTO member_sessions VALUES (?, 'legacy', 1, 9999999999)",
                   (hashlib.sha256(b"legacy-token").hexdigest(),))
    migrated = MemberStore(database)
    assert migrated.authenticate("legacy-token")["is_operator"] is False
    assert migrated.create("new-user")["member"]["is_operator"] is False
    assert len(MemberStore(database).list_members()) == 2
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT password_hash FROM members WHERE member_id = 'legacy'").fetchone()[0] == "existing-hash"


def test_operator_flag_is_admin_only_and_does_not_grant_admin_access(store):
    app = FastAPI()

    def admin(request):
        if request.cookies.get("test-admin") != "yes":
            raise HTTPException(401)

    app.include_router(create_member_router(store, admin))
    with TestClient(app, base_url="https://testserver", headers={"X-Goita-Member": "1"}) as client:
        assert client.post("/admin/api/members", json={"member_id": "operator", "is_operator": True}).status_code == 401
        client.cookies.set("test-admin", "yes")
        issued = client.post("/admin/api/members", json={"member_id": "operator", "is_operator": True}).json()
        assert issued["member"]["is_operator"] is True
        client.cookies.clear()
        credentials = {"member_id": "operator", "password": issued["temporary_password"]}
        assert client.post("/api/member/login", json=dict(credentials, is_operator=True)).status_code == 422
        assert client.post("/api/member/login", json=credentials).status_code == 200
        assert client.get("/admin/api/members").status_code == 401
        assert client.put("/admin/api/members/operator", json={"enabled": True, "paid_enabled": True, "is_operator": False}).status_code == 401
        client.cookies.set("test-admin", "yes")
        # An older admin client omitting the field must preserve it.
        assert client.put("/admin/api/members/operator", json={"enabled": True, "paid_enabled": True}).json()["member"]["is_operator"] is True
        assert client.put("/admin/api/members/operator", json={"enabled": True, "paid_enabled": True, "is_operator": False}).json()["member"]["is_operator"] is False


def test_operator_flag_rechecks_changes_logout_and_expiry(store):
    token = session(store)
    assert store.is_operator_session(token) is True
    store.update("operator", enabled=True, paid_enabled=False, paid_until=None, is_operator=False)
    assert store.is_operator_session(token) is False
    store.update("operator", enabled=True, paid_enabled=False, paid_until=None, is_operator=True)
    assert store.is_operator_session(token) is True  # independent of paid membership
    original_clock = store.clock
    store.clock = lambda: original_clock() + 31 * 86400
    assert store.is_operator_session(token) is False
    store.clock = original_clock
    store.logout(token)
    assert store.is_operator_session(token) is False
    assert store.is_operator_session("forged") is False
    assert store.is_operator_session("") is False


def test_operator_analytics_cookie_blocks_initial_visit_and_beacon(store, tmp_path, monkeypatch):
    token = session(store)
    analytics = AnalyticsStore(tmp_path / "analytics.sqlite3")
    monkeypatch.setattr(app_module, "MEMBER_STORE", store)
    monkeypatch.setattr(app_module, "ANALYTICS_STORE", analytics)
    client = TestClient(app_module.app, base_url="https://testserver")
    client.cookies.set(MEMBER_COOKIE, token)
    event = {"analytics_id": "visitor_operator123456", "session_id": "session_operator123456",
             "event": "site_visit", "room_type": "lobby", "properties": {}}
    for name in ["site_visit", "heartbeat", "room_enter", "room_leave", "kifu_loaded"]:
        assert client.post("/analytics/event", json=dict(event, event=name)).status_code == 200
    assert analytics.snapshot(days=30)["visitors"] == 0
    client.cookies.clear()
    assert client.post("/analytics/event", json=event).status_code == 200
    assert analytics.snapshot(days=30)["visitors"] == 1


class Socket:
    def __init__(self, token="", origin="https://testserver"):
        self.cookies = {MEMBER_COOKIE: token}
        self.headers = {"origin": origin}
        self.url = URL("wss://testserver/ws/lobby")


def test_operator_hidden_from_presence_and_spectator_totals(store, monkeypatch):
    token = session(store)
    ordinary = session(store, operator=False)
    manager = app_module.ConnectionManager()
    monkeypatch.setattr(app_module, "MEMBER_STORE", store)
    monkeypatch.setattr(app_module, "manager", manager)
    monkeypatch.setattr(app_module, "GAMES", {})
    app_module.setup_main_rooms()
    room = next(iter(app_module.MAIN_ROOM_NAMES))
    manager.client_connections = {
        ("lobby", "hidden"): {Socket(token)},
        (room, "hidden"): {Socket(token), Socket(token)},
        (room, "visible"): {Socket(ordinary)},
        ("lobby", "guest"): {Socket()},
    }
    manager.client_names = {key: key[1] for key in manager.client_connections}
    response = app_module.list_rooms()
    assert {p["name"] for p in response["site_people"]} == {"visible", "guest"}
    info = next(r for r in response["rooms"] if r["game_id"] == room)
    assert info["spectator_count"] == 1
    assert info["people_count"] == 1
    assert manager.spectator_count(room, app_module.GAMES[room]) == 1
    store.update("operator", enabled=True, paid_enabled=True, paid_until=None, is_operator=False)
    assert manager.spectator_count(room, app_module.GAMES[room]) == 2
    assert "hidden" in {p["name"] for p in app_module.list_rooms()["site_people"]}
    store.update("operator", enabled=True, paid_enabled=True, paid_until=None, is_operator=True)
    store.logout(token)
    assert manager.hidden_client_ids() == set()


def test_untrusted_socket_cannot_use_operator_cookie_for_hiding(store, monkeypatch):
    token = session(store)
    monkeypatch.setattr(app_module, "MEMBER_STORE", store)
    manager = app_module.ConnectionManager()
    manager.client_connections = {("lobby", "client"): {Socket(token, "https://other.example"), Socket("forged")}}
    assert manager.hidden_client_ids() == set()

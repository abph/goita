from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from backend import app as game_app
from backend.member_api import MEMBER_COOKIE, create_member_router
from backend.member_room_api import create_member_room_router
from backend.member_store import MemberError, MemberStore
from backend.room_settings_persistence import load_room_settings
from test_member_accounts import ready, HEADERS


@pytest.fixture
def managed(tmp_path, monkeypatch):
    store = MemberStore(tmp_path / "members.sqlite3")
    _, token = ready(store, "owner")
    room_id, other_room = list(game_app.PRIVATE_ROOM_NAMES)[:2]
    for key in (room_id, other_room):
        game = game_app._create_game_obj(dealer="A")
        game.update(owner_name="研究室", admin_password="legacy-room-password", password="entry-pass")
        monkeypatch.setitem(game_app.GAMES, key, game)
    monkeypatch.setattr(game_app, "ROOM_SETTINGS_PATH", tmp_path / "rooms.json")
    store.update("owner", enabled=True, paid_enabled=True, paid_until=None,
                 research_enabled=True, managed_room_id=room_id)
    app = FastAPI()
    def admin(request):
        if request.cookies.get("test_admin") != "yes":
            raise HTTPException(401)
    app.include_router(create_member_router(store, admin, room_options=game_app._member_room_options))
    app.include_router(create_member_room_router(store, game_app._member_room_options,
        game_app._room_management_payload, game_app._update_room_management, game_app._vacate_room_seat))
    with TestClient(app, base_url="https://testserver", headers=HEADERS) as client:
        client.cookies.set(MEMBER_COOKIE, token)
        yield store, token, client, room_id, other_room


def settings(room_id, **extra):
    return dict(game_id=room_id, new_owner_name="新しい研究室", ai_profile="current", **extra)


def test_owner_can_read_save_and_persist_without_entering_room(managed):
    store, token, client, room_id, other_room = managed
    response = client.get("/api/member/room")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["room"]["game_id"] == room_id
    assert "legacy-room-password" not in response.text and "entry-pass" not in response.text
    assert client.post("/api/member/room/settings", json=settings(room_id,
        update_password=True, new_password="new-entry", show_log=True)).status_code == 200
    assert game_app.GAMES[room_id]["password"] == "new-entry"
    assert load_room_settings(game_app.ROOM_SETTINGS_PATH)[room_id]["show_log"] is True
    assert game_app.GAMES[other_room]["owner_name"] == "研究室"
    assert client.post("/api/member/room/settings", json=settings(room_id)).status_code == 200
    assert game_app.GAMES[room_id]["password"] == "new-entry"
    assert client.post("/api/member/room/settings", json=settings(room_id, update_password=True)).status_code == 200
    assert game_app.GAMES[room_id]["password"] is None
    assert game_app.verify_admin(room_id, "legacy-room-password")["ok"] is True


@pytest.mark.parametrize("reason", ["ordinary", "unassigned", "expired", "unpaid", "disabled", "temporary", "logout"])
def test_room_access_is_rechecked_for_every_read_and_write(managed, reason):
    store, token, client, room_id, other_room = managed
    update = dict(enabled=True, paid_enabled=True, paid_until=None)
    if reason == "ordinary": update["research_enabled"] = False
    if reason == "unassigned": update["managed_room_id"] = ""
    if reason == "expired": update["paid_until"] = "2000-01-01"
    if reason == "unpaid": update["paid_enabled"] = False
    if reason == "disabled": update["enabled"] = False
    store.update("owner", **update)
    if reason == "logout": store.logout(token)
    if reason == "temporary":
        issued = store.reset_password("owner")
        _, temporary, _ = store.login("owner", issued["temporary_password"])
        client.cookies.set(MEMBER_COOKIE, temporary)
    assert client.get("/api/member/room").status_code in (401, 403)
    assert client.post("/api/member/room/settings", json=settings(room_id)).status_code in (401, 403)
    assert client.post("/api/member/room/vacate", json={"game_id": room_id, "seat": "A", "occupancy_token": "test"}).status_code in (401, 403)
    assert game_app.GAMES[room_id]["owner_name"] == "研究室"


def test_cross_origin_other_room_and_stale_assignment_are_rejected(managed):
    store, token, client, room_id, other_room = managed
    assert client.get("/api/member/room", headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.post("/api/member/room/settings", json=settings(room_id), headers={"Origin": "https://evil.example"}).status_code == 403
    for target in (other_room, game_app.MAIN_GID, game_app.DEBUG_GID):
        assert client.post("/api/member/room/settings", json=settings(target)).status_code == 403
        assert client.post("/api/member/room/vacate", json={"game_id": target, "seat": "A", "occupancy_token": "test"}).status_code == 403
    assert client.post("/api/member/room/settings", json=settings(room_id, admin_password="legacy-room-password")).status_code == 422
    store.update("owner", enabled=True, paid_enabled=True, paid_until=None, managed_room_id=other_room)
    assert client.post("/api/member/room/settings", json=settings(room_id)).status_code == 403
    assert client.get("/api/member/room").json()["room"]["game_id"] == other_room
    assert client.post("/api/member/room/settings", json={**settings(other_room), "ai_profile": "unknown"}).status_code == 400


def test_vacate_requires_current_occupant_token(managed):
    store, token, client, room_id, other_room = managed
    game = game_app.GAMES[room_id]
    game.update(human_seats={"B": "player-one"}, player_names={"B": "田中"})
    seat = client.get("/api/member/room").json()["room"]["managed_human_seats"][0]
    body = dict(game_id=room_id, seat="B", occupancy_token=seat["occupancy_token"])
    game["human_seats"]["B"] = "player-two"
    assert client.post("/api/member/room/vacate", json=body).status_code == 409
    seat = client.get("/api/member/room").json()["room"]["managed_human_seats"][0]
    body["occupancy_token"] = seat["occupancy_token"]
    assert client.post("/api/member/room/vacate", json=body).status_code == 200
    assert game["human_seats"] == {}


def test_assignment_is_admin_only_unique_and_preserved_across_restart(managed):
    store, token, client, room_id, other_room = managed
    body = dict(member_id="second", research_enabled=True, managed_room_id=room_id)
    assert client.post("/admin/api/members", json=body).status_code == 401
    client.cookies.set("test_admin", "yes")
    assert client.post("/admin/api/members", json=body).status_code == 409
    assert len(store.list_members()) == 1  # Failed create rolls back completely.
    for bad in (game_app.MAIN_GID, game_app.DEBUG_GID, "missing"):
        assert client.post("/admin/api/members", json={**body, "managed_room_id": bad}).status_code == 400
    issued = client.post("/admin/api/members", json={**body, "managed_room_id": other_room})
    assert issued.status_code == 200
    assert MemberStore(store.path).authenticate(token)["managed_room_id"] == room_id
    update = dict(enabled=True, paid_enabled=True, paid_until=None)
    assert client.put("/admin/api/members/second", json={**update, "managed_room_id": room_id}).status_code == 409
    assert client.put("/admin/api/members/owner", json=update).json()["member"]["managed_room_id"] == room_id
    assert client.put("/admin/api/members/owner", json={**update, "research_enabled": False}).json()["member"]["managed_room_id"] == ""
    assert client.put("/admin/api/members/second", json={**update, "managed_room_id": room_id}).status_code == 200
    assert client.get("/api/member/room").status_code == 403
    assert client.delete("/admin/api/members/second").status_code == 200
    assert client.put("/admin/api/members/owner", json={**update, "research_enabled": True, "managed_room_id": room_id}).status_code == 200


def test_concurrent_assignment_has_one_owner(managed):
    store, token, client, room_id, other_room = managed
    store.create("second")
    def assign(member_id):
        try:
            MemberStore(store.path).update(member_id, enabled=True, paid_enabled=True, paid_until=None,
                research_enabled=True, managed_room_id=other_room)
            return True
        except MemberError as error:
            assert error.status == 409
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(assign, ["owner", "second"])) == 1

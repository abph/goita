import sqlite3
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient

from backend import app as app_module
from backend.member_api import MEMBER_COOKIE
from backend.member_kifu import MemberKifuStore
from backend.member_kifu_api import create_member_kifu_router
from backend.member_store import MemberError, MemberStore
from backend.retire_room_kifu import retire_room_kifu


HEADERS = {"X-Goita-Member": "1", "Origin": "https://testserver"}


def ready(members, name):
    issued = members.create(name)
    _, temporary, _ = members.login(name, issued["temporary_password"])
    return members.change_password(temporary, issued["temporary_password"], "test-password-2026")[1]


@pytest.fixture
def library(tmp_path, monkeypatch):
    members = MemberStore(tmp_path / "members.sqlite3")
    store = MemberKifuStore(members)
    token = ready(members, "alice")
    other = ready(members, "bobby")
    game = {"state": SimpleNamespace(finished=True), "password": None,
            "last_completed_kifu": {"round_index": 3, "player_names": {s: s + " name" for s in "ABCD"},
                                    "hand": {"p0": "private hand"}, "game": []}}
    monkeypatch.setitem(app_module.GAMES, "member-test-room", game)
    app = FastAPI()
    app.include_router(create_member_kifu_router(store, app_module._member_kifu_snapshot,
                                               app_module._parse_research_kifu_text, persistent=True))

    @app.post("/entry")
    def entry(request: Request, response: Response, password: str):
        return app_module.verify_password("member-test-room", password, request=request, response=response)

    client = TestClient(app, base_url="https://testserver", headers=HEADERS)
    client.cookies.set(MEMBER_COOKIE, token)
    with client:
        yield store, members, token, other, client, game


def save(client, **extra):
    return client.post("/api/member/kifu/save", json={"game_id": "member-test-room", "title": "Test", "tags": ["王玉"], **extra})


def test_round_trip_owner_and_other_device(library):
    store, members, token, other, client, game = library
    result = save(client)
    assert result.status_code == 200
    record = result.json()["record"]
    assert record["payload"]["player_names"]["A"] == "A name"
    listed = client.post("/api/member/kifu/list").json()
    assert listed["records"][0]["id"] == record["id"]
    assert "payload" not in listed["records"][0]
    assert "member_id" not in listed["records"][0]
    _, second, _ = members.login("alice", "test-password-2026")
    client.cookies.set(MEMBER_COOKIE, second)
    url = "/api/member/kifu/" + record["id"]
    assert client.post(url).json()["record"] == record
    edited = client.post(url + "/edit", json={"title": "Edited", "memo": "memo", "tags": ["2香"]})
    assert edited.json()["record"]["tags"] == ["2香"]
    assert client.post(url + "/delete").status_code == 200
    assert client.post(url).status_code == 404


@pytest.mark.parametrize("suffix,body", [("", {}), ("/edit", {"title": "stolen"}), ("/delete", {})])
def test_other_member_cannot_access_record(library, suffix, body):
    store, members, token, other, client, game = library
    record_id = save(client).json()["record"]["id"]
    client.cookies.set(MEMBER_COOKIE, other)
    assert client.post("/api/member/kifu/list").json()["records"] == []
    response = client.post("/api/member/kifu/" + record_id + suffix, json=body)
    assert response.status_code == 404
    assert store.access(token, record_id)["title"] == "Test"


def test_anonymous_save_does_not_mutate_shared_snapshot(library):
    *_, client, game = library
    record = save(client, anonymous=True).json()["record"]
    assert record["payload"]["player_names"] == {s: "プレイヤー" + s for s in "ABCD"}
    assert game["last_completed_kifu"]["player_names"]["A"] == "A name"


def test_live_round_is_rejected_even_with_previous_snapshot(library):
    *_, client, game = library
    game["state"].finished = False
    assert save(client).status_code == 409
    assert client.post("/api/member/kifu/list").json()["records"] == []


def test_locked_room_requires_entry_and_new_password_invalidates_entry(library):
    *_, client, game = library
    game["password"] = "room-secret"
    assert save(client).status_code == 403
    assert client.post("/entry?password=wrong").status_code == 401
    assert client.post("/entry?password=room-secret").status_code == 200
    assert save(client).status_code == 200
    game["password"] = "changed"
    assert save(client).status_code == 403


def test_expired_plan_can_read_edit_delete_but_not_add(library):
    store, members, token, other, client, game = library
    record = save(client).json()["record"]
    members.update("alice", enabled=True, paid_enabled=True, paid_until="2020-01-01")
    assert save(client).status_code == 403
    assert client.post("/api/member/kifu/import", json={"kifu_text": "invalid"}).status_code == 403
    url = "/api/member/kifu/" + record["id"]
    assert client.post(url).status_code == 200
    assert client.post(url + "/edit", json={"memo": "retained"}).status_code == 200
    assert client.post(url + "/delete").status_code == 200


def test_login_temporary_revoked_and_origin_checks(library):
    store, members, token, other, client, game = library
    for headers in ({"X-Goita-Member": "0"}, {"Origin": "https://evil.example"}, {"sec-fetch-site": "cross-site"}):
        assert client.post("/api/member/kifu/list", headers=headers).status_code == 403
    assert client.post("/api/member/kifu/list").headers["cache-control"] == "no-store"
    members.logout(token)
    assert client.post("/api/member/kifu/list").status_code == 401
    issued = members.create("tempuser")
    _, temporary, _ = members.login("tempuser", issued["temporary_password"])
    client.cookies.set(MEMBER_COOKIE, temporary)
    assert client.post("/api/member/kifu/list").status_code == 403
    assert save(client).status_code == 403


def test_limit_is_atomic_and_deletion_frees_space(library):
    store, members, token, other, client, game = library
    store.LIMIT = 2
    def attempt(_):
        try:
            store.save(token, title="x", memo="", tags=[], payload={})
            return True
        except MemberError as error:
            assert error.status == 409
            return False
    with ThreadPoolExecutor(max_workers=4) as executor:
        assert sum(executor.map(attempt, range(4))) == 2
    store.access(token, store.list(token)[0]["id"], action="delete")
    assert attempt(0)


def test_default_limit_accepts_1000_records_and_rejects_1001(library):
    store, members, token, other, client, game = library
    assert store.LIMIT == 1000
    with members._db(write=True) as db:
        db.executemany("INSERT INTO member_kifu VALUES (?, ?, ?, ?, ?, ?, ?)", [
            (f"seed-{i}", "alice", "2026-09-05", "seed", "", "[]", "{}") for i in range(999)
        ])
    record = save(client)
    assert record.status_code == 200
    assert len(store.list(token)) == 1000
    overflow = save(client)
    assert overflow.status_code == 409
    assert "1000件" in overflow.json()["detail"]
    store.access(token, record.json()["record"]["id"], action="delete")
    assert save(client).status_code == 200


def test_member_deletion_removes_records_and_recreated_id_cannot_inherit(library):
    store, members, token, other, client, game = library
    record_id = save(client).json()["record"]["id"]
    members.delete("alice")
    with sqlite3.connect(members.path) as db:
        assert db.execute("SELECT COUNT(*) FROM member_kifu").fetchone()[0] == 0
    new_token = ready(members, "alice")
    assert store.list(new_token) == []
    with pytest.raises(MemberError):
        store.access(new_token, record_id)


def test_retirement_only_drops_room_table_and_is_repeatable(tmp_path):
    path = tmp_path / "old.sqlite3"
    with sqlite3.connect(path) as db:
        db.executescript("CREATE TABLE research_kifu(id TEXT); INSERT INTO research_kifu VALUES ('old');"
                         "CREATE TABLE member_kifu(id TEXT); INSERT INTO member_kifu VALUES ('keep');"
                         "CREATE TABLE ai_dictionary(id TEXT); INSERT INTO ai_dictionary VALUES ('keep');")
    retire_room_kifu(path)
    retire_room_kifu(path)
    with sqlite3.connect(path) as db:
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='research_kifu'").fetchone()
        assert db.execute("SELECT * FROM member_kifu").fetchone() == ("keep",)
        assert db.execute("SELECT * FROM ai_dictionary").fetchone() == ("keep",)


def test_old_room_endpoints_are_removed():
    assert not any("/research_kifu" in getattr(route, "path", "") for route in app_module.app.routes)


def test_invalid_import_and_owner_injection_rejected(library):
    *_, client, game = library
    assert client.post("/api/member/kifu/import", json={"kifu_text": "invalid"}).status_code == 400
    assert save(client, member_id="bobby").status_code == 422
    assert save(client, tags=["unknown"]).status_code == 400


def test_valid_import_is_private_and_survives_store_restart(library):
    from test_research_kifu_library import _valid_kifu_text
    store, members, token, other, client, game = library
    response = client.post("/api/member/kifu/import", json={"kifu_text": _valid_kifu_text(), "title": "Imported"})
    assert response.status_code == 200
    record = response.json()["record"]
    assert record["payload"]["winner"] == "B"
    reopened = MemberKifuStore(MemberStore(members.path))
    assert reopened.access(token, record["id"])["title"] == "Imported"
    assert reopened.list(other) == []


def test_my_seat_save_edit_import_and_legacy(library):
    from test_research_kifu_library import _valid_kifu_text
    store, members, token, other, client, game = library
    record = save(client, my_seat="C", anonymous=True).json()["record"]
    assert record["my_seat"] == record["payload"]["my_seat"] == "C"
    assert "my_seat" not in game["last_completed_kifu"]
    url = "/api/member/kifu/" + record["id"]
    assert client.post(url + "/edit", json={"memo": "keep seat"}).json()["record"]["my_seat"] == "C"
    for seat in ("A", "B", "C", "D", "spectator", ""):
        assert client.post(url + "/edit", json={"my_seat": seat}).json()["record"]["my_seat"] == seat
    for invalid in ("E", 2, "watch", [], {"owner": "bobby"}):
        assert save(client, my_seat=invalid).status_code == 422
        assert client.post(url + "/edit", json={"my_seat": invalid}).status_code == 422
    imported = client.post("/api/member/kifu/import", json={"kifu_text": _valid_kifu_text(), "my_seat": "D"}).json()["record"]
    assert imported["my_seat"] == "D"
    assert store.statistics(token)["partner_finishes"] == 1  # B wins, D is partner.
    legacy = store.save(token, title="legacy", memo="", tags=[], payload={})
    assert legacy["my_seat"] == ""
    assert MemberKifuStore(MemberStore(members.path)).access(token, imported["id"])["my_seat"] == "D"


def test_statistics_are_on_demand_owner_scoped_and_never_persisted(library):
    store, members, token, other, client, game = library
    assert client.post("/api/member/kifu/statistics").json()["statistics"]["win_rate"] is None
    examples = [
        ("A", "A", 20), ("C", "A", 50), ("B", "D", 30),
        ("A", "B", 40), ("D", "C", 20),
        ("", "A", 20), ("spectator", "B", 30), ("A", "", 0), ("D", "D", "bad"),
    ]
    ids = []
    for seat, winner, points in examples:
        ids.append(store.save(token, title="round", memo="", tags=[], payload={
            "my_seat": seat, "winner": winner, "gained_score": points,
        })["id"])
    store.save(other, title="other", memo="", tags=[], payload={"my_seat": "A", "winner": "A", "gained_score": 100})
    with members._db() as db:
        before = [tuple(row) for row in db.execute("SELECT * FROM member_kifu ORDER BY id")]
    response = client.post("/api/member/kifu/statistics")
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["statistics"] == dict(total=9, counted=5, wins=3, losses=2,
        points_for=100, points_against=60, point_difference=40, self_finishes=1,
        partner_finishes=2, unset=1, spectator=1, incomplete=2, win_rate=60.0)
    with members._db() as db:
        assert [tuple(row) for row in db.execute("SELECT * FROM member_kifu ORDER BY id")] == before
    store.access(token, ids[0], action="edit", my_seat="B")
    assert store.statistics(token)["wins"] == 2
    store.access(token, ids[1], action="delete")
    assert store.statistics(token)["wins"] == 1
    client.cookies.set(MEMBER_COOKIE, other)
    assert client.post("/api/member/kifu/statistics").json()["statistics"]["total"] == 1


def test_statistics_access_and_expired_entitlement(library):
    store, members, token, other, client, game = library
    members.update("alice", enabled=True, paid_enabled=True, paid_until="2020-01-01")
    assert client.post("/api/member/kifu/statistics").status_code == 200
    assert client.post("/api/member/kifu/statistics", headers={"Origin": "https://evil.example"}).status_code == 403
    members.logout(token)
    assert client.post("/api/member/kifu/statistics").status_code == 401

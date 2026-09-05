import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from starlette.datastructures import URL

from backend import app as app_module
from backend.member_api import MEMBER_COOKIE, create_member_router
from backend.member_kifu import MemberKifuStore, automatic_hand_tags
from backend.member_kifu_auto import save_connected_round
from backend.member_store import MemberStore
from test_member_kifu import library, ready


def socket(token, origin="https://testserver"):
    return SimpleNamespace(cookies={MEMBER_COOKIE: token}, headers={"origin": origin},
                           url=URL("wss://testserver/ws/member-test-room"))


def setup(library):
    store, members, token, other, client, game = library
    game.update(human_seats={"C": "owner"}, ai_seats=["A", "B", "D"])
    game["last_completed_kifu"].update(winner="A", gained_score=30)
    return store, members, token, game, {"owner": [socket(token)]}


def test_auto_settings_off_by_default_and_authorized_api(library):
    store, members, token, other, client, game = library
    client.app.include_router(create_member_router(members, lambda request: None))
    assert client.get("/api/member/session").json()["member"]["auto_save_kifu"] is False
    url = "/api/member/kifu-auto-save"
    assert client.post(url, json={"enabled": True}, headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.post(url, json={"enabled": "yes"}).status_code == 422
    assert client.post(url, json={"enabled": True, "member_id": "bobby"}).status_code == 422
    assert client.post(url, json={"enabled": True}).json()["member"]["auto_save_kifu"] is True
    assert MemberStore(members.path).kifu_auto_save(token) is True
    assert members.kifu_auto_save(other) is False
    members.update("alice", enabled=True, paid_enabled=False, paid_until=None)
    assert client.post(url, json={"enabled": True}).status_code == 403
    assert client.post(url, json={"enabled": False}).status_code == 200
    members.logout(token)
    assert client.post(url, json={"enabled": True}).status_code == 401


def test_seated_connected_opted_in_named_seat_and_dedup(library):
    store, members, token, game, connections = setup(library)
    game["last_completed_kifu"]["hand"] = {
        "p0": "しししし香馬銀玉", "p1": "しし香馬銀金角飛",
        "p2": "ししし香香馬馬玉", "p3": "しし香馬銀金角飛",
    }
    # The final remaining hand must not replace the saved initial deal.
    game["state"].hands = {"C": []}
    assert save_connected_round(store, game, connections) == []
    members.kifu_auto_save(token, True)
    events = save_connected_round(store, game, connections)
    assert events[0][1]["status"] == "saved"
    record = store.access(token, store.list(token)[0]["id"])
    assert record["my_seat"] == "C"
    assert record["payload"]["anonymous"] is False
    assert record["payload"]["player_names"] == {s: f"{s} name" for s in "ABCD"}
    assert "A name" in json.dumps(record)
    assert game["last_completed_kifu"]["player_names"]["A"] == "A name"
    assert record["title"] == ""
    assert record["tags"] == ["3し", "2香", "2中駒"]
    assert store.statistics(token)["partner_finishes"] == 1
    # Multiple sockets and later reconnects cannot save twice, even after deletion.
    connections["owner"].append(socket(token))
    assert save_connected_round(MemberKifuStore(MemberStore(members.path)), game, connections) == []
    store.access(token, record["id"], action="delete")
    assert save_connected_round(store, game, connections) == []
    assert store.list(token) == []


@pytest.mark.parametrize("hand, expected", [
    ("王玉し香馬銀金角", ["王玉"]),
    ("ししし香馬銀金角", ["3し"]),
    ("しししし香馬銀金", ["4し"]),
    ("ししししし香馬銀", []),
    ("香香し馬銀金角飛", ["2香"]),
    ("香香香し馬銀金角", ["3香"]),
    ("香香香香し馬銀金", ["4香"]),
    ("馬馬銀銀金金し香", ["2中駒"]),
    ("馬馬銀銀銀し香玉", ["2中駒", "3中駒"]),
    ("金金金金し香王玉", ["王玉", "4中駒"]),
    ("角角し香馬銀金玉", ["大駒ペア"]),
    ("飛飛し香馬銀金玉", ["大駒ペア"]),
    ("角角飛飛し香王玉", ["王玉", "大駒ペア"]),
    ("しし香馬銀金角飛", []),
])
def test_automatic_hand_tags_use_exact_counts_and_deduplicate(hand, expected):
    assert automatic_hand_tags({"hand": {"p1": hand}}, "B") == expected


@pytest.mark.parametrize("payload, seat", [
    ({}, "A"), ({"hand": None}, "A"), ({"hand": {"p0": None}}, "A"),
    ({"hand": {"p0": "王玉"}}, "C"), ({"hand": {"p0": "王玉"}}, "spectator"),
])
def test_automatic_hand_tags_missing_own_hand(payload, seat):
    assert automatic_hand_tags(payload, seat) == []


def test_named_auto_save_keeps_existing_anonymous_records(library):
    store, members, token, game, connections = setup(library)
    legacy_payload = dict(game["last_completed_kifu"], anonymous=True,
                          player_names={s: f"プレイヤー{s}" for s in "ABCD"})
    legacy = store.save(token, title="Existing anonymous record", memo="", tags=["し攻め"], payload=legacy_payload)
    members.kifu_auto_save(token, True)
    assert save_connected_round(store, game, connections)[0][1]["status"] == "saved"
    assert store.access(token, legacy["id"]) == legacy
    assert len(store.list(token)) == 2


@pytest.mark.parametrize("case", ["spectator", "disconnected", "left", "ai", "live", "cross_origin", "logout", "expired", "disabled", "temporary"])
def test_ineligible_rounds_are_never_saved(library, case):
    store, members, token, game, connections = setup(library)
    members.kifu_auto_save(token, True)
    if case == "spectator": connections = {"watcher": [socket(token)]}
    if case == "disconnected": connections = {}
    if case == "left": game["human_seats"] = {}
    if case == "ai": game["ai_seats"].append("C")
    if case == "live": game["state"].finished = False
    if case == "cross_origin": connections = {"owner": [socket(token, "https://evil.example")]}
    if case == "logout": members.logout(token)
    if case == "expired": members.update("alice", enabled=True, paid_enabled=True, paid_until="2020-01-01")
    if case == "disabled": members.kifu_auto_save(token, False)
    if case == "temporary":
        issued = members.create("tempuser")
        _, temporary, _ = members.login("tempuser", issued["temporary_password"])
        connections = {"owner": [socket(temporary)]}
    save_connected_round(store, game, connections)
    with members._db() as db:
        assert db.execute("SELECT COUNT(*) FROM member_kifu").fetchone()[0] == 0


def test_capacity_disables_auto_without_deleting_and_error_is_isolated(library, monkeypatch):
    store, members, token, game, connections = setup(library)
    members.kifu_auto_save(token, True)
    store.LIMIT = 1
    save_connected_round(store, game, connections)
    game["member_kifu_round_id"] = "next-round"
    assert save_connected_round(store, game, connections)[0][1]["status"] == "limit"
    assert members.kifu_auto_save(token) is False
    assert len(store.list(token)) == 1
    members.kifu_auto_save(token, True)
    def fail(*args, **kwargs): raise OSError("private storage failure")
    monkeypatch.setattr(store, "save_automatic", fail)
    assert save_connected_round(store, game, connections)[0][1]["status"] == "error"


def test_concurrent_auto_saves_and_member_isolation(library):
    store, members, token, other, client, game = library
    for value in (token, other): members.kifu_auto_save(value, True)
    payload = {"winner": "A", "gained_score": 20}
    def save(_): return store.save_automatic(token, round_id="same", seat="A", payload=payload)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(result is not None for result in pool.map(save, range(8))) == 1
    assert store.save_automatic(other, round_id="same", seat="B", payload=payload)["status"] == "saved"
    assert len(store.list(token)) == len(store.list(other)) == 1
    members.delete("alice")
    new = ready(members, "alice")
    assert members.kifu_auto_save(new) is False


def test_finish_hook_saves_before_next_game_and_sends_only_to_owner(library, monkeypatch):
    store, members, token, game, connections = setup(library)
    members.kifu_auto_save(token, True)
    game.update(current_round_finished=False, total_team_score={"AC": 0, "BD": 0}, log=[])
    game["state"].winner = "C"
    sent = []
    async def send(event): sent.append(event)
    connections["owner"][0].send_json = send
    monkeypatch.setattr(app_module, "MEMBER_STORE", members)
    monkeypatch.setattr(app_module.manager, "client_connections", {("member-test-room", "owner"): connections["owner"]})
    monkeypatch.setattr(app_module, "_research_kifu_snapshot", lambda *_: game["last_completed_kifu"])
    for name in ("checkpoint_ai_search_telemetry", "checkpoint_background_search_value_model", "checkpoint_generic_response_patterns"):
        monkeypatch.setattr(app_module, name, lambda *_: None)
    async def finish():
        app_module._handle_round_finish(game, game["state"], ("attack", None, "2"), [])
        assert len(store.list(token)) == 1
        app_module._handle_round_finish(game, game["state"], ("attack", None, "2"), [])
        await asyncio.sleep(0)
    asyncio.run(finish())
    assert len(sent) == 1 and sent[0]["status"] == "saved"

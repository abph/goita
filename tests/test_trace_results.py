import asyncio
import copy
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend import app as game_app
from backend.kifu_import import parse_kifu_rounds
from backend.trace_results import GUEST_COOKIE, RETENTION, TraceStore


@pytest.fixture
def payload():
    text = (Path(__file__).parent / "fixtures/external_match.yaml").read_text(encoding="utf-8-sig")
    return game_app._canonical_trace_payload(parse_kifu_rounds(text, game_app._parse_research_kifu_text)[2])


def test_rank_first_attempt_ties_and_name_changes(tmp_path, payload):
    store = TraceStore(tmp_path / "trace.sqlite", clock=lambda: 1000)
    payload["score_before"] = {"AC": 20, "BD": 20}
    payload["score_after"] = {"AC": 20, "BD": 50}
    first = store.start("member:one", False, "Before", payload)
    second = store.start("member:two", False, "Two", payload)
    third = store.start("member:three", False, "Three", payload)
    store.finish(first, {"AC": 60, "BD": 20})
    store.finish(second, {"AC": 60, "BD": 20})
    store.finish(third, {"AC": 20, "BD": 50})
    retry = store.start("member:one", False, "After", payload)
    store.finish(retry, {"AC": 100, "BD": 20})
    result = store.read("member:one", first, mode="first")
    assert result["improvement"] == 70
    assert result["actual"] == {"AC": 40, "BD": 0}
    assert [item["rank"] for item in result["ranking"]] == [1, 1, 3]
    assert next(item for item in result["ranking"] if item["self"])["name"] == "After"
    assert result["total"] == 3
    assert store.read("member:one", retry)["ranked"] is False
    store.finish(first, {"AC": 140, "BD": 20})
    assert store.read("member:one", first)["improvement"] == 70
    assert TraceStore(store.path, clock=store.clock).read("member:one", first)["improvement"] == 70
    assert "member:one" not in str(result)
    best = store.read("member:one", first)
    assert best["mode"] == "best" and best["total"] == 3
    assert best["ranking"][0]["improvement"] == 110
    assert best["ranking"][0]["attempt_no"] == 2
    assert not best["is_best"]
    assert store.read("member:one", retry)["is_best"]


def test_history_pagination_names_and_best_after_lower_retry(tmp_path, payload):
    now = [1000]
    store = TraceStore(tmp_path / "trace.sqlite", clock=lambda: now[0])
    attempts = []
    for index, gained in enumerate((40, 60, 20), 1):
        now[0] += 1
        attempt = store.start("member:one", False, "Name", payload)
        store.finish(attempt, {"AC": payload["score_before"]["AC"] + gained, "BD": payload["score_before"]["BD"]})
        attempts.append(attempt)
    other = store.start("member:other", False, "Other", payload)
    store.finish(other, payload["score_after"])
    history = store.history("member:one", limit=2)
    assert history["total"] == 3
    assert [row["attempt_id"] for row in history["records"]] == attempts[:0:-1]
    assert [row["attempt_no"] for row in history["records"]] == [3, 2]
    assert [row["is_best"] for row in history["records"]] == [False, True]
    assert store.history("member:one", offset=2, limit=2)["records"][0]["attempt_id"] == attempts[0]
    assert store.history("member:one", offset=100, limit=2)["offset"] == 2
    assert len({row["challenge_label"] for row in history["records"]}) == 1
    best = store.read("member:one", attempts[-1])
    assert next(row for row in best["ranking"] if row["self"])["attempt_no"] == 2
    with pytest.raises(HTTPException):
        store.history("member:one", offset=-1)


def test_existing_database_migration_preserves_first_and_adds_best(tmp_path, payload):
    import json
    import sqlite3
    store = TraceStore(tmp_path / "legacy.sqlite", clock=lambda: 1000)
    with sqlite3.connect(store.path) as db:
        db.executescript("""
            CREATE TABLE trace_people(owner TEXT PRIMARY KEY,name TEXT NOT NULL,guest INTEGER NOT NULL);
            CREATE TABLE trace_challenges(id TEXT PRIMARY KEY,payload TEXT NOT NULL);
            CREATE TABLE trace_attempts(id TEXT PRIMARY KEY,owner TEXT NOT NULL,challenge TEXT NOT NULL,
                started REAL NOT NULL,finished REAL,expires REAL,ranked INTEGER NOT NULL,
                actual_ac INTEGER,actual_bd INTEGER,improvement INTEGER);
        """)
        db.execute("INSERT INTO trace_people VALUES ('member:one','Name',0)")
        db.execute("INSERT INTO trace_challenges VALUES (?,?)", (store.challenge_id(payload), json.dumps(payload)))
        for index, improvement in enumerate((20,70), 1):
            db.execute("INSERT INTO trace_attempts VALUES (?,?,?,?,?,NULL,?,?,?,?)",
                       (str(index), "member:one", store.challenge_id(payload), index, index + 10, int(index == 1), improvement,0,improvement))
    assert store.read("member:one", "1", mode="first")["ranking"][0]["improvement"] == 20
    best = store.read("member:one", "2")
    assert best["ranking"][0]["improvement"] == 70
    assert best["attempt_no"] == 2 and best["challenge_label"] == "課題001"
    for _ in range(3):
        store.cleanup()
    third = store.start("member:one", False, "New name", payload)
    store.finish(third, payload["score_after"])
    assert store.read("member:one", third)["attempt_no"] == 3


def test_concurrent_starts_reserve_only_one_initial_result(tmp_path, payload):
    from concurrent.futures import ThreadPoolExecutor
    store = TraceStore(tmp_path / "trace.sqlite")
    with ThreadPoolExecutor(max_workers=2) as workers:
        attempts = list(workers.map(lambda _: store.start("member:one", False, "Name", payload), range(2)))
    for attempt in attempts:
        store.finish(attempt, payload["score_after"])
    assert sum(store.read("member:one", attempt)["ranked"] for attempt in attempts) == 1


def test_guest_expiry_deletes_data_but_preserves_member_results(tmp_path, payload):
    now = [1000]
    store = TraceStore(tmp_path / "trace.sqlite", clock=lambda: now[0])
    guest = store.start("guest:test", True, "Guest", payload)
    member = store.start("member:test", False, "Member", payload)
    now[0] += 10
    store.finish(guest, payload["score_after"])
    store.finish(member, payload["score_after"])
    now[0] += RETENTION - 1
    assert store.read("guest:test", guest)["improvement"] == 0
    now[0] += 1
    store.cleanup()
    with pytest.raises(HTTPException) as error:
        store.read("guest:test", guest)
    assert error.value.status_code == 404
    assert store.read("member:test", member)["total"] == 1
    assert store.history("guest:test")["total"] == 0
    with store.db() as db:
        assert db.execute("SELECT COUNT(*) FROM trace_people WHERE guest=1").fetchone()[0] == 0


def test_abandonment_consumes_first_attempt_and_challenges_are_separate(tmp_path, payload):
    store = TraceStore(tmp_path / "trace.sqlite")
    first = store.start("member:test", False, "Name", payload)
    retry = store.start("member:test", False, "Name", payload)
    store.finish(retry, payload["score_after"])
    assert not store.read("member:test", retry)["ranked"]
    with pytest.raises(HTTPException):
        store.read("member:test", first, original=True)
    changed = copy.deepcopy(payload)
    changed["score_before"]["AC"] += 10
    attempt = store.start("member:test", False, "Name", changed)
    store.finish(attempt, changed["score_after"])
    assert store.read("member:test", attempt)["ranked"]


def test_expired_guest_best_falls_back_to_remaining_attempt(tmp_path, payload):
    now = [1000]
    store = TraceStore(tmp_path / "trace.sqlite", clock=lambda: now[0])
    first = store.start("guest:one", True, "Guest", payload)
    store.finish(first, {"AC": payload["score_before"]["AC"]+60, "BD": payload["score_before"]["BD"]})
    now[0] += 86400
    retry = store.start("guest:one", True, "Guest", payload, practice=True)
    store.finish(retry, {"AC": payload["score_before"]["AC"]+20, "BD": payload["score_before"]["BD"]})
    assert not store.read("guest:one", retry)["is_best"]
    now[0] = 1000 + RETENTION
    remaining = store.read("guest:one", retry)
    assert remaining["total"] == 1 and remaining["is_best"]
    assert remaining["ranking"][0]["attempt_no"] == 2
    assert store.read("guest:one", retry, mode="first")["total"] == 0
    assert store.history("guest:one")["total"] == 1


@pytest.fixture
def trace_client(monkeypatch, tmp_path, payload):
    monkeypatch.setenv("GOITA_PERSISTENT_DATA_DIR", str(tmp_path))
    game = game_app._create_game_obj(dealer="A")
    game.update(is_debug_room=True, human_seats={"A": "owner"}, player_names={"A": "挑戦者"},
                debug_auto_next_round=True, debug_auto_new_game=True)
    monkeypatch.setitem(game_app.GAMES, "debug", game)
    monkeypatch.setattr(game_app, "_random_trace_candidates", lambda: [{"payload": payload, "source": {"id": "secret"}}])
    monkeypatch.setattr(game_app.manager, "broadcast_update", lambda *_: asyncio.sleep(0))
    monkeypatch.setattr(game_app, "_arm_turn_timeout", lambda *_: None)
    monkeypatch.setattr(game_app, "_schedule_ai_background_search", lambda *_: None)
    monkeypatch.setattr(game_app, "_auto_save_member_round", lambda *_: None)
    for name in ("checkpoint_ai_search_telemetry", "checkpoint_background_search_value_model", "checkpoint_generic_response_patterns"):
        monkeypatch.setattr(game_app, name, lambda *_: None)
    return TestClient(game_app.app, headers={"X-Goita-Member": "1"})


def start(client):
    response = client.post("/games/debug/trace_random_start", json={"client_id": "owner"})
    assert response.status_code == 200, response.text
    return response.json()["state"]["trace_attempt_id"]


def finish_round():
    game = game_app.GAMES["debug"]
    for _ in range(200):
        if game["state"].finished:
            break
        game_app._apply_agent_turn(game, game["state"].turn)
    assert game["state"].finished
    return game


def test_guest_full_flow_replay_is_private_retry_does_not_replace_ranking(trace_client, payload):
    client = trace_client
    attempt = start(client)
    assert client.cookies.get(GUEST_COOKIE)
    path = f"/games/debug/trace_results/{attempt}"
    assert client.get(path + "/original").status_code == 409
    assert client.get(path).status_code == 409
    game = finish_round()
    assert not game["debug_auto_next_round"] and not game["debug_auto_new_game"]
    result = client.get(path)
    assert result.status_code == 200
    assert result.headers["Cache-Control"] == "no-store"
    assert result.json()["improvement"] == 0
    assert result.json()["ranked"] and result.json()["guest"]
    assert result.json()["total"] == 1
    assert client.get("/games/debug/trace_results/latest").json()["attempt_id"] == attempt
    original = client.get(path + "/original").json()["payload"]
    assert original["hands"] == payload["hands"]
    assert original["anonymous"] and original["my_seat"] == "A"
    outsider = TestClient(game_app.app, headers={"X-Goita-Member": "1"})
    assert outsider.get(path + "/original").status_code == 401
    assert outsider.post(path + "/retry", json={"client_id": "owner"}).status_code == 401
    again = client.post(path + "/retry", json={"client_id": "owner"})
    assert again.status_code == 200, again.text
    retry = again.json()["state"]["trace_attempt_id"]
    assert retry != attempt
    assert game_app.GAMES["debug"]["init_hands"] == payload["hands"]
    assert game_app.GAMES["debug"]["total_team_score"] == payload["score_before"]
    finish_round()
    result = client.get(f"/games/debug/trace_results/{retry}").json()
    assert not result["ranked"] and result["total"] == 1


def test_member_identity_survives_different_browser_and_guest_has_no_access(trace_client, monkeypatch):
    client = trace_client
    monkeypatch.setattr(game_app.MEMBER_STORE, "authenticate", lambda token: {"member_id": "stable-id"})
    client.cookies.set(game_app.MEMBER_COOKIE, "member-token")
    attempt = start(client)
    finish_round()
    other_device = TestClient(game_app.app, headers={"X-Goita-Member": "1"})
    other_device.cookies.set(game_app.MEMBER_COOKIE, "another-session-token")
    result = other_device.get(f"/games/debug/trace_results/{attempt}")
    assert result.status_code == 200 and not result.json()["guest"]
    assert result.json()["expires_at"] is None
    assert "stable-id" not in result.text


def test_trace_endpoints_are_debug_only_and_same_origin(trace_client):
    client = trace_client
    for path in ("trace_random_start", "trace_same_start", "trace_results/abc/retry"):
        assert client.post(f"/games/main/{path}", json={"client_id": "owner"}).status_code == 403
    for path in ("trace_results/latest", "trace_results/history", "trace_results/abc", "trace_results/abc/original"):
        assert client.get(f"/games/main/{path}").status_code == 403
    assert client.post("/games/debug/trace_random_start", json={"client_id": "other"}).status_code == 403
    assert client.post("/games/debug/trace_random_start", json={"client_id": "owner"},
                       headers={"Origin": "https://other.example"}).status_code == 403


def test_different_participants_see_shared_ranking_but_not_each_others_original(trace_client):
    first = start(trace_client)
    finish_round()
    game = game_app.GAMES["debug"]
    game["human_seats"]["A"] = "next-owner"
    view = game_app._state_public_view(game["state"], game_id="debug", viewer="A", game_obj=game, client_id="next-owner")
    assert view["trace_attempt_id"] == ""
    second_client = TestClient(game_app.app, headers={"X-Goita-Member": "1"})
    response = second_client.post("/games/debug/trace_same_start", json={"client_id": "next-owner"})
    assert response.status_code == 200
    second = response.json()["state"]["trace_attempt_id"]
    finish_round()
    result = second_client.get(f"/games/debug/trace_results/{second}").json()
    assert result["total"] == 2
    assert [item["rank"] for item in result["ranking"]] == [1, 1]
    assert second_client.get(f"/games/debug/trace_results/{first}/original").status_code == 404
    history = second_client.get("/games/debug/trace_results/history").json()
    assert history["total"] == 1 and history["records"][0]["attempt_id"] == second
    assert second_client.get(f"/games/debug/trace_results/{second}?mode=first").json()["mode"] == "first"
    assert second_client.get(f"/games/debug/trace_results/{second}?mode=invalid").status_code == 400

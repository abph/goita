"""Exact source moves and team scores earn a repeatable, server-verified bonus."""
import copy
import json
import sqlite3
from datetime import datetime

import pytest

from backend import app as game_app
from backend.trace_results import TraceStore
from goita_ai2.constants import PIECE_TOTALS
from test_trace_results import payload, trace_client
from test_score_rooms import enter


@pytest.mark.parametrize("mode", ["random", "divergent"])
@pytest.mark.parametrize("guest", [False, True])
def test_each_matching_replay_adds_ten_without_changing_the_actual_difference(tmp_path, payload, mode, guest):
    now = [datetime.fromisoformat("2026-11-01T12:00:00+09:00").timestamp()]
    store = TraceStore(tmp_path / "matches.sqlite", clock=lambda: now[0])
    owner = "guest:one" if guest else "member:one"
    for index in range(3):
        attempt = store.start(owner, guest, "Player", payload, selection_mode=mode)
        store.finish(attempt, payload["score_after"], exact_sequence=True)
        result = store.read(owner, attempt)
        assert result["improvement"] == 0
        assert result["ranking"][0]["improvement"] == 0
        assert result["ranking_points"] == result["exact_match_bonus"] == 10
        assert result["ranked"] is (index == 0)
        assert store.period_ranking("daily")["ranking"][0]["score"] == 10 * (index + 1)
        assert store.period_ranking("weekly")["ranking"][0]["score"] == 10 * (index + 1)
        store.finish(attempt, payload["score_after"], exact_sequence=True)
        assert store.period_ranking("daily")["ranking"][0]["score"] == 10 * (index + 1)
        now[0] += 1


@pytest.mark.parametrize("mode", ["random", "divergent"])
@pytest.mark.parametrize("verified,extra_ac,extra_bd,bonus", [
    (False, 0, 0, 0), (True, 0, 0, 10),
    (True, 20, 20, 0),  # Equal margins alone are insufficient.
    (True, 20, 0, 0), (True, 0, 20, 0),
])
def test_both_team_scores_and_the_verified_sequence_are_required(
    tmp_path, payload, mode, verified, extra_ac, extra_bd, bonus
):
    store = TraceStore(tmp_path / "conditions.sqlite")
    attempt = store.start("member:one", False, "Player", payload, selection_mode=mode)
    actual = {"AC": payload["score_after"]["AC"] + extra_ac,
              "BD": payload["score_after"]["BD"] + extra_bd}
    store.finish(attempt, actual, exact_sequence=verified)
    result = store.read("member:one", attempt)
    difference = extra_ac - extra_bd
    base = difference if mode == "random" else 10 if difference > 0 else -10 if difference < 0 else 0
    assert result["improvement"] == difference
    assert result["ranking_points"] == base + bonus
    assert result["exact_match_bonus"] == bonus


def test_legacy_results_are_not_retroactively_given_bonuses(tmp_path, payload):
    store = TraceStore(tmp_path / "old.sqlite", clock=lambda: 1000)
    with sqlite3.connect(store.path) as db:
        db.executescript("""
            CREATE TABLE trace_people(owner TEXT PRIMARY KEY,name TEXT NOT NULL,guest INTEGER NOT NULL);
            CREATE TABLE trace_challenges(id TEXT PRIMARY KEY,payload TEXT NOT NULL);
            CREATE TABLE trace_attempts(id TEXT PRIMARY KEY,owner TEXT NOT NULL,challenge TEXT NOT NULL,
                started REAL NOT NULL,finished REAL,expires REAL,ranked INTEGER NOT NULL,
                actual_ac INTEGER,actual_bd INTEGER,improvement INTEGER);
        """)
        db.execute("INSERT INTO trace_people VALUES ('member:one','Player',0)")
        db.execute("INSERT INTO trace_challenges VALUES (?,?)", (store.challenge_id(payload), json.dumps(payload)))
        db.execute("INSERT INTO trace_attempts VALUES (?,?,?,?,?,NULL,?,?,?,?)",
                   ("old", "member:one", store.challenge_id(payload), 1, 10, 1, 0, 30, 0))
    assert store.read("member:one", "old")["ranking_points"] == 0
    assert store.read("member:one", "old")["exact_match_bonus"] == 0
    # A delayed save cannot change an already completed historical result.
    store.finish("old", payload["score_after"], exact_sequence=True)
    assert store.read("member:one", "old")["ranking_points"] == 0
    with store.db() as db:
        assert "exact_match_bonus" in {row[1] for row in db.execute("PRAGMA table_info(trace_attempts)")}


def test_daily_weekly_rewards_and_uniform_selection_use_the_correct_measures(tmp_path, payload):
    now = [datetime.fromisoformat("2026-10-31T12:00:00+09:00").timestamp()]
    store = TraceStore(tmp_path / "rewards.sqlite", clock=lambda: now[0])
    for _ in range(2):
        attempt = store.start("member:matching", False, "Matching", payload)
        store.finish(attempt, payload["score_after"], exact_sequence=True)
        now[0] += 1
    attempt = store.start("member:improved", False, "Improved", payload)
    actual = dict(payload["score_after"])
    actual["AC"] += 15
    store.finish(attempt, actual)
    assert store.daily_award_candidates("2026-10-31")["candidates"] == [{"member_id":"matching", "rank":1}]
    now[0] = datetime.fromisoformat("2026-11-02T00:00:00+09:00").timestamp()
    assert [row["score"] for row in store.period_ranking("weekly_previous")["ranking"]] == [20, 15]
    assert store.weekly_award_candidates()["candidates"] == [
        {"member_id":"matching", "rank":1}, {"member_id":"improved", "rank":2}]

    other = copy.deepcopy(payload)
    other["score_before"]["AC"] += 100
    other["score_after"]["AC"] += 100
    for index in range(3):
        attempt = store.start(f"member:uniform{index}", False, "Uniform", other)
        store.finish(attempt, other["score_after"], exact_sequence=True)
    assert store.challenge_id(other) in store.challenge_selection()["uniform"]


def controlled_payload(payload):
    # Only A attacks; all other passes are expanded by the normal importer.
    # Swapping the first two hidden pieces leaves the public route and score
    # unchanged, while producing a genuinely different legal sequence.
    source = copy.deepcopy(payload)
    source["hands"] = {"A": list("11113445"), "B": [], "C": [], "D": []}
    pool = [piece for piece, count in PIECE_TOTALS.items()
            for _ in range(count - source["hands"]["A"].count(piece))]
    for index, piece in enumerate(pool):
        source["hands"]["BCD"[index % 3]].append(piece)
    source.update(dealer="A", score_before={"AC":0,"BD":0},
                  moves=[["0","馬","銀"], ["0","し","銀"],
                         ["0","し","金"], ["0","し","し"]])
    return game_app._canonical_trace_payload(source)


def play(client, room, *, swap_hidden=False):
    game = game_app.GAMES[room]
    game["trace_analysis_enabled"] = False
    for _ in range(100):
        state = game["state"]
        if state.finished:
            return game
        if state.turn == "A":
            index = game["trace_move_index"]
            action = game_app._trace_row_to_action(state, "A", game["trace_moves"][index])
            if swap_hidden and index in (0, 4):
                action = (action[0], "1" if index == 0 else "3", action[2])
            assert action in state.legal_actions("A")
            response = client.post(f"/games/{room}/step", json={
                "client_id":"owner", "player":"A", "exact_match_bonus":10,
                "action":{"action_type":action[0], "block":action[1], "attack":action[2]},
            })
        else:
            response = client.post(f"/games/{room}/cpu_step")
        assert response.status_code == 200, response.text
    pytest.fail("source route did not finish")


@pytest.mark.parametrize("swap_hidden,bonus", [(False,10), (True,0)])
def test_full_score_room_flow_checks_actual_hidden_pieces_not_trace_continuation(
    trace_client, payload, monkeypatch, swap_hidden, bonus
):
    source = controlled_payload(payload)
    monkeypatch.setattr(game_app, "_random_trace_candidates", lambda: [{"payload":source, "source":{}}])
    room = enter(trace_client)
    for index in range(2):
        response = trace_client.post(f"/games/{room}/trace_random_start", json={"client_id":"owner"})
        assert response.status_code == 200, response.text
        attempt = response.json()["state"]["trace_attempt_id"]
        game = play(trace_client, room, swap_hidden=swap_hidden)
        assert not game["trace_diverged"]
        result = trace_client.get(f"/games/{room}/trace_results/{attempt}").json()
        assert result["improvement"] == 0
        assert result["ranking_points"] == result["exact_match_bonus"] == bonus
        assert result["actual"] == result["original"]
        assert len(game["kifu_moves"]) == len(game["trace_moves"])
        assert (game["kifu_moves"] == game["trace_moves"]) is (not swap_hidden)
        daily = trace_client.get("/api/score-attack/rankings?period=daily").json()
        assert daily["ranking"][0]["score"] == bonus * (index + 1)


@pytest.mark.parametrize("change", ["pass", "receiver", "attack", "short", "long", "missing"])
def test_server_requires_every_expanded_action_to_match(tmp_path, payload, monkeypatch, change):
    store = TraceStore(tmp_path / "whole-sequence.sqlite")
    monkeypatch.setattr(game_app, "get_trace_store", lambda: store)
    expected = game_app._expand_trace_moves(payload)
    moves = copy.deepcopy(expected)
    if change == "pass":
        index = next(i for i, row in enumerate(moves) if row[1] == "パス")
        moves[index][0] = str((int(moves[index][0]) + 1) % 4)
    elif change == "receiver":
        index = next(i for i, row in enumerate(moves) if row[1] not in ("", "パス") and not row[2])
        moves[index][1] = "王"
    elif change == "attack":
        index = next(i for i, row in enumerate(moves) if row[2])
        moves[index][2] = "し"
    elif change == "short":
        moves.pop()
    elif change == "long":
        moves.append(["0","パス",""])
    else:
        moves = None
    attempt = store.start("member:one", False, "Player", payload)
    game_app._save_trace_result({"trace_attempt_id":attempt, "current_round_finished":True,
                                "total_team_score":payload["score_after"], "trace_moves":expected,
                                "kifu_moves":moves, "trace_diverged":False})
    assert store.read("member:one", attempt)["ranking_points"] == 0

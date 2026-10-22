import json

import pytest

from backend import app as game_app, kifu_timing
from backend.member_kifu_auto import save_connected_round
from backend.kifu_import import parse_kifu_rounds
from goita_ai2.state import GoitaState
from test_member_kifu import library
from test_member_kifu_auto import socket
from test_research_kifu_library import _valid_kifu_text


@pytest.mark.parametrize("seconds, expected", [
    (0, 0), (.04, 0), (.15, .2), (2.34, 2.3), (4, 4),
    (4.49, 4), (4.5, 5), (8.7, 9), (9.9, 10),
    (10, 10), (12.5, 15), (32, 30), (44, 45), (123, 125),
])
def test_rounding(seconds, expected):
    assert kifu_timing.round_seconds(seconds) == expected


def test_clock_only_begins_on_start_and_never_restarts_on_timer_refresh(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(kifu_timing.time, "monotonic", lambda: now[0])
    game = dict(is_started=False, kifu_moves=[])
    kifu_timing.begin(game)
    assert game.get("kifu_clock") is None
    game.update(is_started=True)
    kifu_timing.begin(game)
    now[0] += 3.14
    game_app._reset_turn_deadline(game)
    game_app._record_public_action(game, "A", ("attack_after_block", "1", "1"))
    game["kifu_moves"].append(["0", "し", "し"])
    now[0] += 44
    game_app._record_public_action(game, "B", ("pass", None, None))
    game["kifu_moves"].append(["1", "パス", ""])
    assert kifu_timing.saved_times(game) == {"move_times": [3.1, 45]}
    # Only accepted game actions call the recorder. Polling cannot consume time.
    now[0] += 5
    game_app._reset_turn_deadline(game)
    assert game["kifu_move_times"] == [3.1, 45]
    kifu_timing.reset(game)
    assert game["kifu_move_times"] == [None, None]
    assert game["kifu_clock"] is None


def test_legacy_prefix_does_not_get_invented_intervals(monkeypatch):
    monkeypatch.setattr(kifu_timing.time, "monotonic", lambda: 10)
    game = dict(kifu_moves=[["0", "し", "し"]])
    kifu_timing.record(game)
    game["kifu_moves"].append(["1", "パス", ""])
    assert game["kifu_move_times"] == [None, None]
    assert kifu_timing.saved_times(game) == {}


def recorded_game(monkeypatch):
    payload = game_app._parse_research_kifu_text(_valid_kifu_text())
    rows = game_app._expand_trace_moves(payload)
    game = game_app._create_game_obj(dealer="A")
    game.update(state=GoitaState(hands=payload["hands"], dealer="A"),
                init_hands=payload["hands"], is_started=True)
    now = [100.0]
    monkeypatch.setattr(kifu_timing.time, "monotonic", lambda: now[0])
    for name in ("_handle_round_finish", "_schedule_ai_background_search"):
        monkeypatch.setattr(game_app, name, lambda *args: None)
    kifu_timing.begin(game)
    intervals = [2.3, 32, 44]
    for index, row in enumerate(rows):
        actor = "ABCD"[int(row[0])]
        action = game_app._trace_row_to_action(game["state"], actor, row)
        now[0] += intervals[index % 3]
        # Exercises the accepted AI/timeout action path without running a search.
        assert game_app._apply_agent_turn(game, actor, forced_action=action)["status"] == "ok"
    game["total_team_score"] = dict(game["state"].team_score)
    game["last_round_score"] = sum(game["state"].team_score.values())
    return game, rows


def test_snapshot_download_import_preserve_passes_and_separate_actions(monkeypatch):
    game, rows = recorded_game(monkeypatch)
    assert game["state"].finished
    assert any(row[1] == "パス" for row in rows)
    assert any(not row[1] and row[2] for row in rows)
    snapshot = game_app._research_kifu_snapshot(game, game["state"])
    assert snapshot["moves"] == rows
    assert snapshot["move_times"] == [2.3, 30, 45] * (len(rows) // 3) + [2.3, 30, 45][:len(rows) % 3]
    monkeypatch.setitem(game_app.GAMES, "timed-kifu-test", game)
    text = game_app.get_kifu_yaml("timed-kifu-test")
    assert "move_times:" in text
    restored = game_app._parse_research_kifu_text(text)
    assert restored["move_times"] == snapshot["move_times"]
    assert restored["moves"] == snapshot["moves"]
    assert restored["winner"] == snapshot["winner"]
    # Exporting an older game retains the previous format and needs no migration.
    game.pop("kifu_move_times")
    old_text = game_app.get_kifu_yaml("timed-kifu-test")
    assert "move_times:" not in old_text
    old = game_app._parse_research_kifu_text(old_text)
    assert "move_times" not in old
    assert old["winner"] == restored["winner"]


@pytest.mark.parametrize("values", [[1], [-1] * 11, [True] * 11, ["5"] * 11, [float("nan")] * 11])
def test_invalid_times_are_rejected_without_accepting_a_broken_replay(values):
    text = _valid_kifu_text().replace("  game:", "  move_times: " + json.dumps(values) + "\n  game:")
    with pytest.raises(ValueError, match="時間情報"):
        game_app._parse_research_kifu_text(text)


def test_legacy_and_timed_rounds_can_share_one_import_file():
    text = _valid_kifu_text()
    timed = text.replace("  game:", "  move_times: " + json.dumps([2.3] * 11) + "\n  game:").replace("score: [0,80]", "score: [0,60]")
    second_round = text[text.index("- hand:"):]
    rounds = parse_kifu_rounds(timed + second_round, game_app._parse_research_kifu_text)
    assert rounds[0]["move_times"] == [2.3] * 11
    assert "move_times" not in rounds[1]


def test_automatic_and_manual_library_saves_keep_times_and_respect_opt_in(library):
    store, members, token, other, client, game = library
    payload = game_app._parse_research_kifu_text(_valid_kifu_text())
    payload["move_times"] = [2.3] * len(payload["moves"])
    game.update(last_completed_kifu=payload, human_seats={"A": "owner"}, ai_seats=["B", "C", "D"])
    connections = {"owner": [socket(token)]}
    assert save_connected_round(store, game, connections) == []
    assert store.list(token) == []
    members.kifu_auto_save(token, True)
    assert save_connected_round(store, game, connections)[0][1]["status"] == "saved"
    saved = store.access(token, store.list(token)[0]["id"])
    assert saved["payload"]["move_times"] == payload["move_times"]
    response = client.post("/api/member/kifu/save", json={"game_id": "member-test-room"})
    assert response.status_code == 200
    assert response.json()["record"]["payload"]["move_times"] == payload["move_times"]
    assert store.list(other) == []


def test_next_round_starts_fresh(monkeypatch):
    monkeypatch.setattr(kifu_timing.time, "monotonic", lambda: 100)
    new_game = game_app._create_game_obj(dealer="B")
    game_app._set_reset_start_state(new_game, True)
    assert new_game["kifu_move_times"] == []
    assert new_game["kifu_clock"] == 100
    game_app._apply_deal_mode_to_unstarted_game(new_game, "normal")
    assert new_game["kifu_clock"] is None
    assert new_game["kifu_move_times"] == []

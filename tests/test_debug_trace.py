import asyncio
from pathlib import Path

import pytest
from fastapi import HTTPException

from backend import app as game_app


FIXTURE = (Path(__file__).parent / "fixtures" / "external_match.yaml").read_text(encoding="utf-8-sig")


@pytest.fixture
def debug_trace(monkeypatch):
    game = game_app._create_game_obj(dealer="A")
    game.update(is_debug_room=True, human_seats={"A": "owner"}, ai_seats=[])
    monkeypatch.setitem(game_app.GAMES, game_app.DEBUG_GID, game)
    monkeypatch.setattr(game_app.manager, "broadcast_update", lambda *_: asyncio.sleep(0))
    monkeypatch.setattr(game_app, "_schedule_debug_auto_next_round", lambda *_: None)
    monkeypatch.setattr(game_app, "_arm_turn_timeout", lambda *_: None)
    monkeypatch.setattr(game_app, "_schedule_ai_background_search", lambda *_: None)
    return game


def test_debug_trace_starts_from_selected_round_and_reuses_trace_action(debug_trace):
    request = game_app.DebugTraceStartRequest(kifu_text=FIXTURE, round_index=3, client_id="owner")
    result = asyncio.run(game_app.start_debug_trace(game_app.DEBUG_GID, request))
    assert result["ok"] is True
    game = debug_trace = game_app.GAMES[game_app.DEBUG_GID]
    assert game["trace_mode"] is True
    assert game["trace_original_round"] == 3
    assert game["human_seats"] == {"A": "owner"}
    assert game["ai_seats"] == ["B", "C", "D"]
    assert game["state"].turn == "C"
    assert game["total_team_score"] == {"AC": 20, "BD": 20}
    assert game["trace_moves"][:5] == [
        ["2", "し", "金"],
        ["3", "パス", ""],
        ["0", "パス", ""],
        ["1", "金", ""],
        ["1", "", "角"],
    ]
    result = game_app._apply_agent_turn(game, "C")
    assert result["status"] == "ok"
    assert game["trace_move_index"] == 1
    assert game["log"][-1].endswith("[TRACE]")
    # The imported format omits D/A's passes before B's next recorded move.
    # They must be replayed by the live engine to keep the trace cursor aligned.
    assert game["state"].turn == "D"
    result = game_app._apply_agent_turn(game, "D")
    assert result["status"] == "ok"
    assert game["trace_move_index"] == 2
    assert game["log"][-1].endswith("[TRACE]")
    assert game["state"].turn == "A"


def test_trace_route_is_debug_only_and_requires_a_owner(debug_trace):
    request = game_app.DebugTraceStartRequest(kifu_text=FIXTURE, client_id="owner")
    with pytest.raises(HTTPException) as error:
        asyncio.run(game_app.start_debug_trace("main", request))
    assert error.value.status_code == 403
    request.requester = "B"
    with pytest.raises(HTTPException) as error:
        asyncio.run(game_app.start_debug_trace(game_app.DEBUG_GID, request))
    assert error.value.status_code == 403


def test_random_trace_candidates_are_limited_to_fifty_starting_points(monkeypatch):
    import yaml
    rounds = yaml.safe_load(FIXTURE)["log"]
    for index, item in enumerate(rounds, 1):
        item["round_index"] = index
    monkeypatch.setattr(game_app, "_load_kifu_archive", lambda: {
        "matches": [{"id": "fixture", "rounds": rounds}],
    })
    monkeypatch.setattr(game_app, "_KIFU_RANDOM_TRACE_CACHE", None)
    candidates = game_app._random_trace_candidates()
    assert candidates
    assert all(
        item["payload"]["score_before"]["AC"] <= 50
        and item["payload"]["score_before"]["BD"] <= 50
        for item in candidates
    )
    assert all(item["payload"]["hands"] for item in candidates)

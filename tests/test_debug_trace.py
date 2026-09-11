import asyncio
from pathlib import Path

import pytest
from fastapi import HTTPException, Request, Response

from backend import app as game_app


FIXTURE = (Path(__file__).parent / "fixtures" / "external_match.yaml").read_text(encoding="utf-8-sig")


@pytest.fixture
def debug_trace(monkeypatch, tmp_path):
    monkeypatch.setenv("GOITA_PERSISTENT_DATA_DIR", str(tmp_path))
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
    http = Request({"type": "http", "scheme": "http", "server": ("testserver", 80), "path": "/", "headers": []})
    result = asyncio.run(game_app.start_debug_trace(game_app.DEBUG_GID, request, http, Response()))
    assert result["ok"] is True
    game = debug_trace = game_app.GAMES[game_app.DEBUG_GID]
    assert game["trace_mode"] is True
    assert game["trace_analysis_enabled"] is True
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
    assert "[TRACE]" in game["log"][-1]
    assert "TRACE-ANALYSIS:C" in game["log"][-1]
    # The imported format omits D/A's passes before B's next recorded move.
    # They must be replayed by the live engine to keep the trace cursor aligned.
    assert game["state"].turn == "D"
    result = game_app._apply_agent_turn(game, "D")
    assert result["status"] == "ok"
    assert game["trace_move_index"] == 2
    assert "[TRACE]" in game["log"][-1]
    assert "TRACE-ANALYSIS:D" in game["log"][-1]
    assert game["state"].turn == "A"


def test_trace_route_is_debug_only_and_requires_a_owner(debug_trace):
    request = game_app.DebugTraceStartRequest(kifu_text=FIXTURE, client_id="owner")
    with pytest.raises(HTTPException) as error:
        asyncio.run(game_app.start_debug_trace("main", request, None, Response()))
    assert error.value.status_code == 403
    request.requester = "B"
    with pytest.raises(HTTPException) as error:
        asyncio.run(game_app.start_debug_trace(game_app.DEBUG_GID, request, None, Response()))
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


def test_trace_keeps_following_public_attack_when_human_hidden_block_differs():
    state = game_app.GoitaState(
        hands={seat: ["1"] * 8 for seat in game_app.ALL_SEATS},
        dealer="A",
    )
    game = {
        "trace_mode": True,
        "trace_diverged": False,
        "trace_move_index": 0,
        "trace_moves": [["0", "角", "馬"]],
        "human_seats": {"A": "owner"},
    }
    actual = ("attack_after_block", "7", "3")

    assert game_app._debug_trace_action(
        game,
        state,
        "A",
        [actual],
        allow_hidden_block_variation=True,
    ) == actual
    assert game["trace_diverged"] is False


def test_trace_still_diverges_when_human_public_attack_differs():
    state = game_app.GoitaState(
        hands={seat: ["1"] * 8 for seat in game_app.ALL_SEATS},
        dealer="A",
    )
    game = {
        "trace_mode": True,
        "trace_diverged": False,
        "trace_move_index": 0,
        "trace_moves": [["0", "角", "馬"]],
        "human_seats": {"A": "owner"},
    }
    actual = ("attack_after_block", "7", "4")

    assert game_app._debug_trace_action(
        game,
        state,
        "A",
        [actual],
        allow_hidden_block_variation=True,
    ) is None
    assert game["trace_diverged"] is True


def test_trace_still_diverges_when_human_public_receive_differs():
    state = game_app.GoitaState(
        hands={seat: ["1"] * 8 for seat in game_app.ALL_SEATS},
        dealer="A",
    )
    game = {
        "trace_mode": True,
        "trace_diverged": False,
        "trace_move_index": 0,
        "trace_moves": [["0", "し", ""]],
        "human_seats": {"A": "owner"},
    }
    actual = ("receive", "2", None)

    assert game_app._debug_trace_action(
        game,
        state,
        "A",
        [actual],
        allow_hidden_block_variation=True,
    ) is None
    assert game["trace_diverged"] is True


def test_trace_shadow_analysis_compares_forced_move_without_mutating_live_agent():
    state = game_app.GoitaState(
        hands={
            "A": ["1", "1", "2", "3", "4", "4", "7", "8"],
            "B": ["1", "1", "1", "3", "5", "5", "6", "9"],
            "C": ["1", "1", "2", "2", "4", "5", "5", "6"],
            "D": ["1", "1", "1", "2", "3", "3", "4", "7"],
        },
        dealer="C",
    )

    class FakeAgent:
        def __init__(self):
            self._track = {id(state): {"marker": "live"}}
            self._my_initial_hands_by_state_id = {id(state): ["1"]}
            self.last_decision_reason = "live-reason"
            self.last_score_fallback_detail = ""
            self.last_attack_candidate_snapshot = {}

        def select_action(self, shadow_state, player, actions):
            assert id(shadow_state) in self._track
            self.last_decision_reason = "shadow-reason"
            self.last_score_fallback_detail = "shadow-detail"
            self.last_attack_candidate_snapshot = {
                "chosen": {"attack": "5", "score": 12.5},
                "alternatives": [{"attack": "4", "score": 8.0}],
            }
            return actions[0]

        def cancel_background_search(self):
            return None

    agent = FakeAgent()
    state.current_attack = "1"
    state.attacker = "D"
    state.phase = "receive"
    state.turn = "C"
    source = ("pass", None, None)
    analysis = game_app._analyze_trace_action(agent, state, "C", source)

    assert analysis["source_action"] == source
    assert analysis["ai_action"] == state.legal_actions("C")[0]
    assert analysis["match"] is True
    assert analysis["reason"] == "shadow-reason"
    assert analysis["candidates"] == "第一候補=金(12.5), 代替=銀(8.0)"
    assert agent.last_decision_reason == "live-reason"
    formatted = game_app._format_trace_analysis(analysis, "C")
    assert "TRACE-ANALYSIS:C" in formatted
    assert "元棋譜=パス" in formatted
    assert "現AI候補=パス" in formatted

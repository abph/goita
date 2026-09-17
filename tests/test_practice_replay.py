from __future__ import annotations

import asyncio
import copy

import pytest
from fastapi import HTTPException

from backend import app as app_module


def _disable_finish_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "_auto_save_member_round", lambda *_: None)
    monkeypatch.setattr(app_module, "_save_trace_result", lambda *_: None)
    for name in (
        "checkpoint_ai_search_telemetry",
        "checkpoint_background_search_value_model",
        "checkpoint_generic_response_patterns",
    ):
        monkeypatch.setattr(app_module, name, lambda *_: None)


def test_practice_replay_reuses_hands_and_dealer_without_adding_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_finish_side_effects(monkeypatch)

    async def scenario() -> None:
        game_id = app_module.PRIVATE_A_GID
        client_id = "practice-host"
        previous_game = app_module.GAMES.get(game_id)
        game = app_module._create_game_obj(dealer="C")
        game["human_seats"] = {"A": client_id}
        game["ai_seats"] = ["B", "C", "D"]
        game["is_started"] = True
        game["round_count"] = 4
        game["total_team_score"] = {"AC": 40, "BD": 30}
        initial_hands = copy.deepcopy(game["init_hands"])
        game["state"].finished = True
        game["state"].winner = "A"
        app_module.GAMES[game_id] = game

        try:
            app_module._handle_round_finish(
                game,
                game["state"],
                ("attack", None, "2"),
                [],
            )
            original_score = copy.deepcopy(game["total_team_score"])
            original_log = copy.deepcopy(game["log"])
            original_kifu = copy.deepcopy(game["last_completed_kifu"])
            assert game["practice_replay_source"]["hands"] == initial_hands
            assert game["practice_replay_source"]["dealer"] == "C"
            normal_view = app_module.get_state(
                game_id, viewer="A", client_id=client_id
            )
            assert normal_view["practice_replay_available"] is True
            assert normal_view["practice_replay_active"] is False

            started = await app_module.practice_replay(
                game_id,
                app_module.PracticeReplayRequest(
                    requester="A",
                    client_id=client_id,
                    action="start",
                ),
            )
            assert started["practice_replay_active"] is True
            practice = app_module.GAMES[game_id]
            assert practice["dealer"] == "C"
            assert practice["init_hands"] == initial_hands
            assert practice["total_team_score"] == original_score
            assert practice["round_count"] == 4
            assert practice["is_started"] is True
            assert practice["state"].finished is False
            practice_view = app_module.get_state(
                game_id, viewer="A", client_id=client_id
            )
            assert practice_view["practice_replay_active"] is True
            assert practice_view["practice_replay_available"] is False

            practice["state"].finished = True
            practice["state"].winner = "B"
            app_module._handle_round_finish(
                practice,
                practice["state"],
                ("attack", None, "3"),
                [],
            )
            assert practice["total_team_score"] == original_score
            assert practice["last_completed_kifu"] == original_kifu
            assert practice["last_round_score"] > 0
            finished_practice_view = app_module.get_state(
                game_id, viewer="A", client_id=client_id
            )
            assert finished_practice_view["practice_replay_available"] is True

            retried = await app_module.practice_replay(
                game_id,
                app_module.PracticeReplayRequest(
                    requester="A",
                    client_id=client_id,
                    action="retry",
                ),
            )
            assert retried["practice_replay_active"] is True
            retry_game = app_module.GAMES[game_id]
            assert retry_game["init_hands"] == initial_hands
            assert retry_game["dealer"] == "C"
            assert retry_game["total_team_score"] == original_score

            retry_game["state"].finished = True
            retry_game["state"].winner = "D"
            app_module._handle_round_finish(
                retry_game,
                retry_game["state"],
                ("attack", None, "4"),
                [],
            )
            returned = await app_module.practice_replay(
                game_id,
                app_module.PracticeReplayRequest(
                    requester="A",
                    client_id=client_id,
                    action="return",
                ),
            )
            assert returned["practice_replay_active"] is False
            restored = app_module.GAMES[game_id]
            assert restored["state"].finished is True
            assert restored["state"].winner == "A"
            assert restored["total_team_score"] == original_score
            assert restored["round_count"] == 4
            assert restored["log"] == original_log
            assert restored["last_completed_kifu"] == original_kifu
            assert restored["practice_replay_source"]["hands"] == initial_hands
        finally:
            app_module._cancel_turn_timeout_task(game_id)
            if previous_game is None:
                app_module.GAMES.pop(game_id, None)
            else:
                app_module.GAMES[game_id] = previous_game
            app_module.GAME_TURN_LOCKS.pop(game_id, None)

    asyncio.run(scenario())


def test_practice_replay_requires_finished_round_and_host_owner() -> None:
    async def scenario() -> None:
        game_id = app_module.PRIVATE_A_GID
        previous_game = app_module.GAMES.get(game_id)
        game = app_module._create_game_obj(dealer="A")
        game["human_seats"] = {"A": "owner"}
        game["is_started"] = True
        app_module.GAMES[game_id] = game
        try:
            with pytest.raises(HTTPException) as unfinished:
                await app_module.practice_replay(
                    game_id,
                    app_module.PracticeReplayRequest(
                        requester="A", client_id="owner", action="start"
                    ),
                )
            assert unfinished.value.status_code == 409

            game["state"].finished = True
            game["practice_replay_source"] = app_module._practice_replay_source(game)
            with pytest.raises(HTTPException) as not_owner:
                await app_module.practice_replay(
                    game_id,
                    app_module.PracticeReplayRequest(
                        requester="A", client_id="other", action="start"
                    ),
                )
            assert not_owner.value.status_code == 403
        finally:
            app_module._cancel_turn_timeout_task(game_id)
            if previous_game is None:
                app_module.GAMES.pop(game_id, None)
            else:
                app_module.GAMES[game_id] = previous_game
            app_module.GAME_TURN_LOCKS.pop(game_id, None)

    asyncio.run(scenario())


def test_practice_replay_can_return_before_practice_round_finishes() -> None:
    async def scenario() -> None:
        game_id = app_module.PRIVATE_A_GID
        client_id = "practice-early-return-host"
        previous_game = app_module.GAMES.get(game_id)
        game = app_module._create_game_obj(dealer="B")
        game["human_seats"] = {"A": client_id}
        game["ai_seats"] = ["B", "C", "D"]
        game["is_started"] = True
        game["total_team_score"] = {"AC": 30, "BD": 20}
        game["state"].finished = True
        game["state"].winner = "C"
        game["practice_replay_source"] = app_module._practice_replay_source(game)
        original_state = copy.deepcopy(game["state"])
        app_module.GAMES[game_id] = game
        try:
            await app_module.practice_replay(
                game_id,
                app_module.PracticeReplayRequest(
                    requester="A", client_id=client_id, action="start"
                ),
            )
            assert app_module.GAMES[game_id]["state"].finished is False

            returned = await app_module.practice_replay(
                game_id,
                app_module.PracticeReplayRequest(
                    requester="A", client_id=client_id, action="return"
                ),
            )
            restored = app_module.GAMES[game_id]
            assert returned["practice_replay_active"] is False
            assert restored["state"].finished is True
            assert restored["state"].winner == original_state.winner
            assert restored["total_team_score"] == {"AC": 30, "BD": 20}
        finally:
            app_module._cancel_turn_timeout_task(game_id)
            if previous_game is None:
                app_module.GAMES.pop(game_id, None)
            else:
                app_module.GAMES[game_id] = previous_game
            app_module.GAME_TURN_LOCKS.pop(game_id, None)

    asyncio.run(scenario())


def test_practice_replay_ui_has_all_controls_and_state_flags() -> None:
    html = app_module.FRONTEND_DIR.joinpath("index.html").read_text(encoding="utf-8")
    script = app_module.FRONTEND_DIR.joinpath("trace-results.js").read_text(encoding="utf-8")
    assert 'id="btnPracticeReplay"' in html
    assert 'id="btnPracticeReturn"' in html
    assert 'id="practiceExitButton"' in html
    assert 'id="btnPracticeScene"' in html
    assert 'id="practiceSceneModal"' in html
    assert 'fetch(`${API}/games/${gid}/practice_replay`' in html
    assert 'practiceReplayAction("scene", sceneIndex)' in html
    assert 'returnFromPracticeReplay()' in html
    assert 'この局の得点は加算されません' in html
    assert "updatePracticeReplayButtons(state, isHost, autoNextRoundPending)" in html
    assert "state.practice_replay_available === true" in script
    assert "PRIVATE_ROOM_IDS.has(gid)" in script
    assert "active ? 'retry' : 'start'" in script


def test_practice_replay_scene_reconstructs_state_before_selected_turn() -> None:
    async def scenario() -> None:
        game_id = app_module.PRIVATE_A_GID
        client_id = "practice-scene-host"
        previous_game = app_module.GAMES.get(game_id)
        source_game = app_module._create_game_obj(dealer="A")
        source_game["human_seats"] = {"A": client_id}
        source_game["ai_seats"] = ["B", "C", "D"]
        source_game["is_started"] = True
        source_game["round_count"] = 3
        source_game["total_team_score"] = {"AC": 20, "BD": 40}

        simulation_state = app_module.GoitaState(
            hands=copy.deepcopy(source_game["init_hands"]), dealer="A"
        )
        moves = []
        while len(app_module._practice_replay_turn_groups(moves)) < 3:
            state = simulation_state
            actor = state.turn
            action = state.legal_actions(actor)[0]
            moves.append(app_module._action_to_kifu_row(actor, action))
            app_module._apply_action(state, actor, action)
            assert not state.finished

        source_game["kifu_moves"] = copy.deepcopy(moves)
        source_game["practice_replay_source"] = app_module._practice_replay_source(
            source_game
        )
        source_game["state"].finished = True
        source_game["state"].winner = "A"
        app_module.GAMES[game_id] = source_game

        expected_state = app_module.GoitaState(
            hands=copy.deepcopy(source_game["init_hands"]), dealer="A"
        )
        prefix_rows = []
        for group in app_module._practice_replay_turn_groups(moves)[:2]:
            for row in group:
                actor = app_module.ALL_SEATS[int(row[0])]
                action = app_module._trace_row_to_action(expected_state, actor, row)
                assert action is not None
                prefix_rows.append(row)
                app_module._apply_action(expected_state, actor, action)

        try:
            result = await app_module.practice_replay(
                game_id,
                app_module.PracticeReplayRequest(
                    requester="A",
                    client_id=client_id,
                    action="scene",
                    scene_index=2,
                ),
            )
            replay = app_module.GAMES[game_id]
            assert result["practice_replay_active"] is True
            assert replay["practice_scene_index"] == 2
            assert replay["kifu_moves"] == prefix_rows
            assert replay["state"].turn == expected_state.turn
            assert replay["state"].phase == expected_state.phase
            assert replay["state"].current_attack == expected_state.current_attack
            assert replay["state"].hands == expected_state.hands
            assert replay["state"].face_down_hidden == expected_state.face_down_hidden
            assert replay["total_team_score"] == {"AC": 20, "BD": 40}

            replay["state"].finished = True
            replay["state"].winner = "B"
            retried = await app_module.practice_replay(
                game_id,
                app_module.PracticeReplayRequest(
                    requester="A", client_id=client_id, action="retry"
                ),
            )
            retried_game = app_module.GAMES[game_id]
            assert retried["practice_replay_active"] is True
            assert retried_game["practice_scene_index"] == 2
            assert retried_game["kifu_moves"] == prefix_rows
            assert retried_game["state"].turn == expected_state.turn
            assert retried_game["state"].hands == expected_state.hands
        finally:
            app_module._cancel_turn_timeout_task(game_id)
            if previous_game is None:
                app_module.GAMES.pop(game_id, None)
            else:
                app_module.GAMES[game_id] = previous_game
            app_module.GAME_TURN_LOCKS.pop(game_id, None)

    asyncio.run(scenario())


def test_practice_replay_is_not_available_in_public_rooms() -> None:
    async def scenario() -> None:
        game_id = app_module.MAIN_GID
        client_id = "public-host"
        previous_game = app_module.GAMES.get(game_id)
        game = app_module._create_game_obj(dealer="A")
        game["human_seats"] = {"A": client_id}
        game["is_started"] = True
        game["state"].finished = True
        game["state"].winner = "A"
        game["practice_replay_source"] = app_module._practice_replay_source(game)
        app_module.GAMES[game_id] = game
        try:
            view = app_module.get_state(game_id, viewer="A", client_id=client_id)
            assert view["practice_replay_available"] is False
            with pytest.raises(HTTPException) as unavailable:
                await app_module.practice_replay(
                    game_id,
                    app_module.PracticeReplayRequest(
                        requester="A", client_id=client_id, action="start"
                    ),
                )
            assert unavailable.value.status_code == 403
        finally:
            app_module._cancel_turn_timeout_task(game_id)
            if previous_game is None:
                app_module.GAMES.pop(game_id, None)
            else:
                app_module.GAMES[game_id] = previous_game
            app_module.GAME_TURN_LOCKS.pop(game_id, None)

    asyncio.run(scenario())

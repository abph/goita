from __future__ import annotations

import asyncio
from collections import Counter

import pytest
from fastapi import HTTPException

from backend import app as app_module


def _step_request(client_id: str, action):
    return app_module.StepRequest(
        player="A",
        client_id=client_id,
        action=app_module.ActionModel(
            action_type=action[0],
            block=action[1],
            attack=action[2],
        ),
    )


def _clean_room(room_id: str) -> None:
    app_module._cancel_turn_timeout_task(room_id)
    app_module.GAMES.pop(room_id, None)
    app_module.GAME_TURN_LOCKS.pop(room_id, None)


def test_tutorial_hands_use_the_complete_piece_set() -> None:
    for hands in (app_module.TUTORIAL_HANDS, app_module.ROYAL_TUTORIAL_HANDS):
        actual = Counter(piece for hand in hands.values() for piece in hand)
        assert actual == Counter(app_module.PIECE_TOTALS)
        assert all(len(hand) == 8 for hand in hands.values())


def test_tutorial_runs_all_five_steps_without_score_or_kifu_save() -> None:
    async def scenario() -> None:
        client_id = "tutorial-full-flow"
        room_id = app_module._tutorial_room_id(client_id)
        _clean_room(room_id)
        try:
            started = await app_module.start_tutorial(
                app_module.TutorialStartRequest(client_id=client_id, start_step=1)
            )
            assert started["game_id"] == room_id
            game = app_module.GAMES[room_id]
            assert game["human_seats"] == {"A": client_id}
            assert set(game["ai_seats"]) == {"B", "C", "D"}
            assert game["hidden_from_lobby"] is True
            assert game["tutorial_mode"] is True
            assert game["is_started"] is True

            expected_human_actions = [
                ("attack_after_block", "3", "4"),
                ("pass", None, None),
                ("receive", "3", None),
                ("attack", None, "2"),
                ("attack_after_block", "1", "7"),
                ("receive", "6", None),
                ("attack", None, "5"),
            ]
            for action in expected_human_actions:
                legal = app_module.get_legal_actions(
                    room_id, player="A", client_id=client_id
                )
                assert legal == [
                    {
                        "action_type": action[0],
                        "block": action[1],
                        "attack": action[2],
                    }
                ]
                await app_module.step(room_id, _step_request(client_id, action))
                game = app_module.GAMES[room_id]
                if not game["state"].finished and game["state"].turn != "A":
                    cpu_result = await app_module.cpu_step(room_id)
                    assert cpu_result["status"] == "ok"

            game = app_module.GAMES[room_id]
            assert game["state"].finished is True
            assert game["state"].winner == "A"
            assert game["tutorial_completed"] is True
            assert game["total_team_score"] == {"AC": 0, "BD": 0}
            assert game["last_round_score"] == 0
            assert game["last_completed_kifu"] is None
            view = app_module.get_state(room_id, viewer="A", client_id=client_id)
            assert view["tutorial_step"] == 5
            assert view["tutorial_completed_step"] == 5
            assert view["tutorial_completed"] is True
        finally:
            _clean_room(room_id)

    asyncio.run(scenario())


def test_tutorial_rejects_actions_outside_the_script() -> None:
    async def scenario() -> None:
        client_id = "tutorial-invalid-action"
        room_id = app_module._tutorial_room_id(client_id)
        _clean_room(room_id)
        try:
            await app_module.start_tutorial(
                app_module.TutorialStartRequest(client_id=client_id, start_step=1)
            )
            with pytest.raises(HTTPException) as error:
                await app_module.step(
                    room_id,
                    _step_request(client_id, ("attack_after_block", "3", "2")),
                )
            assert error.value.status_code == 409
            assert app_module.GAMES[room_id]["tutorial_cursor"] == 0
        finally:
            _clean_room(room_id)

    asyncio.run(scenario())


def test_royal_tutorial_practices_receive_pass_and_attack() -> None:
    async def scenario() -> None:
        client_id = "tutorial-royal-flow"
        room_id = app_module._tutorial_room_id(client_id)
        _clean_room(room_id)
        try:
            started = await app_module.start_tutorial(
                app_module.TutorialStartRequest(
                    client_id=client_id,
                    start_step=1,
                    chapter="royal",
                )
            )
            assert started["chapter"] == "royal"
            game = app_module.GAMES[room_id]
            assert game["dealer"] == "B"
            assert game["tutorial_chapter"] == "royal"
            assert game["state"].turn == "A"

            expected_human_actions = [
                ("receive", "9", None),
                ("attack", None, "5"),
                ("pass", None, None),
                ("receive", "4", None),
                ("attack", None, "8"),
            ]
            for action in expected_human_actions:
                legal = app_module.get_legal_actions(
                    room_id, player="A", client_id=client_id
                )
                assert legal == [
                    {
                        "action_type": action[0],
                        "block": action[1],
                        "attack": action[2],
                    }
                ]
                await app_module.step(room_id, _step_request(client_id, action))
                game = app_module.GAMES[room_id]
                if not game["state"].finished and game["state"].turn != "A":
                    result = await app_module.cpu_step(room_id)
                    assert result["status"] == "ok"

            game = app_module.GAMES[room_id]
            assert game["tutorial_completed"] is True
            assert game["state"].finished is True
            assert game["state"].winner is None
            assert game["total_team_score"] == {"AC": 0, "BD": 0}
            assert game["last_completed_kifu"] is None
            view = app_module.get_state(room_id, viewer="A", client_id=client_id)
            assert view["tutorial_chapter"] == "royal"
            assert view["tutorial_step"] == 3
            assert view["tutorial_total_steps"] == 3
            assert view["tutorial_completed_step"] == 3
        finally:
            _clean_room(room_id)

    asyncio.run(scenario())


def test_tutorial_can_resume_at_a_saved_step_and_restart() -> None:
    async def scenario() -> None:
        client_id = "tutorial-resume"
        room_id = app_module._tutorial_room_id(client_id)
        _clean_room(room_id)
        try:
            started = await app_module.start_tutorial(
                app_module.TutorialStartRequest(client_id=client_id, start_step=4)
            )
            assert started["start_step"] == 4
            game = app_module.GAMES[room_id]
            assert game["state"].turn == "A"
            assert app_module._tutorial_expected_action(game, "A") == (
                "attack_after_block", "1", "7"
            )

            restarted = await app_module.restart_tutorial(
                room_id,
                app_module.TutorialStartRequest(client_id=client_id, start_step=1),
            )
            assert restarted["start_step"] == 1
            assert app_module.GAMES[room_id]["tutorial_cursor"] == 0
        finally:
            _clean_room(room_id)

    asyncio.run(scenario())


def test_tutorial_room_is_removed_when_its_player_leaves() -> None:
    async def scenario() -> None:
        client_id = "tutorial-leave"
        room_id = app_module._tutorial_room_id(client_id)
        _clean_room(room_id)
        await app_module.start_tutorial(
            app_module.TutorialStartRequest(client_id=client_id, start_step=1)
        )
        result = await app_module.release_seat(
            room_id, seat="A", client_id=client_id
        )
        assert result["human_seats"] == []
        assert room_id not in app_module.GAMES
        app_module.GAME_TURN_LOCKS.pop(room_id, None)

    asyncio.run(scenario())


def test_tutorial_frontend_entry_and_guide_are_present() -> None:
    html = app_module.FRONTEND_DIR.joinpath("index.html").read_text(encoding="utf-8")
    assert 'onclick="startTutorial()"' in html
    assert 'id="tutorialGuide"' in html
    assert 'id="tutorialProgressBar"' in html
    assert 'fetch(`${API}/tutorial/start`' in html
    assert 'fetch(`${API}/games/${gid}/tutorial/restart`' in html
    assert 'goita_tutorial_progress_v1' in html
    assert 'goita_tutorial_royal_progress_v1' in html
    assert 'id="tutorialRoyalExplanation"' in html
    assert 'onclick="startRoyalTutorial()"' in html
    assert 'classList.toggle("tutorial-room"' in html
    assert "state.tutorial_mode" in html
    assert "if(state.tutorial_mode){" in html
    assert "return actions.length === 1" in html

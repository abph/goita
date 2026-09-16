from __future__ import annotations

import asyncio

from backend import app as app_module


PRIVATE_B_GID = "room-silver-02"


def _practice_room(name: str):
    room = app_module._create_game_obj(dealer="A")
    room["owner_name"] = name
    return room


def _restore_rooms(previous: dict[str, object]) -> None:
    for room_id in (app_module.PRIVATE_A_GID, PRIVATE_B_GID):
        app_module._cancel_turn_timeout_task(room_id)
        old = previous[room_id]
        if old is None:
            app_module.GAMES.pop(room_id, None)
        else:
            app_module.GAMES[room_id] = old
        app_module.GAME_TURN_LOCKS.pop(room_id, None)


def test_guided_practice_prepares_private_a_with_ai_and_selected_deal() -> None:
    async def scenario() -> None:
        previous = {
            app_module.PRIVATE_A_GID: app_module.GAMES.get(app_module.PRIVATE_A_GID),
            PRIVATE_B_GID: app_module.GAMES.get(PRIVATE_B_GID),
        }
        app_module.GAMES[app_module.PRIVATE_A_GID] = _practice_room("プライベートA")
        app_module.GAMES[PRIVATE_B_GID] = _practice_room("プライベートB")
        try:
            result = await app_module.start_guided_practice(
                app_module.GuidedPracticeRequest(
                    client_id="guided-a",
                    practice_type="balanced",
                )
            )
            assert result["ok"] is True
            assert result["game_id"] == app_module.PRIVATE_A_GID
            game = app_module.GAMES[app_module.PRIVATE_A_GID]
            assert game["human_seats"] == {"A": "guided-a"}
            assert set(game["ai_seats"]) == {"B", "C", "D"}
            assert game["ai_profile"] == "intermediate_middle2"
            assert game["deal_mode"] == "balanced"
            assert game["next_deal_mode"] == "balanced"
            assert game["is_started"] is False
        finally:
            _restore_rooms(previous)

    asyncio.run(scenario())


def test_guided_practice_uses_private_b_when_any_human_is_in_private_a() -> None:
    async def scenario() -> None:
        previous = {
            app_module.PRIVATE_A_GID: app_module.GAMES.get(app_module.PRIVATE_A_GID),
            PRIVATE_B_GID: app_module.GAMES.get(PRIVATE_B_GID),
        }
        private_a = _practice_room("プライベートA")
        private_a["human_seats"] = {"C": "someone-else"}
        app_module.GAMES[app_module.PRIVATE_A_GID] = private_a
        app_module.GAMES[PRIVATE_B_GID] = _practice_room("プライベートB")
        try:
            result = await app_module.start_guided_practice(
                app_module.GuidedPracticeRequest(
                    client_id="guided-b",
                    practice_type="frequent",
                )
            )
            assert result["game_id"] == PRIVATE_B_GID
            assert app_module.GAMES[app_module.PRIVATE_A_GID]["human_seats"] == {
                "C": "someone-else"
            }
            game = app_module.GAMES[PRIVATE_B_GID]
            assert game["human_seats"] == {"A": "guided-b"}
            assert set(game["ai_seats"]) == {"B", "C", "D"}
            assert game["deal_mode"] == "frequent"
        finally:
            _restore_rooms(previous)

    asyncio.run(scenario())


def test_guided_practice_only_explains_when_both_rooms_have_humans() -> None:
    async def scenario() -> None:
        previous = {
            app_module.PRIVATE_A_GID: app_module.GAMES.get(app_module.PRIVATE_A_GID),
            PRIVATE_B_GID: app_module.GAMES.get(PRIVATE_B_GID),
        }
        private_a = _practice_room("プライベートA")
        private_b = _practice_room("プライベートB")
        private_a["human_seats"] = {"D": "busy-a"}
        private_b["human_seats"] = {"B": "busy-b"}
        app_module.GAMES[app_module.PRIVATE_A_GID] = private_a
        app_module.GAMES[PRIVATE_B_GID] = private_b
        try:
            result = await app_module.start_guided_practice(
                app_module.GuidedPracticeRequest(
                    client_id="guided-waiting",
                    practice_type="preset",
                )
            )
            assert result == {"ok": False, "status": "occupied"}
            assert private_a["human_seats"] == {"D": "busy-a"}
            assert private_b["human_seats"] == {"B": "busy-b"}
        finally:
            _restore_rooms(previous)

    asyncio.run(scenario())


def test_private_b_defaults_to_three_ai_players_when_first_created() -> None:
    previous = app_module.GAMES.pop(PRIVATE_B_GID, None)
    try:
        app_module.setup_supporter_rooms()
        assert set(app_module.GAMES[PRIVATE_B_GID]["ai_seats"]) == {"B", "C", "D"}
    finally:
        if previous is None:
            app_module.GAMES.pop(PRIVATE_B_GID, None)
        else:
            app_module.GAMES[PRIVATE_B_GID] = previous
        app_module.GAME_TURN_LOCKS.pop(PRIVATE_B_GID, None)


def test_guided_practice_ui_contains_all_three_choices() -> None:
    html = app_module.FRONTEND_DIR.joinpath("index.html").read_text(encoding="utf-8")
    assert 'onclick="startGuidedPractice(\'frequent\')"' in html
    assert 'onclick="startGuidedPractice(\'balanced\')"' in html
    assert 'onclick="startGuidedPractice(\'preset\')"' in html
    assert "極端な強弱を除いて練習する" in html
    assert 'fetch(`${API}/guided-practice`' in html

from __future__ import annotations

import asyncio

from fastapi import HTTPException

import backend.app as app_module


def test_lobby_shows_configured_main_rooms_and_two_private_rooms() -> None:
    app_module.setup_main_rooms()
    app_module.setup_supporter_rooms()
    rooms = app_module.list_rooms()["rooms"]

    main_rooms = [room for room in rooms if room["is_main_room"]]
    private_rooms = [room for room in rooms if not room["is_main_room"]]

    assert [room["game_id"] for room in main_rooms] == list(
        app_module.MAIN_ROOM_NAMES
    )
    assert [room["owner_name"] for room in main_rooms] == list(
        app_module.MAIN_ROOM_NAMES.values()
    )
    assert {room["game_id"] for room in private_rooms} == {
        "room-gold-01",
        "room-silver-02",
    }
    assert len(rooms) == len(app_module.MAIN_ROOM_NAMES) + 2
    assert app_module.DEBUG_GID not in {room["game_id"] for room in rooms}


def test_private_c_and_d_exist_but_are_hidden_from_lobby() -> None:
    room_ids = {room["game_id"] for room in app_module.list_rooms()["rooms"]}

    for game_id in ("room-bronze-03", "room-copper-04"):
        assert game_id in app_module.GAMES
        assert app_module.GAMES[game_id]["hidden_from_lobby"] is True
        assert game_id not in room_ids


def test_every_main_room_disables_beginner_support() -> None:
    for game_id in app_module.MAIN_GIDS:
        try:
            app_module.get_beginner_recommendation(
                game_id,
                player="A",
                client_id="test-client",
            )
        except HTTPException as exc:
            assert exc.status_code == 403
        else:
            raise AssertionError(f"{game_id} must reject beginner support")


def test_main_room_host_can_toggle_all_hands() -> None:
    game_id = next(iter(app_module.MAIN_ROOM_NAMES))
    client_id = "main-reveal-test-client"
    game = app_module.GAMES[game_id]
    old_human_seats = game.get("human_seats")
    old_reveal_hands = game.get("reveal_hands", False)

    try:
        game["human_seats"] = {"A": client_id}
        game["reveal_hands"] = False
        result = asyncio.run(
            app_module.toggle_reveal_hands(
                game_id,
                requester="A",
                client_id=client_id,
            )
        )
        assert result == {"ok": True, "reveal_hands": True}
        assert game["reveal_hands"] is True
    finally:
        game["human_seats"] = old_human_seats
        game["reveal_hands"] = old_reveal_hands


def test_frontend_recognizes_all_main_room_ids() -> None:
    html = (app_module.FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    frontend_ids = ["MAIN_GID", *[
        f'"{game_id}"'
        for game_id in app_module.MAIN_ROOM_NAMES
        if game_id != app_module.MAIN_GID
    ]]
    expected_set = f"const MAIN_ROOM_IDS = new Set([{', '.join(frontend_ids)}]);"
    assert expected_set in html
    assert "room.is_main_room === true" in html
    assert "<h2>🌐 公開部屋</h2>" in html
    assert 'toggleBtn.style.display = isHost ? "" : "none"' in html


if __name__ == "__main__":
    test_lobby_shows_configured_main_rooms_and_two_private_rooms()
    test_private_c_and_d_exist_but_are_hidden_from_lobby()
    test_every_main_room_disables_beginner_support()
    test_main_room_host_can_toggle_all_hands()
    test_frontend_recognizes_all_main_room_ids()
    print("ROOM_INVENTORY_TEST_OK")

from __future__ import annotations

import asyncio

from fastapi import HTTPException

import backend.app as app_module


def _settings_request(
    game_id: str,
    *,
    background_image: str | None,
) -> app_module.SettingsUpdateRequest:
    game = app_module.GAMES[game_id]
    return app_module.SettingsUpdateRequest(
        admin_password=str(game["admin_password"]),
        new_owner_name=str(game["owner_name"]),
        ai_profile=str(game["ai_profile"]),
        show_legal_actions=bool(game["show_legal_actions"]),
        show_log=bool(game["show_log"]),
        room_background_image=background_image,
    )


def test_private_a_background_is_exposed_and_survives_reset() -> None:
    game_id = app_module.PRIVATE_A_GID
    original_game = app_module.GAMES[game_id]
    original_background = original_game.get("room_background_image", "")
    original_human_seats = original_game.get("human_seats", {})
    client_id = "private-a-background-test"

    try:
        result = asyncio.run(
            app_module.update_settings(
                game_id,
                _settings_request(
                    game_id,
                    background_image="/static/private-a-background.webp",
                ),
            )
        )
        assert result["room_background_image"] == "/static/private-a-background.webp"

        public_state = app_module.get_state(game_id, viewer="W")
        assert public_state["room_background_image"] == "/static/private-a-background.webp"

        app_module.GAMES[game_id]["human_seats"] = {"A": client_id}
        asyncio.run(
            app_module.reset_game(
                game_id,
                dealer="A",
                requester="A",
                client_id=client_id,
            )
        )
        assert (
            app_module.GAMES[game_id]["room_background_image"]
            == "/static/private-a-background.webp"
        )
    finally:
        original_game["room_background_image"] = original_background
        original_game["human_seats"] = original_human_seats
        app_module.GAMES[game_id] = original_game


def test_other_rooms_reject_background_images() -> None:
    game_id = "room-silver-02"
    original_background = app_module.GAMES[game_id].get("room_background_image", "")

    try:
        try:
            asyncio.run(
                app_module.update_settings(
                    game_id,
                    _settings_request(
                        game_id,
                        background_image="/static/private-a-background.webp",
                    ),
                )
            )
        except HTTPException as exc:
            assert exc.status_code == 400
        else:
            raise AssertionError("Private B must reject room background settings")
    finally:
        app_module.GAMES[game_id]["room_background_image"] = original_background


def test_background_path_must_be_a_same_origin_static_image() -> None:
    for value in (
        "https://example.com/background.webp",
        "/static/../secret.webp",
        "/static/background.svg",
    ):
        try:
            app_module._normalize_room_background_image(
                app_module.PRIVATE_A_GID,
                value,
            )
        except HTTPException as exc:
            assert exc.status_code == 400
        else:
            raise AssertionError(f"Unsafe background path accepted: {value}")


def test_frontend_applies_and_clears_private_a_background() -> None:
    html = (app_module.FRONTEND_DIR / "index.html").read_text(encoding="utf-8")

    assert 'const PRIVATE_A_GID = "room-gold-01";' in html
    assert 'id="roomBackgroundSettingRow"' not in html
    assert 'id="setRoomBackgroundImage"' not in html
    assert "function applyRoomBackground(state)" in html
    assert "function clearRoomBackground()" in html
    assert "applyRoomBackground(state);" in html
    assert "body.room-background-active" in html
    assert "body.room-background-active .board-wrap" in html
    assert "background-color: rgba(209, 171, 117, 0.84)" in html


if __name__ == "__main__":
    test_private_a_background_is_exposed_and_survives_reset()
    test_other_rooms_reject_background_images()
    test_background_path_must_be_a_same_origin_static_image()
    test_frontend_applies_and_clears_private_a_background()
    print("PRIVATE_A_BACKGROUND_TEST_OK")

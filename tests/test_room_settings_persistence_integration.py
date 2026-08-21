import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import HTTPException

import backend.app as app_module
from backend.room_settings_persistence import load_room_settings


def _request(game: dict, **changes) -> app_module.SettingsUpdateRequest:
    return app_module.SettingsUpdateRequest(
        admin_password=str(game["admin_password"]),
        new_owner_name=str(changes.get("owner_name", game["owner_name"])),
        update_password=bool(changes.get("update_password", False)),
        new_password=changes.get("password"),
        ai_profile=str(changes.get("ai_profile", game["ai_profile"])),
        show_legal_actions=bool(changes.get("show_legal_actions", game["show_legal_actions"])),
        show_log=bool(changes.get("show_log", game["show_log"])),
        room_background_image=None,
    )


def test_update_settings_persists_and_restores_room_management_values() -> None:
    game_id = "room-silver-02"
    game = app_module.GAMES[game_id]
    original_path = app_module.ROOM_SETTINGS_PATH
    original_settings = app_module._room_management_settings(game)

    with TemporaryDirectory() as directory:
        path = Path(directory) / "goita-room-settings.json"
        app_module.ROOM_SETTINGS_PATH = path
        try:
            result = asyncio.run(app_module.update_settings(
                game_id,
                _request(
                    game,
                    owner_name="永続テスト部屋",
                    update_password=True,
                    password="persist-pass",
                    show_log=True,
                ),
            ))
            assert result["room_settings_persistent"] is True
            assert load_room_settings(path)[game_id]["password"] == "persist-pass"
            assert '"admin_password"' not in path.read_text(encoding="utf-8")

            app_module._apply_room_management_settings(game_id, game, original_settings)
            app_module._load_persisted_room_management_settings()
            assert game["owner_name"] == "永続テスト部屋"
            assert game["password"] == "persist-pass"
            assert game["show_log"] is True
        finally:
            app_module._apply_room_management_settings(game_id, game, original_settings)
            app_module.ROOM_SETTINGS_PATH = original_path


def test_update_settings_rolls_back_when_persistent_write_fails() -> None:
    game_id = "room-silver-02"
    game = app_module.GAMES[game_id]
    original_path = app_module.ROOM_SETTINGS_PATH
    original_save = app_module.save_room_settings
    original_settings = app_module._room_management_settings(game)

    app_module.ROOM_SETTINGS_PATH = Path("unwritable-room-settings.json")
    app_module.save_room_settings = lambda _path, _rooms: False
    try:
        try:
            asyncio.run(app_module.update_settings(
                game_id,
                _request(
                    game,
                    owner_name="保存失敗",
                    update_password=True,
                    password="must-not-remain",
                ),
            ))
        except HTTPException as error:
            assert error.status_code == 500
        else:
            raise AssertionError("A failed persistent write must reject the update")

        assert app_module._room_management_settings(game) == original_settings
    finally:
        app_module._apply_room_management_settings(game_id, game, original_settings)
        app_module.save_room_settings = original_save
        app_module.ROOM_SETTINGS_PATH = original_path


def test_lobby_admin_changes_and_resets_private_room_admin_password() -> None:
    game_id = "room-silver-02"
    game = app_module.GAMES[game_id]
    initial_password = str(game["admin_password"])
    original_path = app_module.ROOM_SETTINGS_PATH
    original_settings = app_module._room_management_settings(game)

    with TemporaryDirectory() as directory:
        path = Path(directory) / "goita-room-settings.json"
        app_module.ROOM_SETTINGS_PATH = path
        try:
            changed = app_module.update_private_room_admin_password(
                app_module.PrivateRoomAdminPasswordUpdateRequest(
                    admin_password=app_module.LOBBY_ADMIN_PASSWORD,
                    game_id=game_id,
                    new_password="new-room-admin",
                )
            )
            assert changed["ok"] is True
            assert changed["room_settings_persistent"] is True
            assert app_module.verify_admin(game_id, "new-room-admin")["ok"] is True
            try:
                app_module.verify_admin(game_id, initial_password)
            except HTTPException as error:
                assert error.status_code == 401
            else:
                raise AssertionError("The initial password must stop working after a change")

            stored_text = path.read_text(encoding="utf-8")
            stored = load_room_settings(path)[game_id]
            assert stored["admin_password_hash"].startswith("pbkdf2_sha256$")
            assert "new-room-admin" not in stored_text
            assert '"admin_password"' not in stored_text

            game["admin_password_hash"] = ""
            app_module._load_persisted_room_management_settings()
            assert app_module.verify_admin(game_id, "new-room-admin")["ok"] is True

            reset = app_module.update_private_room_admin_password(
                app_module.PrivateRoomAdminPasswordUpdateRequest(
                    admin_password=app_module.LOBBY_ADMIN_PASSWORD,
                    game_id=game_id,
                    reset_to_default=True,
                )
            )
            assert reset["reset_to_default"] is True
            assert app_module.verify_admin(game_id, initial_password)["ok"] is True
            assert load_room_settings(path)[game_id]["admin_password_hash"] == ""
        finally:
            app_module._apply_room_management_settings(game_id, game, original_settings)
            app_module.ROOM_SETTINGS_PATH = original_path


def test_lobby_admin_password_change_rolls_back_when_save_fails() -> None:
    game_id = "room-silver-02"
    game = app_module.GAMES[game_id]
    original_path = app_module.ROOM_SETTINGS_PATH
    original_save = app_module.save_room_settings
    original_hash = str(game.get("admin_password_hash", ""))

    app_module.ROOM_SETTINGS_PATH = Path("unwritable-room-settings.json")
    app_module.save_room_settings = lambda _path, _rooms: False
    try:
        try:
            app_module.update_private_room_admin_password(
                app_module.PrivateRoomAdminPasswordUpdateRequest(
                    admin_password=app_module.LOBBY_ADMIN_PASSWORD,
                    game_id=game_id,
                    new_password="must-not-remain",
                )
            )
        except HTTPException as error:
            assert error.status_code == 500
        else:
            raise AssertionError("A failed password save must reject the update")
        assert game.get("admin_password_hash", "") == original_hash
    finally:
        game["admin_password_hash"] = original_hash
        app_module.save_room_settings = original_save
        app_module.ROOM_SETTINGS_PATH = original_path


if __name__ == "__main__":
    test_update_settings_persists_and_restores_room_management_values()
    test_update_settings_rolls_back_when_persistent_write_fails()
    test_lobby_admin_changes_and_resets_private_room_admin_password()
    test_lobby_admin_password_change_rolls_back_when_save_fails()
    print("ROOM_SETTINGS_PERSISTENCE_INTEGRATION_TEST_OK")

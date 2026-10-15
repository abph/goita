from __future__ import annotations

from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import app as app_module


def test_public_board_preview_setting_is_exposed_and_persisted() -> None:
    old_enabled = app_module.LOBBY_ROOM_SETTINGS["public_board_previews_enabled"]
    old_path = app_module.ROOM_SETTINGS_PATH
    try:
        app_module._apply_lobby_management_settings(
            {"public_board_previews_enabled": False}
        )
        assert app_module._lobby_management_settings()[
            "public_board_previews_enabled"
        ] is False
        assert app_module._lobby_admin_payload()[
            "public_board_previews_enabled"
        ] is False
        assert app_module.list_rooms()["public_board_previews_enabled"] is False

        with tempfile.TemporaryDirectory() as directory:
            app_module.ROOM_SETTINGS_PATH = Path(directory) / "room-settings.json"
            assert app_module._save_persisted_room_management_settings() is True
            stored = app_module.load_room_settings(app_module.ROOM_SETTINGS_PATH)
            assert stored[app_module.LOBBY_SETTINGS_STORAGE_KEY][
                "public_board_previews_enabled"
            ] is False
    finally:
        app_module.ROOM_SETTINGS_PATH = old_path
        app_module.LOBBY_ROOM_SETTINGS["public_board_previews_enabled"] = old_enabled


def test_admin_settings_request_defaults_preview_to_disabled() -> None:
    request = app_module.AdminLobbySettingsUpdateRequest(
        main_room_count=4,
        private_room_count=4,
    )
    assert request.public_board_previews_enabled is False

    enabled = app_module.AdminLobbySettingsUpdateRequest(
        main_room_count=4,
        private_room_count=4,
        public_board_previews_enabled=True,
    )
    assert enabled.public_board_previews_enabled is True


if __name__ == "__main__":
    test_public_board_preview_setting_is_exposed_and_persisted()
    test_admin_settings_request_defaults_preview_to_disabled()
    print("LOBBY_PUBLIC_PIXEL_SETTING_TEST_OK")

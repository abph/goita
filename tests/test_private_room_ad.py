from pathlib import Path
import asyncio
import copy
import tempfile

import backend.app as app_module
from backend.room_settings_persistence import load_room_settings
from fastapi import HTTPException


ROOT = Path(__file__).parents[1]
HTML = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
WHISPER_JS = (ROOT / "frontend" / "lobbyWhisper.js").read_text(encoding="utf-8")


def test_private_room_ad_accepts_an_empty_optional_url():
    room_id = next(iter(app_module.PRIVATE_ROOM_NAMES))
    settings = app_module._normalize_private_room_ad_settings({
        "enabled": True,
        "message": "大会のお知らせ",
        "url": "",
        "room_ids": [room_id],
    })

    assert settings["url"] == ""
    assert settings["title"] == "お知らせ"
    assert settings["room_ids"] == [room_id]


def test_private_room_ad_rejects_non_http_urls():
    room_id = next(iter(app_module.PRIVATE_ROOM_NAMES))
    try:
        app_module._normalize_private_room_ad_settings({
            "enabled": True,
            "message": "大会のお知らせ",
            "url": "javascript:alert(1)",
            "room_ids": [room_id],
        })
    except HTTPException as error:
        assert error.status_code == 400
    else:
        raise AssertionError("unsafe URL was accepted")


def test_private_room_ad_admin_controls_and_payload_are_connected():
    assert app_module._lobby_admin_payload()["private_room_ad"] == app_module.PRIVATE_ROOM_AD_SETTINGS
    assert 'id="lobbyPrivateAdEnabled"' in HTML
    assert 'id="lobbyPrivateAdTitle"' in HTML
    assert 'id="lobbyPrivateAdMessage"' in HTML
    assert 'id="lobbyPrivateAdUrl"' in HTML
    assert 'id="lobbyPrivateAdRoomList"' in HTML
    assert "private_ad_room_ids: privateAdRoomIds" in HTML
    assert "state.private_room_ad || null" in HTML


def test_private_room_ad_only_acts_like_a_link_when_url_exists():
    assert 'whisper.classList.toggle("has-link", Boolean(activeUrl));' in WHISPER_JS
    assert 'window.open(activeUrl, "_blank", "noopener,noreferrer");' in WHISPER_JS
    assert 'activeMessages = isPublicRoom ? PUBLIC_MESSAGES : [privateMessage];' in WHISPER_JS


def test_private_room_ad_is_written_to_the_persistent_room_settings_file():
    previous_path = app_module.ROOM_SETTINGS_PATH
    previous_ad = copy.deepcopy(app_module.PRIVATE_ROOM_AD_SETTINGS)
    room_id = next(iter(app_module.PRIVATE_ROOM_NAMES))
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            app_module.ROOM_SETTINGS_PATH = Path(temp_dir) / "room-settings.json"
            app_module.PRIVATE_ROOM_AD_SETTINGS.update({
                "enabled": True,
                "title": "研究会のお知らせ",
                "message": "永続保存の確認",
                "url": "",
                "room_ids": [room_id],
            })

            assert app_module._save_persisted_room_management_settings() is True
            stored = load_room_settings(app_module.ROOM_SETTINGS_PATH)
            assert stored[app_module.LOBBY_SETTINGS_STORAGE_KEY]["private_room_ad"] == {
                "enabled": True,
                "title": "研究会のお知らせ",
                "message": "永続保存の確認",
                "url": "",
                "room_ids": [room_id],
            }
    finally:
        app_module.ROOM_SETTINGS_PATH = previous_path
        app_module.PRIVATE_ROOM_AD_SETTINGS.clear()
        app_module.PRIVATE_ROOM_AD_SETTINGS.update(previous_ad)


def test_lobby_admin_can_save_and_publish_a_private_room_ad():
    previous_path = app_module.ROOM_SETTINGS_PATH
    previous_lobby = app_module._lobby_management_settings()
    room_id = next(iter(app_module.PRIVATE_ROOM_NAMES))
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            app_module.ROOM_SETTINGS_PATH = Path(temp_dir) / "room-settings.json"
            result = asyncio.run(app_module.update_lobby_admin_settings(
                app_module.LobbySettingsUpdateRequest(
                    admin_password=app_module.LOBBY_ADMIN_PASSWORD,
                    main_room_count=previous_lobby["main_room_count"],
                    private_room_count=previous_lobby["private_room_count"],
                    private_ad_enabled=True,
                    private_ad_title="大会のお知らせ",
                    private_ad_message="大会ページをご覧ください",
                    private_ad_url="https://example.com/event",
                    private_ad_room_ids=[room_id],
                )
            ))

            assert result["private_room_ad"]["enabled"] is True
            assert app_module._private_room_ad_public_payload(room_id) == {
                "enabled": True,
                "label": "大会のお知らせ",
                "message": "大会ページをご覧ください",
                "url": "https://example.com/event",
            }
    finally:
        app_module.ROOM_SETTINGS_PATH = previous_path
        app_module._apply_lobby_management_settings(previous_lobby)
        app_module.setup_main_rooms()
        app_module.setup_supporter_rooms()

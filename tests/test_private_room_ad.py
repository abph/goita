from pathlib import Path
import asyncio
import copy
import tempfile

import backend.app as app_module
from backend.room_settings_persistence import load_room_settings
from fastapi import HTTPException


ROOT = Path(__file__).parents[1]
HTML = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
ADMIN_HTML = (ROOT / "frontend" / "admin.html").read_text(encoding="utf-8")
WHISPER_JS = (ROOT / "frontend" / "lobbyWhisper.js").read_text(
    encoding="utf-8"
)


def _empty_ads():
    return {
        game_id: {
            "enabled": False,
            "title": "お知らせ",
            "message": "",
            "url": "",
        }
        for game_id in app_module.PRIVATE_ROOM_NAMES
    }


def test_private_room_ad_accepts_an_empty_optional_url():
    settings = app_module._normalize_private_room_ad_settings({
        "enabled": True,
        "message": "大会のお知らせ",
        "url": "",
    })

    assert settings == {
        "enabled": True,
        "title": "お知らせ",
        "message": "大会のお知らせ",
        "url": "",
    }


def test_private_room_ad_rejects_non_http_urls():
    try:
        app_module._normalize_private_room_ad_settings({
            "enabled": True,
            "message": "大会のお知らせ",
            "url": "javascript:alert(1)",
        })
    except HTTPException as error:
        assert error.status_code == 400
    else:
        raise AssertionError("unsafe URL was accepted")


def test_private_room_ad_admin_controls_and_payload_are_connected():
    payload = app_module._lobby_admin_payload()
    assert payload["private_room_ads"] == app_module.PRIVATE_ROOM_AD_SETTINGS
    assert 'id="privateAdRoomSelect"' in ADMIN_HTML
    assert 'id="privateAdEnabled"' in ADMIN_HTML
    assert 'id="privateAdTitle"' in ADMIN_HTML
    assert 'id="privateAdMessage"' in ADMIN_HTML
    assert 'id="privateAdUrl"' in ADMIN_HTML
    assert 'id="privateAdSummary"' in ADMIN_HTML
    assert "private_room_ads: privateRoomAds" in ADMIN_HTML
    assert "state.private_room_ad || null" in HTML


def test_private_room_ad_only_acts_like_a_link_when_url_exists():
    assert 'whisper.classList.toggle("has-link", Boolean(activeUrl));' in WHISPER_JS
    assert 'window.open(activeUrl, "_blank", "noopener,noreferrer");' in WHISPER_JS
    assert 'activeMessages = isPublicRoom ? PUBLIC_MESSAGES : [privateMessage];' in WHISPER_JS


def test_private_room_ads_are_written_per_room_to_persistent_settings():
    previous_path = app_module.ROOM_SETTINGS_PATH
    previous_ads = copy.deepcopy(app_module.PRIVATE_ROOM_AD_SETTINGS)
    room_ids = list(app_module.PRIVATE_ROOM_NAMES)[:2]
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            app_module.ROOM_SETTINGS_PATH = Path(temp_dir) / "room-settings.json"
            ads = _empty_ads()
            ads[room_ids[0]] = {
                "enabled": True,
                "title": "研究会",
                "message": "Aのお知らせ",
                "url": "",
            }
            ads[room_ids[1]] = {
                "enabled": True,
                "title": "大会",
                "message": "Bのお知らせ",
                "url": "https://example.com/event",
            }
            app_module.PRIVATE_ROOM_AD_SETTINGS.clear()
            app_module.PRIVATE_ROOM_AD_SETTINGS.update(ads)

            assert app_module._save_persisted_room_management_settings() is True
            stored = load_room_settings(app_module.ROOM_SETTINGS_PATH)
            lobby = stored[app_module.LOBBY_SETTINGS_STORAGE_KEY]
            persisted = lobby["private_room_ads"]
            assert persisted[room_ids[0]]["message"] == "Aのお知らせ"
            assert persisted[room_ids[1]]["message"] == "Bのお知らせ"
            assert "private_room_ad" not in lobby
    finally:
        app_module.ROOM_SETTINGS_PATH = previous_path
        app_module.PRIVATE_ROOM_AD_SETTINGS.clear()
        app_module.PRIVATE_ROOM_AD_SETTINGS.update(previous_ads)


def test_legacy_private_room_ad_is_migrated_to_selected_rooms():
    previous_ads = copy.deepcopy(app_module.PRIVATE_ROOM_AD_SETTINGS)
    room_ids = list(app_module.PRIVATE_ROOM_NAMES)[:3]
    try:
        app_module._apply_lobby_management_settings({
            "private_room_ad": {
                "enabled": True,
                "title": "以前のお知らせ",
                "message": "引き継ぐ文章",
                "url": "",
                "room_ids": room_ids[:2],
            },
        })
        assert app_module.PRIVATE_ROOM_AD_SETTINGS[room_ids[0]]["enabled"] is True
        assert app_module.PRIVATE_ROOM_AD_SETTINGS[room_ids[1]]["enabled"] is True
        assert app_module.PRIVATE_ROOM_AD_SETTINGS[room_ids[2]]["enabled"] is False
        assert app_module.PRIVATE_ROOM_AD_SETTINGS[room_ids[0]]["message"] == "引き継ぐ文章"
    finally:
        app_module.PRIVATE_ROOM_AD_SETTINGS.clear()
        app_module.PRIVATE_ROOM_AD_SETTINGS.update(previous_ads)


def test_lobby_admin_can_save_distinct_private_room_ads():
    previous_path = app_module.ROOM_SETTINGS_PATH
    previous_lobby = app_module._lobby_management_settings()
    room_ids = list(app_module.PRIVATE_ROOM_NAMES)[:2]
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            app_module.ROOM_SETTINGS_PATH = Path(temp_dir) / "room-settings.json"
            ads = _empty_ads()
            ads[room_ids[0]] = {
                "enabled": True,
                "title": "A専用",
                "message": "Aだけに表示",
                "url": "",
            }
            ads[room_ids[1]] = {
                "enabled": True,
                "title": "B専用",
                "message": "Bだけに表示",
                "url": "https://example.com/b",
            }
            result = asyncio.run(app_module.update_lobby_admin_settings(
                app_module.LobbySettingsUpdateRequest(
                    admin_password=app_module.LOBBY_ADMIN_PASSWORD,
                    main_room_count=previous_lobby["main_room_count"],
                    private_room_count=previous_lobby["private_room_count"],
                    private_room_ads=ads,
                )
            ))

            assert result["private_room_ads"][room_ids[0]]["title"] == "A専用"
            assert result["private_room_ads"][room_ids[1]]["title"] == "B専用"
            assert app_module._private_room_ad_public_payload(room_ids[0]) == {
                "enabled": True,
                "label": "A専用",
                "message": "Aだけに表示",
                "url": "",
            }
            assert app_module._private_room_ad_public_payload(room_ids[1]) == {
                "enabled": True,
                "label": "B専用",
                "message": "Bだけに表示",
                "url": "https://example.com/b",
            }
    finally:
        app_module.ROOM_SETTINGS_PATH = previous_path
        app_module._apply_lobby_management_settings(previous_lobby)
        app_module.setup_main_rooms()
        app_module.setup_supporter_rooms()


if __name__ == "__main__":
    test_private_room_ad_accepts_an_empty_optional_url()
    test_private_room_ad_rejects_non_http_urls()
    test_private_room_ad_admin_controls_and_payload_are_connected()
    test_private_room_ad_only_acts_like_a_link_when_url_exists()
    test_private_room_ads_are_written_per_room_to_persistent_settings()
    test_legacy_private_room_ad_is_migrated_to_selected_rooms()
    test_lobby_admin_can_save_distinct_private_room_ads()
    print("PRIVATE_ROOM_AD_TEST_OK")

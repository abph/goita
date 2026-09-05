import asyncio
import copy

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from backend import app as app_module
from backend.room_settings_persistence import load_room_settings


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    for name in ["PUBLIC_ROOM_AD_SETTINGS", "PRIVATE_ROOM_AD_SETTINGS", "LOBBY_ROOM_SETTINGS", "MAIN_ROOM_NAMES"]:
        monkeypatch.setattr(app_module, name, copy.deepcopy(getattr(app_module, name)))
    monkeypatch.setattr(app_module, "GAMES", {})
    monkeypatch.setattr(app_module, "manager", app_module.ConnectionManager())
    monkeypatch.setattr(app_module, "ROOM_SETTINGS_PATH", tmp_path / "settings.json")
    app_module.setup_main_rooms()
    app_module.setup_supporter_rooms()
    return list(app_module.MAIN_ROOM_NAMES)[:2]


def update(ads=None, **overrides):
    values = dict(admin_password=app_module.LOBBY_ADMIN_PASSWORD,
                  main_room_count=4, private_room_count=4,
                  private_room_ads=app_module.PRIVATE_ROOM_AD_SETTINGS)
    if ads is not None:
        values["public_room_ads"] = ads
    values.update(overrides)
    return asyncio.run(app_module.update_lobby_admin_settings(app_module.LobbySettingsUpdateRequest(**values)))


def test_public_notices_save_per_room_restore_and_keep_special_mode(isolated):
    first, second = isolated
    ads = {
        first: {"enabled": True, "mode": "custom", "title": "大会", "message": "来週開催", "url": "https://example.com/event"},
        second: {"enabled": True, "mode": "whisper", "title": "保存する下書き", "message": "通常表示の下書き"},
    }
    response = update(ads)
    assert response["public_room_ads"][first]["message"] == "来週開催"
    assert app_module._public_room_ad_public_payload(first)["url"] == "https://example.com/event"
    assert app_module._public_room_ad_public_payload(second)["mode"] == "whisper"
    assert app_module._public_room_ad_public_payload(second)["message"] == ""
    private = next(iter(app_module.PRIVATE_ROOM_NAMES))
    assert app_module._public_room_ad_public_payload(private)["enabled"] is False
    saved = load_room_settings(app_module.ROOM_SETTINGS_PATH)[app_module.LOBBY_SETTINGS_STORAGE_KEY]
    app_module.PUBLIC_ROOM_AD_SETTINGS[first]["message"] = "changed"
    app_module._apply_lobby_management_settings(saved)
    assert app_module.PUBLIC_ROOM_AD_SETTINGS[first]["message"] == "来週開催"
    assert app_module.PUBLIC_ROOM_AD_SETTINGS[second]["message"] == "通常表示の下書き"
    # Older clients omit this new property when saving unrelated settings.
    update()
    assert app_module.PUBLIC_ROOM_AD_SETTINGS[first]["message"] == "来週開催"


def test_disable_notice_and_failed_save_rollback(isolated, monkeypatch):
    first, _ = isolated
    update({first: {"enabled": False, "mode": "custom", "message": "非表示の文章"}})
    public = app_module._public_room_ad_public_payload(first)
    assert public["enabled"] is False and public["message"] == ""
    before = app_module._lobby_management_settings()
    monkeypatch.setattr(app_module, "_save_persisted_room_management_settings", lambda: False)
    with pytest.raises(HTTPException) as error:
        update({first: {"enabled": True, "mode": "custom", "message": "保存失敗"}})
    assert error.value.status_code == 500
    assert app_module._lobby_management_settings() == before


def test_validation_and_admin_authorization(isolated):
    first, _ = isolated
    before = app_module._lobby_management_settings()
    for ads in [
        {"not-a-room": {"enabled": False}},
        {first: {"enabled": True, "mode": "custom", "message": ""}},
        {first: {"enabled": True, "mode": "custom", "message": "x", "url": "javascript:alert(1)"}},
    ]:
        with pytest.raises(HTTPException):
            update(ads)
        assert app_module._lobby_management_settings() == before
    with pytest.raises(ValidationError):
        update({first: {"mode": "unknown"}})
    with pytest.raises(HTTPException) as error:
        update({first: {"enabled": False}}, admin_password="invalid-test-password")
    assert error.value.status_code == 401
    assert app_module._lobby_management_settings() == before

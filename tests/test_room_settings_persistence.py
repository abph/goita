import json
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.room_settings_persistence import (
    ROOM_SETTINGS_FILENAME,
    hash_admin_password,
    is_admin_password_hash,
    load_room_settings,
    resolve_room_settings_path,
    save_room_settings,
    verify_admin_password,
)


ROOT = Path(__file__).parents[1]
APP_SOURCE = (ROOT / "backend" / "app.py").read_text(encoding="utf-8")


def test_room_settings_path_uses_explicit_then_persistent_directory() -> None:
    assert resolve_room_settings_path({}) is None
    assert resolve_room_settings_path({
        "GOITA_PERSISTENT_DATA_DIR": "/var/data",
    }) == Path("/var/data") / ROOM_SETTINGS_FILENAME
    assert resolve_room_settings_path({
        "GOITA_ROOM_SETTINGS_PATH": "/tmp/custom-room-settings.json",
        "GOITA_PERSISTENT_DATA_DIR": "/var/data",
    }) == Path("/tmp/custom-room-settings.json")


def test_room_settings_round_trip_is_atomic_and_keeps_unicode() -> None:
    rooms = {
        "room-silver-02": {
            "owner_name": "プライベートB",
            "password": "新しい合言葉",
            "ai_profile": "current",
            "show_legal_actions": False,
            "show_log": True,
            "room_background_image": "",
        },
        "room-gold-01": {
            "owner_name": "プライベートA",
            "password": None,
        },
    }
    with TemporaryDirectory() as directory:
        path = Path(directory) / "nested" / ROOM_SETTINGS_FILENAME
        assert save_room_settings(path, rooms) is True
        assert load_room_settings(path) == rooms
        assert not path.with_name(f".{path.name}.tmp").exists()


def test_bad_or_unsupported_room_settings_are_ignored() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / ROOM_SETTINGS_FILENAME
        path.write_text("not-json", encoding="utf-8")
        assert load_room_settings(path) == {}

        path.write_text(json.dumps({"version": 999, "rooms": {"x": {}}}), encoding="utf-8")
        assert load_room_settings(path) == {}


def test_admin_password_hash_can_be_verified_without_storing_plaintext() -> None:
    stored_hash = hash_admin_password("room-secret")

    assert is_admin_password_hash(stored_hash)
    assert "room-secret" not in stored_hash
    assert verify_admin_password("room-secret", stored_hash)
    assert not verify_admin_password("wrong-secret", stored_hash)
    assert not verify_admin_password("room-secret", "not-a-supported-hash")

def test_app_loads_and_saves_only_room_management_settings() -> None:
    setup_section = APP_SOURCE[
        APP_SOURCE.index("def _persisted_room_ids"):
        APP_SOURCE.index("def _check_effects")
    ]
    snapshot_section = APP_SOURCE[
        APP_SOURCE.index("def _room_management_settings"):
        APP_SOURCE.index("def _apply_room_management_settings")
    ]
    update_section = APP_SOURCE[
        APP_SOURCE.index('@app.post("/games/{game_id}/update_settings")'):
        APP_SOURCE.index('@app.post("/games/{game_id}/start")')
    ]

    assert "_load_persisted_room_management_settings()" in setup_section
    assert "_save_persisted_room_management_settings()" in update_section
    assert "ルーム設定を永続保存できませんでした" in update_section
    assert '"admin_password"' not in snapshot_section
    assert '"admin_password_hash"' in snapshot_section
    assert '"password"' in snapshot_section
    assert '"owner_name"' in snapshot_section
    assert '"ai_profile"' in snapshot_section


if __name__ == "__main__":
    test_room_settings_path_uses_explicit_then_persistent_directory()
    test_room_settings_round_trip_is_atomic_and_keeps_unicode()
    test_bad_or_unsupported_room_settings_are_ignored()
    test_admin_password_hash_can_be_verified_without_storing_plaintext()
    test_app_loads_and_saves_only_room_management_settings()
    print("ROOM_SETTINGS_PERSISTENCE_TEST_OK")

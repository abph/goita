from __future__ import annotations

import os
from pathlib import Path


os.environ.pop("DEBUG_ROOM_PASSWORD", None)

import backend.app as app_module


ROOT = Path(__file__).resolve().parents[1]


def _run() -> None:
    debug_room = app_module.GAMES.get(app_module.DEBUG_GID)
    assert debug_room is not None
    assert debug_room["owner_name"] == "デバッグルーム"
    assert debug_room["password"] == "goita-debug"
    assert debug_room["admin_password"] == "goita-debug"
    assert debug_room["ai_seats"] == ["B", "C", "D"]
    assert debug_room["show_legal_actions"] is True
    assert debug_room["show_log"] is True
    assert debug_room["hidden_from_lobby"] is True
    assert debug_room["is_debug_room"] is True

    listed_ids = {room["game_id"] for room in app_module.list_rooms()["rooms"]}
    assert app_module.DEBUG_GID not in listed_ids

    verified = app_module.verify_admin(app_module.DEBUG_GID, "goita-debug")
    assert verified["ok"] is True
    assert verified["show_legal_actions"] is True
    assert verified["show_log"] is True

    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert 'params.get("password")' in html
    assert "autoEnterDebugRoomFromUrl" in html
    assert 'url.searchParams.delete("password")' in html


if __name__ == "__main__":
    _run()
    print("DEBUG_ROOM_TEST_OK")

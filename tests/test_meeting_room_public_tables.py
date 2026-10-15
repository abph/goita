from pathlib import Path
import asyncio
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import app as app_module


def test_public_table_snapshot_never_exposes_hidden_piece_identity() -> None:
    game = app_module._create_game_obj(dealer="A")
    game["is_started"] = True
    game["board"]["A"]["receive"][0] = "7"
    game["board"]["A"]["receive_hidden"][0] = True
    game["board"]["A"]["attack"][0] = "2"
    game["last_public_action"] = {
        "player": "A",
        "type": "attack_after_block",
        "at_ms": 123,
    }

    snapshot = app_module._meeting_room_public_table_snapshot("main", game)

    assert snapshot["board_public"]["A"]["receive"][0] == "■"
    assert snapshot["board_public"]["A"]["attack"][0] == "2"
    assert snapshot["last_public_action"] == {
        "player": "A",
        "type": "attack_after_block",
        "at_ms": 123,
    }
    assert "hands" not in snapshot
    assert "init_hands" not in snapshot
    assert "face_down_pieces" not in snapshot
    assert "log" not in snapshot


def test_public_table_list_excludes_meeting_room_and_hidden_rooms() -> None:
    old_settings = dict(app_module.LOBBY_ROOM_SETTINGS)
    try:
        app_module.LOBBY_ROOM_SETTINGS["main_room_count"] = 4
        app_module.setup_main_rooms()

        tables = app_module.list_public_tables()["tables"]
        table_ids = [table["game_id"] for table in tables]

        assert table_ids == ["main", "main-b", "main-c", "main-e"]
        assert app_module.MEETING_ROOM_GID not in table_ids
        assert "main-f" not in table_ids
    finally:
        app_module.LOBBY_ROOM_SETTINGS.update(old_settings)
        app_module.setup_main_rooms()


def test_public_room_update_notifies_meeting_room_and_lobby_viewers() -> None:
    class FakeWebSocket:
        def __init__(self) -> None:
            self.messages = []

        async def send_json(self, payload) -> None:
            self.messages.append(payload)

    meeting_socket = FakeWebSocket()
    lobby_socket = FakeWebSocket()
    old_connections = app_module.manager.active_connections.get(
        app_module.MEETING_ROOM_GID
    )
    old_lobby_connections = app_module.manager.active_connections.get("lobby")
    old_preview_setting = app_module.LOBBY_ROOM_SETTINGS[
        "public_board_previews_enabled"
    ]
    app_module.LOBBY_ROOM_SETTINGS["public_board_previews_enabled"] = True
    app_module.manager.active_connections[app_module.MEETING_ROOM_GID] = [
        meeting_socket
    ]
    app_module.manager.active_connections["lobby"] = [lobby_socket]
    try:
        asyncio.run(app_module.manager.broadcast_update("main"))
        assert meeting_socket.messages == [
            {"type": "public_table_update", "game_id": "main"}
        ]
        assert {"type": "public_table_update", "game_id": "main"} in lobby_socket.messages
    finally:
        if old_connections is None:
            app_module.manager.active_connections.pop(
                app_module.MEETING_ROOM_GID, None
            )
        else:
            app_module.manager.active_connections[
                app_module.MEETING_ROOM_GID
            ] = old_connections
        if old_lobby_connections is None:
            app_module.manager.active_connections.pop("lobby", None)
        else:
            app_module.manager.active_connections["lobby"] = old_lobby_connections
        app_module.LOBBY_ROOM_SETTINGS["public_board_previews_enabled"] = (
            old_preview_setting
        )


def test_gathering_table_update_always_notifies_lobby_viewers() -> None:
    class FakeWebSocket:
        def __init__(self) -> None:
            self.messages = []

        async def send_json(self, payload) -> None:
            self.messages.append(payload)

    table_id = app_module.GATHERING_TABLE_IDS[0]
    meeting_socket = FakeWebSocket()
    lobby_socket = FakeWebSocket()
    old_connections = app_module.manager.active_connections.get(
        app_module.MEETING_ROOM_GID
    )
    old_lobby_connections = app_module.manager.active_connections.get("lobby")
    old_preview_setting = app_module.LOBBY_ROOM_SETTINGS[
        "public_board_previews_enabled"
    ]
    app_module.LOBBY_ROOM_SETTINGS["public_board_previews_enabled"] = False
    app_module.manager.active_connections[app_module.MEETING_ROOM_GID] = [
        meeting_socket
    ]
    app_module.manager.active_connections["lobby"] = [lobby_socket]
    try:
        asyncio.run(app_module.manager.broadcast_update(table_id))
        expected = {"type": "public_table_update", "game_id": table_id}
        assert expected in meeting_socket.messages
        assert expected in lobby_socket.messages
    finally:
        if old_connections is None:
            app_module.manager.active_connections.pop(
                app_module.MEETING_ROOM_GID, None
            )
        else:
            app_module.manager.active_connections[
                app_module.MEETING_ROOM_GID
            ] = old_connections
        if old_lobby_connections is None:
            app_module.manager.active_connections.pop("lobby", None)
        else:
            app_module.manager.active_connections["lobby"] = old_lobby_connections
        app_module.LOBBY_ROOM_SETTINGS["public_board_previews_enabled"] = (
            old_preview_setting
        )


if __name__ == "__main__":
    test_public_table_snapshot_never_exposes_hidden_piece_identity()
    test_public_table_list_excludes_meeting_room_and_hidden_rooms()
    test_public_room_update_notifies_meeting_room_and_lobby_viewers()
    test_gathering_table_update_always_notifies_lobby_viewers()
    print("MEETING_ROOM_PUBLIC_TABLES_TEST_OK")

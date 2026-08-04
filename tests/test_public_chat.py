from __future__ import annotations

import asyncio

import backend.app as app_module


async def _run() -> None:
    app_module.setup_main_rooms()
    app_module.setup_supporter_rooms()
    original_public_messages = list(app_module.PUBLIC_CHAT_MESSAGES)
    original_broadcast = app_module.manager.broadcast_update
    private_id = "room-gold-01"
    private_messages = list(app_module.GAMES[private_id].get("chat_messages", []))
    main_id = next(iter(app_module.MAIN_ROOM_NAMES))
    other_main_id = next(
        game_id for game_id in app_module.MAIN_ROOM_NAMES if game_id != main_id
    )
    main_messages = list(app_module.GAMES[main_id].get("chat_messages", []))
    broadcasts: list[str] = []

    async def record_broadcast(channel: str) -> None:
        broadcasts.append(channel)

    try:
        app_module.PUBLIC_CHAT_MESSAGES.clear()
        app_module.GAMES[private_id]["chat_messages"] = []
        app_module.GAMES[main_id]["chat_messages"] = []
        app_module.manager.broadcast_update = record_broadcast

        lobby_result = await app_module.post_lobby_chat_message(
            app_module.ChatRequest(name="Lobby", message="hello from lobby")
        )
        lobby_item = lobby_result["chat_messages"][-1]
        assert lobby_item["origin"] == "lobby"
        assert lobby_item["sender"] == "Lobby"
        assert lobby_item["message"] == "hello from lobby"
        assert set(broadcasts) == {"lobby", *app_module.MAIN_ROOM_NAMES.keys()}
        assert app_module.list_rooms()["public_chat_messages"][-1] == lobby_item
        assert lobby_item in app_module._chat_messages_for_game(
            main_id, app_module.GAMES[main_id]
        )

        broadcasts.clear()
        room_result = await app_module.post_chat_message(
            main_id,
            app_module.ChatRequest(name="Room User", message="hello from room"),
        )
        room_item = room_result["chat_messages"][-1]
        assert room_item["origin"] == "public_room"
        assert room_item["room_name"] == app_module.MAIN_ROOM_NAMES[main_id]
        assert room_item["message"] == "hello from room"
        assert broadcasts == [main_id]
        assert all(
            item.get("message") != "hello from room"
            for item in app_module.list_rooms()["public_chat_messages"]
        )
        assert all(
            item.get("message") != "hello from room"
            for item in app_module._chat_messages_for_game(
                other_main_id, app_module.GAMES[other_main_id]
            )
        )

        broadcasts.clear()
        private_result = await app_module.post_chat_message(
            private_id,
            app_module.ChatRequest(name="Private User", message="private only"),
        )
        private_item = private_result["chat_messages"][-1]
        assert private_item["message"] == "private only"
        assert "origin" not in private_item
        assert broadcasts == [private_id]
        assert all(
            item.get("message") != "private only"
            for item in app_module.PUBLIC_CHAT_MESSAGES
        )
    finally:
        app_module.PUBLIC_CHAT_MESSAGES[:] = original_public_messages
        app_module.GAMES[private_id]["chat_messages"] = private_messages
        app_module.GAMES[main_id]["chat_messages"] = main_messages
        app_module.manager.broadcast_update = original_broadcast


if __name__ == "__main__":
    asyncio.run(_run())
    print("PUBLIC_CHAT_TEST_OK")

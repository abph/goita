from __future__ import annotations

import asyncio
from pathlib import Path

import backend.app as app_module


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")


def test_stamp_controls_are_available_in_lobby_and_room_chat() -> None:
    assert 'id="lobbyChatStampButton"' in HTML
    assert 'id="lobbyChatStampPicker"' in HTML
    assert 'id="chatStampButton"' in HTML
    assert 'id="chatStampPicker"' in HTML
    assert "const CHAT_STAMP_DEFINITIONS" in HTML
    assert "function initializeChatStampPickers()" in HTML
    assert "function sendChatStamp(kind, stampId)" in HTML
    assert "function buildChatStampVisual(definition, owner, decorative = false)" in HTML
    assert "/static/stamps/${encodeURIComponent(definition.id)}.png" in HTML
    assert (ROOT / "frontend" / "stamps" / "greeting.png").is_file()
    assert 'message_type === "stamp"' in HTML


def test_initial_stamp_catalog_matches_the_ten_requested_stamps() -> None:
    assert app_module.CHAT_STAMPS == {
        "greeting": "よろしくおねがいします！",
        "thanks": "ありがとうございました！",
        "thinking": "考え中です",
        "nice": "ナイス！",
        "sorry": "ごめん！",
        "surprised": "えっ！？",
        "happy": "やった！",
        "leave_it": "あとはまかせた！",
        "got_me": "やられた！",
        "goita_fun": "ごいたのしい！",
    }


async def _test_stamp_delivery() -> None:
    app_module.setup_main_rooms()
    main_id = next(iter(app_module.MAIN_ROOM_NAMES))
    game = app_module.GAMES[main_id]
    original_room_messages = list(game.get("chat_messages", []))
    original_public_messages = list(app_module.PUBLIC_CHAT_MESSAGES)
    original_everyone_messages = list(app_module.EVERYONE_CHAT_MESSAGES)
    original_lobby_here_messages = list(app_module.LOBBY_HERE_CHAT_MESSAGES)
    original_last_timestamp = app_module.LAST_CHAT_TIMESTAMP
    original_broadcast = app_module.manager.broadcast_update
    broadcasts: list[str] = []

    async def record_broadcast(channel: str) -> None:
        broadcasts.append(channel)

    try:
        game["chat_messages"] = []
        app_module.PUBLIC_CHAT_MESSAGES.clear()
        app_module.EVERYONE_CHAT_MESSAGES.clear()
        app_module.LOBBY_HERE_CHAT_MESSAGES.clear()
        app_module.manager.broadcast_update = record_broadcast

        room_result = await app_module.post_chat_message(
            main_id,
            app_module.ChatRequest(
                name="Stamp User",
                message="@here altered label",
                stamp_id="nice",
            ),
        )
        room_stamp = room_result["chat_messages"][-1]
        assert room_stamp["message"] == "@here ナイス！"
        assert room_stamp["message_type"] == "stamp"
        assert room_stamp["stamp_id"] == "nice"
        assert room_stamp["mention_scope"] == "here"
        assert broadcasts == [main_id]

        broadcasts.clear()
        lobby_result = await app_module.post_lobby_chat_message(
            app_module.ChatRequest(
                name="Lobby Stamp User",
                message="",
                stamp_id="thanks",
            ),
        )
        lobby_stamp = lobby_result["chat_messages"][-1]
        assert lobby_stamp["message"] == "ありがとうございました！"
        assert lobby_stamp["message_type"] == "stamp"
        assert lobby_stamp["stamp_id"] == "thanks"
        assert set(broadcasts) == {"lobby", *app_module.MAIN_ROOM_NAMES.keys()}

        try:
            await app_module.post_lobby_chat_message(
                app_module.ChatRequest(message="", stamp_id="unknown")
            )
        except app_module.HTTPException as error:
            assert error.status_code == 400
        else:
            raise AssertionError("unknown stamp must be rejected")
    finally:
        game["chat_messages"] = original_room_messages
        app_module.PUBLIC_CHAT_MESSAGES[:] = original_public_messages
        app_module.EVERYONE_CHAT_MESSAGES[:] = original_everyone_messages
        app_module.LOBBY_HERE_CHAT_MESSAGES[:] = original_lobby_here_messages
        app_module.LAST_CHAT_TIMESTAMP = original_last_timestamp
        app_module.manager.broadcast_update = original_broadcast


if __name__ == "__main__":
    test_stamp_controls_are_available_in_lobby_and_room_chat()
    test_initial_stamp_catalog_matches_the_ten_requested_stamps()
    asyncio.run(_test_stamp_delivery())
    print("CHAT_STAMPS_TEST_OK")

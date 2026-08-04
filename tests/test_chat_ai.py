from __future__ import annotations

import asyncio
import os

from fastapi import HTTPException
from starlette.requests import Request

import backend.app as app_module


async def _run() -> None:
    original_request = app_module._request_gemini_help
    original_api_key = os.environ.get("GEMINI_API_KEY")
    original_public_messages = list(app_module.PUBLIC_CHAT_MESSAGES)
    try:
        app_module.GAMES.clear()
        app_module.PUBLIC_CHAT_MESSAGES.clear()
        app_module.AI_HELP_LAST_REQUEST.clear()
        assert "https://vrcgoita.com/goita/rule/" in app_module.AI_HELP_SYSTEM_PROMPT
        assert "https://vrcgoita.com/goita/strategy/" in app_module.AI_HELP_SYSTEM_PROMPT
        assert "操作回答の末尾" in app_module.AI_HELP_SYSTEM_PROMPT
        assert "初心者サポートを有効にする" in app_module.AI_HELP_SYSTEM_PROMPT
        assert "分類名、見出しとして回答に表示しない" in app_module.AI_HELP_SYSTEM_PROMPT
        basic_usage_answer = app_module._lobby_basic_usage_answer("使い方が分からない")
        assert basic_usage_answer is not None
        assert basic_usage_answer.startswith("まず、遊びたい部屋を選んで入ってください。")
        assert "ホストが「開始」を押す" in basic_usage_answer
        assert "1. そろうごいたのページ操作" not in basic_usage_answer
        assert "房主点击“开始”" in app_module._lobby_basic_usage_answer(
            "不知道怎么用", "zh"
        )
        assert "host can press Start" in app_module._lobby_basic_usage_answer(
            "How do I use this?", "en"
        )
        assert app_module._beginner_support_move_answer("どの駒を出せばいい？") is not None
        assert app_module._beginner_support_move_answer("何を伏せるべき？") is not None
        assert app_module._beginner_support_move_answer("受けるべき？それともパス？") is not None
        assert app_module._beginner_support_move_answer("ごいたの攻め方を教えて") is None
        os.environ["GEMINI_API_KEY"] = "test-key"
        app_module._request_gemini_help = lambda question, language="ja": f"案内回答: {question}"

        request = Request({
            "type": "http",
            "method": "POST",
            "path": "/games/main/chat/ask_ai",
            "headers": [],
            "client": ("127.0.0.1", 12345),
        })
        payload = app_module.ChatAiRequest(
            seat="W",
            client_id="test-client",
            name="テスト観戦者",
            message="Autoはどう使いますか？",
        )

        result = await app_module.ask_chat_ai("main", payload, request)
        messages = result["chat_messages"]
        assert result["ok"] is True
        assert messages[-2]["sender"] == "観戦: テスト観戦者"
        assert messages[-1]["sender"] == "案内AI"
        assert messages[-1]["ai_answer"] is True
        assert "Auto" in messages[-1]["message"]

        lobby_payload = payload.model_copy(update={
            "client_id": "lobby-help-client",
            "name": "ロビー利用者",
            "message": "設定はどこですか？",
        })
        lobby_result = await app_module.ask_lobby_chat_ai(lobby_payload, request)
        lobby_messages = lobby_result["chat_messages"]
        assert lobby_result["ok"] is True
        assert lobby_messages[-2]["sender"] == "ロビー利用者"
        assert lobby_messages[-2]["origin"] == "lobby"
        assert lobby_messages[-1]["sender"] == "案内AI"
        assert lobby_messages[-1]["origin"] == "lobby"
        assert lobby_messages[-1]["ai_answer"] is True
        assert lobby_messages[-1] in app_module._chat_messages_for_game(
            "main", app_module.GAMES["main"]
        )

        basic_lobby_payload = payload.model_copy(update={
            "client_id": "lobby-basic-help-client",
            "message": "どうすればいいかわからない",
        })
        basic_lobby_result = await app_module.ask_lobby_chat_ai(
            basic_lobby_payload, request
        )
        assert basic_lobby_result["answer"] == app_module._lobby_basic_usage_answer(
            basic_lobby_payload.message
        )

        try:
            await app_module.ask_chat_ai("main", payload, request)
        except HTTPException as exc:
            assert exc.status_code == 429
        else:
            raise AssertionError("cooldown was not enforced")

        app_module._request_gemini_help = lambda question, language="ja": (_ for _ in ()).throw(
            AssertionError("concrete move questions must not call Gemini")
        )
        move_payload = payload.model_copy(update={
            "client_id": "move-help-client",
            "message": "この場面では、どの駒を出せばいい？",
        })
        move_result = await app_module.ask_chat_ai("main", move_payload, request)
        assert "初心者サポートを有効にする" in move_result["answer"]
        assert "おすすめの駒が強調表示" in move_result["answer"]
        assert "strategy" not in move_result["answer"]

        zh_move_payload = payload.model_copy(update={
            "client_id": "zh-move-help-client",
            "message": "这时应该出哪张棋子？",
            "language": "zh",
        })
        zh_move_result = await app_module.ask_chat_ai("main", zh_move_payload, request)
        assert "新手辅助" in zh_move_result["answer"]
        assert "推荐棋子" in zh_move_result["answer"]

        en_move_payload = payload.model_copy(update={
            "client_id": "en-move-help-client",
            "message": "Which piece should I play here?",
            "language": "en",
        })
        en_move_result = await app_module.ask_chat_ai("main", en_move_payload, request)
        assert "Beginner Support" in en_move_result["answer"]
        assert "recommended piece" in en_move_result["answer"]

        os.environ.pop("GEMINI_API_KEY", None)
        no_key_payload = payload.model_copy(update={"client_id": "no-key-client"})
        try:
            await app_module.ask_chat_ai("main", no_key_payload, request)
        except HTTPException as exc:
            assert exc.status_code == 503
        else:
            raise AssertionError("missing API key was not rejected")
    finally:
        app_module._request_gemini_help = original_request
        app_module.PUBLIC_CHAT_MESSAGES[:] = original_public_messages
        if original_api_key is None:
            os.environ.pop("GEMINI_API_KEY", None)
        else:
            os.environ["GEMINI_API_KEY"] = original_api_key


if __name__ == "__main__":
    asyncio.run(_run())
    print("AI_CHAT_API_TEST_OK")

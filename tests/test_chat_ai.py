from __future__ import annotations

import asyncio
import os

from fastapi import HTTPException
from starlette.requests import Request

import backend.app as app_module


async def _run() -> None:
    original_request = app_module._request_gemini_help
    original_api_key = os.environ.get("GEMINI_API_KEY")
    try:
        app_module.GAMES.clear()
        app_module.AI_HELP_LAST_REQUEST.clear()
        assert "https://vrcgoita.com/goita/rule/" in app_module.AI_HELP_SYSTEM_PROMPT
        os.environ["GEMINI_API_KEY"] = "test-key"
        app_module._request_gemini_help = lambda question: f"案内回答: {question}"

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

        try:
            await app_module.ask_chat_ai("main", payload, request)
        except HTTPException as exc:
            assert exc.status_code == 429
        else:
            raise AssertionError("cooldown was not enforced")

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
        if original_api_key is None:
            os.environ.pop("GEMINI_API_KEY", None)
        else:
            os.environ["GEMINI_API_KEY"] = original_api_key


if __name__ == "__main__":
    asyncio.run(_run())
    print("AI_CHAT_API_TEST_OK")

import asyncio
from urllib.request import Request

import pytest
from fastapi import HTTPException

from backend import app as app_module


def test_memory_payload_prefers_container_usage_and_reports_percentage(monkeypatch) -> None:
    monkeypatch.delenv("MEMORY_LIMIT_BYTES", raising=False)
    monkeypatch.delenv("RENDER_API_KEY", raising=False)
    monkeypatch.setattr(app_module, "_process_rss_bytes", lambda: 100)
    monkeypatch.setattr(app_module, "_cgroup_memory_bytes", lambda: (200, 400))
    monkeypatch.setattr(app_module, "GAMES", {})
    monkeypatch.setattr(app_module.manager, "client_connections", {})
    monkeypatch.setattr(app_module.manager, "active_connections", {})
    monkeypatch.setattr(app_module.voice_manager, "connections", {})

    payload = app_module._admin_memory_payload()

    assert payload["used_bytes"] == 200
    assert payload["limit_bytes"] == 400
    assert payload["usage_percent"] == 50.0
    assert payload["process_rss_bytes"] == 100
    assert payload["measurement_source"] == "container"
    assert payload["restart_configured"] is False


def test_render_restart_uses_configured_service_without_exposing_key(monkeypatch) -> None:
    captured = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(request: Request, timeout: int):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setenv("RENDER_API_KEY", "secret-test-key")
    monkeypatch.setenv("RENDER_SERVICE_ID", "srv-test123")
    monkeypatch.setattr(app_module.urllib.request, "urlopen", fake_urlopen)

    app_module._request_render_restart()

    request = captured["request"]
    assert request.full_url == "https://api.render.com/v1/services/srv-test123/restart"
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Bearer secret-test-key"
    assert captured["timeout"] == 20


def test_render_restart_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("RENDER_API_KEY", raising=False)
    monkeypatch.setenv("RENDER_SERVICE_ID", "srv-test123")

    with pytest.raises(HTTPException) as error:
        app_module._request_render_restart()

    assert error.value.status_code == 503


def test_forced_unused_room_cleanup_does_not_wait_sixty_seconds(monkeypatch) -> None:
    async def scenario() -> None:
        game_id = app_module.PRIVATE_A_GID
        game = app_module._create_game_obj()
        game["empty_room_cleanup_armed"] = True
        games = {game_id: game}
        monkeypatch.setattr(app_module, "GAMES", games)
        monkeypatch.setattr(app_module.manager, "active_connections", {})
        monkeypatch.setattr(app_module.voice_manager, "connections", {})
        monkeypatch.setattr(app_module, "_cancel_turn_timeout_task", lambda *_: None)
        monkeypatch.setattr(app_module, "_cancel_debug_auto_next_round_task", lambda *_: None)

        async def no_broadcast(_game_id):
            return None

        monkeypatch.setattr(app_module.manager, "broadcast_update", no_broadcast)

        assert await app_module._reset_empty_room(
            game_id,
            force=True,
            collect_memory=False,
        )
        assert games[game_id] is not game
        assert games[game_id]["empty_room_cleanup_armed"] is False

    asyncio.run(scenario())

from __future__ import annotations

import asyncio
import copy
from pathlib import Path

from fastapi import HTTPException

import backend.app as app_module


ROOT = Path(__file__).resolve().parents[1]


class FakeWebSocket:
    def __init__(self) -> None:
        self.accepted = False
        self.sent: list[dict] = []
        self.closed: tuple[int, str] | None = None

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, payload: dict) -> None:
        self.sent.append(copy.deepcopy(payload))

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = (code, reason)


async def test_voice_manager_relays_only_signaling_and_tracks_roster() -> None:
    manager = app_module.VoiceConnectionManager()
    socket_a = FakeWebSocket()
    socket_b = FakeWebSocket()

    await manager.connect(socket_a, "debug", "A", "client-a")
    await manager.connect(socket_b, "debug", "B", "client-b")
    assert socket_a.accepted and socket_b.accepted
    assert manager.has_client_connection("debug", "client-a")
    assert socket_a.sent[-1]["type"] == "voice_roster"
    assert [item["seat"] for item in socket_a.sent[-1]["participants"]] == ["A", "B"]

    offer = {"type": "offer", "sdp": "test-sdp"}
    await manager.relay("debug", "A", "B", "offer", offer)
    assert socket_b.sent[-1] == {"type": "offer", "from": "A", "data": offer}

    await manager.update_state("debug", "A", muted=False, speaking=True)
    participant_a = next(
        item
        for item in socket_b.sent[-1]["participants"]
        if item["seat"] == "A"
    )
    assert participant_a == {"seat": "A", "muted": False, "speaking": True}

    await manager.disconnect(socket_b, "debug", "B")
    assert [item["seat"] for item in socket_a.sent[-1]["participants"]] == ["A"]


def test_voice_config_requires_owned_debug_seat() -> None:
    app_module.setup_debug_room()
    game = app_module.GAMES[app_module.DEBUG_GID]
    original_human_seats = copy.deepcopy(game.get("human_seats", {}))
    try:
        game["human_seats"] = {"A": "voice-owner"}
        config = app_module.get_voice_config(
            app_module.DEBUG_GID,
            seat="A",
            client_id="voice-owner",
        )
        assert config["enabled"] is True
        assert config["recording"] is False
        assert config["iceServers"]

        try:
            app_module.get_voice_config(
                app_module.DEBUG_GID,
                seat="A",
                client_id="other-client",
            )
        except HTTPException as exc:
            assert exc.status_code == 403
        else:
            raise AssertionError("voice config must reject a non-owner")
    finally:
        game["human_seats"] = original_human_seats


def test_frontend_has_debug_only_voice_controls() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    module = (ROOT / "frontend" / "voiceChat.js").read_text(encoding="utf-8")
    backend = (ROOT / "backend" / "app.py").read_text(encoding="utf-8")

    assert 'id="voiceChatBar"' in html
    assert 'id="voiceChatJoinButton"' in html
    assert 'id="voiceChatMuteButton"' in html
    assert "音声は録音・保存されません。" in html
    assert "const VOICE_CHAT_UI_ENABLED = false;" in html
    assert 'bar.style.display = VOICE_CHAT_UI_ENABLED && gid === DEBUG_GID ? "flex" : "none"' in html
    assert "await voiceChatController?.leave?.();" in html
    assert 'import("/static/voiceChat.js?v=20260730a")' in html
    assert "navigator.mediaDevices.getUserMedia" in module
    assert "new RTCPeerConnection" in module
    assert "echoCancellation: true" in module
    assert "track.enabled = false" in module
    assert "VOICE_RECONNECT_DELAY_MS" in module
    assert '@app.websocket("/voice/{game_id}")' in backend
    assert "VOICE_SIGNAL_TYPES" in backend
    assert "voice chat requires an owned debug-room seat" in backend


async def _run() -> None:
    await test_voice_manager_relays_only_signaling_and_tracks_roster()
    test_voice_config_requires_owned_debug_seat()
    test_frontend_has_debug_only_voice_controls()


if __name__ == "__main__":
    asyncio.run(_run())
    print("VOICE_CHAT_TEST_OK")

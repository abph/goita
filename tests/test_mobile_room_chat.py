from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mobile_room_chat_matches_lobby_toggle_and_clears_settings() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert 'id="chatToggle"' in html
    assert 'id="chatUnread" class="lobby-chat-unread"' in html
    assert html.count('<span aria-hidden="true">💬</span>') >= 2
    assert "btn.style.display = inRoom && !docked ? \"flex\" : \"none\"" in html
    assert "badge.hidden = chatVisible || unread === 0" in html
    assert "#chatPanel,\n      #lobbyChatPanel {\n        top: 54px;" in html
    assert "width: 42px" in html
    assert "height: 42px" in html


if __name__ == "__main__":
    test_mobile_room_chat_matches_lobby_toggle_and_clears_settings()
    print("MOBILE_ROOM_CHAT_TEST_OK")

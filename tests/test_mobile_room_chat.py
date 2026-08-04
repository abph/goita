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


def test_mobile_chat_scroll_stays_inside_the_open_panel() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert "function updateMobileChatPageScrollLock()" in html
    assert 'document.body.classList.add("mobile-chat-scroll-locked")' in html
    assert 'document.documentElement.classList.add("mobile-chat-scroll-locked")' in html
    assert 'document.body.style.top = `-${mobileChatLockedScrollY}px`' in html
    assert "overscroll-behavior-y: contain;" in html
    assert "-webkit-overflow-scrolling: touch;" in html


def test_chat_people_rows_use_compact_spacing() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert ".room-chat-people .lobby-person-row {\n      display: block;\n      padding: 4px 1px;" in html
    assert "#lobbyChatPanel.chat-people-visible .lobby-person-row {\n        padding: 3px 0;" in html


if __name__ == "__main__":
    test_mobile_room_chat_matches_lobby_toggle_and_clears_settings()
    print("MOBILE_ROOM_CHAT_TEST_OK")

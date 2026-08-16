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
    panel_css = html.split("    .chat-panel {", 1)[1].split("    }", 1)[0]
    assert "border: 3px solid #8b5a2b;" in panel_css
    assert "border-right: none;" not in panel_css
    assert "border-radius: 10px;" in panel_css


def test_mobile_chat_scroll_stays_inside_the_open_panel() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert "function updateMobileChatPageScrollLock()" in html
    assert 'document.body.classList.add("mobile-chat-scroll-locked")' in html
    assert 'document.documentElement.classList.add("mobile-chat-scroll-locked")' in html
    assert 'document.body.style.top = `-${mobileChatLockedScrollY}px`' in html
    assert "overscroll-behavior-y: contain;" in html
    assert "-webkit-overflow-scrolling: touch;" in html
    lock_css = html.split("      body.mobile-chat-scroll-locked {", 2)[2].split("      }", 1)[0]
    assert "right: 0;" in lock_css
    assert "left: 0;" in lock_css
    assert "width: auto;" in lock_css
    assert "width: 100%;" not in lock_css


def test_mobile_chat_close_is_easy_to_tap_after_typing() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert 'onpointerdown="closeChatPanelOnPointerDown(event, \'room\')"' in html
    assert 'onpointerdown="closeChatPanelOnPointerDown(event, \'lobby\')"' in html
    assert "function closeChatPanelOnPointerDown(event, kind)" in html
    assert "activeElement.blur();" in html
    assert "touch-action: manipulation;" in html
    mobile_close_css = html.split("      .chat-close {", 1)[1].split("      }", 1)[0]
    assert "width: 44px;" in mobile_close_css
    assert "height: 44px;" in mobile_close_css


def test_chat_people_rows_use_compact_spacing() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert ".room-chat-people .lobby-person-row {\n      display: block;\n      padding: 1px;" in html
    assert "#lobbyChatPanel.chat-people-visible .lobby-person-row {\n        padding: 1px 0;" in html


if __name__ == "__main__":
    test_mobile_room_chat_matches_lobby_toggle_and_clears_settings()
    test_mobile_chat_scroll_stays_inside_the_open_panel()
    test_chat_people_rows_use_compact_spacing()
    test_mobile_chat_close_is_easy_to_tap_after_typing()
    print("MOBILE_ROOM_CHAT_TEST_OK")

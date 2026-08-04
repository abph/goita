from pathlib import Path


HTML = (Path(__file__).parents[1] / "frontend" / "index.html").read_text(encoding="utf-8")


def test_board_and_hands_share_the_desktop_main_column():
    column_start = HTML.index('<div class="play-main-column">')
    board_start = HTML.index('<div class="board-wrap">', column_start)
    hands_start = HTML.index('<div id="handsArea" class="hands-area"></div>', board_start)
    chat_start = HTML.index('<div id="chatPanel" class="chat-panel"', hands_start)

    assert column_start < board_start < hands_start < chat_start


def test_desktop_chat_stretches_to_board_and_hands_height():
    assert ".play-layout .board-wrap,\n      .play-layout .hands-area" in HTML
    assert "height: var(--desktop-room-chat-height, auto);" in HTML
    assert "max-height: var(--desktop-room-chat-height, none);" in HTML
    assert "align-self: flex-start;" in HTML
    assert "function syncDesktopRoomChatHeight()" in HTML
    assert 'document.querySelector(".play-main-column")' in HTML
    assert 'panel.style.setProperty("--desktop-room-chat-height", `${height}px`);' in HTML
    assert "desktopRoomChatHeightObserver = new ResizeObserver(() => syncDesktopRoomChatHeight())" in HTML
    assert "height: calc((var(--cell) * 8) + (var(--gap) * 7) + 32px);" not in HTML


def test_chat_messages_scroll_inside_the_fixed_desktop_panel():
    desktop_chat_start = HTML.index(".play-layout .chat-messages {")
    desktop_chat_end = HTML.index("}", desktop_chat_start)
    desktop_chat_css = HTML[desktop_chat_start:desktop_chat_end]

    assert "min-height: 0;" in desktop_chat_css
    assert "overflow-y: auto;" in desktop_chat_css


def test_chat_does_not_render_timestamps():
    assert 'id="chatToastTime"' not in HTML
    assert 'className = "chat-time"' not in HTML
    assert "function formatChatTime(" not in HTML

from pathlib import Path


HTML = (Path(__file__).parents[1] / "frontend" / "index.html").read_text(encoding="utf-8")


def test_board_and_hands_share_the_desktop_main_column():
    column_start = HTML.index('<div class="play-main-column">')
    board_start = HTML.index('<div class="board-wrap">', column_start)
    hands_start = HTML.index('<div id="handsArea" class="hands-area"></div>', board_start)
    chat_start = HTML.index('<div id="chatPanel" class="chat-panel"', hands_start)

    assert column_start < board_start < hands_start < chat_start


def test_desktop_chat_stretches_to_board_and_hands_height():
    desktop_layout_start = HTML.index(".play-layout {")
    desktop_layout_end = HTML.index("}", desktop_layout_start)
    desktop_layout_css = HTML[desktop_layout_start:desktop_layout_end]
    main_column_start = HTML.index(".play-main-column {", desktop_layout_end)
    main_column_end = HTML.index("}", main_column_start)
    main_column_css = HTML[main_column_start:main_column_end]

    assert "align-items: flex-start;" in desktop_layout_css
    assert "align-items: stretch;" not in desktop_layout_css
    assert "align-self: flex-start;" in main_column_css
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


def test_desktop_log_matches_the_full_control_column_height():
    assert "--desktop-log-panel-width: 280px;" in HTML
    assert "--chat-panel-width: 220px;" in HTML
    assert "var(--chat-panel-width) + var(--desktop-log-panel-width)" in HTML
    assert "grid-template-columns: minmax(0, 1fr) var(--desktop-log-panel-width);" in HTML
    assert "width: var(--desktop-log-panel-width);" in HTML
    assert "height: var(--desktop-room-log-height, auto);" in HTML
    assert "max-height: var(--desktop-room-log-height, none);" in HTML
    assert "function syncDesktopRoomLogHeight()" in HTML
    assert 'document.querySelector(".game-content-main")' in HTML
    assert 'card.style.setProperty("--desktop-room-log-height", `${height}px`);' in HTML
    assert "desktopRoomLogHeightObserver = new ResizeObserver(() => syncDesktopRoomLogHeight())" in HTML
    assert "setupDesktopRoomLogHeightSync();" in HTML

    log_layout_start = HTML.index(".game-content-layout.log-visible {")
    log_layout_end = HTML.index("}", log_layout_start)
    log_layout_css = HTML[log_layout_start:log_layout_end]
    log_card_start = HTML.index(".game-content-layout.log-visible #logCard {")
    log_card_end = HTML.index("}", log_card_start)
    log_card_css = HTML[log_card_start:log_card_end]

    assert "align-items: start;" in log_layout_css
    assert "overflow: hidden;" in log_card_css
    assert "position: absolute;" in HTML
    assert "inset: 68px 0 0;" in HTML
    assert "overflow-y: scroll;" in HTML
    assert "scrollbar-gutter: stable;" in HTML
    assert "scrollbar-color: #9b744d #eee4d2;" in HTML


def test_chat_does_not_render_timestamps():
    assert 'id="chatToastTime"' not in HTML
    assert 'className = "chat-time"' not in HTML
    assert "function formatChatTime(" not in HTML

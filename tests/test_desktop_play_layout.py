from pathlib import Path


HTML = (Path(__file__).parents[1] / "frontend" / "index.html").read_text(encoding="utf-8")


def test_board_and_hands_share_the_desktop_main_column():
    column_start = HTML.index('<div class="play-main-column">')
    board_start = HTML.index('<div class="board-wrap">', column_start)
    hands_start = HTML.index('<div id="handsArea" class="hands-area"></div>', board_start)
    chat_start = HTML.index('<div id="chatPanel" class="chat-panel"', hands_start)

    assert column_start < board_start < hands_start < chat_start


def test_desktop_chat_stretches_to_board_and_hands_height():
    assert "align-items: stretch;" in HTML
    assert ".play-layout .board-wrap,\n      .play-layout .hands-area" in HTML
    assert "height: auto;\n        align-self: stretch;" in HTML
    assert "height: calc((var(--cell) * 8) + (var(--gap) * 7) + 32px);" not in HTML

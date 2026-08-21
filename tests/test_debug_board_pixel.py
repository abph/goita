from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pixel_board_is_available_in_private_and_debug_rooms() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert 'id="boardPixel"' in html
    assert 'id="boardPixelCanvas"' in html
    assert 'import("/static/boardPixel.js?v=20260821b")' in html
    assert "setBoardViewControl('pixel-color')" in html
    assert "setBoardViewControl('pixel-mono')" in html
    assert 'if (mode === "pixel") return "pixel-color"' in html
    assert "const requestedPixel = boardViewsEnabled && PIXEL_BOARD_MODES.has(boardViewMode);" in html
    assert "if(!supportsAlternateBoardViews(gid) || !PIXEL_BOARD_MODES.has(boardViewMode)) return;" in html
    assert 'boardViewMode: "2d"' in html


def test_pixel_board_uses_low_resolution_canvas_and_public_snapshot() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    board_module = (ROOT / "frontend" / "boardPixel.js").read_text(encoding="utf-8")

    assert 'width="320" height="320"' in html
    assert "image-rendering: pixelated" in html
    assert "const SIZE = 320" in board_module
    assert "context.imageSmoothingEnabled = false" in board_module
    assert "requestAnimationFrame(drawScene)" not in board_module
    assert "enteringPieces" not in board_module
    assert "if (piece.current || piece.pending)" not in board_module
    assert 'context.fillStyle = "rgba(35, 23, 16, 0.38)"' not in board_module
    assert "palette.teal" not in board_module
    assert "buildBoard3DSnapshot(state)" in html
    assert "piece.ownHidden) context.globalAlpha = 0.32" in board_module
    assert "window.goitaBoardPixel.render" in html
    assert "window.goitaBoardPixel.setTheme" in html
    assert "const palettes" in board_module
    assert 'setTheme(theme)' in board_module
    assert 'theme === "mono" ? palettes.mono : palettes.color' in board_module
    assert 'board: "#ffffff"' in board_module
    assert 'humanPiece: "#ffffff"' in board_module
    assert 'hiddenPiece: "#ffffff"' in board_module
    assert "body.board-view-pixel .chat-panel" in html
    assert "body.board-view-pixel .chat-form input" in html
    assert "body.board-view-pixel .chat-toast" in html
    assert "body.board-view-pixel .hand-row" in html
    assert "body.board-view-pixel .hand-row-pieces .hand::before" in html
    assert "body.board-view-pixel .pass" in html
    assert "body.board-view-pixel .auto-play" in html
    assert "body.board-view-pixel:not(.board-view-pixel-mono) .special-anim-text" in html
    assert "@keyframes pixelColorSpecial" in html
    assert "animation: pixelColorSpecial 1.5s steps(1, end) forwards" in html
    assert "body.board-view-pixel .board-wrap" in html
    assert "background: transparent" in html
    assert "body.board-view-pixel #boardPixel" in html
    assert "width: calc((var(--cell) * 8) + (var(--gap) * 7) + 32px)" in html
    assert "body.board-view-pixel-mono .chat-panel" in html
    assert "body.board-view-pixel-mono .hand-row" in html
    assert "body.board-view-pixel-mono .special-anim-text" in html
    assert "@keyframes pixelMonoSpecial" in html
    assert "animation: pixelMonoSpecial 1.5s steps(1, end) forwards" in html
    assert "background: rgba(255, 255, 255, var(--mobile-chat-opacity, 1))" in html
    assert "border-radius: 0" in html
    assert 'document.querySelector(".board-wrap"), overlay: true' in html
    assert "wrapper.dataset.passSeat = String(phys);" in html
    assert "body.board-view-pixel .pass-anim-overlay .pass-anim-text" in html
    assert "body.board-view-pixel-mono .pass-anim-overlay .pass-anim-text" in html


if __name__ == "__main__":
    test_pixel_board_is_available_in_private_and_debug_rooms()
    test_pixel_board_uses_low_resolution_canvas_and_public_snapshot()
    print("DEBUG_BOARD_PIXEL_TEST_OK")

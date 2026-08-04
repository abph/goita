from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_own_hidden_piece_remains_as_faint_text_in_its_hand_slot() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    backend = (ROOT / "backend" / "app.py").read_text(encoding="utf-8")

    assert '"face_down_pieces": face_down_pieces_view' in backend
    assert "face_down_pieces_view[p] = list(state.face_down_hidden[p])" in backend
    assert "function reconcileFixedHandMemory(mem, current, faceDownPieces = [])" in html
    assert 'pieceDiv.classList.add("used-hidden")' in html
    assert ".hand-row-pieces .hand.face-down.used-hidden .val" in html
    assert "opacity: 0.32;" in html
    assert "if(isPendingHidden || isUsedHidden) delete valDiv.dataset.pieceEn;" in html


def test_own_hidden_piece_is_faintly_labeled_on_every_board_view() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    board_3d = (ROOT / "frontend" / "board3d.js").read_text(encoding="utf-8")
    board_pixel = (ROOT / "frontend" / "boardPixel.js").read_text(encoding="utf-8")

    assert "Array.isArray(state.face_down_pieces?.[slot.p])" in html
    assert 'div.classList.add("own-hidden")' in html
    assert ".slot.own-hidden .val" in html
    assert "ownHidden: isOwnHidden" in html
    assert "(!piece.hidden || piece.ownHidden)" in board_3d
    assert "piece.ownHidden ? 0.32" in board_3d
    assert "(!piece.hidden || piece.ownHidden)" in board_pixel
    assert "if (piece.ownHidden) context.globalAlpha = 0.32" in board_pixel


if __name__ == "__main__":
    test_own_hidden_piece_remains_as_faint_text_in_its_hand_slot()
    test_own_hidden_piece_is_faintly_labeled_on_every_board_view()
    print("OWN_HIDDEN_HAND_UI_TEST_OK")

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


if __name__ == "__main__":
    test_own_hidden_piece_remains_as_faint_text_in_its_hand_slot()
    print("OWN_HIDDEN_HAND_UI_TEST_OK")

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_official_english_piece_names_are_defined() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    expected = {
        "1": "Pawn",
        "2": "Lance",
        "3": "Knight",
        "4": "Silver",
        "5": "Gold",
        "6": "Bishop",
        "7": "Rook",
        "8": "King",
        "9": "King",
    }
    for piece, english_name in expected.items():
        assert f'"{piece}": "{english_name}"' in html


def test_english_piece_names_are_rendered_across_board_modes() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    board_3d = (ROOT / "frontend" / "board3d.js").read_text(encoding="utf-8")
    board_pixel = (ROOT / "frontend" / "boardPixel.js").read_text(encoding="utf-8")

    assert 'html[lang="en"] .lobby-piece-background span::after' in html
    assert "piece.dataset.pieceEn = pieceEnglishName(label);" in html
    assert "function setPieceValue(element, piece" in html
    assert "setPieceValue(val, piece" in html
    assert "setPieceValue(" in html.split("function renderHands", 1)[1]
    assert 'englishLabel: currentUiLanguage() === "en"' in html
    assert "piece.englishLabel || \"\"" in board_3d
    assert "context.fillText(englishLabel, 128, 249)" in board_3d
    assert "if (piece.englishLabel)" in board_pixel
    assert "drawPixelText(piece.englishLabel" in board_pixel


if __name__ == "__main__":
    test_official_english_piece_names_are_defined()
    test_english_piece_names_are_rendered_across_board_modes()
    print("ENGLISH_PIECE_LABELS_TEST_OK")

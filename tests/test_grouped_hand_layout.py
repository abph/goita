from pathlib import Path


HTML = (Path(__file__).parents[1] / "frontend" / "index.html").read_text(encoding="utf-8")


def test_hand_layout_keeps_four_pieces_on_each_row():
    assert "function arrangeHandInGroupedRows(hand)" in HTML
    assert "if(topCount !== 4) continue;" in HTML
    assert "return [...top, ...bottom];" in HTML


def test_hand_layout_prefers_complete_duplicate_groups():
    assert "candidate.groupedPiecesOnTop > best.groupedPiecesOnTop" in HTML
    assert "return arrangeHandInGroupedRows(init);" in HTML
    assert "return arrangeHandInGroupedRows(raw);" in HTML


def test_unavoidable_split_uses_a_connected_snake_fallback():
    assert "sorted[7], sorted[6], sorted[5], sorted[4]" in HTML

from pathlib import Path


HTML = (Path(__file__).parents[1] / "frontend" / "index.html").read_text(encoding="utf-8")


def test_lobby_intro_uses_scoped_typography_classes():
    assert '<div class="lobby-intro">' in HTML
    assert '<p class="lobby-intro-lead">' in HTML
    assert '<div class="lobby-intro-notes">' in HTML


def test_lobby_intro_has_smaller_mobile_typography():
    assert ".lobby-intro h1 { font-size: 24px; }" in HTML
    assert ".lobby-intro-lead { font-size: 13px; }" in HTML
    assert "font-size: 11px;" in HTML


def test_lobby_room_section_headings_and_descriptions_are_compact():
    assert ".lobby-room-section h2" in HTML
    assert "font-size: 20px;" in HTML
    assert ".lobby-room-section > p" in HTML
    assert "font-size: 13px;" in HTML

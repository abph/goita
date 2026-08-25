from pathlib import Path


HTML = (Path(__file__).parents[1] / "frontend" / "index.html").read_text(encoding="utf-8")


def test_mobile_lobby_room_list_uses_two_columns():
    assert "grid-template-columns: repeat(2, 154px);" in HTML
    assert "margin-inline: -17px;" in HTML
    assert "justify-content: center;" in HTML
    assert HTML.count('class="card lobby-room-section"') == 2


def test_mobile_room_cards_shrink_without_losing_square_shape():
    assert "aspect-ratio: 1;" in HTML
    assert ".room-card .seat-badge" in HTML
    assert "max-width: 39px;" in HTML
    assert "font-size: 7.5px;" in HTML


def test_private_room_password_note_uses_compact_mobile_text():
    assert 'class="count password-note">※合言葉が必要です</div>' in HTML
    assert ".room-card .center-info .password-note" in HTML
    assert "font-size: 7px;" in HTML

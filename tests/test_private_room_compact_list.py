from pathlib import Path


ROOT = Path(__file__).parents[1]
HTML = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")


def test_all_private_rooms_use_the_regular_card_layout():
    assert 'id="lobbyRoomList" class="room-card-container"' in HTML
    assert "privateContainer.appendChild(card);" in HTML
    assert "FEATURED_PRIVATE_ROOM_IDS" not in HTML
    assert "private-room-compact" not in HTML
    assert "lobbyPrivateRoomCompact" not in HTML


def test_private_a_and_b_open_notice_remains_visible():
    assert "支援していただいた方のための専用ルームです。" in HTML
    assert "＊<b>プライベートA・Bは、どなたでも自由に使えます。</b>" in HTML
    assert "プライベートA・Bは、どなたでも自由に使えます。" in HTML

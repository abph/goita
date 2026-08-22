from pathlib import Path


ROOT = Path(__file__).parents[1]
HTML = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
ZH = (ROOT / "frontend" / "i18n.js").read_text(encoding="utf-8")
EN = (ROOT / "frontend" / "i18n-en.js").read_text(encoding="utf-8")


def test_private_a_and_b_remain_featured_cards():
    assert 'const FEATURED_PRIVATE_ROOM_IDS = new Set([PRIVATE_A_GID, "room-silver-02"]);' in HTML
    assert "!FEATURED_PRIVATE_ROOM_IDS.has(room.game_id)" in HTML


def test_private_c_through_f_use_the_compact_list():
    assert 'id="lobbyPrivateRoomCompactSection"' in HTML
    assert 'id="lobbyPrivateRoomCompactList"' in HTML
    assert "width: min(100%, 560px);" in HTML
    assert 'row.className = "private-room-compact-row";' in HTML
    assert "compactPrivateSection.hidden = false;" in HTML


def test_compact_private_room_action_is_always_enter():
    assert 'enterButton.textContent = uiText("入室");' in HTML
    assert "enterButton.onclick = event =>" in HTML
    assert "tryJoinRoom(room.game_id, requiresPassword);" in HTML


def test_compact_private_room_heading_is_localized():
    assert '"その他のプライベートルーム": "其他私人房间"' in ZH
    assert '"その他のプライベートルーム": "Other private rooms"' in EN
    assert 'i18n-en.js?v=20260822a' in HTML
    assert 'i18n.js?v=20260822a' in HTML

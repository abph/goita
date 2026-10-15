from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_lobby_uses_pixel_board_as_public_room_card_background() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert 'id="publicRoomPixelSection"' not in html
    assert 'card.dataset.publicRoomId = room.game_id' in html
    assert 'canvas.className = "public-room-pixel-background"' in html
    assert 'card.classList.add("has-pixel-board-background")' in html
    assert ".room-card.has-pixel-board-background .seat-badge" in html


def test_lobby_pixel_boards_reuse_safe_public_table_snapshots() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert "function buildMeetingPublicTableSnapshot(table)" in html
    assert "async function refreshLobbyPublicPixelBoards(force = false)" in html
    assert "`${API}/games/public-tables?${query.toString()}`" in html
    assert "(data.tables || []).map(buildMeetingPublicTableSnapshot)" in html
    assert "{...snapshot, backgroundOnly: true}" in html
    assert "refreshLobbyPublicPixelBoards(true);" in html
    assert "lobbyPublicPixelEnabled = data.public_board_previews_enabled === true" in html
    assert "&& lobbyPublicPixelEnabled" in html


def test_gathering_mode_always_builds_four_table_cards_with_boards() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert "function gatheringLobbyRooms(event)" in html
    assert "function buildGatheringLobbyTableSnapshots(event)" in html
    assert html.count("return Array.from(GATHERING_TABLE_IDS).map") >= 2
    assert "...gatheringLobbyRooms(latestGatheringEvent)" in html
    assert "is_gathering_table: true" in html
    assert "(gatheringActive && gatheringCard)" in html
    assert 'card.classList.add("is-display-only")' in html
    assert 'room.gathering_table_status === "playing"' in html
    assert "openGatheringTable(room.game_id, room.owner_name)" in html


def test_admin_can_toggle_public_room_pixel_boards() -> None:
    admin = (ROOT / "frontend" / "admin.html").read_text(encoding="utf-8")

    assert 'id="publicBoardPreviewsEnabled"' in admin
    assert "公開部屋カードに盤面を表示する" in admin
    assert "data.public_board_previews_enabled === true" in admin
    assert "public_board_previews_enabled: document.getElementById(" in admin


if __name__ == "__main__":
    test_lobby_uses_pixel_board_as_public_room_card_background()
    test_lobby_pixel_boards_reuse_safe_public_table_snapshots()
    test_gathering_mode_always_builds_four_table_cards_with_boards()
    test_admin_can_toggle_public_room_pixel_boards()
    print("LOBBY_PUBLIC_PIXEL_BOARDS_TEST_OK")

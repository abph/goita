from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_room_header_places_back_arrow_before_room_name() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    header_start = html.index('<div class="game-header">')
    header_end = html.index('<div class="top-controls">', header_start)
    header = html[header_start:header_end]

    assert 'class="game-header-title"' in header
    assert 'class="lobby-return-btn"' in header
    assert 'aria-label="ロビーへ"' in header
    assert 'title="ロビーへ"' in header
    assert '<span aria-hidden="true">←</span>' in header
    assert header.index('class="lobby-return-btn"') < header.index('id="currentRoomName"')
    assert header.index('id="currentRoomName"') < header.index('id="roomSettingsBtn"')
    assert "🏠 ロビーへ" not in header


def test_room_header_back_arrow_has_stable_responsive_size() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert ".game-header-title" in html
    assert "flex: 0 0 40px" in html
    assert "flex-basis: 34px" in html


if __name__ == "__main__":
    test_room_header_places_back_arrow_before_room_name()
    test_room_header_back_arrow_has_stable_responsive_size()
    print("GAME_HEADER_NAVIGATION_TEST_OK")

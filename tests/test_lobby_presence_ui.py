from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_lobby_presence_is_shown_only_inside_chat() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert 'id="lobbyPeopleToggle"' not in html
    assert '<span aria-hidden="true">👥</span>' not in html
    assert 'id="lobbyPeopleCount"' not in html
    assert 'id="lobbyPeoplePanel"' not in html
    assert 'id="lobbyPeopleList"' not in html
    assert "function toggleLobbyPeoplePanel(open)" not in html
    assert 'id="lobbyChatPeople"' in html
    assert 'id="lobbyChatPeopleList"' in html
    assert "function renderLobbyPeople(sitePeople)" in html
    assert "renderLobbyPeople(data.site_people || []);" in html
    assert "new URLSearchParams({viewer_game_id: gid, client_id: clientId})" in html
    assert "new URLSearchParams({client_id: clientId})" in html
    assert 'wsParams.set("name", personalSettings.playerName)' in html


def test_lobby_chat_keeps_its_mobile_position() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    chat_start = html.index("    .lobby-chat-toggle {")
    chat_end = html.index("    .lobby-chat-toggle.open", chat_start)

    assert "top: 142px" in html[chat_start:chat_end]


if __name__ == "__main__":
    test_lobby_presence_is_shown_only_inside_chat()
    test_lobby_chat_keeps_its_mobile_position()
    print("LOBBY_PRESENCE_UI_TEST_OK")

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_lobby_has_site_presence_button_and_panel() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert 'id="lobbyPeopleToggle"' in html
    assert '<span aria-hidden="true">👥</span>' in html
    assert 'id="lobbyPeopleCount"' in html
    assert 'id="lobbyPeoplePanel"' in html
    assert 'id="lobbyPeopleList"' in html
    assert "function toggleLobbyPeoplePanel(open)" in html
    assert "function renderLobbyPeople(sitePeople)" in html
    assert "renderLobbyPeople(data.site_people || []);" in html
    assert "new URLSearchParams({client_id: clientId})" in html
    assert 'wsParams.set("name", personalSettings.playerName)' in html


def test_lobby_site_presence_is_positioned_below_chat() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    chat_start = html.index("    .lobby-chat-toggle {")
    chat_end = html.index("    .lobby-chat-toggle.open", chat_start)
    people_start = html.index("    .lobby-people-toggle {")
    people_end = html.index("    .lobby-people-toggle.open", people_start)

    assert "top: 142px" in html[chat_start:chat_end]
    assert "top: 198px" in html[people_start:people_end]


if __name__ == "__main__":
    test_lobby_has_site_presence_button_and_panel()
    test_lobby_site_presence_is_positioned_below_chat()
    print("LOBBY_PRESENCE_UI_TEST_OK")

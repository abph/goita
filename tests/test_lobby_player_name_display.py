from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "frontend" / "index.html"


def test_lobby_room_cards_truncate_only_long_player_names() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")

    assert "function truncateLobbyPlayerName(playerName, maxCharacters = 5)" in html
    assert 'characters.length <= maxCharacters' in html
    assert 'characters.slice(0, maxCharacters).join("")}…' in html
    assert "(!isAi && !isEmpty)" in html
    assert "? truncateLobbyPlayerName(fullDisplayName)" in html


def test_lobby_room_cards_keep_full_name_for_hover_and_accessibility() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")

    assert 'title="${escapeLobbyHtml(fullText)}"' in html
    assert 'aria-label="${escapeLobbyHtml(fullText)}"' in html
    assert ">${escapeLobbyHtml(text)}</div>`" in html


if __name__ == "__main__":
    test_lobby_room_cards_truncate_only_long_player_names()
    test_lobby_room_cards_keep_full_name_for_hover_and_accessibility()
    print("LOBBY_PLAYER_NAME_DISPLAY_TEST_OK")

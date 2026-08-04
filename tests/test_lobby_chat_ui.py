from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_lobby_chat_uses_shared_public_channel_and_bubble_button() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert 'id="lobbyChatToggle"' in html
    assert '<span aria-hidden="true">💬</span>' in html
    assert 'id="lobbyChatPanel"' in html
    assert 'id="lobbyChatMessages"' in html
    assert 'id="lobbyChatInput"' in html
    assert 'id="lobbyChatUnread"' in html
    assert "function toggleLobbyChatPanel(open)" in html
    assert "function renderLobbyChat(serverMessages)" in html
    assert "async function sendLobbyChatMessage(event)" in html
    assert "`${API}/lobby/chat`" in html
    assert "data.public_chat_messages" in html
    assert html.index('id="lobbyChatToggle"') < html.index('id="gameView"')
    assert ".lobby-chat-panel[hidden]" in html
    assert ".lobby-chat-unread[hidden]" in html


def test_lobby_chat_reuses_mobile_chat_appearance_settings() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert 'class="chat-panel lobby-chat-panel"' in html
    assert ".lobby-chat-toggle" in html
    assert "--mobile-chat-width" in html
    assert "--mobile-chat-opacity" in html
    assert "body.mobile-chat-placement-top .chat-panel" in html


if __name__ == "__main__":
    test_lobby_chat_uses_shared_public_channel_and_bubble_button()
    test_lobby_chat_reuses_mobile_chat_appearance_settings()
    print("LOBBY_CHAT_UI_TEST_OK")

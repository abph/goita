from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_lobby_chat_uses_shared_public_channel_and_bubble_button() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert 'id="lobbyChatToggle"' in html
    assert '<span aria-hidden="true">💬</span>' in html
    assert 'id="lobbyChatPanel"' in html
    assert 'id="lobbyChatMessages"' in html
    assert 'id="lobbyChatInput"' in html
    assert 'id="lobbyChatModeChatButton"' in html
    assert 'id="lobbyChatModeAiButton"' in html
    assert 'id="lobbyChatUnread"' in html
    assert "function toggleLobbyChatPanel(open)" in html
    assert "function renderLobbyChat(serverMessages)" in html
    assert "function setLobbyChatComposeMode(mode, focusInput = true)" in html
    assert "function submitLobbyChatInput(event)" in html
    assert "async function sendLobbyChatMessage(event)" in html
    assert "async function askLobbyChatAi()" in html
    assert "function showLobbyWelcomeToast()" in html
    assert "公開部屋またはプライベートルームを選んで入室してください。" in html
    assert "let lobbyChatNotices = [];" in html
    assert "lobbyChatNotices = [notice];" in html
    assert "renderLobbyChat(lobbyChatMessages);" in html
    assert "lobbyChatMessages = [...sharedMessages, ...lobbyChatNotices]" in html
    assert "await fetchRoomList();\n  showLobbyWelcomeToast();" in html
    assert "`${API}/lobby/chat`" in html
    assert "`${API}/lobby/chat/ask_ai`" in html
    assert "data.public_chat_messages" in html
    assert html.index('id="lobbyChatToggle"') < html.index('id="gameView"')
    assert ".lobby-chat-panel[hidden]" in html
    assert ".lobby-chat-unread[hidden]" in html


def test_lobby_chat_reuses_mobile_chat_appearance_settings() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert 'class="chat-panel lobby-chat-panel"' in html
    assert ".lobby-chat-toggle" in html
    assert "right: 14px;\n      top: 142px;\n      bottom: auto;" in html
    assert "--mobile-chat-width" in html
    assert "--mobile-chat-opacity" in html
    assert "body.mobile-chat-placement-top .chat-panel" in html
    assert "top: 142px;\n        bottom: auto;" in html


if __name__ == "__main__":
    test_lobby_chat_uses_shared_public_channel_and_bubble_button()
    test_lobby_chat_reuses_mobile_chat_appearance_settings()
    print("LOBBY_CHAT_UI_TEST_OK")

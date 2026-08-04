from pathlib import Path


ROOT = Path(__file__).parents[1]
HTML = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
ZH = (ROOT / "frontend" / "i18n.js").read_text(encoding="utf-8")
EN = (ROOT / "frontend" / "i18n-en.js").read_text(encoding="utf-8")


def test_personal_settings_include_chat_people_toggle_in_both_modals():
    assert 'id="checkShowPlayersInChat"' in HTML
    assert 'id="lobbyCheckShowPlayersInChat"' in HTML
    assert HTML.count("チャットの中にプレイヤー表示を追加") == 2


def test_chat_people_setting_is_persisted_and_defaults_on():
    assert "showPlayersInChat: saved.showPlayersInChat !== false" in HTML
    assert "showPlayersInChat: true" in HTML
    assert 'showPlayersInChat: document.getElementById("checkShowPlayersInChat").checked' in HTML
    assert 'showPlayersInChat: document.getElementById("lobbyCheckShowPlayersInChat").checked' in HTML


def test_room_and_lobby_chat_have_people_sidebars_and_refresh_them():
    assert 'id="roomChatPeople" class="room-chat-people"' in HTML
    assert 'id="roomChatPeopleList" class="room-chat-people-list"' in HTML
    assert 'id="lobbyChatPeople" class="room-chat-people"' in HTML
    assert 'id="lobbyChatPeopleList" class="room-chat-people-list"' in HTML
    assert 'class="room-chat-people-title"' not in HTML
    assert "function refreshRoomChatPeople()" in HTML
    assert "window.setInterval(refreshRoomChatPeople, 10000)" in HTML
    assert ".chat-panel.chat-people-visible .room-chat-people" in HTML
    assert 'document.getElementById("lobbyChatPanel")?.classList.toggle("chat-people-visible", enabled)' in HTML
    assert '["roomChatPeopleList", "lobbyChatPeopleList"]' in HTML


def test_mobile_chat_uses_a_compact_people_column():
    assert "--mobile-chat-people-width: 92px" in HTML
    assert "#chatPanel.chat-people-visible .room-chat-people" in HTML
    assert "#lobbyChatPanel.chat-people-visible .room-chat-people" in HTML
    assert "body.mobile-chat-placement-top #chatPanel.chat-people-visible" in HTML


def test_chat_height_setting_is_available_and_persisted():
    assert 'id="mobileChatHeight"' in HTML
    assert 'id="lobbyMobileChatHeight"' in HTML
    assert "mobileChatHeight: [\"tall\", \"normal\", \"short\"].includes(saved.mobileChatHeight)" in HTML
    assert "--mobile-chat-height" in HTML
    assert "--mobile-chat-top-height" in HTML
    assert '["tall", "normal", "short"].includes(document.getElementById("mobileChatHeight").value)' in HTML
    assert '["tall", "normal", "short"].includes(document.getElementById("lobbyMobileChatHeight").value)' in HTML


def test_new_labels_are_translated():
    for source in (ZH, EN):
        assert '"▶ チャットの設定を開く"' in source
        assert '"▼ チャットの設定を閉じる"' in source
        assert '"チャットの中にプレイヤー表示を追加"' in source
        assert '"高さ"' in source
        assert '"高い"' in source
        assert '"低い"' in source

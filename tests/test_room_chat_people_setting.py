from pathlib import Path


ROOT = Path(__file__).parents[1]
HTML = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
ZH = (ROOT / "frontend" / "i18n.js").read_text(encoding="utf-8")
EN = (ROOT / "frontend" / "i18n-en.js").read_text(encoding="utf-8")


def test_personal_settings_include_chat_people_toggle_in_both_modals():
    assert 'id="checkShowPlayersInChat"' in HTML
    assert 'id="lobbyCheckShowPlayersInChat"' in HTML
    assert HTML.count("チャットの中にプレイヤー表示を追加") == 2


def test_chat_people_setting_is_persisted_and_defaults_off():
    assert "showPlayersInChat: saved.showPlayersInChat === true" in HTML
    assert "showPlayersInChat: false" in HTML
    assert 'showPlayersInChat: document.getElementById("checkShowPlayersInChat").checked' in HTML
    assert 'showPlayersInChat: document.getElementById("lobbyCheckShowPlayersInChat").checked' in HTML


def test_room_chat_has_people_sidebar_and_refreshes_it():
    assert 'id="roomChatPeople" class="room-chat-people"' in HTML
    assert 'id="roomChatPeopleList" class="room-chat-people-list"' in HTML
    assert "function refreshRoomChatPeople()" in HTML
    assert "window.setInterval(refreshRoomChatPeople, 10000)" in HTML
    assert ".chat-panel.chat-people-visible .room-chat-people" in HTML


def test_mobile_chat_uses_a_compact_people_column():
    assert "--mobile-chat-people-width: 92px" in HTML
    assert "#chatPanel.chat-people-visible .room-chat-people" in HTML
    assert "body.mobile-chat-placement-top #chatPanel.chat-people-visible" in HTML


def test_new_labels_are_translated():
    for source in (ZH, EN):
        assert '"▶ チャットの設定を開く"' in source
        assert '"▼ チャットの設定を閉じる"' in source
        assert '"チャットの中にプレイヤー表示を追加"' in source

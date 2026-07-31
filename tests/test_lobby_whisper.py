from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "frontend" / "index.html"
SCRIPT = ROOT / "frontend" / "lobbyWhisper.js"
ZH_I18N = ROOT / "frontend" / "i18n.js"
EN_I18N = ROOT / "frontend" / "i18n-en.js"


def test_whisper_appears_inside_public_game_rooms() -> None:
    html = INDEX.read_text(encoding="utf-8")

    assert 'id="lobbyWhisper"' in html
    assert html.index('id="gameView"') < html.index('id="lobbyWhisper"')
    assert html.index('id="lobbyWhisper"') < html.index('id="chatToast"')
    assert 'aria-label="1222のつぶやき" hidden' in html
    assert "window.goitaLobbyWhisper?.setRoomVisibility?.(isMainRoomId(gid))" in html
    assert "window.goitaLobbyWhisper?.setRoomVisibility?.(false)" in html
    assert 'id="lobbyWhisperClose"' in html
    assert 'aria-label="つぶやきを閉じる"' in html
    assert "1222のつぶやき" in html
    assert "こんにちは！" in html
    assert ".lobby-whisper-close" in html
    assert "min-height: 156px" in html
    assert "border-radius: 0" in html
    assert "background: #ffffff" in html
    assert "@keyframes lobby-whisper-enter" in html
    assert "translate(-105px, 62px)" in html


def test_whisper_switches_message_and_can_be_closed() -> None:
    html = INDEX.read_text(encoding="utf-8")
    script = SCRIPT.read_text(encoding="utf-8")

    assert '/static/lobbyWhisper.js?v=20260731c' in html
    assert '"こんにちは！"' in script
    assert '"最近の広告は姑息すぎて"' in script
    assert '"その類の広告は絶滅すべきだと思います！"' in script
    assert "MESSAGE_HOLD_MS = Object.freeze([1150, 1650])" in script
    assert "function showMessage(index)" in script
    assert 'message.classList.add("is-entering")' in script
    assert 'message.classList.add("is-leaving")' in script
    assert 'closeButton.addEventListener("click", dismiss)' in script
    assert "function setRoomVisibility(isPublicRoom)" in script
    assert "function startSequence()" in script
    assert "if (!isPublicRoom || dismissed)" in script
    assert "whisper.hidden = true" in script


def test_whisper_supports_chinese_and_english() -> None:
    zh = ZH_I18N.read_text(encoding="utf-8")
    en = EN_I18N.read_text(encoding="utf-8")

    for source in (
        "1222のつぶやき",
        "つぶやきを閉じる",
        "こんにちは！",
        "最近の広告は姑息すぎて",
        "その類の広告は絶滅すべきだと思います！",
    ):
        assert source in zh
        assert source in en


if __name__ == "__main__":
    test_whisper_appears_inside_public_game_rooms()
    test_whisper_switches_message_and_can_be_closed()
    test_whisper_supports_chinese_and_english()
    print("Lobby whisper tests passed")

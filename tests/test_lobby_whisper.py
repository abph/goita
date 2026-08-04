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
    assert "min-height: 116px" in html
    assert "min-height: 112px" in html
    assert "font-size: 19px" in html
    assert "border-radius: 0" in html
    assert "background: #ffffff" in html
    assert "-webkit-user-select: none" in html
    assert "user-select: none" in html
    assert "@keyframes lobby-whisper-enter" in html
    assert "translate(-105px, 62px)" in html


def test_whisper_switches_message_and_can_be_closed() -> None:
    html = INDEX.read_text(encoding="utf-8")
    script = SCRIPT.read_text(encoding="utf-8")

    assert '/static/lobbyWhisper.js?v=20260731f' in html
    assert '"こんにちは！"' in script
    assert '"最近の広告は姑息すぎて、ほんと嫌ですね。"' in script
    assert '"その類の広告は絶滅すべきだと思います！"' not in script
    assert "MESSAGE_HOLD_MS = Object.freeze([1150])" in script
    assert "function showMessage(index)" in script
    assert 'SURPRISE_MESSAGE = "なにもありませんよ笑"' in script
    assert 'MILESTONE_MESSAGE = "100回目おめでとう。' in script
    assert "let surpriseClickCount = 0" in script
    assert "surpriseClickCount = Math.min(surpriseClickCount + 1, 100)" in script
    assert "if (surpriseClickCount >= 100) return MILESTONE_MESSAGE" in script
    assert "if (surpriseClickCount >= 10)" in script
    assert "なにもありませんよ（笑）${surpriseClickCount}回目" in script
    assert "function showSurpriseMessage()" in script
    assert 'whisper.addEventListener("click", showSurpriseMessage)' in script
    assert "event.stopPropagation()" in script
    assert 'message.classList.add("is-entering")' in script
    assert 'message.classList.add("is-leaving")' in script
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
        "最近の広告は姑息すぎて、ほんと嫌ですね。",
        "なにもありませんよ笑",
        "100回目おめでとう。そんな暇なあなたには、プライベートルームを一つ、1年間、授けます。希望するなら、連絡をください。",
    ):
        assert source in zh
        assert source in en


if __name__ == "__main__":
    test_whisper_appears_inside_public_game_rooms()
    test_whisper_switches_message_and_can_be_closed()
    test_whisper_supports_chinese_and_english()
    print("Lobby whisper tests passed")

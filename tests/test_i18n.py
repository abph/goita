from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_language_switcher_and_translation_runtime_are_loaded() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    i18n = (ROOT / "frontend" / "i18n.js").read_text(encoding="utf-8")

    assert 'data-language-choice="ja"' in html
    assert 'data-language-choice="zh"' in html
    assert 'data-language-choice="en"' in html
    assert "setSiteLanguage('ja')" in html
    assert "setSiteLanguage('zh')" in html
    assert "setSiteLanguage('en')" in html
    assert "openSiteInfo('support')" in html
    assert '"https://vrcgoita.com/support/"' in html
    assert '<script src="/static/i18n-en.js?v=20260901a"></script>' in html
    assert '<script src="/static/i18n.js?v=20260901a"></script>' in html
    assert 'const STORAGE_KEY = "goita-ui-language"' in i18n
    assert 'const SUPPORTED_LANGUAGES = new Set(["ja", "zh", "en"])' in i18n
    assert 'new URLSearchParams(window.location.search).get("lang")' in i18n
    assert "let currentLanguage = urlLanguage || readStoredLanguage()" in i18n
    assert "if (options.persist !== false)" in i18n
    assert "setLanguage(currentLanguage, { dispatch: false, persist: false })" in i18n
    assert 'currentLanguage === "en" ? "en" : "ja"' in i18n
    assert "new MutationObserver" in i18n
    assert 'parent.closest(".hand .val, .cell .val, .piece-value")' in i18n
    assert 'parent.closest(".chat-message")' in i18n
    assert '"そろうごいた": "Solo Goita"' in i18n
    assert '"支援について": "关于支持"' in i18n
    assert '"请选择公开房间或私人房间进入。' in i18n
    assert '"支援していただいた方のための専用ルームです。": "这是为支持本项目的朋友准备的专用房间。"' in i18n
    assert '"プライベートA・Bは、どなたでも自由に使えます。": "任何人都可以自由使用私人房间A和B。"' in i18n
    assert '"研究用棋譜ライブラリ": "研究棋谱库"' in i18n
    assert "凑齐Goita" not in i18n


def test_footer_information_modal_has_close_button_and_current_notice() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    chinese = (ROOT / "frontend" / "i18n.js").read_text(encoding="utf-8")
    english = (ROOT / "frontend" / "i18n-en.js").read_text(encoding="utf-8")

    assert 'onclick="closeSiteInfo()" aria-label="案内を閉じる"' in html
    assert ".site-info-modal-content > h3 { padding-right: 36px; }" in html
    assert "そろうごいたは現在ベータ版です" not in html
    for source in [
        "そろうごいたでは、サービスの改善や保守のため、予告なく機能の変更、一時停止、メンテナンスなどを行う場合があります。",
        "保存した棋譜などのデータが、障害や仕様変更によって利用できなくなる場合があります。大切な棋譜は「棋譜DL」で端末にも保存してください。",
    ]:
        assert source in html
        assert source in chinese
        assert source in english


def test_dynamic_ui_and_ai_help_follow_selected_language() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    backend = (ROOT / "backend" / "app.py").read_text(encoding="utf-8")
    board_3d = (ROOT / "frontend" / "board3d.js").read_text(encoding="utf-8")
    board_pixel = (ROOT / "frontend" / "boardPixel.js").read_text(encoding="utf-8")
    english = (ROOT / "frontend" / "i18n-en.js").read_text(encoding="utf-8")
    ask_chat_block = html.split("async function askChatAi(){", 1)[1].split(
        "function seatTurnLabel", 1
    )[0]
    ask_lobby_chat_block = html.split("async function askLobbyChatAi(){", 1)[1].split(
        "async function askChatAi(){", 1
    )[0]

    assert "window.goitaI18n?.translate?.(source)" in html
    assert 'language: window.goitaI18n?.getLanguage?.() || "ja"' in ask_chat_block
    assert 'language: window.goitaI18n?.getLanguage?.() || "ja"' in ask_lobby_chat_block
    assert 'window.addEventListener("goita-language-change"' in html
    assert "function shouldTranslateChatItem(item)" in html
    assert "function localizedChatSender(item)" in html
    assert "window.goitaI18n?.translate?.(text)" in board_3d
    assert "window.goitaI18n?.translate?.(text)" in board_pixel
    assert '"そろうごいた": "Solo Goita"' in english
    assert '"支援について": "Support"' in english
    assert '"Choose a public or private room to enter.' in english
    assert '"支援ページを開く": "Open Support Page"' in english
    assert '"支援していただいた方のための専用ルームです。": "These rooms are reserved for supporters."' in english
    assert '"プライベートA・Bは、どなたでも自由に使えます。": "Private A and B are open for everyone to use."' in english
    assert '"研究用棋譜ライブラリ": "Research Game Record Library"' in english
    assert '"空席": "Open"' in english
    assert '"現在: 空席": "Current: Open"' in english
    assert '["空席", "Open"]' in english
    assert '"Vacant"' not in english
    assert '"設定": "Settings"' in english
    assert "window.GOITA_I18N_EN" in english
    assert 'language: str = "ja"' in backend
    assert "_normalize_ui_language(req.language)" in backend
    assert "请仅使用简体中文" in backend
    assert "日文说明只作为内部参考" in backend
    assert "Answer the user in concise, natural English" in backend


def test_language_packs_cover_site_presence_ui() -> None:
    chinese = (ROOT / "frontend" / "i18n.js").read_text(encoding="utf-8")
    english = (ROOT / "frontend" / "i18n-en.js").read_text(encoding="utf-8")

    for source in [
        "このサイトにいる人",
        "現在いる人を表示",
        "人の一覧を閉じる",
        "トップページ",
        "観戦者",
        "現在、このサイトにいる人はいません。",
    ]:
        assert source in chinese
        assert source in english


def test_hand_reveal_confirmation_is_translated() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    chinese = (ROOT / "frontend" / "i18n.js").read_text(encoding="utf-8")
    english = (ROOT / "frontend" / "i18n-en.js").read_text(encoding="utf-8")

    assert 'id="handRevealConfirmModal"' in html
    assert 'id="handRevealConfirmAccept"' in html
    for source in ["手札を公開しますか？", "公開する", "キャンセル"]:
        assert source in chinese
        assert source in english


def test_lobby_certification_and_hand_limit_are_translated() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    chinese = (ROOT / "frontend" / "i18n.js").read_text(encoding="utf-8")
    english = (ROOT / "frontend" / "i18n-en.js").read_text(encoding="utf-8")

    assert "日本ごいた協会に認定されました。" in html
    assert "「5し」以上の配牌はまだ実装していません" in html
    assert "※「5し」" not in html
    assert '"日本ごいた協会に認定されました。": "已获得日本Goita协会认证。"' in chinese
    assert '"日本ごいた協会に認定されました。": "Recognized by the JAPAN GOITA ASSOCIATION."' in english
    assert '"「5し」以上の配牌はまだ実装していません（必ず「4し」以下になります）。"' in chinese
    assert '"「5し」以上の配牌はまだ実装していません（必ず「4し」以下になります）。"' in english


def test_research_kifu_tags_are_translated() -> None:
    chinese = (ROOT / "frontend" / "i18n.js").read_text(encoding="utf-8")
    english = (ROOT / "frontend" / "i18n-en.js").read_text(encoding="utf-8")

    for source in [
        "タグで絞り込み",
        "すべてのタグ",
        "棋譜をサーバーに保存",
        "タイトル　例：終盤の王受け",
        "気になった点をメモできます",
        "棋譜一覧",
        "棋譜再生",
        "停止",
        "もう一度再生",
        "棋譜を再生しています。",
        "再生が終了しました。",
        "再生できる手順がありません。",
        "棋譜読込",
        "棋譜を読み込みました。",
        "棋譜を読み込めませんでした。",
        "棋譜ファイルは200KB以下にしてください。",
        "匿名で保存",
        "匿名で保存しました。",
        "棋譜一覧へ",
        "棋譜DL",
        "王玉",
        "3し",
        "4し",
        "2香",
        "3香",
        "4香",
        "2中駒",
        "3中駒",
        "4中駒",
        "大駒ペア",
        "し攻め",
        "差し込み",
        "ダブル狙い",
        "だまし香",
        "だましし",
    ]:
        assert source in chinese
        assert source in english


if __name__ == "__main__":
    test_language_switcher_and_translation_runtime_are_loaded()
    test_dynamic_ui_and_ai_help_follow_selected_language()
    test_language_packs_cover_site_presence_ui()
    test_hand_reveal_confirmation_is_translated()
    test_lobby_certification_and_hand_limit_are_translated()
    test_research_kifu_tags_are_translated()
    print("I18N_TEST_OK")

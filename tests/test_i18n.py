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
    assert '"https://i22.fanbox.cc/plans"' in html
    assert '<script src="/static/i18n-en.js?v=20260731a"></script>' in html
    assert '<script src="/static/i18n.js?v=20260731a"></script>' in html
    assert 'const STORAGE_KEY = "goita-ui-language"' in i18n
    assert 'const SUPPORTED_LANGUAGES = new Set(["ja", "zh", "en"])' in i18n
    assert 'currentLanguage === "en" ? "en" : "ja"' in i18n
    assert "new MutationObserver" in i18n
    assert 'parent.closest(".hand .val, .cell .val, .piece-value")' in i18n
    assert 'parent.closest(".chat-message")' in i18n
    assert '"そろうごいた": "Solo Goita"' in i18n
    assert '"支援について": "关于支持"' in i18n
    assert '"请选择公开房间或私人房间进入。' in i18n
    assert '"支援していただいた人のための専用の部屋です。": "这是为支持本项目的人准备的专用房间。"' in i18n
    assert "凑齐Goita" not in i18n


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
    assert '"支援していただいた人のための専用の部屋です。": "Dedicated rooms for people who support the project."' in english
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


if __name__ == "__main__":
    test_language_switcher_and_translation_runtime_are_loaded()
    test_dynamic_ui_and_ai_help_follow_selected_language()
    print("I18N_TEST_OK")

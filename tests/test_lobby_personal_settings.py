from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_lobby_exposes_shared_personal_settings() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert 'id="lobbyPersonalSettingsPanel"' in html
    assert 'id="lobbyAdminSettingsPanel"' not in html
    assert 'id="lobbyAdminSettingsTab"' not in html
    assert 'id="lobbyPersonalPlayerName"' in html
    assert 'id="lobbyMobileChatPlacement"' in html
    assert 'id="lobbyMobileChatWidth"' in html
    assert 'id="lobbyMobileChatTransparency"' in html
    assert 'id="lobbyBoardViewMode"' in html
    assert 'id="lobbyCheckEnableEffects"' in html
    assert 'id="lobbyCheckEnableBeginnerSupport"' in html
    assert 'id="lobbyCheckEnableSoundEffects"' in html
    assert 'id="lobbyPieceSoundChoice"' in html
    assert 'id="lobbyCheckEnableCVoice"' in html
    assert 'id="lobbyCheckEnableAnalytics"' in html
    assert 'id="lobbyDataSharingSettingsDetails"' in html
    assert 'id="dataSharingSettingsDetails"' in html
    assert "▶ データ提供の設定を開く" in html
    assert "▼ データ提供の設定を閉じる" in html
    assert 'onclick="saveLobbyPersonalSettings()"' in html
    assert 'localStorage.setItem(PERSONAL_SETTINGS_KEY, JSON.stringify(personalSettings))' in html
    assert '.settings-disclosure[open] > summary .when-open' in html


def test_lobby_language_switcher_is_inside_personal_settings() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    toolbar_start = html.index('<div class="lobby-toolbar">')
    toolbar_end = html.index("</div>", toolbar_start)
    toolbar = html[toolbar_start:toolbar_end]
    personal_start = html.index('id="lobbyPersonalSettingsPanel"')
    personal_end = html.index('<div class="settings-footer">', personal_start)
    personal = html[personal_start:personal_end]

    assert "language-switcher" not in toolbar
    assert 'class="lobby-settings-button header-menu-trigger"' in toolbar
    assert 'data-language-choice="ja"' in personal
    assert 'data-language-choice="zh"' in personal
    assert 'data-language-choice="en"' in personal
    assert "justify-content: flex-end" in html


def test_lobby_does_not_expose_kifu_save_actions_in_personal_settings() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    start = html.index('id="lobbyPersonalSettingsPanel"')
    end = html.index('<div class="settings-footer">', start)
    lobby_personal_settings = html[start:end]

    assert "downloadKifu" not in lobby_personal_settings


def test_analytics_cannot_block_lobby_initialization() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert "function analyticsLanguage()" in html
    assert "window.goitaI18n?.getLanguage?.()" in html
    assert "normalizeSiteLanguage" not in html
    assert "SITE_LANGUAGE_KEY" not in html
    assert 'console.warn("利用状況の記録をスキップしました:", error)' in html

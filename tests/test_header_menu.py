from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MenuParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.toggles = []
        self.popups = {}
        self.current_popup = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "data-header-menu-toggle" in attrs:
            self.toggles.append(attrs)
        if attrs.get("class") == "header-menu-popup":
            self.current_popup = attrs["id"]
            self.popups[self.current_popup] = {"hidden": "hidden" in attrs, "buttons": []}
        elif self.current_popup and tag == "button":
            self.popups[self.current_popup]["buttons"].append(attrs)

    def handle_endtag(self, tag):
        if tag == "div":
            self.current_popup = None


def test_both_header_menus_group_existing_actions_and_start_closed():
    parser = MenuParser()
    parser.feed((ROOT / "frontend/index.html").read_text(encoding="utf-8"))
    assert len(parser.toggles) == 2
    for toggle in parser.toggles:
        assert toggle["aria-expanded"] == "false"
        popup = parser.popups[toggle["aria-controls"]]
        assert popup["hidden"]
        assert len(popup["buttons"]) == 2
        member, settings = popup["buttons"]
        assert "data-member-entry" in member
        assert member["onclick"] == "openMemberPage()"
        assert settings["onclick"] in ("openLobbySettings()", "openSettingsModal(gid)")


def test_menu_asset_and_translations_are_available():
    html = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
    assert '/static/header-menu.js?v=20260904a' in html
    assert '.header-menu-popup[hidden] { display: none; }' in html
    for name in ("i18n.js", "i18n-en.js"):
        assert '"メニュー":' in (ROOT / "frontend" / name).read_text(encoding="utf-8")

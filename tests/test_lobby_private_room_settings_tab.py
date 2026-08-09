from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "frontend" / "index.html"


def test_private_room_card_opens_room_management_first() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")

    assert (
        "onclick=\"openSettingsModal('${room.game_id}', event, 'management')\""
        in html
    )
    assert (
        'function openSettingsModal(targetGid, event, initialTab = "personal")'
        in html
    )
    assert (
        'showSettingsTab(initialTab === "management" ? "management" : "personal");'
        in html
    )
    assert 'onclick="openSettingsModal(gid)"' in html


if __name__ == "__main__":
    test_private_room_card_opens_room_management_first()
    print("LOBBY_PRIVATE_ROOM_SETTINGS_TAB_TEST_OK")

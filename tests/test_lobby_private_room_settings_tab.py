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
    assert "showSettingsTab(tabToOpen);" in html
    assert 'onclick="openSettingsModal(gid)"' in html


def test_room_admin_session_and_last_settings_screen_survive_modal_close() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")
    close_block = html.split("function closeSettings() {", 1)[1].split(
        "function clearRoomBackground", 1
    )[0]

    assert 'let lastSettingsTab = "personal";' in html
    assert "function clearRoomAdminSession" in html
    assert "const sameTarget = settingTargetGid === targetGid;" in html
    assert "sameTarget ? lastSettingsTab : \"personal\"" in html
    assert "unlockRoomManagement(currentAdminPass, true);" in html
    assert "researchKifuAdminPassword" not in html
    assert "clearRoomAdminSession({clearTarget: true});" in html
    assert "if (response.status === 401) clearRoomAdminSession();" in html
    assert "currentAdminPass = \"\";" not in close_block
    assert "settingTargetGid = null;" not in close_block
    assert "selectedResearchKifu = null;" not in close_block


if __name__ == "__main__":
    test_private_room_card_opens_room_management_first()
    test_room_admin_session_and_last_settings_screen_survive_modal_close()
    print("LOBBY_PRIVATE_ROOM_SETTINGS_TAB_TEST_OK")

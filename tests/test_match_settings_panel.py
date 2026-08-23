from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_match_setup_is_grouped_in_the_settings_modal() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    panel_start = html.index('<section id="matchSettingsPanel"')
    panel_end = html.index("</section>", panel_start)
    panel = html[panel_start:panel_end]

    assert 'id="matchSettingsTab"' in html
    assert 'id="dealerDetails"' in panel
    assert 'id="presetHandsDetails"' in panel
    assert 'id="kifuPresetId"' in panel
    assert 'id="kifuPresetRound"' in panel
    assert 'onsubmit="loadPresetFromKifu(event)"' in panel
    assert 'id="forceResetSettingsRow"' in panel
    assert 'id="forceResetSettingsButton"' in panel
    assert 'onclick="confirmForceReset()"' in panel
    assert html.count('id="dealerDetails"') == 1
    assert html.count('id="presetHandsDetails"') == 1


def test_match_setup_is_host_only_and_private_hands_stay_private() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert '(targetGid === gid && isCurrentClientHost()) ? "block" : "none"' in html
    assert 'presetDetails.style.display = (!isMain && isHost) ? "block" : "none"' in html
    assert 'hostToolsGrid.style.display = showActions ? "grid" : "none"' in html
    assert "grid-template-columns: 22px repeat(9, 28px)" in html
    assert "grid-template-columns: 18px repeat(9, 24px)" in html
    assert 'fetch("/static/kifu_data.json", {cache: "force-cache"})' in html
    assert 'const seatKeys = {A: "p0", B: "p1", C: "p2", D: "p3"}' in html
    assert "Number(item?.round_index) === requestedRound" in html


def test_force_reset_requires_confirmation_and_is_removed_from_the_play_controls() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert "async function confirmForceReset()" in html
    assert 'window.confirm(uiText("現在の対局と得点をリセットします。よろしいですか？"))' in html
    assert 'await startNewGame(false);' in html
    assert 'btnNewGame.style.display = (isHost && state.finished) ? "" : "none";' in html
    assert 'forceResetSettingsRow.style.display = (isHost && !state.finished) ? "block" : "none";' in html
    assert 'btnNewGame.textContent = "強制リセット"' not in html


def test_kifu_0001_round_1_has_the_expected_four_hands() -> None:
    archive = json.loads(
        (ROOT / "frontend" / "kifu_data.json").read_text(encoding="utf-8")
    )
    match = next(item for item in archive["matches"] if item["id"] == "0001")
    round_data = next(
        item for item in match["rounds"] if item["round_index"] == 1
    )

    assert round_data["hand"] == {
        "p0": "ししし香馬金金飛",
        "p1": "ししし銀銀銀飛玉",
        "p2": "し香馬馬銀金金角",
        "p3": "ししし香香馬角王",
    }


def test_personal_kifu_save_actions_are_grouped() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    details_start = html.index('<details id="kifuSaveSettingsDetails"')
    details_end = html.index("</details>", details_start)
    details = html[details_start:details_end]

    assert "棋譜を保存する" in details
    assert "匿名で棋譜を保存する" in details
    assert html.count('id="kifuSaveSettingsDetails"') == 1
    assert '"kifuSaveSettingsDetails"' in html


def test_effects_and_beginner_support_are_grouped() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    details_start = html.index('<details id="playSupportSettingsDetails"')
    details_end = html.index("</details>", details_start)
    details = html[details_start:details_end]

    assert 'id="boardViewSettingRow"' in details
    assert 'id="boardViewMode"' in details
    assert 'id="checkEnableEffects"' in details
    assert 'id="beginnerSupportSettingRow"' in details
    assert 'id="checkEnableBeginnerSupport"' in details
    assert html.count('id="playSupportSettingsDetails"') == 1
    assert '"playSupportSettingsDetails"' in html


def test_settings_modals_keep_a_fixed_height_while_panels_scroll() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert html.count("height: min(720px, calc(100dvh - 24px));") >= 2
    assert ".settings-modal-content > .settings-panel.active" in html
    assert ".lobby-admin-modal-content > .settings-panel.active" in html
    assert "flex: 1 1 auto;" in html
    assert "min-height: 0;" in html
    assert "overflow-y: auto;" in html
    assert html.count("height: calc(100dvh - 12px);") >= 2


def test_room_settings_header_and_footer_use_compact_actions() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert ".settings-modal-content > .settings-footer {" in html
    assert "display: flex;" in html
    assert 'id="personalSettingsFooterActions"' in html
    assert "font-size: 31px;" in html
    assert "font-size: 15px;" in html
    assert ".settings-modal-content > .settings-footer button {" in html
    assert "padding: 6px 8px;" in html
    assert '.settings-inline-action input[type="password"]:focus' in html
    assert "outline: none;" in html


if __name__ == "__main__":
    test_match_setup_is_grouped_in_the_settings_modal()
    test_match_setup_is_host_only_and_private_hands_stay_private()
    test_kifu_0001_round_1_has_the_expected_four_hands()
    test_personal_kifu_save_actions_are_grouped()
    test_effects_and_beginner_support_are_grouped()
    test_settings_modals_keep_a_fixed_height_while_panels_scroll()
    test_room_settings_header_and_footer_use_compact_actions()
    print("MATCH_SETTINGS_PANEL_TEST_OK")

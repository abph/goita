from pathlib import Path


HTML = (Path(__file__).parents[1] / "frontend" / "index.html").read_text(encoding="utf-8")


def test_lobby_intro_uses_scoped_typography_classes():
    assert '<div class="lobby-intro">' in HTML
    assert '<p class="lobby-intro-lead">' in HTML
    assert '<section class="lobby-purpose-guide"' in HTML
    assert HTML.count('class="lobby-purpose-guide"') == 1
    assert HTML.count('class="lobby-purpose-option"') == 7


def test_lobby_intro_has_smaller_mobile_typography():
    assert ".lobby-intro h1 { font-size: 24px; }" in HTML
    assert ".lobby-intro-lead { font-size: 13px; }" in HTML
    assert ".lobby-purpose-option { min-height: 62px;" in HTML


def test_lobby_feature_guide_has_play_and_study_steps():
    assert 'id="lobbyPurposeHome"' in HTML
    assert 'id="lobbyPurposePlay"' in HTML
    assert 'id="lobbyPurposeStudy"' in HTML
    assert 'onclick="showLobbyPurposeStep(\'play\')"' in HTML
    assert 'onclick="showLobbyPurposeStep(\'study\')"' in HTML
    assert 'onclick="startGuidedMatch(\'ai\')"' in HTML
    assert 'onclick="startGuidedMatch(\'human\')"' in HTML
    assert 'onclick="openFeatureGuideLibrary()"' in HTML
    assert 'onclick="guideToPrivateRooms(\'deal\')"' in HTML
    assert 'onclick="guideToPrivateRooms(\'hand\')"' in HTML
    assert 'id="lobbyPrivateRoomsSection"' in HTML


def test_lobby_guide_and_score_attack_are_separate_stable_columns():
    assert '<div class="lobby-feature-row">' in HTML
    assert "grid-template-columns: minmax(0, 2fr) minmax(0, 3fr);" in HTML
    assert "gap: 18px;" in HTML
    assert ".lobby-purpose-stage { display: grid; }" in HTML
    assert "grid-area: 1 / 1;" in HTML
    assert 'class="lobby-purpose-panel is-inactive"' in HTML
    assert 'panel.toggleAttribute("inert", inactive);' in HTML


def test_lobby_submenus_use_compact_arrow_back_controls():
    assert HTML.count('class="lobby-purpose-back-arrow"') == 2
    assert HTML.count('aria-label="最初の選択に戻る"') == 2
    assert 'class="lobby-purpose-back"' not in HTML
    assert ".lobby-purpose-option strong" in HTML
    assert "font-size: 14px;" in HTML
    assert "font-size: 11px;" in HTML


def test_guided_match_uses_temporary_player_tags():
    assert 'featureGuideRoomTag = {roomId, tag};' in HTML
    assert 'const tag = aiMatch ? "ai_practice" : "human_match";' in HTML
    assert 'const roomId = aiMatch ? "main-e" : MAIN_GID;' in HTML
    assert 'featureGuideRoomTag = null;' in HTML
    assert 'localStorage.setItem(PERSONAL_SETTINGS_KEY' not in HTML.split("async function startGuidedMatch", 1)[1].split("function guideToPrivateRooms", 1)[0]


def test_preset_hand_indicator_is_rendered_below_seat_a():
    assert 'className = "hand-preset-indicator"' in HTML
    assert 'presetIndicator.textContent = uiText("手札指定 ON")' in HTML
    assert 'seat === "A" && presetIsEnabled()' in HTML


def test_lobby_room_section_headings_and_descriptions_are_compact():
    assert ".lobby-room-section h2" in HTML
    assert "font-size: 20px;" in HTML
    assert ".lobby-room-section > p" in HTML
    assert "font-size: 13px;" in HTML

from pathlib import Path


HTML = (Path(__file__).parents[1] / "frontend" / "index.html").read_text(encoding="utf-8")


def test_lobby_intro_uses_scoped_typography_classes():
    assert '<div class="lobby-intro">' in HTML
    assert '<div class="lobby-brand-lockup">' in HTML
    assert '<p class="lobby-brand-certification">日本ごいた協会認定</p>' in HTML
    assert '<p class="lobby-brand-subtitle">オンライン対局室</p>' in HTML
    assert "grid-template-columns: auto auto;" in HTML
    assert "grid-row: 1 / 3;" in HTML
    assert '<section class="lobby-purpose-guide lobby-section-card"' in HTML
    assert HTML.count('class="lobby-purpose-guide lobby-section-card"') == 1
    assert HTML.count('class="lobby-purpose-option"') == 7
    assert "border-style: dashed;" in HTML
    assert "border-color: rgba(139, 90, 43, 0.55);" in HTML
    assert "box-shadow: 0 2px 7px rgba(79, 52, 29, 0.055);" in HTML


def test_lobby_intro_has_smaller_mobile_typography():
    assert ".lobby-intro {\n        padding: 28px 0 34px;" in HTML
    assert ".lobby-intro { padding: 20px 0 28px; }" in HTML
    assert ".lobby-intro h1 { font-size: 28px; }" in HTML
    assert ".lobby-brand-certification { font-size: 9px; }" in HTML
    assert ".lobby-brand-subtitle { font-size: 11px; }" in HTML
    assert ".lobby-purpose-option { min-height: 58px;" in HTML


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
    assert 'element.toggleAttribute("inert", inactive);' in HTML


def test_lobby_submenus_use_compact_arrow_back_controls():
    assert HTML.count('class="lobby-purpose-back-arrow"') == 2
    assert HTML.count('aria-label="最初の選択に戻る"') == 2
    assert 'class="lobby-purpose-back"' not in HTML
    assert ".lobby-purpose-option strong" in HTML
    assert "font-size: 13px;" in HTML
    assert "font-size: 10px;" in HTML


def test_lobby_questions_replace_the_main_guide_heading():
    assert 'id="lobbyPurposeHomeHeader"' in HTML
    assert 'id="lobbyPurposePlayHeader"' in HTML
    assert 'id="lobbyPurposeStudyHeader"' in HTML
    assert '<h2 class="lobby-purpose-subtitle" tabindex="-1">どのように研究しますか？</h2>' in HTML
    assert '<h2 class="lobby-purpose-subtitle" tabindex="-1">人間とAI、どちらと対局しますか？</h2>' in HTML
    assert '.lobby-purpose-title {' in HTML
    assert 'font-size: 20px;' in HTML


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
    assert ".lobby-room-panel-description" in HTML
    assert "font-size: 13px;" in HTML


def test_public_and_private_rooms_share_one_switchable_section():
    assert HTML.count('class="card lobby-room-section lobby-section-card"') == 1
    assert 'id="lobbyRoomsSection"' in HTML
    assert 'id="lobbyPublicRoomsPanel"' in HTML
    assert 'id="lobbyPrivateRoomsSection"' in HTML
    assert '← 公開部屋</button>' in HTML
    assert 'プライベート →</button>' in HTML
    assert '<h2 class="lobby-section-heading lobby-room-category-title">対局ルーム</h2>' in HTML
    assert 'function showLobbyRoomCategory(category, focusTab = false)' in HTML
    assert 'function handleLobbyRoomCategoryKeydown(event)' in HTML
    assert 'showLobbyRoomCategory("private");' in HTML
    assert ".lobby-room-category-tab {\n        min-height: 53px;" in HTML
    assert "text-decoration-thickness: 2px;" in HTML
    assert "text-underline-offset: 6px;" in HTML
    assert "border-right: 1px solid rgba(139, 90, 43, 0.28);" not in HTML


def test_three_lobby_features_use_consistent_section_cards():
    assert HTML.count("lobby-section-card") >= 4
    assert '<h2 id="scoreAttackSectionTitle" class="lobby-section-heading">' in HTML
    assert '<h2 class="lobby-section-heading lobby-room-category-title">対局ルーム</h2>' in HTML
    assert 'class="score-attack-entry-body"' in HTML
    assert 'class="lobby-room-section-body"' in HTML

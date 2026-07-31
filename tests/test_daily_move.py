from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "frontend" / "index.html"
SCRIPT = ROOT / "frontend" / "dailyMove.js"
STYLES = ROOT / "frontend" / "dailyMove.css"


def test_daily_move_assets_and_lobby_placement() -> None:
    html = INDEX.read_text(encoding="utf-8")

    assert STYLES.exists()
    assert SCRIPT.exists()
    assert '/static/dailyMove.css?v=20260731b' in html
    assert '/static/dailyMove.js?v=20260731b' in html
    assert 'id="dailyMoveCard" class="daily-move-card" data-i18n-ignore hidden' in html
    assert ".daily-move-card[hidden]" in STYLES.read_text(encoding="utf-8")
    assert 'id="dailyMoveModal"' in html
    assert html.index('id="dailyMoveCard"') < html.index('id="lobbyMainRoom"')


def test_first_daily_move_uses_only_curated_position_data() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert 'id: "0206-r6-t12"' in script
    assert 'kifuId: "0206"' in script
    assert "round: 6" in script
    assert "turn: 12" in script
    assert 'seat: "D"' in script
    assert 'hand: ["し", "し", "香"]' in script
    assert '{ receive: "王", attack: "?", turn: 12, question: true }' in script
    assert 'correctChoice: "pawn"' in script
    assert "kifu_data.json" not in script


def test_daily_move_shows_the_board_position_and_two_attack_choices() -> None:
    html = INDEX.read_text(encoding="utf-8")
    script = SCRIPT.read_text(encoding="utf-8")
    styles = STYLES.read_text(encoding="utf-8")

    assert 'data-daily-move-choice="pawn"' in html
    assert 'data-daily-move-choice="lance"' in html
    assert 'data-daily-move-choice="pass"' not in html
    assert "BOARD_SLOTS" in script
    assert "faceDown: true" in script
    assert "if (options.faceDown)" in script
    assert "grid-template-columns: repeat(8, var(--cell))" in styles
    assert "color: transparent" in styles
    assert "Dは王で銀を受け、しを出しました。" in script
    assert "Bがしで受け、香で20点上がり" in script
    assert "パスするとAが銀を受けて香で上がる" in script
    assert "Aが香で受けて銀で上がります" in script


def test_daily_move_supports_all_site_languages_and_completion_state() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert "ja: {" in script
    assert "zh: {" in script
    assert "en: {" in script
    assert '"Move of the Day"' in script
    assert "goita-language-change" in script
    assert "localStorage.setItem(STORAGE_KEY" in script
    assert "localStorage.getItem(STORAGE_KEY)" in script


if __name__ == "__main__":
    test_daily_move_assets_and_lobby_placement()
    test_first_daily_move_uses_only_curated_position_data()
    test_daily_move_shows_the_board_position_and_two_attack_choices()
    test_daily_move_supports_all_site_languages_and_completion_state()
    print("daily move tests passed")

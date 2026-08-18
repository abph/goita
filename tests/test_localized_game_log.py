from pathlib import Path


HTML = (Path(__file__).parents[1] / "frontend" / "index.html").read_text(encoding="utf-8")


def test_game_log_is_localized_at_render_time_without_changing_server_logs():
    assert "function localizeGameLogLine(" in HTML
    assert "function localizeGameLogSuffix(" in HTML
    assert "function renderGameLog(" in HTML
    assert "renderGameLog(state.log || []);" in HTML
    assert "renderGameLog(latestState.log || []);" in HTML
    assert '(state.log||[]).join("\\n")' not in HTML


def test_japanese_game_log_covers_actions_results_and_effects():
    for expected in (
        "対局開始。親=",
        "局終了。勝者=",
        "対戦終了！勝利ペア=",
        "を伏せる → ",
        "で受ける",
        "で攻める",
        "パス",
        'reach: "リーチ"',
        'kakarigotae: "かかりごたえ"',
    ):
        assert expected in HTML


def test_english_game_log_keeps_the_original_debug_format():
    localization_start = HTML.index("function localizeGameLogLine(")
    localization_end = HTML.index("function renderGameLog(", localization_start)
    localization = HTML[localization_start:localization_end]

    assert 'if(language === "en") return source;' in localization
    assert "[AI:" not in localization

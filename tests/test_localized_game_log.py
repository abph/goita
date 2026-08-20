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


def test_japanese_game_log_explains_ai_reason_and_performance_fields():
    for expected in (
        'shi_signal: "し攻めに賛同する意思表示"',
        'kakari: "味方の攻めに合わせる判断"',
        "function localizeGameLogAiDetail(",
        "function localizeGameLogAiDecision(",
        "function formatGameLogSeconds(",
        "function localizeGameLogPerformance(",
        'infer: "手駒推定"',
        'cache: "先読みキャッシュ"',
        'sample: "候補生成"',
        'search: "探索"',
        "【AI判断：",
        "【思考時間：${total}】",
        "【思考時間：合計",
    ):
        assert expected in HTML

    suffix_start = HTML.index("function localizeGameLogSuffix(")
    suffix_end = HTML.index("function localizeGameLogLine(", suffix_start)
    suffix = HTML[suffix_start:suffix_end]
    assert r"/\[AI:([^\]]+)\]/g" in suffix
    assert r"/\[PERF\(ms\):([^\]]+)\]/g" in suffix


def test_ai_thinking_breakdown_is_a_personal_setting_and_defaults_to_off():
    for expected in (
        'id="lobbyCheckShowAiThinkingBreakdown"',
        'id="checkShowAiThinkingBreakdown"',
        "AIの思考時間の内訳を表示する",
        "showAiThinkingBreakdown: saved.showAiThinkingBreakdown === true",
        "showAiThinkingBreakdown: false",
        "personalSettings.showAiThinkingBreakdown !== true",
    ):
        assert expected in HTML

    assert "Math.max(0, value) / 1000" in HTML


def test_attack_candidates_are_localized_from_structured_log_data():
    for expected in (
        "function localizeGameLogAttackCandidates(",
        "【攻め候補：${chosen}を採用",
        "評価値が${amount}低い",
        "優先戦略により不採用",
        "item?.score_gap !== null",
        "[Attack candidates: chose",
    ):
        assert expected in HTML

    suffix_start = HTML.index("function localizeGameLogSuffix(")
    suffix_end = HTML.index("function localizeGameLogLine(", suffix_start)
    suffix = HTML[suffix_start:suffix_end]
    assert r"/\[AI-CANDIDATES:([^\]]+)\]/g" in suffix


def test_english_game_log_keeps_debug_reasons_but_displays_time_in_seconds():
    localization_start = HTML.index("function localizeGameLogLine(")
    localization_end = HTML.index("function renderGameLog(", localization_start)
    localization = HTML[localization_start:localization_end]

    assert 'if(language === "en") {' in localization
    assert "[Thinking time:" in HTML
    assert 'language === "en" ? `${rounded || "0"} sec`' in HTML
    assert "[AI:" not in localization


def test_ai_board_piece_opens_local_candidate_comparison_in_chat():
    for expected in (
        "function showAiBoardThought(",
        "function attachAiBoardThoughtInteraction(",
        "ai_board_explanations",
        "localAiThoughtMessages",
        "この手の判断と候補比較：",
        'window.addEventListener("goita-ai-piece-thought"',
        "toggleChatPanel(true, false)",
    ):
        assert expected in HTML

    board_3d = (Path(__file__).parents[1] / "frontend" / "board3d.js").read_text(encoding="utf-8")
    board_pixel = (Path(__file__).parents[1] / "frontend" / "boardPixel.js").read_text(encoding="utf-8")
    assert "function aiThoughtKeyAtPointer(" in board_3d
    assert 'new CustomEvent("goita-ai-piece-thought"' in board_3d
    assert "function thoughtKeyAtPointer(" in board_pixel
    assert 'new CustomEvent("goita-ai-piece-thought"' in board_pixel

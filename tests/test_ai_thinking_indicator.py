from __future__ import annotations

from pathlib import Path

import backend.app as app_module


def test_ai_thinking_indicator_is_wired_to_cpu_and_auto_turns() -> None:
    html = (app_module.FRONTEND_DIR / "index.html").read_text(encoding="utf-8")

    assert "let aiThinkingSeat = null;" in html
    assert 'spinner.className = "name-thinking-spinner"' in html
    assert 'spinner.src = "/static/svg-loading-spinner.svg"' in html
    assert "function setAiThinking(active, seat = \"\")" in html
    assert "function scheduleAiThinking(seat)" in html
    assert "let aiThinkingTimer = null;" in html
    assert "if (!active && aiThinkingTimer)" in html
    assert "if (isProcessingCpu && latestState && latestState.turn === seat)" in html
    assert "}, 2000);" in html
    assert "aiThinkingSeat = active && seat ? seat : null;" in html
    assert "if (aiThinkingSeat === seat)" in html
    assert "scheduleAiThinking(state.turn);" in html
    assert "setAiThinking(false);" in html
    assert html.count("scheduleAiThinking(state.turn);") == 2


def test_spinner_asset_contains_animation() -> None:
    spinner_path = Path(app_module.FRONTEND_DIR / "svg-loading-spinner.svg")
    spinner = spinner_path.read_text(encoding="utf-8")

    assert spinner.startswith("<svg ")
    assert 'repeatCount="indefinite"' in spinner
    assert 'type="rotate"' in spinner


if __name__ == "__main__":
    test_ai_thinking_indicator_is_wired_to_cpu_and_auto_turns()
    test_spinner_asset_contains_animation()
    print("AI_THINKING_INDICATOR_TEST_OK")

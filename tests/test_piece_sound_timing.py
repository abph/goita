"""Keep piece sounds synchronized with confirmed board rendering."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")


def test_piece_sound_runs_before_followup_requests() -> None:
    refresh = HTML.split("async function refresh(){", 1)[1]
    early_render = refresh.index(
        "renderBoard(state);\n  playNewPieceSoundsAfterBoardRender(state, currentLogCount);"
    )
    legal_request = refresh.index("const fetchedLegal = await fetchLegal();")

    assert early_render < legal_request


def test_early_piece_sound_is_not_replayed_by_log_animation() -> None:
    assert "const earlyPieceSoundLogKeys = new Set();" in HTML
    assert "earlyPieceSoundLogKeys.clear();" in HTML
    assert "pieceSoundWasPlayedEarly" in HTML
    assert "!pieceSoundWasPlayedEarly" in HTML


if __name__ == "__main__":
    test_piece_sound_runs_before_followup_requests()
    test_early_piece_sound_is_not_replayed_by_log_animation()
    print("PIECE_SOUND_TIMING_TEST_OK")

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_linked_kifu_query_is_presented_in_the_lobby() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert 'id="linkedKifuNotice"' in html
    assert 'params.get("kifu")' in html
    assert 'params.get("round")' in html
    assert 'params.get("seat")' in html
    assert 'params.get("hands")' in html
    assert "function parseLinkedKifuHands(value)" in html
    assert "LINKED_KIFU_HAND_COUNTS" in html
    assert "function renderLinkedKifuNotice()" in html
    assert "棋譜の手駒で遊ぶ" in html
    assert "使用棋谱手牌进行游戏" in html
    assert "Play with hands from a game record" in html


def test_linked_kifu_auto_load_is_private_host_only() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    auto_load = html.split(
        "async function maybeApplyLinkedKifuPreset(state){", 1
    )[1].split("function renderPresetCopyButtons", 1)[0]

    assert "if(isMainRoomId(gid) || !isCurrentClientHost()) return false;" in auto_load
    assert "if(state?.is_started || state?.finished) return false;" in auto_load
    assert "autoLoadAttempted" in auto_load
    assert "applyLinkedKifuHandsToPreset" in auto_load
    assert "loadKifuPresetByValues" not in auto_load
    assert "getKifuArchive" not in auto_load
    assert "applyCurrentSetupToWaitingGame" in auto_load
    assert "keepScore: false" in auto_load
    assert "clearLinkedKifuQueryParams()" in auto_load


def test_manual_and_linked_kifu_loading_share_one_loader() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert "async function loadKifuPresetByValues(" in html
    assert "return loadKifuPresetByValues(idInput?.value, roundInput?.value);" in html
    assert "applyKifuHandsToPreset(round, {scheduleApply: options.scheduleApply !== false})" in html
    assert "const linkedKifuAppliedNow = await maybeApplyLinkedKifuPreset(state);" in html
    assert "if (linkedKifuPreset)" in html
    assert "renderLinkedKifuNotice();" in html


if __name__ == "__main__":
    test_linked_kifu_query_is_presented_in_the_lobby()
    test_linked_kifu_auto_load_is_private_host_only()
    test_manual_and_linked_kifu_loading_share_one_loader()
    print("LINKED_KIFU_PRESET_TEST_OK")

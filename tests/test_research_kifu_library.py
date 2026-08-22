from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from fastapi import HTTPException

from backend import app as app_module
from backend.research_kifu_store import (
    RESEARCH_KIFU_FILENAME,
    ResearchKifuStore,
    resolve_research_kifu_path,
)


ROOT = Path(__file__).parents[1]
HTML = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")


def _payload(round_index: int = 3):
    return {
        "version": 1,
        "round_index": round_index,
        "dealer": "C",
        "winner": "A",
        "gained_score": 30,
        "hand": {
            "p0": "しししし香馬銀王",
            "p1": "しし香馬銀金角飛",
            "p2": "しし香馬銀金金玉",
            "p3": "しし香馬銀金角飛",
        },
        "moves": [["0", "し", "馬"]],
    }


def test_path_prefers_explicit_then_persistent_then_local_fallback():
    fallback = Path("results/local.sqlite3")
    assert resolve_research_kifu_path(
        {"GOITA_KIFU_DB_PATH": "/tmp/custom.sqlite3", "GOITA_PERSISTENT_DATA_DIR": "/var/data"},
        local_fallback=fallback,
    ) == Path("/tmp/custom.sqlite3")
    assert resolve_research_kifu_path(
        {"GOITA_PERSISTENT_DATA_DIR": "/var/data"},
        local_fallback=fallback,
    ) == Path("/var/data") / RESEARCH_KIFU_FILENAME
    assert resolve_research_kifu_path({}, local_fallback=fallback) == fallback


def test_store_keeps_room_libraries_isolated_and_supports_crud():
    with TemporaryDirectory() as directory:
        store = ResearchKifuStore(Path(directory) / "library.sqlite3")
        saved = store.save("private-a", title="終盤研究", memo="王受け", payload=_payload())
        record_id = saved["id"]

        assert record_id.startswith("K-")
        assert store.list("private-a")[0]["title"] == "終盤研究"
        assert store.list("private-b") == []
        assert store.get("private-a", record_id)["payload"]["dealer"] == "C"
        assert store.get("private-b", record_id) is None
        assert store.delete("private-b", record_id) is False
        assert store.delete("private-a", record_id) is True
        assert store.get("private-a", record_id) is None


def test_completed_round_snapshot_is_preserved_for_the_next_round():
    old_game = app_module._create_game_obj(dealer="C")
    old_game["round_count"] = 4
    old_game["is_started"] = True
    old_game["last_completed_kifu"] = _payload(round_index=4)
    new_game = app_module._create_game_obj(dealer="A")

    app_module._preserve_match_progress(new_game, old_game)

    assert new_game["round_count"] == 5
    assert new_game["last_completed_kifu"]["round_index"] == 4


def test_private_room_api_requires_admin_and_round_trips_records():
    room_id = app_module.PRIVATE_A_GID
    original_game = app_module.GAMES[room_id]
    original_store = app_module.RESEARCH_KIFU_STORE
    with TemporaryDirectory() as directory:
        app_module.RESEARCH_KIFU_STORE = ResearchKifuStore(
            Path(directory) / "library.sqlite3"
        )
        game = app_module._create_game_obj(dealer="B")
        game["admin_password"] = "research-secret"
        game["last_completed_kifu"] = _payload(round_index=7)
        app_module.GAMES[room_id] = game
        try:
            try:
                app_module.list_research_kifu(
                    room_id,
                    app_module.ResearchKifuAuthRequest(admin_password="wrong"),
                )
                raise AssertionError("wrong admin password should be rejected")
            except HTTPException as error:
                assert error.status_code == 401

            saved = app_module.save_research_kifu(
                room_id,
                app_module.ResearchKifuSaveRequest(
                    admin_password="research-secret",
                    title="角を残す",
                    memo="3つ目の攻めを比較",
                ),
            )["record"]
            listed = app_module.list_research_kifu(
                room_id,
                app_module.ResearchKifuAuthRequest(admin_password="research-secret"),
            )["records"]
            assert listed[0]["id"] == saved["id"]
            detail = app_module.get_research_kifu(
                room_id,
                saved["id"],
                app_module.ResearchKifuAuthRequest(admin_password="research-secret"),
            )["record"]
            assert detail["payload"]["round_index"] == 7
            assert app_module.delete_research_kifu(
                room_id,
                saved["id"],
                app_module.ResearchKifuAuthRequest(admin_password="research-secret"),
            ) == {"ok": True}
        finally:
            app_module.GAMES[room_id] = original_game
            app_module.RESEARCH_KIFU_STORE = original_store


def test_settings_popup_contains_the_research_library_workflow():
    assert 'id="researchKifuTab"' in HTML
    assert 'id="researchKifuPanel"' in HTML
    assert "unlockResearchKifuLibrary()" in HTML
    assert "saveCurrentResearchKifu()" in HTML
    assert "openResearchKifu(record.id)" in HTML
    assert "applySelectedResearchKifu()" in HTML
    assert "deleteSelectedResearchKifu()" in HTML
    assert 'id="researchKifuBoard"' in HTML
    assert "buildResearchKifuFinalState(payload)" in HTML
    assert "renderResearchKifuBoard(payload)" in HTML
    assert "research-kifu-move-number" in HTML
    assert "薄い駒：伏せた駒・残った駒" in HTML


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
    print("RESEARCH_KIFU_LIBRARY_TEST_OK")

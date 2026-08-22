import json
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from fastapi import HTTPException

from backend import app as app_module
from backend.research_kifu_store import (
    RESEARCH_KIFU_FILENAME,
    RESEARCH_KIFU_TAGS,
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
        saved = store.save(
            "private-a",
            title="終盤研究",
            memo="王受け",
            tags=["王玉", "2香", "2中駒", "王玉"],
            payload=_payload(),
        )
        record_id = saved["id"]

        assert record_id.startswith("K-")
        assert store.list("private-a")[0]["title"] == "終盤研究"
        assert store.list("private-a")[0]["tags"] == ["王玉", "2香", "2中駒"]
        assert store.list("private-b") == []
        assert store.get("private-a", record_id)["payload"]["dealer"] == "C"
        assert store.get("private-b", record_id) is None
        edited = store.update_details(
            "private-a",
            record_id,
            title="高得点ルート",
            memo="王上がりも比較",
            tags=["4し", "し攻め", "差し込み"],
        )
        assert edited["title"] == "高得点ルート"
        assert edited["memo"] == "王上がりも比較"
        assert edited["tags"] == ["4し", "し攻め", "差し込み"]
        assert store.delete("private-b", record_id) is False
        assert store.delete("private-a", record_id) is True
        assert store.get("private-a", record_id) is None


def test_store_adds_tags_to_an_existing_database_without_losing_records():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "old-library.sqlite3"
        payload = _payload()
        with closing(sqlite3.connect(path)) as connection:
            connection.executescript(
                """
                CREATE TABLE research_kifu (
                    id TEXT PRIMARY KEY,
                    room_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    title TEXT NOT NULL,
                    memo TEXT NOT NULL,
                    round_index INTEGER NOT NULL,
                    dealer TEXT NOT NULL,
                    winner TEXT,
                    gained_score INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )
            connection.execute(
                """
                INSERT INTO research_kifu (
                    id, room_id, created_at, title, memo, round_index,
                    dealer, winner, gained_score, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "K-23456789AB",
                    "private-a",
                    "2026-08-22T00:00:00+00:00",
                    "既存棋譜",
                    "移行確認",
                    3,
                    "C",
                    "A",
                    30,
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
            connection.commit()

        store = ResearchKifuStore(path)
        records = store.list("private-a")
        assert records[0]["title"] == "既存棋譜"
        assert records[0]["tags"] == []
        with closing(sqlite3.connect(path)) as connection:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(research_kifu)")
            }
        assert "tags_json" in columns


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
                    tags=["3中駒", "大駒ペア"],
                ),
            )["record"]
            listed = app_module.list_research_kifu(
                room_id,
                app_module.ResearchKifuAuthRequest(admin_password="research-secret"),
            )["records"]
            assert listed[0]["id"] == saved["id"]
            assert listed[0]["tags"] == ["3中駒", "大駒ペア"]
            detail = app_module.get_research_kifu(
                room_id,
                saved["id"],
                app_module.ResearchKifuAuthRequest(admin_password="research-secret"),
            )["record"]
            assert detail["payload"]["round_index"] == 7
            updated = app_module.update_research_kifu_memo(
                room_id,
                saved["id"],
                app_module.ResearchKifuMemoUpdateRequest(
                    admin_password="research-secret",
                    memo="王を残す形も比較する",
                ),
            )["record"]
            assert updated["memo"] == "王を残す形も比較する"
            assert updated["payload"]["round_index"] == 7
            edited = app_module.update_research_kifu(
                room_id,
                saved["id"],
                app_module.ResearchKifuUpdateRequest(
                    admin_password="research-secret",
                    title="王を残す終盤",
                    memo="角上がりとの点数比較",
                    tags=["王玉", "ダブル狙い"],
                ),
            )["record"]
            assert edited["title"] == "王を残す終盤"
            assert edited["memo"] == "角上がりとの点数比較"
            assert edited["tags"] == ["王玉", "ダブル狙い"]
            assert edited["payload"]["round_index"] == 7
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
    assert "startResearchKifuEdit()" in HTML
    assert "saveResearchKifuEdit()" in HTML
    assert 'id="researchKifuTitleEditInput"' in HTML
    assert 'id="researchKifuSaveTags"' in HTML
    assert 'id="researchKifuEditTags"' in HTML
    assert 'id="researchKifuTagFilter"' in HTML
    assert '<h4 class="research-kifu-save-heading">棋譜をサーバーに保存</h4>' in HTML
    assert 'placeholder="タイトル　例：終盤の王受け"' in HTML
    assert 'placeholder="気になった点をメモできます"' in HTML
    assert '<strong class="research-kifu-list-heading">棋譜一覧</strong>' in HTML
    assert '<label for="researchKifuTitle">タイトル</label>' not in HTML
    assert '<label for="researchKifuMemo">研究メモ</label>' not in HTML
    assert '<h4 style="margin:0 0 10px; color:#6d461f;">研究用棋譜ライブラリ</h4>' not in HTML
    for tag in RESEARCH_KIFU_TAGS:
        assert f'"{tag}"' in HTML
    assert "← 棋譜一覧へ" in HTML
    assert "この配牌で対局</button>" in HTML
    assert "この配牌で対局する" not in HTML
    assert 'class="research-kifu-detail-title-line"' in HTML
    assert 'className = "research-kifu-item-title-line"' in HTML
    assert ">編集</button>" in HTML
    assert "deleteSelectedResearchKifu()" in HTML
    assert 'id="researchKifuBoard"' in HTML
    assert "buildResearchKifuFinalState(payload)" in HTML
    assert "renderResearchKifuBoard(payload)" in HTML
    assert "research-kifu-move-number" in HTML
    assert "薄い駒：伏せた駒・残った駒" not in HTML
    assert "初期手駒・手順" not in HTML


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
    print("RESEARCH_KIFU_LIBRARY_TEST_OK")

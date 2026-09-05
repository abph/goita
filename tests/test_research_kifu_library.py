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


def _valid_kifu_text():
    return '''version: 1.0
p0: "プレイヤーA"
p1: "プレイヤーB"
p2: "プレイヤーC"
p3: "プレイヤーD"
log:
- hand:
  p0: "しししし馬銀金玉"
  p1: "ししし香馬銀金王"
  p2: "しし香香馬金角飛"
  p3: "し香馬銀銀金角飛"
  uchidashi: 0
  score: [0,80]
  game:
  - ["0","し","し"]
  - ["1","し","銀"]
  - ["0","銀","し"]
  - ["1","し","金"]
  - ["2","金","飛"]
  - ["3","飛","銀"]
  - ["3","馬","金"]
  - ["3","角","銀"]
  - ["0","玉","し"]
  - ["1","し","王"]
  - ["1","馬","香"]
'''


def test_import_parser_replays_a_completed_downloaded_record():
    payload = app_module._parse_research_kifu_text(_valid_kifu_text())

    assert payload["dealer"] == "A"
    assert payload["winner"] == "B"
    assert payload["score_after"] == {"AC": 0, "BD": 80}
    assert payload["hand"]["p0"] == "しししし馬銀金玉"
    assert payload["game"][-1] == ["1", "馬", "香"]
    assert payload["player_names"]["A"] == "プレイヤーA"


def test_import_parser_rejects_an_impossible_piece_inventory():
    invalid = _valid_kifu_text().replace(
        'p3: "し香馬銀銀金角飛"',
        'p3: "しし馬銀銀金角飛"',
    )
    try:
        app_module._parse_research_kifu_text(invalid)
        raise AssertionError("an impossible piece inventory should be rejected")
    except ValueError as error:
        assert "32枚の駒構成" in str(error)


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




def test_settings_popup_contains_the_research_library_workflow():
    assert 'id="researchKifuTab"' not in HTML
    assert 'id="researchKifuPanel"' in HTML
    assert "unlockResearchKifuLibrary()" not in HTML
    assert 'id="memberKifuParking"' in HTML
    assert '/api/member/kifu' in HTML
    assert "saveCurrentResearchKifu(false)" in HTML
    assert "saveCurrentResearchKifu(true)" in HTML
    assert "async function saveCurrentResearchKifu(anonymous = false)" in HTML
    assert "openResearchKifu(record.id)" in HTML
    assert "applySelectedResearchKifu()" in HTML
    assert "startResearchKifuEdit()" in HTML
    assert "saveResearchKifuEdit()" in HTML
    assert 'id="researchKifuTitleEditInput"' in HTML
    assert 'id="researchKifuSaveTags"' in HTML
    assert 'id="researchKifuEditTags"' in HTML
    assert 'id="researchKifuTagFilter"' in HTML
    assert 'id="researchKifuImportButton"' in HTML
    assert 'id="researchKifuImportInput"' in HTML
    assert ">棋譜読込</button>" in HTML
    assert "openResearchKifuImport()" in HTML
    assert "importResearchKifuFile(event)" in HTML
    assert 'researchKifuApi("/import"' in HTML
    assert '<h4 id="memberKifuSaveTitle">棋譜をサーバーに保存</h4>' in HTML
    assert 'placeholder="タイトル　例：終盤の王受け"' in HTML
    assert 'placeholder="気になった点をメモできます"' in HTML
    assert '<strong class="research-kifu-list-heading">棋譜一覧</strong>' in HTML
    assert '<label for="researchKifuTitle">タイトル</label>' not in HTML
    assert '<label for="researchKifuMemo">研究メモ</label>' not in HTML
    assert '<h4 style="margin:0 0 10px; color:#6d461f;">研究用棋譜ライブラリ</h4>' not in HTML
    for tag in RESEARCH_KIFU_TAGS:
        assert f'"{tag}"' in HTML
    assert 'class="research-kifu-back-button"' in HTML
    assert '>←</button>' in HTML
    assert 'class="research-kifu-play-actions"' in HTML
    assert 'id="researchKifuReplayButton"' in HTML
    assert ">棋譜再生</button>" in HTML
    assert "toggleResearchKifuReplay()" in HTML
    assert "researchKifuReplayFrames(payload)" in HTML
    assert "researchKifuFaceDownLabel(seat, receive)" in HTML
    assert "attackNumbers.set(`${seat}:${attackIndex}`, attackSequenceNumber)" in HTML
    assert "attackNumbers.set(`${seat}:${attackIndex}`, rowIndex + 1)" not in HTML
    assert 'id="researchKifuApplyButton"' in HTML
    assert 'id="researchKifuEditButton"' in HTML
    assert 'id="researchKifuDownloadButton"' in HTML
    assert ">棋譜DL</button>" in HTML
    assert "downloadSelectedResearchKifu()" in HTML
    assert "researchKifuDownloadText(record)" in HTML
    assert "researchKifuDownloadFilename(record)" in HTML
    assert '<button class="danger" type="button" onclick="deleteSelectedResearchKifu()">削除</button>' in HTML
    assert "research-kifu-detail-heading" not in HTML
    assert "この配牌で対局</button>" in HTML
    assert "この配牌で対局する" not in HTML
    assert 'class="research-kifu-detail-title-line"' in HTML
    assert 'className = "research-kifu-item-title-line"' in HTML
    assert ">編集</button>" in HTML
    assert "deleteSelectedResearchKifu()" in HTML
    assert 'id="researchKifuBoard"' in HTML
    assert "buildResearchKifuFinalState(payload, rows)" in HTML
    assert "renderResearchKifuBoard(payload)" in HTML
    assert "research-kifu-move-number" in HTML
    assert "薄い駒：伏せた駒・残った駒" not in HTML
    assert "初期手駒・手順" not in HTML


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
    print("RESEARCH_KIFU_LIBRARY_TEST_OK")

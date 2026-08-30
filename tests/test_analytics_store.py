import sqlite3
from pathlib import Path

from backend.analytics_store import AnalyticsStore, resolve_analytics_path


def _event(**overrides):
    payload = {
        "analytics_id": "visitor_1234567890abcdef",
        "session_id": "session_1234567890abcdef",
        "event": "site_visit",
        "room_type": "lobby",
        "source": "direct",
        "device": "desktop",
        "language": "ja",
        "properties": {},
    }
    payload.update(overrides)
    return payload


def test_analytics_store_records_only_allowed_product_properties(tmp_path: Path) -> None:
    store = AnalyticsStore(tmp_path / "analytics.sqlite3")

    assert store.record_event(_event(prefecture="埼玉県")) is True
    assert store.record_event(_event(
        event="game_started",
        room_type="private",
        properties={
            "role": "host",
            "human_count": 2,
            "ai_count": 2,
            "pair_practice": True,
            "name": "記録してはいけない名前",
            "hand": "しししし",
        },
    )) is True

    snapshot = store.snapshot(days=30)
    assert snapshot["visitors"] == 1
    assert snapshot["game_started"] == 1
    assert snapshot["host_game_starts"] == 1
    assert snapshot["pair_practice_games"] == 1
    assert snapshot["regions"] == [{
        "prefecture": "埼玉県",
        "visitors": 1,
        "sessions": 1,
    }]
    started = next(
        event
        for session in snapshot["recent_sessions"]
        for event in session["events"]
        if event["event"] == "game_started"
    )
    assert started["properties"] == {
        "role": "host",
        "human_count": 2,
        "ai_count": 2,
        "pair_practice": True,
    }


def test_analytics_opt_out_deletes_browser_history(tmp_path: Path) -> None:
    store = AnalyticsStore(tmp_path / "analytics.sqlite3")
    payload = _event()
    assert store.record_event(payload) is True
    assert store.delete_visitor(payload["analytics_id"]) is True

    snapshot = store.snapshot(days=30)
    assert snapshot["visitors"] == 0
    assert snapshot["sessions"] == 0
    assert snapshot["recent_sessions"] == []


def test_analytics_rejects_unknown_events_and_resolves_persistent_path(tmp_path: Path) -> None:
    store = AnalyticsStore(tmp_path / "analytics.sqlite3")
    assert store.record_event(_event(event="arbitrary_payload")) is False
    assert resolve_analytics_path({"GOITA_PERSISTENT_DATA_DIR": "/var/data"}) == Path(
        "/var/data/goita-analytics.sqlite3"
    )


def test_analytics_rejects_untrusted_prefecture_values(tmp_path: Path) -> None:
    store = AnalyticsStore(tmp_path / "analytics.sqlite3")
    assert store.record_event(_event(prefecture="細かすぎる住所")) is True

    assert store.snapshot(days=30)["regions"] == [{
        "prefecture": "不明",
        "visitors": 1,
        "sessions": 1,
    }]


def test_existing_analytics_database_adds_prefecture_column(tmp_path: Path) -> None:
    database = tmp_path / "analytics.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE analytics_sessions (
                session_id TEXT PRIMARY KEY,
                analytics_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                ended_at TEXT,
                source TEXT NOT NULL DEFAULT '',
                medium TEXT NOT NULL DEFAULT '',
                campaign TEXT NOT NULL DEFAULT '',
                device TEXT NOT NULL DEFAULT 'unknown',
                language TEXT NOT NULL DEFAULT 'other',
                event_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )

    store = AnalyticsStore(database)
    store._ensure_schema()
    with sqlite3.connect(database) as connection:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(analytics_sessions)")
        }
    assert "prefecture" in columns

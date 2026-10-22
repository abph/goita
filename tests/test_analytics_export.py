import json
import sqlite3
import time
from datetime import date

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import app as game_app
from backend import analytics_store as analytics_module
from backend.analytics_store import AnalyticsStore


@pytest.fixture
def usage(tmp_path, monkeypatch):
    store = AnalyticsStore(tmp_path / "usage.sqlite3")

    def emit(when, visitor="visitor_private_12345678", session="session_private_12345678",
             event="site_visit", **extra):
        monkeypatch.setattr(analytics_module, "_utc_now_text", lambda: when)
        assert store.record_event({"analytics_id": visitor, "session_id": session,
                                   "event": event, "room_type": "public_ai",
                                   "device": "desktop", "language": "ja", **extra})
    return store, emit


def export(store):
    return store.export_usage(start_date=date(2026, 10, 20), end_date=date(2026, 10, 21))


def test_groups_repeat_visits_and_exports_only_allowed_information(usage):
    store, emit = usage
    emit("2026-10-10T03:00:00+00:00", session="session_old_123456789012")
    emit("2026-10-20T03:00:00+00:00", source="secret-source", campaign="secret-campaign",
         prefecture="東京都", country_code="JP")
    emit("2026-10-20T03:01:00+00:00", event="room_enter")
    emit("2026-10-20T03:02:00+00:00", event="game_started",
         properties={"role": "host", "human_count": 1, "ai_count": 3,
                     "name": "private-name", "hand": "private-hand"})
    emit("2026-10-20T03:10:00+00:00", event="game_completed")
    emit("2026-10-20T03:20:00+00:00", event="heartbeat")
    emit("2026-10-21T03:00:00+00:00", session="session_repeat_1234567890", event="room_enter")
    emit("2026-10-21T04:00:00+00:00", visitor="visitor_second_1234567890",
         session="session_second_1234567890", device="mobile", event="game_started",
         properties={"role": "joined", "pair_practice": True})
    data = export(store)
    assert data["format"] == "sorou-goita-usage"
    assert data["summary"]["visitors"] == 2
    assert data["summary"]["new_visitors"] == 1
    assert data["summary"]["returning_visitors"] == 1
    assert data["summary"]["sessions"] == 3
    assert data["summary"]["game_started"] == 2
    assert data["summary"]["game_completed"] == 1
    assert data["summary"]["host_game_starts"] == 1
    assert data["summary"]["joined_game_starts"] == 1
    assert data["summary"]["pair_practice_games"] == 1
    assert data["summary"]["room_entries"] == {"public_ai": 2}
    first, second = data["visitors"]
    assert first["sessions"] == 2 and second["sessions"] == 1
    assert first["duration_seconds_in_period"] == 1200
    assert data["sessions"][0]["visitor_number"] == data["sessions"][1]["visitor_number"]
    assert data["sessions"][2]["visitor_number"] != data["sessions"][0]["visitor_number"]
    encoded = json.dumps(data, ensure_ascii=False)
    for private in ("visitor_private_12345678", "session_private_12345678",
                    "visitor_second_1234567890", "session_second_1234567890",
                    "private-name", "private-hand", "secret-source", "secret-campaign", "東京都"):
        assert private not in encoded
    for key in ("analytics_id", "session_id", "source", "campaign", "prefecture", "country_code"):
        assert f'"{key}":' not in encoded
    assert len({session["session_number"] for session in data["sessions"]}) == 3


def test_export_contains_all_visits_beyond_recent_list_limit(usage):
    store, emit = usage
    for n in range(270):
        emit("2026-10-20T03:00:00+00:00", session=f"session_export_{n:012d}")
    assert len(store.snapshot(start_date=date(2026, 10, 20), end_date=date(2026, 10, 21),
                              recent_limit=100)["recent_sessions"]) == 100
    data = export(store)
    assert len(data["sessions"]) == data["summary"]["sessions"] == 270
    assert data["visitors"][0]["sessions"] == 270
    assert len(data["visitors"]) == 1


def test_jst_boundaries_and_cross_midnight_visits_are_explicit(usage):
    store, emit = usage
    emit("2026-10-19T14:50:00+00:00", session="session_carryover_1234567")
    emit("2026-10-19T15:10:00+00:00", session="session_carryover_1234567", event="room_enter")
    emit("2026-10-19T15:00:00+00:00", session="session_start_1234567890")
    emit("2026-10-20T14:59:59+00:00", session="session_end_123456789012")
    emit("2026-10-20T15:00:00+00:00", session="session_excluded_1234567")
    data = store.export_usage(start_date=date(2026, 10, 20), end_date=date(2026, 10, 20))
    assert data["timezone"] == "Asia/Tokyo"
    assert data["period_start"] == data["period_end"] == "2026-10-20"
    assert data["summary"]["sessions"] == 3
    assert data["summary"]["sessions_started_in_period"] == 2
    carried = data["sessions"][0]
    assert carried["started_in_period"] is False
    assert carried["duration_seconds_in_period"] == 600
    assert [event["event"] for event in carried["events"]] == ["room_enter"]
    assert all(event["occurred_at"] < "2026-10-20T15:00:00+00:00"
               for session in data["sessions"] for event in session["events"])


def test_export_resanitizes_legacy_properties_and_respects_deletion(usage):
    store, emit = usage
    emit("2026-10-20T03:00:00+00:00")
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE analytics_events SET properties_json = ?", (json.dumps({
            "name": "secret-player", "raw_ip": "secret-ip", "role": "secret-role",
            "human_count": 99, "ai_count": 0, "chat": "secret-chat"}),))
        counts_before = db.execute("SELECT COUNT(*) FROM analytics_events").fetchone()[0]
    data = export(store)
    assert "secret-" not in json.dumps(data)
    assert data["sessions"][0]["events"][0]["properties"] == {"human_count": 4, "ai_count": 0}
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT COUNT(*) FROM analytics_events").fetchone()[0] == counts_before
    assert store.delete_visitor("visitor_private_12345678")
    assert export(store)["sessions"] == []
    assert export(store)["visitors"] == []


@pytest.fixture
def export_client(usage, monkeypatch):
    store, _ = usage
    monkeypatch.setattr(game_app, "ANALYTICS_STORE", store)
    app = FastAPI()
    app.get("/admin/api/analytics/export")(game_app.admin_analytics_export)
    return TestClient(app)


def login(client):
    client.cookies.set(game_app.ADMIN_SESSION_COOKIE,
                       game_app._admin_session_token(int(time.time()) + 60))


def test_only_site_admin_can_download_and_response_is_not_cached(export_client, usage):
    client = export_client
    store, emit = usage
    emit("2026-10-20T03:00:00+00:00")
    url = "/admin/api/analytics/export?start_date=2026-10-20&end_date=2026-10-21"
    assert client.get(url).status_code == 401
    client.cookies.set(game_app.ADMIN_SESSION_COOKIE, "forged")
    assert client.get(url).status_code == 401
    login(client)
    response = client.get(url)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert "goita-usage-2026-10-20-2026-10-21.json" in response.headers["content-disposition"]
    assert response.json() == export(store)


@pytest.mark.parametrize("query,expected", [
    ("start_date=2026-10-20", 400),
    ("end_date=2026-10-21", 400),
    ("start_date=2026-10-21&end_date=2026-10-20", 400),
    ("start_date=2000-01-01&end_date=2026-10-21", 400),
    ("start_date=invalid&end_date=2026-10-21", 422),
])
def test_export_validates_periods(export_client, query, expected):
    login(export_client)
    assert export_client.get("/admin/api/analytics/export?" + query).status_code == expected

import sqlite3
import time
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from backend import app as game_app
from backend.analytics_store import AnalyticsStore
from backend.regional_ad_report import build_regional_ad_report


# 00:30 on November 6 in Japan; the UTC date is still November 5.
NOW = datetime(2026, 11, 5, 15, 30, tzinfo=timezone.utc)
ADS = [
    dict(id="report-ad-a", title="はまりばカフェ", enabled=True,
         starts_at="2026-09-01T00:00:00+09:00", ends_at="2026-12-01T00:00:00+09:00"),
    dict(id="report-ad-b", title="停止中の広告", enabled=False,
         starts_at="2026-09-01T00:00:00+09:00", ends_at="2026-12-01T00:00:00+09:00"),
    dict(id="report-ad-zero", title="自動スライド経由", enabled=True,
         starts_at="2026-09-01T00:00:00+09:00", ends_at="2026-11-01T00:00:00+09:00"),
]


@pytest.fixture
def store(tmp_path):
    store = AnalyticsStore(tmp_path / "analytics.sqlite3")
    def add(ad_id, day, impressions, clicks, region="関東", surface="public_room"):
        now = datetime.fromisoformat(day + "T12:00:00+09:00")
        if impressions:
            assert store.record_regional_ad_metric(ad_id, region, "impression", surface=surface,
                viewer_measurement=True, viewer_first=True, surface_viewer_first=True, now=now)
        if clicks:
            assert store.record_regional_ad_metric(ad_id, region, "click", surface=surface, now=now)
        with sqlite3.connect(store.path) as db:
            db.execute("UPDATE regional_ad_daily_metrics SET impressions=?, clicks=?, measured_impressions=? WHERE ad_id=? AND metric_date=? AND region=?",
                       (impressions, clicks, impressions, ad_id, day, region))
            db.execute("UPDATE regional_ad_surface_daily_metrics SET impressions=?, clicks=?, measured_impressions=? WHERE ad_id=? AND metric_date=? AND region=? AND surface=?",
                       (impressions, clicks, impressions, ad_id, day, region, surface))
    for day, impressions, clicks in [
        ("2026-09-01", 1000, 2), ("2026-10-07", 100, 1), ("2026-10-08", 100, 2),
        ("2026-10-30", 50, 0), ("2026-10-31", 10, 1), ("2026-11-06", 10, 1),
    ]:
        add("report-ad-a", day, impressions, clicks)
    add("report-ad-a", "2026-11-06", 5, 1, "関西", "score_attack")
    add("report-ad-b", "2026-11-06", 100, 0)
    add("report-ad-zero", "2026-11-06", 0, 3)
    # Future records must not leak into a report ending today.
    add("report-ad-a", "2026-11-07", 999, 9)
    # An old recording may not have a display-surface breakdown at all.
    with sqlite3.connect(store.path) as db:
        db.execute("DELETE FROM regional_ad_surface_daily_metrics WHERE metric_date='2026-09-01'")
    return store


@pytest.mark.parametrize("period,start,impressions,clicks", [
    ("7", "2026-10-31", 125, 6), ("30", "2026-10-08", 275, 8),
    ("all", "2026-09-01", 1375, 11),
])
def test_periods_use_japanese_calendar_boundaries_and_weighted_click_rate(store, period, start, impressions, clicks):
    report = build_regional_ad_report(store, ADS, period, now=NOW)
    assert report["period_start"] == start
    assert report["period_end"] == "2026-11-06"
    assert report["summary"] == dict(impressions=impressions, clicks=clicks, click_rate=round(clicks / impressions * 100, 2))
    assert {item["status"] for item in report["items"]} == {"掲載中", "停止中", "終了"}
    assert all(row["date"] <= "2026-11-06" for item in report["items"] for row in item["daily"])


def test_daily_series_merge_regions_without_double_counting_surfaces(store):
    report = build_regional_ad_report(store, ADS, "7", now=NOW)
    item = report["items"][0]
    assert item["daily"] == [
        dict(date="2026-10-31", impressions=10, clicks=1),
        dict(date="2026-11-06", impressions=15, clicks=2),
    ]
    assert item["impressions"] == sum(row["impressions"] for row in item["daily"]) == 25
    assert item["clicks"] == sum(row["clicks"] for row in item["regions"]) == 3
    assert sum(row["impressions"] for row in item["surfaces"]) == 25
    assert item["click_rate"] == 12.0
    # These are first sightings, not a purported active-user count.
    assert item["viewers"] == 3


def test_all_period_includes_legacy_totals_without_inventing_surface_data(store):
    item = build_regional_ad_report(store, ADS, "all", now=NOW)["items"][0]
    assert item["daily"][0]["date"] == "2026-09-01"
    assert item["impressions"] == 1275
    assert sum(row["impressions"] for row in item["surfaces"]) == 275


def test_zero_impressions_have_no_fabricated_ctr(store):
    item = build_regional_ad_report(store, ADS, "7", now=NOW)["items"][2]
    assert item["clicks"] == 3 and item["impressions"] == 0 and item["click_rate"] is None
    assert item["regions"][0]["click_rate"] is None


def test_empty_ads_and_unused_ad_are_included_as_zero_without_exposing_other_ids(store):
    empty = build_regional_ad_report(store, [], "all", now=NOW)
    assert empty["items"] == [] and empty["period_start"] is None
    assert empty["summary"] == dict(impressions=0, clicks=0, click_rate=None)
    ad = dict(ADS[0], id="report-new-ad", title="新しい広告")
    report = build_regional_ad_report(store, [ad], "30", now=NOW)
    assert len(report["items"]) == 1 and report["items"][0]["daily"] == []
    assert report["summary"]["impressions"] == 0


def test_invalid_period_is_rejected_and_report_does_not_modify_counters(store):
    with pytest.raises(HTTPException):
        build_regional_ad_report(store, ADS, "90", now=NOW)
    with sqlite3.connect(store.path) as db:
        before = db.execute("SELECT * FROM regional_ad_daily_metrics ORDER BY ad_id,metric_date,region").fetchall()
    build_regional_ad_report(store, ADS, "30", now=NOW)
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT * FROM regional_ad_daily_metrics ORDER BY ad_id,metric_date,region").fetchall() == before


def test_old_metrics_call_keeps_its_legacy_all_time_contract(store):
    old = store.regional_ad_metrics(["report-ad-a"], now=NOW)["report-ad-a"]
    assert old["impressions"] == 2274  # Existing unbounded callers remain unbounded.
    assert len(old["daily"]) == 6


def test_admin_only_endpoint_and_validation(store, monkeypatch):
    app = FastAPI()
    app.get("/admin/api/regional-ads/report")(game_app.admin_regional_ad_report)
    monkeypatch.setattr(game_app, "ANALYTICS_STORE", store)
    monkeypatch.setattr(game_app, "REGIONAL_AD_SETTINGS", ADS)
    client = TestClient(app)
    assert client.get("/admin/api/regional-ads/report").status_code == 401
    client.cookies.set(game_app.ADMIN_SESSION_COOKIE, game_app._admin_session_token(int(time.time()) + 60))
    response = client.get("/admin/api/regional-ads/report?period=7")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store" and response.json()["period"] == "7"
    assert client.get("/admin/api/regional-ads/report?period=90").status_code == 400

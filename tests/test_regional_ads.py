from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from backend import app as app_module
from backend.analytics_geo import PREFECTURES_BY_CODE
from backend.analytics_store import AnalyticsStore
from backend.regional_ads import PREFECTURE_REGIONS, infer_region, normalize_ads, select_ad
from backend.room_settings_persistence import load_room_settings


NOW = datetime(2026, 9, 22, 3, 0, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[1]


def ad(ad_id: str, regions: list[str], **changes: object) -> dict:
    result = {
        "id": ad_id,
        "title": ad_id,
        "message": "本文",
        "url": "https://example.com/",
        "regions": regions,
        "starts_at": "2026-09-22T00:00:00+00:00",
        "ends_at": "2026-09-23T00:00:00+00:00",
        "enabled": True,
    }
    result.update(changes)
    return result


def test_every_prefecture_is_assigned_and_edge_headers_select_region() -> None:
    assert set(PREFECTURE_REGIONS) == set(PREFECTURES_BY_CODE.values())
    assert infer_region({"cf-ipcountry": "JP", "cf-region-code": "JP-13"}) == "関東"
    assert infer_region({"cf-ipcountry": "JP", "cf-region-code": "JP-27"}) == "関西"
    assert infer_region({"cf-ipcountry": "US", "cf-region-code": "JP-13"}) == ""
    assert infer_region({}) == ""


def test_regional_ad_takes_priority_and_unknown_location_uses_common_ad() -> None:
    ads = normalize_ads([ad("common-ad", []), ad("kanto-ad", ["関東"])])
    assert select_ad(ads, "関東", now=NOW)["id"] == "kanto-ad"
    assert select_ad(ads, "関西", now=NOW)["id"] == "common-ad"
    assert select_ad(ads, "", now=NOW)["id"] == "common-ad"
    assert select_ad(ads, "関東", now=datetime(2026, 9, 24, tzinfo=timezone.utc)) is None


def test_overlapping_enabled_ads_in_same_region_are_rejected() -> None:
    with pytest.raises(HTTPException) as error:
        normalize_ads([ad("first-ad", ["関東"]), ad("second-ad", ["関東", "関西"])])
    assert error.value.status_code == 400
    normalize_ads([ad("first-ad", ["関東"]), ad("second-ad", ["関東"], enabled=False)])
    normalize_ads([ad("first-ad", ["関東"]), ad("second-ad", ["関東"], starts_at="2026-09-23T00:00:00+00:00", ends_at="2026-09-24T00:00:00+00:00")])


def test_invalid_region_payload_is_rejected_cleanly() -> None:
    with pytest.raises(HTTPException) as error:
        normalize_ads([ad("broken-ad", [["関東"]])])
    assert error.value.status_code == 400


def test_admin_save_persists_and_public_endpoint_uses_region(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "REGIONAL_AD_SETTINGS", [])
    monkeypatch.setattr(app_module, "ROOM_SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(app_module, "_require_site_admin", lambda _request: None)
    monkeypatch.setattr(app_module, "ANALYTICS_STORE", AnalyticsStore(tmp_path / "analytics.sqlite3"))
    request = Request({"type": "http", "method": "GET", "path": "/api/regional-ad", "headers": [(b"cf-ipcountry", b"JP"), (b"cf-region-code", b"JP-13")], "query_string": b""})
    now = datetime.now(timezone.utc)
    period = {"starts_at": (now - timedelta(days=1)).isoformat(), "ends_at": (now + timedelta(days=1)).isoformat()}
    saved = app_module.admin_update_regional_ads(request, {"ads": [ad("common-ad", [], **period), ad("kanto-ad", ["関東"], **period)]})
    assert len(saved["ads"]) == 2
    persisted = load_room_settings(app_module.ROOM_SETTINGS_PATH)[app_module.LOBBY_SETTINGS_STORAGE_KEY]
    assert len(persisted["regional_ads"]) == 2
    response = app_module.public_regional_ad(request)
    assert response.headers["cache-control"] == "no-store"
    assert b'"id":"kanto-ad"' in response.body
    app_module.REGIONAL_AD_SETTINGS[:] = []
    app_module._apply_lobby_management_settings({"regional_ads": persisted["regional_ads"]})
    assert len(app_module.REGIONAL_AD_SETTINGS) == 2


def test_public_metric_endpoint_records_active_ad_and_admin_returns_totals(tmp_path, monkeypatch) -> None:
    store = AnalyticsStore(tmp_path / "analytics.sqlite3")
    now = datetime.now(timezone.utc)
    period = {
        "starts_at": (now - timedelta(days=1)).isoformat(),
        "ends_at": (now + timedelta(days=1)).isoformat(),
    }
    monkeypatch.setattr(
        app_module,
        "REGIONAL_AD_SETTINGS",
        normalize_ads([ad("kanto-ad", ["関東"], **period)]),
    )
    monkeypatch.setattr(app_module, "ANALYTICS_STORE", store)
    monkeypatch.setattr(app_module, "_require_site_admin", lambda _request: None)
    request = Request({
        "type": "http",
        "method": "POST",
        "path": "/api/regional-ad/metric",
        "headers": [(b"cf-ipcountry", b"JP"), (b"cf-region-code", b"JP-13")],
        "query_string": b"",
    })

    assert app_module.record_regional_ad_metric(
        app_module.RegionalAdMetricRequest(ad_id="kanto-ad", event="impression"),
        request,
    ) == {"ok": True}
    assert app_module.record_regional_ad_metric(
        app_module.RegionalAdMetricRequest(ad_id="kanto-ad", event="click"),
        request,
    ) == {"ok": True}
    payload = app_module.admin_regional_ads(request)
    assert payload["metrics_recent_days"] == 30
    assert payload["metrics"]["kanto-ad"]["impressions"] == 1
    assert payload["metrics"]["kanto-ad"]["clicks"] == 1
    assert payload["metrics"]["kanto-ad"]["click_rate"] == 100.0

    with pytest.raises(HTTPException) as error:
        app_module.record_regional_ad_metric(
            app_module.RegionalAdMetricRequest(ad_id="another-ad", event="click"),
            request,
        )
    assert error.value.status_code == 400


def test_regional_ad_frontend_reports_and_admin_displays_metrics() -> None:
    script = (ROOT / "frontend" / "regionalAds.js").read_text(encoding="utf-8")
    admin = (ROOT / "frontend" / "admin.html").read_text(encoding="utf-8")
    assert 'recordMetric(ad.id, "impression")' in script
    assert 'recordMetric(ad.dataset.adId, "click", true)' in script
    assert '"/api/regional-ad/metric"' in script
    assert "表示回数・クリック回数・クリック率" in admin
    assert "CTR" in admin
    assert "regionalAdMetrics" in admin


def test_admin_save_failure_restores_previous_ads(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "REGIONAL_AD_SETTINGS", normalize_ads([ad("first-ad", [])]))
    monkeypatch.setattr(app_module, "ROOM_SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(app_module, "_require_site_admin", lambda _request: None)
    monkeypatch.setattr(app_module, "_save_persisted_room_management_settings", lambda: False)
    with pytest.raises(HTTPException) as error:
        app_module.admin_update_regional_ads(None, {"ads": [ad("second-ad", [])]})
    assert error.value.status_code == 500
    assert app_module.REGIONAL_AD_SETTINGS[0]["id"] == "first-ad"

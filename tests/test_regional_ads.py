from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from backend import app as app_module
from backend.analytics_geo import PREFECTURES_BY_CODE
from backend.analytics_store import AnalyticsStore
from backend.regional_ads import PREFECTURE_REGIONS, infer_region, normalize_ads, select_ad, select_ads
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


def test_carousel_selects_up_to_three_regional_ads_before_common_ads() -> None:
    ads = normalize_ads([
        ad("common-01", []),
        ad("kanto-01", ["関東"]),
        ad("common-02", []),
        ad("kanto-02", ["関東"]),
    ])
    assert [item["id"] for item in select_ads(ads, "関東", now=NOW)] == [
        "kanto-01", "kanto-02", "common-01",
    ]
    assert [item["id"] for item in select_ads(ads, "関西", now=NOW)] == [
        "common-01", "common-02",
    ]


def test_regional_ads_can_target_public_rooms_and_score_attack_separately() -> None:
    ads = normalize_ads([
        ad("public-ad", ["関東"], surfaces=["public_room"]),
        ad("score-ad", ["関東"], surfaces=["score_attack"]),
    ])
    assert select_ad(ads, "関東", surface="public_room", now=NOW)["id"] == "public-ad"
    assert select_ad(ads, "関東", surface="score_attack", now=NOW)["id"] == "score-ad"
    assert select_ad(ads, "関東", surface="unknown", now=NOW) is None


def test_existing_regional_ads_default_to_public_rooms() -> None:
    normalized = normalize_ads([ad("public-ad", [])])
    assert normalized[0]["surfaces"] == ["public_room"]
    assert select_ad(normalized, "", surface="score_attack", now=NOW) is None


def test_up_to_three_overlapping_enabled_ads_are_allowed() -> None:
    normalize_ads([
        ad("first-ad", ["関東"]),
        ad("second-ad", ["関東", "関西"]),
        ad("third-ad", ["関東"]),
    ])
    with pytest.raises(HTTPException) as error:
        normalize_ads([
            ad("first-ad", ["関東"]),
            ad("second-ad", ["関東", "関西"]),
            ad("third-ad", ["関東"]),
            ad("fourth-ad", ["関東"]),
        ])
    assert error.value.status_code == 400
    assert "3件まで" in error.value.detail
    normalize_ads([ad("first-ad", ["関東"]), ad("second-ad", ["関東"], enabled=False)])
    normalize_ads([ad("first-ad", ["関東"]), ad("second-ad", ["関東"], starts_at="2026-09-23T00:00:00+00:00", ends_at="2026-09-24T00:00:00+00:00")])
    normalize_ads([
        ad("first-ad", ["関東"], surfaces=["public_room"]),
        ad("second-ad", ["関東"], surfaces=["score_attack"]),
    ])


def test_invalid_region_payload_is_rejected_cleanly() -> None:
    with pytest.raises(HTTPException) as error:
        normalize_ads([ad("broken-ad", [["関東"]])])
    assert error.value.status_code == 400
    with pytest.raises(HTTPException) as error:
        normalize_ads([ad("broken-ad", ["関東"], surfaces=["score_attack", "unknown"])])
    assert error.value.status_code == 400


def test_admin_save_persists_and_public_endpoint_uses_region(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "REGIONAL_AD_SETTINGS", [])
    monkeypatch.setattr(app_module, "REGIONAL_AD_POSITION", "top")
    monkeypatch.setattr(app_module, "ROOM_SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(app_module, "_require_site_admin", lambda _request: None)
    monkeypatch.setattr(app_module, "ANALYTICS_STORE", AnalyticsStore(tmp_path / "analytics.sqlite3"))
    request = Request({"type": "http", "method": "GET", "path": "/api/regional-ad", "headers": [(b"cf-ipcountry", b"JP"), (b"cf-region-code", b"JP-13")], "query_string": b""})
    now = datetime.now(timezone.utc)
    period = {"starts_at": (now - timedelta(days=1)).isoformat(), "ends_at": (now + timedelta(days=1)).isoformat()}
    saved = app_module.admin_update_regional_ads(request, {
        "ads": [ad("common-ad", [], **period), ad("kanto-ad", ["関東"], **period)],
        "position": "bottom",
    })
    assert len(saved["ads"]) == 2
    assert saved["position"] == "bottom"
    persisted = load_room_settings(app_module.ROOM_SETTINGS_PATH)[app_module.LOBBY_SETTINGS_STORAGE_KEY]
    assert len(persisted["regional_ads"]) == 2
    assert persisted["regional_ad_position"] == "bottom"
    response = app_module.public_regional_ad(request)
    assert response.headers["cache-control"] == "no-store"
    assert b'"id":"kanto-ad"' in response.body
    assert b'"ads":[' in response.body
    assert b'"position":"bottom"' in response.body
    app_module.REGIONAL_AD_SETTINGS[:] = []
    app_module.REGIONAL_AD_POSITION = "top"
    app_module._apply_lobby_management_settings(persisted)
    assert len(app_module.REGIONAL_AD_SETTINGS) == 2
    assert app_module.REGIONAL_AD_POSITION == "bottom"


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
        normalize_ads([ad("kanto-ad", ["関東"], surfaces=["public_room", "score_attack"], **period)]),
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
        app_module.RegionalAdMetricRequest(
            ad_id="kanto-ad",
            event="impression",
            surface="score_attack",
            viewer_measurement=True,
            viewer_first=True,
            surface_viewer_first=True,
        ),
        request,
    ) == {"ok": True}
    assert app_module.record_regional_ad_metric(
        app_module.RegionalAdMetricRequest(ad_id="kanto-ad", event="click", surface="score_attack"),
        request,
    ) == {"ok": True}
    payload = app_module.admin_regional_ads(request)
    assert payload["metrics_recent_days"] == 30
    assert payload["metrics"]["kanto-ad"]["impressions"] == 1
    assert payload["metrics"]["kanto-ad"]["clicks"] == 1
    assert payload["metrics"]["kanto-ad"]["viewers"] == 1
    assert payload["metrics"]["kanto-ad"]["measured_impressions"] == 1
    assert payload["metrics"]["kanto-ad"]["impressions_per_viewer"] == 1.0
    assert payload["metrics"]["kanto-ad"]["click_rate"] == 100.0
    assert payload["metrics"]["kanto-ad"]["surfaces"] == [{
        "surface": "score_attack",
        "impressions": 1,
        "clicks": 1,
        "viewers": 1,
        "measured_impressions": 1,
        "click_rate": 100.0,
        "impressions_per_viewer": 1.0,
    }]

    with pytest.raises(HTTPException) as error:
        app_module.record_regional_ad_metric(
            app_module.RegionalAdMetricRequest(ad_id="another-ad", event="click"),
            request,
        )
    assert error.value.status_code == 400


def test_metric_endpoint_accepts_every_ad_in_current_carousel(tmp_path, monkeypatch) -> None:
    now = datetime.now(timezone.utc)
    period = {
        "starts_at": (now - timedelta(days=1)).isoformat(),
        "ends_at": (now + timedelta(days=1)).isoformat(),
    }
    monkeypatch.setattr(
        app_module,
        "REGIONAL_AD_SETTINGS",
        normalize_ads([
            ad("kanto-01", ["関東"], **period),
            ad("kanto-02", ["関東"], **period),
            ad("kanto-03", ["関東"], **period),
        ]),
    )
    monkeypatch.setattr(app_module, "ANALYTICS_STORE", AnalyticsStore(tmp_path / "analytics.sqlite3"))
    request = Request({
        "type": "http",
        "method": "POST",
        "path": "/api/regional-ad/metric",
        "headers": [(b"cf-ipcountry", b"JP"), (b"cf-region-code", b"JP-13")],
        "query_string": b"",
    })
    assert app_module.record_regional_ad_metric(
        app_module.RegionalAdMetricRequest(ad_id="kanto-03", event="impression"),
        request,
    ) == {"ok": True}


def test_regional_ad_frontend_reports_and_admin_displays_metrics() -> None:
    script = (ROOT / "frontend" / "regionalAds.js").read_text(encoding="utf-8")
    admin = (ROOT / "frontend" / "admin.html").read_text(encoding="utf-8")
    index = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert 'recordMetric(ad.id, "impression", currentSurface, false' in script
    assert "const DAILY_LIMIT = 3" in script
    assert "const AUTO_SLIDE_MS = 3 * 60 * 1000" in script
    assert "rotateStartingAd" in script
    assert "shownInContext.has(ad.id)" in script
    assert "viewer_first: exposure.viewerFirst" in script
    assert "viewer_measurement: true" in script
    assert "surface_viewer_first: exposure.surfaceViewerFirst" in script
    assert 'recordMetric(link.dataset.adId, "click", currentSurface, true)' in script
    assert '"/api/regional-ad/metric"' in script
    assert "表示場所別・地方別・直近30日" in admin
    assert 'value="score_attack"' in admin
    assert "CTR" in admin
    assert "閲覧ブラウザー" in admin
    assert "regionalAdMetrics" in admin
    assert ".regional-ad strong { display:block; font-size:12px; }" in index
    assert ".regional-ad p { margin:4px 0 0; font-size:13px;" in index
    assert 'openMark.className = "regional-ad-open"' in script
    assert 'id="regionalAdTrack"' in index
    assert 'id="regionalAdDots"' in index
    assert 'id="regionalAdTopAnchor"' in index
    assert 'id="regionalAdBottomAnchor"' in index
    assert 'id="regionalAdPosition"' in admin
    assert 'id="regionalAdPositionSave"' in admin
    assert 'id="regionalAdCarouselSets"' in admin
    assert "function renderRegionalAdCarouselSets()" in admin
    assert "あと${remaining}件、同じ条件で広告を登録できます。" in admin
    assert "3件の横スライド広告が設定されています。" in admin
    assert "applyPosition(data.position)" in script
    assert 'state.trace_attempt_id || "active"' in index
    assert '"score_attack"' in index


def test_admin_save_failure_restores_previous_ads(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "REGIONAL_AD_SETTINGS", normalize_ads([ad("first-ad", [])]))
    monkeypatch.setattr(app_module, "REGIONAL_AD_POSITION", "top")
    monkeypatch.setattr(app_module, "ROOM_SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(app_module, "_require_site_admin", lambda _request: None)
    monkeypatch.setattr(app_module, "_save_persisted_room_management_settings", lambda: False)
    with pytest.raises(HTTPException) as error:
        app_module.admin_update_regional_ads(None, {
            "ads": [ad("second-ad", [])],
            "position": "bottom",
        })
    assert error.value.status_code == 500
    assert app_module.REGIONAL_AD_SETTINGS[0]["id"] == "first-ad"
    assert app_module.REGIONAL_AD_POSITION == "top"


def test_invalid_regional_ad_position_is_rejected(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "REGIONAL_AD_SETTINGS", [])
    monkeypatch.setattr(app_module, "REGIONAL_AD_POSITION", "top")
    monkeypatch.setattr(app_module, "ROOM_SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(app_module, "_require_site_admin", lambda _request: None)
    with pytest.raises(HTTPException) as error:
        app_module.admin_update_regional_ads(None, {"ads": [], "position": "center"})
    assert error.value.status_code == 400
    assert app_module.REGIONAL_AD_POSITION == "top"

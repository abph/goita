import io
import copy
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.message import Message

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from backend import app as game_app
from backend import twipla_import as twipla
from backend.regional_ads import normalize_ads, select_ads


URL = "https://twipla.jp/events/747979"
NOW = datetime(2026, 11, 5, 0, 0, tzinfo=timezone.utc)


def event_html(*, dates="20261122T020000Z/20261122T020000Z", detail=None, title="はまりばカフェ ごいた会"):
    if detail is None:
        detail = ('<p><span>初心者歓迎。</span>ルールから説明します。</p>'
                  '<p><span>日時</span><span>：2026年11月22日(日) 11:00～16:30</span></p>')
    return f'''<!doctype html><meta property="og:title" content="{title} - TwiPla">
      <meta property="og:description" content="途中で切れたメタデータの文">
      <div class="arrow_box"><span>2026<span>年</span>11月22日[日] 11:00</span><h1>ごいた会</h1></div>
      <a href="http://www.google.com/calendar/event?action=TEMPLATE&amp;dates={dates}&amp;location=%E3%81%AF%E3%81%BE%E3%82%8A%E3%81%B0%E3%82%AB%E3%83%95%E3%82%A7">カレンダー</a>
      <div class="desc content_width" id="desc">{detail}</div>
      <div>参加者のコメント：2026年11月22日 11:00～23:00</div>'''


@pytest.mark.parametrize("raw", [URL, URL + "/", URL + "?utm_source=site#info", "http://www.twipla.jp/events/747979"])
def test_event_url_is_canonicalized(raw):
    assert twipla.normalize_event_url(raw) == URL


@pytest.mark.parametrize("raw", [None, 123, "", "http://127.0.0.1/events/747979", "file:///etc/passwd",
    "https://twipla.jp.evil.example/events/747979", "https://twipla.jp@evil.example/events/747979",
    "https://twipla.jp:8443/events/747979", "https://twipla.jp/users/someone", URL + "/.ics",
    "https://twipla.jp/events/../admin", "https://twipla.jp/events/%31", URL + "\r\nX-Test: value",
    "https://twipla.jp\\@localhost/events/747979"])
def test_only_public_twipla_event_urls_are_accepted(raw):
    with pytest.raises(HTTPException) as error:
        twipla.normalize_event_url(raw)
    assert error.value.status_code == 400


def test_live_page_structure_imports_end_from_body_not_zero_length_calendar():
    result = twipla.parse_event_html(event_html(), URL, now=NOW)
    event, draft = result["event"], result["draft"]
    assert event["starts_at"] == "2026-11-22T11:00:00+09:00"
    assert event["ends_at"] == "2026-11-22T16:30:00+09:00"
    assert event["venue"] == "はまりばカフェ" and not event["ended"]
    assert draft["starts_at"] == "2026-11-05T09:00:00+09:00"  # Advertisement starts now.
    assert draft["ends_at"] == event["ends_at"] and draft["url"] == URL
    assert "初心者歓迎。" in draft["message"] and "途中で切れた" not in draft["message"]
    assert "参加者" not in json.dumps(result, ensure_ascii=False)
    assert result["warnings"] == []
    saved = normalize_ads([dict(id="imported-001", regions=["関東"], enabled=True, **draft)])
    assert select_ads(saved, "関東", now=NOW)
    assert not select_ads(saved, "関東", now=datetime(2026, 11, 22, 7, 30, tzinfo=timezone.utc))


def test_real_calendar_end_is_used_when_present():
    data = twipla.parse_event_html(event_html(dates="20261122T020000Z/20261122T073000Z"), URL, now=NOW)
    assert data["event"]["ends_at"] == "2026-11-22T16:30:00+09:00"


def test_missing_end_is_not_guessed_from_comments_or_mismatched_body_dates():
    html = event_html(detail='<p>2025年11月22日 11:00～23:00</p><p>開催時間：10:00～18:00</p>')
    data = twipla.parse_event_html(html, URL, now=NOW)
    assert data["draft"]["ends_at"] is None
    assert any("掲載終了を入力" in warning for warning in data["warnings"])


def test_header_fallback_and_next_day_time():
    html = event_html(detail='<p>日時：2026年11月22日 11:00～翌日01:00</p><p>会場：テスト会場</p>')
    html = html.replace("/calendar/event", "/not-a-calendar")
    data = twipla.parse_event_html(html, URL, now=NOW)
    assert data["event"]["starts_at"] == "2026-11-22T11:00:00+09:00"
    assert data["event"]["ends_at"] == "2026-11-23T01:00:00+09:00"
    assert data["event"]["venue"] == "テスト会場"


def test_ended_event_is_marked_and_draft_remains_valid_but_expired():
    data = twipla.parse_event_html(event_html(), URL, now=datetime(2026, 12, 1, tzinfo=timezone.utc))
    assert data["event"]["ended"]
    assert any("終了しています" in warning for warning in data["warnings"])
    saved = normalize_ads([dict(id="imported-001", regions=[], enabled=True, **data["draft"])])
    assert not select_ads(saved, "", now=datetime(2026, 12, 1, tzinfo=timezone.utc))


def test_unreadable_page_requires_manual_review_instead_of_fabricating_dates():
    with pytest.raises(HTTPException):
        twipla.parse_event_html('<title>ログインしてください</title>', URL, now=NOW)
    data = twipla.parse_event_html('<meta property="og:title" content="日時未定の会"><div id="desc">開催を準備中です。</div>', URL, now=NOW)
    assert data["event"]["starts_at"] is None and data["draft"]["ends_at"] is None
    assert len(data["warnings"]) == 3


def test_plain_text_and_length_limits_prevent_importing_markup_or_long_full_posts():
    html = event_html(title="長いタイトル" * 15,
        detail='<p>本文。<script>秘密を表示()</script><img src=x onerror=alert(1)>' + "説明" * 300 + '</p>')
    data = twipla.parse_event_html(html, URL, now=NOW)
    assert len(data["draft"]["title"]) <= 40 and len(data["draft"]["message"]) <= 200
    assert "<" not in data["draft"]["message"] and "秘密を表示" not in data["draft"]["message"]
    assert len(data["event"]["title"]) > 40


@pytest.mark.parametrize("destination", ["https://localhost/events/747979", "https://twipla.jp.evil/events/747979",
    "http://twipla.jp/events/747979", "https://twipla.jp/events/123", "https://twipla.jp/accounts/login"])
def test_redirects_cannot_leave_the_requested_https_event(destination):
    handler = twipla._EventRedirect(URL)
    with pytest.raises(HTTPException) as error:
        handler.redirect_request(urllib.request.Request(URL), None, 302, "redirect", {}, destination)
    assert error.value.status_code == 502


def test_same_event_www_redirect_is_supported():
    request = twipla._EventRedirect(URL).redirect_request(
        urllib.request.Request(URL), None, 302, "redirect", {}, "https://www.twipla.jp/events/747979")
    assert request.full_url == "https://www.twipla.jp/events/747979"


class PageResponse(io.BytesIO):
    def __init__(self, body, content_type="text/html"):
        super().__init__(body)
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def geturl(self):
        return URL


def test_fetcher_limits_response_size_and_requires_html(monkeypatch):
    response = PageResponse(event_html().encode())
    calls = []
    def open_page(request, timeout):
        calls.append((request.full_url, timeout))
        return response
    monkeypatch.setattr(twipla.urllib.request, "build_opener", lambda handler: type("Opener", (), {"open": staticmethod(open_page)})())
    assert "カレンダー" in twipla.fetch_event_html(URL)
    assert calls == [(URL, 5)]
    for response in (PageResponse(b"x" * (twipla.MAX_PAGE_BYTES + 1)), PageResponse(b"{}", "application/json")):
        with pytest.raises(HTTPException) as error:
            twipla.fetch_event_html(URL)
        assert error.value.status_code == 502


def test_fetcher_handles_time_limit(monkeypatch):
    response = PageResponse(b"x")
    monkeypatch.setattr(twipla.urllib.request, "build_opener", lambda handler: type("Opener", (), {"open": lambda *args, **kwargs: response})())
    clock = iter((0, 9))
    monkeypatch.setattr(twipla.time, "monotonic", lambda: next(clock))
    with pytest.raises(HTTPException) as error:
        twipla.fetch_event_html(URL)
    assert error.value.status_code == 504


@pytest.mark.parametrize("error", [TimeoutError(), urllib.error.URLError("offline"),
    urllib.error.HTTPError(URL, 404, "deleted", {}, None)])
def test_fetcher_reports_failed_or_deleted_events_without_server_traceback(monkeypatch, error):
    def failing_open(*args, **kwargs):
        raise error
    monkeypatch.setattr(twipla.urllib.request, "build_opener", lambda handler:
                        type("Opener", (), {"open": staticmethod(failing_open)})())
    with pytest.raises(HTTPException) as caught:
        twipla.fetch_event_html(URL)
    assert caught.value.status_code == 502


def test_admin_endpoint_does_not_publish_and_requires_admin(monkeypatch):
    app = FastAPI()
    app.post("/admin/api/regional-ads/import-twipla")(game_app.admin_import_twipla_ad)
    client = TestClient(app)
    calls = []
    ads_before = copy.deepcopy(game_app.REGIONAL_AD_SETTINGS)
    def imported(url):
        calls.append(url)
        return twipla.parse_event_html(event_html(), URL, now=NOW)
    monkeypatch.setattr(game_app, "import_twipla_event", imported)
    assert client.post("/admin/api/regional-ads/import-twipla", json={"url": URL}).status_code == 401
    assert not calls
    client.cookies.set(game_app.ADMIN_SESSION_COOKIE, game_app._admin_session_token(int(time.time()) + 60))
    response = client.post("/admin/api/regional-ads/import-twipla", json={"url": URL})
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    assert response.json()["draft"]["ends_at"] == "2026-11-22T16:30:00+09:00"
    assert "regions" not in response.json()["draft"]  # Do not guess a region from the game's origin.
    assert game_app.REGIONAL_AD_SETTINGS == ads_before

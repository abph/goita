"""Read one public TwiPla event and propose an editable advertisement draft."""
from __future__ import annotations

import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser

from fastapi import HTTPException


JST = timezone(timedelta(hours=9))
MAX_PAGE_BYTES = 1024 * 1024
EVENT_PATH = re.compile(r"/events/([1-9][0-9]{0,11})/?\Z")
DATE = re.compile(r"(20\d{2})\s*(?:年|/|-)\s*(\d{1,2})\s*(?:月|/|-)\s*(\d{1,2})\s*日?")
TIME_RANGE = re.compile(
    r"(?<!\d)(\d{1,2})[:：](\d{2})\s*[～〜~\-－–—]\s*(翌(?:日)?\s*)?(\d{1,2})[:：](\d{2})(?!\d)"
)


def normalize_event_url(raw: object) -> str:
    if not isinstance(raw, str) or len(raw) > 2048:
        raise HTTPException(400, "ツイプラのイベントURLを入力してください")
    value = raw.strip()
    if re.search(r"[\x00-\x20\\]", value):
        raise HTTPException(400, "ツイプラのイベントURLが正しくありません")
    try:
        parsed = urllib.parse.urlsplit(value)
    except ValueError as error:
        raise HTTPException(400, "ツイプラのイベントURLが正しくありません") from error
    match = EVENT_PATH.fullmatch(parsed.path)
    if (parsed.scheme.lower() not in ("http", "https")
            or parsed.netloc.lower() not in ("twipla.jp", "www.twipla.jp") or not match):
        raise HTTPException(400, "https://twipla.jp/events/数字 の形式で入力してください")
    return f"https://twipla.jp/events/{match[1]}"


class _EventRedirect(urllib.request.HTTPRedirectHandler):
    max_redirections = 3
    max_repeats = 2

    def __init__(self, event_url):
        self.event_url = event_url

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Validate BEFORE following a redirect; never fetch an arbitrary URL.
        try:
            allowed = (urllib.parse.urlsplit(newurl).scheme.lower() == "https"
                       and normalize_event_url(newurl) == self.event_url)
        except (HTTPException, ValueError):
            allowed = False
        if not allowed:
            raise HTTPException(502, "ツイプラのイベントページを取得できませんでした")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_event_html(url: str) -> str:
    deadline = time.monotonic() + 8
    request = urllib.request.Request(url, headers={
        "User-Agent": "SorouGoita-EventImport/1.0", "Accept": "text/html",
        "Accept-Encoding": "identity",
    })
    try:
        opener = urllib.request.build_opener(_EventRedirect(url))
        with opener.open(request, timeout=5) as response:
            if normalize_event_url(response.geturl()) != url:
                raise HTTPException(502, "ツイプラのイベントページを取得できませんでした")
            if response.headers.get_content_type() not in ("text/html", "application/xhtml+xml"):
                raise HTTPException(502, "ツイプラからイベント情報を取得できませんでした")
            chunks, size = [], 0
            while True:
                if time.monotonic() >= deadline:
                    raise HTTPException(504, "情報の取得に時間がかかっています。しばらくしてからお試しください")
                chunk = response.read(min(65536, MAX_PAGE_BYTES + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > MAX_PAGE_BYTES:
                    raise HTTPException(502, "イベントページが大きすぎるため取得できませんでした")
            return b"".join(chunks).decode("utf-8-sig", errors="replace")
    except urllib.error.HTTPError as error:
        detail = ("イベントが見つかりません。URLと公開状態を確認してください" if error.code in (404, 410)
                  else "ツイプラのページを取得できませんでした。公開状態を確認してください")
        raise HTTPException(502, detail) from error
    except (TimeoutError, urllib.error.URLError, OSError) as error:
        raise HTTPException(502, "ツイプラに接続できませんでした。しばらくしてからお試しください") from error


class _EventPage(HTMLParser):
    """Only read metadata, the event header and #desc, never attendee lists."""
    VOID = frozenset(("area", "base", "br", "col", "embed", "hr", "img", "input",
                      "link", "meta", "param", "source", "track", "wbr"))
    BLOCK = frozenset(("p", "div", "li", "h1", "h2", "h3", "h4", "tr", "br"))

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.meta, self.calendar = [], {}, {}
        self.capture, self.parts = {}, {"description": [], "header": [], "title": []}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta":
            key = (attrs.get("property") or attrs.get("name") or "").lower()
            if key in ("og:title", "twitter:title", "og:description", "description"):
                self.meta.setdefault(key, attrs.get("content", ""))
        if tag == "a" and not self.calendar:
            link = urllib.parse.urlsplit(attrs.get("href", ""))
            if link.hostname in ("www.google.com", "calendar.google.com") and link.path == "/calendar/event":
                self.calendar = urllib.parse.parse_qs(link.query)
        depth = len(self.stack)
        if attrs.get("id") == "desc":
            self.capture["description"] = depth
        if "arrow_box" in attrs.get("class", "").split():
            self.capture["header"] = depth
        if tag == "title":
            self.capture["title"] = depth
        if tag in self.BLOCK:
            self._text("\n")
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag not in self.stack:
            return
        if tag in self.BLOCK:
            self._text("\n")
        index = len(self.stack) - 1 - self.stack[::-1].index(tag)
        del self.stack[index:]
        for name, depth in list(self.capture.items()):
            if depth >= len(self.stack):
                del self.capture[name]

    def handle_data(self, data):
        self._text(data)

    def _text(self, text):
        if not any(tag in ("script", "style") for tag in self.stack):
            for name in self.capture:
                self.parts[name].append(text)


def _clean(text: str) -> str:
    return " ".join(text.split())


def _short(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def _calendar_date(value: str) -> datetime | None:
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%dT%H%M%S"):
        try:
            parsed = datetime.strptime(value, fmt)
            return parsed.replace(tzinfo=timezone.utc if value.endswith("Z") else JST).astimezone(JST)
        except ValueError:
            pass
    return None


def parse_event_html(html: str, url: str, *, now: datetime | None = None) -> dict:
    page = _EventPage()
    try:
        page.feed(html)
    except (ValueError, RecursionError) as error:
        raise HTTPException(502, "イベントページを読み取れませんでした") from error
    description = "".join(page.parts["description"])
    title = _clean(page.meta.get("og:title") or page.meta.get("twitter:title") or "".join(page.parts["title"]))
    title = re.sub(r"\s*[-|｜]\s*TwiPla\s*$", "", title, flags=re.I)
    if not title or not (description.strip() or page.calendar):
        raise HTTPException(502, "公開されたイベント情報を確認できませんでした。URLと公開状態を確認してください")
    warnings = []
    dates = page.calendar.get("dates", [""])[0].split("/")
    start = _calendar_date(dates[0])
    end = _calendar_date(dates[1]) if len(dates) == 2 else None
    if start is None:
        header = _clean("".join(page.parts["header"]))
        date_match = DATE.search(header)
        time_match = re.search(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)", header)
        if date_match and time_match:
            try:
                start = datetime(*map(int, date_match.groups()), *map(int, time_match.groups()), tzinfo=JST)
            except ValueError:
                pass
    if end is not None and (start is None or end <= start):
        end = None
    if start is not None and end is None:
        # TwiPla's calendar link often repeats DTSTART as DTEND. Read the
        # event's own date/time line instead, not unrelated comments/times.
        for line in description.splitlines():
            date_match = DATE.search(line)
            labelled = re.match(r"\s*(?:開催日時|開催時間|日時|日程|時間)\s*[;；:：]", line)
            if date_match:
                if tuple(map(int, date_match.groups())) != (start.year, start.month, start.day):
                    continue
            elif not labelled:
                continue
            match = TIME_RANGE.search(line)
            if not match:
                continue
            hour, minute, next_day, end_hour, end_minute = match.groups()
            hour, minute, end_hour, end_minute = map(int, (hour, minute, end_hour, end_minute))
            if (hour, minute) != (start.hour, start.minute) or end_hour > 29 or end_minute > 59:
                continue
            end = start.replace(hour=0, minute=0) + timedelta(hours=end_hour, minutes=end_minute)
            if (next_day and end_hour < 24) or end < start:
                end += timedelta(days=1)
            if end <= start:
                end = None
            break
    venue = _clean(page.calendar.get("location", [""])[0])
    if not venue:
        match = re.search(r"(?:^|\n)\s*(?:会場|場所)\s*[;；:：]\s*([^\n]+)", description)
        venue = _clean(match.group(1)) if match else ""
    summary = _clean(description or page.meta.get("og:description") or page.meta.get("description") or "")
    # Prefer complete opening sentences to TwiPla's mid-word metadata excerpt.
    opening = ""
    for sentence in re.split(r"(?<=[。！？!?])\s*", summary):
        if len(opening + sentence) > 120:
            break
        opening += sentence
    summary = opening or _short(summary, 120)
    lines = []
    if start:
        label = f"{start.month}/{start.day}（{'月火水木金土日'[start.weekday()]}）{start:%H:%M}"
        if end:
            label += f"～{end:%H:%M}" if start.date() == end.date() else f"～{end.month}/{end.day} {end:%H:%M}"
        lines.append(label)
    if venue:
        lines.append(_short(venue, 70))
    if summary:
        lines.append(summary)
    if len(title) > 40:
        warnings.append("見出しを40文字以内に短縮しました。元のイベント名を確認してください。")
    if start is None:
        warnings.append("開催日時を取得できませんでした。ツイプラで日時を確認してください。")
    if end is None:
        warnings.append("終了時刻を取得できませんでした。掲載終了を入力してください。")
    if not venue:
        warnings.append("会場を取得できませんでした。本文と対象地方を確認してください。")
    current = (now or datetime.now(timezone.utc)).astimezone(JST)
    ended = bool(end and end <= current)
    if ended:
        warnings.append("このイベントは終了しています。掲載期間を確認してください。")
    return {
        "event": {"title": title, "starts_at": start.isoformat() if start else None,
                  "ends_at": end.isoformat() if end else None, "venue": venue, "ended": ended},
        "draft": {"title": _short(title, 40), "message": _short("\n".join(lines), 200), "url": url,
                  "starts_at": (start if ended and start else current.replace(second=0, microsecond=0)).isoformat(),
                  "ends_at": end.isoformat() if end else None},
        "warnings": warnings,
    }


def import_event(raw_url: object) -> dict:
    url = normalize_event_url(raw_url)
    return parse_event_html(fetch_event_html(url), url)

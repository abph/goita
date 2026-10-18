"""Regional public-room advertisements selected from coarse edge location."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Mapping
from urllib.parse import urlparse

from fastapi import HTTPException

from backend.analytics_geo import infer_country_code, infer_prefecture


REGIONS = ("北海道", "東北", "関東", "中部", "関西", "中国", "四国", "九州", "沖縄")
SURFACES = ("public_room", "score_attack")
PREFECTURE_REGIONS = {
    "北海道": "北海道",
    **{name + "県": "東北" for name in ("青森", "岩手", "宮城", "秋田", "山形", "福島")},
    **{name + "県": "関東" for name in ("茨城", "栃木", "群馬", "埼玉", "千葉", "神奈川")},
    "東京都": "関東",
    **{name + "県": "中部" for name in ("新潟", "富山", "石川", "福井", "山梨", "長野", "岐阜", "静岡", "愛知")},
    **{name + "県": "関西" for name in ("三重", "滋賀", "兵庫", "奈良", "和歌山")},
    "京都府": "関西", "大阪府": "関西",
    **{name + "県": "中国" for name in ("鳥取", "島根", "岡山", "広島", "山口")},
    **{name + "県": "四国" for name in ("徳島", "香川", "愛媛", "高知")},
    **{name + "県": "九州" for name in ("福岡", "佐賀", "長崎", "熊本", "大分", "宮崎", "鹿児島")},
    "沖縄県": "沖縄",
}
AD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def infer_region(headers: Mapping[str, str]) -> str:
    if infer_country_code(headers) not in ("", "JP"):
        return ""
    return PREFECTURE_REGIONS.get(infer_prefecture(headers), "")


def _date(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise HTTPException(400, "掲載日時が正しくありません") from error
    if parsed.tzinfo is None:
        raise HTTPException(400, "掲載日時には時差情報が必要です")
    return parsed.astimezone(timezone.utc)


def normalize_ads(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or len(raw) > 30:
        raise HTTPException(400, "地域別広告は30件以内にしてください")
    result = []
    ids = set()
    for item in raw:
        if not isinstance(item, dict):
            raise HTTPException(400, "地域別広告の形式が正しくありません")
        ad_id = str(item.get("id") or "")
        title = str(item.get("title") or "").strip()
        message = str(item.get("message") or "").strip()
        url = str(item.get("url") or "").strip()
        regions = item.get("regions")
        surfaces = item.get("surfaces", ["public_room"])
        if not AD_ID_RE.fullmatch(ad_id) or ad_id in ids:
            raise HTTPException(400, "広告IDが正しくありません")
        if not title or len(title) > 40 or not message or len(message) > 200:
            raise HTTPException(400, "広告の見出しと本文を入力してください")
        if (not isinstance(regions, list)
                or any(not isinstance(region, str) for region in regions)
                or len(regions) != len(set(regions))
                or any(region not in REGIONS for region in regions)):
            raise HTTPException(400, "対象地方が正しくありません")
        if (not isinstance(surfaces, list)
                or not surfaces
                or any(not isinstance(surface, str) for surface in surfaces)
                or len(surfaces) != len(set(surfaces))
                or any(surface not in SURFACES for surface in surfaces)):
            raise HTTPException(400, "広告の表示場所が正しくありません")
        if url:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https") or not parsed.netloc or len(url) > 2048:
                raise HTTPException(400, "リンク先URLが正しくありません")
        start, end = _date(item.get("starts_at")), _date(item.get("ends_at"))
        if end <= start:
            raise HTTPException(400, "終了日時は開始日時より後にしてください")
        ids.add(ad_id)
        result.append({
            "id": ad_id, "title": title, "message": message, "url": url,
            "regions": regions, "starts_at": start.isoformat(), "ends_at": end.isoformat(),
            "surfaces": surfaces,
            "enabled": bool(item.get("enabled", False)),
        })
    for target in (*REGIONS, "全国共通"):
        for surface in SURFACES:
            events: list[tuple[str, int]] = []
            for item in result:
                targets = set(item["regions"]) or {"全国共通"}
                if not item["enabled"] or target not in targets or surface not in item["surfaces"]:
                    continue
                events.append((item["starts_at"], 1))
                events.append((item["ends_at"], -1))
            active_count = 0
            for _timestamp, delta in sorted(events, key=lambda event: (event[0], event[1])):
                active_count += delta
                if active_count > 3:
                    raise HTTPException(
                        400,
                        "同じ期間・同じ地方・同じ表示場所に掲載できる広告は3件までです",
                    )
    return result


def select_ads(
    ads: list[dict[str, Any]],
    region: str,
    *,
    surface: str = "public_room",
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    if surface not in SURFACES:
        return []
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    active = [
        ad for ad in ads
        if ad["enabled"]
        and surface in ad.get("surfaces", ["public_room"])
        and ad["starts_at"] <= current < ad["ends_at"]
    ]
    regional = [ad for ad in active if region and region in ad["regions"]]
    common = [ad for ad in active if not ad["regions"]]
    return (regional + common)[:3]


def select_ad(
    ads: list[dict[str, Any]],
    region: str,
    *,
    surface: str = "public_room",
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Return the first carousel advertisement for older internal callers."""

    selected = select_ads(ads, region, surface=surface, now=now)
    return selected[0] if selected else None

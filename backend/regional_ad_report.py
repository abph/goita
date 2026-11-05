"""Period-specific advertisement reports from existing aggregate counters."""
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException


JST = timezone(timedelta(hours=9))


def _rate(impressions, clicks):
    # Auto-slides do not count as impressions, but can still lead to clicks.
    return round(clicks / impressions * 100, 2) if impressions else None


def _status(ad, now):
    if not ad.get("enabled"):
        return "停止中"
    if now < datetime.fromisoformat(ad["starts_at"]):
        return "開始前"
    if now >= datetime.fromisoformat(ad["ends_at"]):
        return "終了"
    return "掲載中"


def build_regional_ad_report(store, ads, period="30", *, now=None):
    if period not in ("7", "30", "all"):
        raise HTTPException(400, "集計期間は7日・30日・全期間から選んでください")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    today = current.astimezone(JST).date()
    days = None if period == "all" else int(period)
    metrics = store.regional_ad_metrics(
        [ad["id"] for ad in ads], now=current, period_days=days,
        recent_days=days or 30, include_all_days=days is None,
    )
    items = []
    for ad in ads:
        values = metrics.get(ad["id"], {})
        impressions, clicks = int(values.get("impressions", 0)), int(values.get("clicks", 0))
        daily = {}
        for row in values.get("daily", []):
            counts = daily.setdefault(row["date"], {"date": row["date"], "impressions": 0, "clicks": 0})
            counts["impressions"] += row["impressions"]
            counts["clicks"] += row["clicks"]
        item = {
            "id": ad["id"], "title": ad["title"], "status": _status(ad, current),
            "impressions": impressions, "clicks": clicks, "click_rate": _rate(impressions, clicks),
            # On finite periods these are FIRST sightings within the period,
            # not all returning browsers active during that period.
            "viewers": int(values.get("viewers", 0)),
            "measured_impressions": int(values.get("measured_impressions", 0)),
            "daily": [daily[key] for key in sorted(daily)],
        }
        for field in ("regions", "surfaces"):
            item[field] = [dict(row, click_rate=_rate(row["impressions"], row["clicks"]))
                           for row in values.get(field, [])]
        items.append(item)
    impressions = sum(item["impressions"] for item in items)
    clicks = sum(item["clicks"] for item in items)
    available_dates = [row["date"] for item in items for row in item["daily"]]
    return {
        "period": period,
        "period_start": (today - timedelta(days=days - 1)).isoformat() if days else min(available_dates, default=None),
        "period_end": today.isoformat(),
        "summary": {"impressions": impressions, "clicks": clicks, "click_rate": _rate(impressions, clicks)},
        "items": items,
    }

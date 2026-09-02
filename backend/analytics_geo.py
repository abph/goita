"""Convert trusted edge location headers into coarse analytics regions."""

from __future__ import annotations

from typing import Mapping


UNKNOWN_PREFECTURE = "不明"
OVERSEAS_PREFECTURE = "国外"
UNKNOWN_COUNTRY_CODE = ""

PREFECTURES_BY_CODE = {
    "01": "北海道",
    "02": "青森県",
    "03": "岩手県",
    "04": "宮城県",
    "05": "秋田県",
    "06": "山形県",
    "07": "福島県",
    "08": "茨城県",
    "09": "栃木県",
    "10": "群馬県",
    "11": "埼玉県",
    "12": "千葉県",
    "13": "東京都",
    "14": "神奈川県",
    "15": "新潟県",
    "16": "富山県",
    "17": "石川県",
    "18": "福井県",
    "19": "山梨県",
    "20": "長野県",
    "21": "岐阜県",
    "22": "静岡県",
    "23": "愛知県",
    "24": "三重県",
    "25": "滋賀県",
    "26": "京都府",
    "27": "大阪府",
    "28": "兵庫県",
    "29": "奈良県",
    "30": "和歌山県",
    "31": "鳥取県",
    "32": "島根県",
    "33": "岡山県",
    "34": "広島県",
    "35": "山口県",
    "36": "徳島県",
    "37": "香川県",
    "38": "愛媛県",
    "39": "高知県",
    "40": "福岡県",
    "41": "佐賀県",
    "42": "長崎県",
    "43": "熊本県",
    "44": "大分県",
    "45": "宮崎県",
    "46": "鹿児島県",
    "47": "沖縄県",
}

_ENGLISH_PREFECTURES = {
    name.lower(): prefecture
    for name, prefecture in {
        "Hokkaido": "北海道", "Aomori": "青森県", "Iwate": "岩手県",
        "Miyagi": "宮城県", "Akita": "秋田県", "Yamagata": "山形県",
        "Fukushima": "福島県", "Ibaraki": "茨城県", "Tochigi": "栃木県",
        "Gunma": "群馬県", "Saitama": "埼玉県", "Chiba": "千葉県",
        "Tokyo": "東京都", "Kanagawa": "神奈川県", "Niigata": "新潟県",
        "Toyama": "富山県", "Ishikawa": "石川県", "Fukui": "福井県",
        "Yamanashi": "山梨県", "Nagano": "長野県", "Gifu": "岐阜県",
        "Shizuoka": "静岡県", "Aichi": "愛知県", "Mie": "三重県",
        "Shiga": "滋賀県", "Kyoto": "京都府", "Osaka": "大阪府",
        "Hyogo": "兵庫県", "Nara": "奈良県", "Wakayama": "和歌山県",
        "Tottori": "鳥取県", "Shimane": "島根県", "Okayama": "岡山県",
        "Hiroshima": "広島県", "Yamaguchi": "山口県", "Tokushima": "徳島県",
        "Kagawa": "香川県", "Ehime": "愛媛県", "Kochi": "高知県",
        "Fukuoka": "福岡県", "Saga": "佐賀県", "Nagasaki": "長崎県",
        "Kumamoto": "熊本県", "Oita": "大分県", "Miyazaki": "宮崎県",
        "Kagoshima": "鹿児島県", "Okinawa": "沖縄県",
    }.items()
}

ALLOWED_ANALYTICS_REGIONS = frozenset({
    *PREFECTURES_BY_CODE.values(),
    UNKNOWN_PREFECTURE,
    OVERSEAS_PREFECTURE,
})


def normalize_prefecture(value: object) -> str:
    text = str(value or "").strip()
    return text if text in ALLOWED_ANALYTICS_REGIONS else UNKNOWN_PREFECTURE


def normalize_country_code(value: object) -> str:
    """Keep only a coarse ISO-style country code, never an address or IP."""
    code = str(value or "").strip().upper()
    if len(code) == 2 and code.isalpha() and code not in {"XX", "T1"}:
        return code
    return UNKNOWN_COUNTRY_CODE


def infer_country_code(headers: Mapping[str, str]) -> str:
    """Read the Cloudflare country code without retaining the source IP."""
    return normalize_country_code(headers.get("cf-ipcountry", ""))


def infer_prefecture(headers: Mapping[str, str]) -> str:
    """Read Cloudflare location headers without retaining the source IP."""

    country = infer_country_code(headers)
    if country and country != "JP":
        return OVERSEAS_PREFECTURE

    region_code = str(headers.get("cf-region-code", "") or "").strip().upper()
    if region_code.startswith("JP-"):
        region_code = region_code[3:]
    if region_code.isdigit():
        region_code = region_code.zfill(2)
    if region_code in PREFECTURES_BY_CODE:
        return PREFECTURES_BY_CODE[region_code]

    region = str(headers.get("cf-region", "") or "").strip()
    if region in ALLOWED_ANALYTICS_REGIONS:
        return region
    normalized_region = region.lower().removesuffix(" prefecture").strip()
    return _ENGLISH_PREFECTURES.get(normalized_region, UNKNOWN_PREFECTURE)

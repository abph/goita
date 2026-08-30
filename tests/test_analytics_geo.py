from backend.analytics_geo import infer_prefecture


def test_cloudflare_region_code_is_converted_to_prefecture() -> None:
    assert infer_prefecture({
        "cf-ipcountry": "JP",
        "cf-region-code": "JP-11",
    }) == "埼玉県"


def test_cloudflare_english_region_name_is_supported() -> None:
    assert infer_prefecture({
        "cf-ipcountry": "JP",
        "cf-region": "Ishikawa Prefecture",
    }) == "石川県"


def test_overseas_and_missing_locations_are_coarsened() -> None:
    assert infer_prefecture({"cf-ipcountry": "US"}) == "国外"
    assert infer_prefecture({}) == "不明"

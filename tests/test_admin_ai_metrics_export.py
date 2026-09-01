import backend.app as app_module


def test_ai_metrics_export_contains_only_anonymous_ai_aggregates() -> None:
    payload = app_module._ai_metrics_export_payload()

    assert set(payload) == {
        "format",
        "schema_version",
        "exported_at",
        "versions",
        "privacy",
        "ai_response_dictionary",
        "generic_patterns",
    }
    assert payload["format"] == "sorou-goita-ai-metrics"
    assert payload["schema_version"] == 1
    assert payload["versions"]["ai_profile"] == "current"
    assert payload["versions"]["ai_profile_label"] == "強化中AI"
    assert payload["privacy"] == {
        "player_names_included": False,
        "hands_included": False,
        "kifu_included": False,
        "chat_included": False,
        "passwords_included": False,
    }
    assert isinstance(payload["ai_response_dictionary"], dict)
    assert isinstance(payload["generic_patterns"], dict)
    assert "private_rooms" not in payload
    assert "private_room_ads" not in payload
    assert "room_totals" not in payload


def test_ai_metrics_export_requests_all_anonymous_pattern_details() -> None:
    original = app_module.generic_response_pattern_snapshot
    detail_limits = []

    def fake_snapshot(*, detail_limit=50):
        detail_limits.append(detail_limit)
        return {"requested_detail_limit": detail_limit}

    try:
        app_module.generic_response_pattern_snapshot = fake_snapshot
        payload = app_module._ai_metrics_export_payload()
    finally:
        app_module.generic_response_pattern_snapshot = original

    assert detail_limits == [None]
    assert payload["generic_patterns"] == {
        "requested_detail_limit": None,
    }


if __name__ == "__main__":
    test_ai_metrics_export_contains_only_anonymous_ai_aggregates()
    test_ai_metrics_export_requests_all_anonymous_pattern_details()
    print("ADMIN_AI_METRICS_EXPORT_TEST_OK")

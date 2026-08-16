from __future__ import annotations

from pathlib import Path

from goita_ai2.current_ai.persistence import resolve_adaptive_value_storage


def test_explicit_adaptive_value_path_has_priority() -> None:
    storage = resolve_adaptive_value_storage(
        {
            "RENDER": "true",
            "GOITA_AI_ADAPTIVE_VALUE_PATH": "custom/value.json",
            "GOITA_PERSISTENT_DATA_DIR": "/var/data",
        }
    )

    assert storage.path == "custom/value.json"
    assert storage.source == "explicit_path"
    assert storage.render_detected is True
    assert storage.persistent_configured is False
    assert storage.warning == ""


def test_persistent_directory_builds_adaptive_value_path() -> None:
    storage = resolve_adaptive_value_storage(
        {
            "RENDER": "true",
            "GOITA_PERSISTENT_DATA_DIR": "/var/data",
        }
    )

    assert storage.path == str(
        Path("/var/data") / "goita-ai" / "background_search_value.json"
    )
    assert storage.source == "persistent_directory"
    assert storage.render_detected is True
    assert storage.persistent_configured is True
    assert storage.warning == ""


def test_render_without_persistent_directory_reports_warning() -> None:
    storage = resolve_adaptive_value_storage({"RENDER": "true"})

    assert storage.path == "results/background_search_value.json"
    assert storage.source == "local_default"
    assert storage.render_detected is True
    assert storage.persistent_configured is False
    assert storage.warning == (
        "render_persistent_data_directory_not_configured"
    )


def test_local_default_does_not_report_render_warning() -> None:
    storage = resolve_adaptive_value_storage({})

    assert storage.path == "results/background_search_value.json"
    assert storage.source == "local_default"
    assert storage.render_detected is False
    assert storage.persistent_configured is False
    assert storage.warning == ""

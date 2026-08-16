"""Resolves durable storage locations for current-AI runtime data.

Explicit file paths keep priority, while a shared persistent-data directory
lets Render and local deployments use the same storage-selection contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


DEFAULT_ADAPTIVE_VALUE_PATH = "results/background_search_value.json"
ADAPTIVE_VALUE_FILENAME = "background_search_value.json"


def _is_true(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AdaptiveValueStorage:
    """Describes the selected adaptive-value checkpoint location."""

    path: str
    source: str
    render_detected: bool
    persistent_configured: bool
    warning: str = ""

    def snapshot(self) -> dict:
        return {
            "path": self.path,
            "storage_source": self.source,
            "render_detected": self.render_detected,
            "persistent_configured": self.persistent_configured,
            "storage_warning": self.warning,
        }


def resolve_adaptive_value_storage(
    environ: Mapping[str, str],
    *,
    default_path: str = DEFAULT_ADAPTIVE_VALUE_PATH,
) -> AdaptiveValueStorage:
    """Resolve the checkpoint path without touching the filesystem."""

    render_detected = _is_true(environ.get("RENDER"))
    explicit_path = str(
        environ.get("GOITA_AI_ADAPTIVE_VALUE_PATH", "") or ""
    ).strip()
    if explicit_path:
        return AdaptiveValueStorage(
            path=explicit_path,
            source="explicit_path",
            render_detected=render_detected,
            persistent_configured=False,
        )

    persistent_directory = str(
        environ.get("GOITA_PERSISTENT_DATA_DIR", "") or ""
    ).strip()
    if persistent_directory:
        return AdaptiveValueStorage(
            path=str(
                Path(persistent_directory)
                / "goita-ai"
                / ADAPTIVE_VALUE_FILENAME
            ),
            source="persistent_directory",
            render_detected=render_detected,
            persistent_configured=True,
        )

    warning = (
        "render_persistent_data_directory_not_configured"
        if render_detected
        else ""
    )
    return AdaptiveValueStorage(
        path=str(default_path),
        source="local_default",
        render_detected=render_detected,
        persistent_configured=False,
        warning=warning,
    )

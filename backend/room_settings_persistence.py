"""Persist editable room-management settings across server restarts.

The storage path can share Render's persistent-data directory with the AI
checkpoint, while local and test environments remain opt-in by default.
Writes use an atomic replacement so an interrupted save keeps the old file.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Mapping, Optional


ROOM_SETTINGS_FILENAME = "goita-room-settings.json"
ROOM_SETTINGS_VERSION = 1
_SAVE_LOCK = threading.Lock()
_LOGGER = logging.getLogger(__name__)


def resolve_room_settings_path(environ: Mapping[str, str]) -> Optional[Path]:
    """Select an explicit path or a file under the shared persistent directory."""

    explicit_path = str(environ.get("GOITA_ROOM_SETTINGS_PATH", "") or "").strip()
    if explicit_path:
        return Path(explicit_path)

    persistent_directory = str(
        environ.get("GOITA_PERSISTENT_DATA_DIR", "") or ""
    ).strip()
    if persistent_directory:
        return Path(persistent_directory) / ROOM_SETTINGS_FILENAME
    return None


def load_room_settings(path: Optional[Path]) -> dict[str, dict[str, Any]]:
    """Load the room mapping, returning an empty mapping for absent/bad files."""

    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _LOGGER.error("Unable to load persistent room settings from %s: %s", path, error)
        return {}

    if not isinstance(payload, dict) or payload.get("version") != ROOM_SETTINGS_VERSION:
        _LOGGER.warning("Ignoring unsupported room settings file: %s", path)
        return {}
    rooms = payload.get("rooms")
    if not isinstance(rooms, dict):
        return {}
    return {
        str(game_id): dict(settings)
        for game_id, settings in rooms.items()
        if isinstance(game_id, str) and isinstance(settings, dict)
    }


def save_room_settings(
    path: Optional[Path],
    rooms: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Atomically save room settings; return False when storage is unavailable."""

    if path is None:
        return False

    payload = {
        "version": ROOM_SETTINGS_VERSION,
        "rooms": {
            str(game_id): dict(settings)
            for game_id, settings in rooms.items()
        },
    }
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with _SAVE_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            try:
                path.chmod(0o600)
            except OSError:
                pass
        return True
    except (OSError, TypeError, ValueError) as error:
        _LOGGER.error("Unable to save persistent room settings to %s: %s", path, error)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        return False

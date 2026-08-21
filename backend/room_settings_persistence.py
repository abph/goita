"""Persist editable room-management settings across server restarts.

The storage path can share Render's persistent-data directory with the AI
checkpoint, while local and test environments remain opt-in by default.
Writes use an atomic replacement so an interrupted save keeps the old file.
"""

from __future__ import annotations

import json
import hashlib
import hmac
import logging
import os
import secrets
import threading
from pathlib import Path
from typing import Any, Mapping, Optional


ROOM_SETTINGS_FILENAME = "goita-room-settings.json"
ROOM_SETTINGS_VERSION = 1
ADMIN_PASSWORD_HASH_SCHEME = "pbkdf2_sha256"
ADMIN_PASSWORD_HASH_ITERATIONS = 260_000
_SAVE_LOCK = threading.Lock()
_LOGGER = logging.getLogger(__name__)


def hash_admin_password(password: str) -> str:
    """Hash a room administrator password for persistent storage."""

    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        ADMIN_PASSWORD_HASH_ITERATIONS,
    )
    return (
        f"{ADMIN_PASSWORD_HASH_SCHEME}${ADMIN_PASSWORD_HASH_ITERATIONS}"
        f"${salt.hex()}${digest.hex()}"
    )


def is_admin_password_hash(value: Any) -> bool:
    """Return whether a value has the supported stored-password format."""

    if not isinstance(value, str):
        return False
    parts = value.split("$")
    if len(parts) != 4 or parts[0] != ADMIN_PASSWORD_HASH_SCHEME:
        return False
    try:
        iterations = int(parts[1])
        salt = bytes.fromhex(parts[2])
        digest = bytes.fromhex(parts[3])
    except (TypeError, ValueError):
        return False
    return iterations > 0 and len(salt) >= 16 and len(digest) == 32


def verify_admin_password(password: str, stored_hash: str) -> bool:
    """Verify a submitted password without exposing the stored value."""

    if not is_admin_password_hash(stored_hash):
        return False
    _scheme, iterations_text, salt_hex, digest_hex = stored_hash.split("$")
    candidate = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        int(iterations_text),
    )
    return hmac.compare_digest(candidate, bytes.fromhex(digest_hex))


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

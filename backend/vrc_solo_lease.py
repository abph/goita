"""Exclusive leases for pre-authored VRChat connection URLs.

Claims are deliberately non-idempotent: even an identical connection key cannot
claim an occupied room. Only a successful claim response enables the Udon client.
The keys are routing identifiers in a public world, not authentication secrets.
"""

import os
import re
import secrets
import sqlite3
import time
from pathlib import Path


LEASE_SECONDS = 180
RETIRED_SECONDS = 600
KEY_RE = re.compile(r"^[a-f0-9]{32}$")


class LeaseError(ValueError):
    pass


class SoloLeaseStore:
    def __init__(self, path, clock=time.time):
        self.path = Path(path)
        self.clock = clock
        self.epoch = secrets.token_hex(16)

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("CREATE TABLE IF NOT EXISTS leases (room TEXT PRIMARY KEY, connection TEXT, epoch TEXT, expires REAL)")
        connection.execute("CREATE TABLE IF NOT EXISTS retired (room TEXT, connection TEXT, until REAL, PRIMARY KEY(room, connection))")
        return connection

    def _transaction(self, room, key, operation):
        if not KEY_RE.fullmatch(key or ""):
            raise LeaseError("upgrade_required")
        now = self.clock()
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM retired WHERE until <= ?", (now,))
            row = db.execute("SELECT connection, epoch, expires FROM leases WHERE room=?", (room,)).fetchone()
            if row and (row[2] <= now or row[1] != self.epoch):
                # A restart loses the in-memory game; old requests must not resume it.
                db.execute("INSERT OR REPLACE INTO retired VALUES (?, ?, ?)", (room, row[0], now + RETIRED_SECONDS))
                db.execute("DELETE FROM leases WHERE room=?", (room,))
                row = None
            if operation == "claim":
                if row:
                    raise LeaseError("busy")
                retired = db.execute("SELECT 1 FROM retired WHERE room=? AND connection=?", (room, key)).fetchone()
                if retired:
                    raise LeaseError("connection_retired")
                db.execute("INSERT INTO leases VALUES (?, ?, ?, ?)", (room, key, self.epoch, now + LEASE_SECONDS))
            else:
                if not row or row[0] != key:
                    raise LeaseError("lease_lost" if not row else "busy")
                if operation == "release":
                    db.execute("DELETE FROM leases WHERE room=?", (room,))
                    db.execute("INSERT OR REPLACE INTO retired VALUES (?, ?, ?)", (room, key, now + RETIRED_SECONDS))
                elif operation == "touch":
                    db.execute("UPDATE leases SET expires=? WHERE room=?", (now + LEASE_SECONDS, room))
            db.execute("COMMIT")
        except LeaseError:
            # Expiration/retirement must persist even when the request is rejected.
            db.execute("COMMIT")
            raise
        except Exception:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        finally:
            db.close()

    def claim(self, room, key):
        self._transaction(room, key, "claim")

    def require(self, room, key):
        self._transaction(room, key, "require")

    def touch(self, room, key):
        self._transaction(room, key, "touch")

    def release(self, room, key):
        self._transaction(room, key, "release")


def default_store():
    configured = os.getenv("GOITA_VRC_LEASE_DB_PATH", "").strip()
    persistent = os.getenv("GOITA_PERSISTENT_DATA_DIR", "").strip()
    path = configured or str((Path(persistent) if persistent else Path(__file__).resolve().parents[1] / "results") / "goita-vrc-leases.sqlite3")
    return SoloLeaseStore(path)

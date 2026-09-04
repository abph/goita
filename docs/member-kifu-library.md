# Member Kifu Library

## Storage and permissions

- The My Page library is owned by the authenticated member, not by a room.
- New saves and imports require active paid access and a completed initial password change.
- Reading, editing, exporting, and deleting existing records remain available after paid access expires.
- The maximum is 1000 records per member, enforced inside the write transaction. The library does not display a limit notice until a save exceeds the limit.
- Records use the `member_kifu` table in the existing member database (`GOITA_MEMBER_DB_PATH`, or the member DB under `GOITA_PERSISTENT_DATA_DIR`). Keep this database on the persistent disk.
- Deleting a member cascades to their records. Reissuing the same member ID does not restore them.
- Library APIs require the member session cookie and same-origin custom header. A room administrator password grants no library access.
- Saving a room snapshot requires a finished round. Locked rooms additionally require the entry cookie issued by successful passphrase verification. Password changes or server restarts invalidate that entry proof; re-enter the room to save again.
- Import and download retain the existing text kifu format. Applying a deal remains limited to a private-room host and uses the existing match settings.

## Retiring the old room library

The owner explicitly requested deletion of all old room-library records, identified as test data.
At startup, `retire_room_kifu` drops only the `research_kifu` table from the configured old library database (`GOITA_KIFU_DB_PATH` or its existing fallback). This is irreversible and idempotent.

The old room-scoped HTTP endpoints have been removed. No old record is copied into a member account.
The cleanup does not read or modify `frontend/kifu_data.json`, `frontend/kifu_data_raw.json`, AI dictionaries, or the new `member_kifu` table.
Historical external backups are not modified by this migration.

## Verification

Run `tests/test_member_kifu.py` for ownership, locked-room entry, live-round refusal, subscription expiry, concurrency, persistence, and cleanup scope.
Existing record parsing/viewer tests and membership/stamp tests remain part of the regression suite.

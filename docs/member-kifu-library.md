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

## Personal seats and on-demand statistics

- Saves/imports accept an explicitly selected `my_seat`: A, B, C, D, `spectator`, or an empty (unset) value. Editing can correct or clear it. It is not inferred from a name or from a seat occupied after the round.
- The seat is stored only in that member's existing record payload; old records remain unset. Anonymous saves retain the selected seat without retaining player names.
- POST `/api/member/kifu/statistics` reads all of the authenticated owner's records on demand. It uses the winner and points already present in each kifu to calculate pair wins/losses, points, and individual/partner finishes. No derived statistic is persisted or sent to analytics/AI training.
- Unset seats, spectators and unfinished/unknown results are excluded. Zero counted records produce no win-rate percentage. Each saved record counts once, including separately saved copies of the same round. This is a per-round statistic of the saved library, not a full match history.
- The tag filter does not limit statistics. Edits, saves, imports, deletion, reloading and logout invalidate the displayed statistics; press the button again to recalculate. Requests returning after invalidation cannot restore outdated results.
- The standard downloadable kifu format remains unchanged; personal seat metadata is not exported.

## Tests

Run `tests/test_member_kifu.py` for ownership, locked-room entry, live-round refusal, subscription expiry, concurrency, persistence, and cleanup scope.
Existing record parsing/viewer tests and membership/stamp tests remain part of the regression suite.

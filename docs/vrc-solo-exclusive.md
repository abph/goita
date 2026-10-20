# AI Solo Goita Table: exclusive use (0.2.0)

## Behavior

- A room (normally `table1`) has one active holder. The first successful START claims it.
- START in another instance displays `現在使用中です` and does not receive a hand.
- A completed round/match does not release the room. NEXT ROUND keeps the score and lease.
- The A-seat player presses `利用終了` twice within 10 seconds to release the room. The board is cleared locally.
- While A is occupied, the table owner renews the lease every 30 seconds, including during thinking and after a round.
- A vacant seat does not renew the lease, even if spectators remain.
- The server releases the lease after 180 seconds without an accepted holder request. The client stops using its connection after 150 seconds without confirmation, leaving a buffer for delayed traffic.
- Ownership transfer within a VRC instance reuses the synchronized connection and refreshes the board. Expired connections cannot resume another user's game.
- The browser spectator link remains `https://solo.vrcgoita.com/?room=vrc-solo-table1`.

## Unity upgrade

Update the package `com.sorougoita.ai-solo-goita-table` to 0.2.0. In the existing scene select the table root, then run:

`Tools > AI Solo Goita Table > Upgrade Selected Table`

This preserves the table's position and existing presentation references, generates the connection URL groups, and adds the release button. It reads the existing base URL/room from the START URL. Confirm the base is `https://solo.vrcgoita.com` and the room is `table1`. Save the scene and upload the updated world with the VRChat SDK. Old world versions cannot use the new server's action endpoints: they fail closed with `upgrade_required`.

## Connection identifiers

Udon cannot construct VRCUrls at runtime. The editor pre-generates 128 groups of 33 URLs with random 128-bit connection identifiers. START chooses a group at runtime and calls `/claim?session=...`. A claim always fails when occupied, including an identical identifier. Only a successful claim response enables subsequent operations. A collision therefore refuses the later client instead of joining the earlier game.

The identifiers are public routing keys, not authentication credentials. This prevents interference by ordinary world clients; it does not authenticate arbitrary external HTTP callers. Explicitly released and expired keys are retired for 10 minutes. Late callbacks are discarded locally, and requests are serialized with at least 5.2 seconds between downloads. A failed claim is never automatically treated as a successful resume.

## Server

The lease is stored in SQLite at `GOITA_VRC_LEASE_DB_PATH` if configured, otherwise `${GOITA_PERSISTENT_DATA_DIR}/goita-vrc-leases.sqlite3`, or `results/goita-vrc-leases.sqlite3` locally. Render's existing `/var/data` configuration works without an additional environment variable.

Run **one process/one worker/one service instance**. The Goita games and turn locks are in process memory. SQLite makes lease changes atomic but does not make game state shareable across workers. A server restart invalidates earlier leases because the corresponding games no longer exist.

The existing Render start command can remain:

```sh
uvicorn backend.app:app --host 0.0.0.0 --port $PORT --workers 1
```

From PowerShell in this repository, commit only the server changes for this feature:

```powershell
git add -- backend/app.py backend/vrc_solo_lease.py tests/test_vrc_solo_bridge.py docs/vrc-solo-exclusive.md
git commit --only -m "Add exclusive leases for VRChat solo tables" -- backend/app.py backend/vrc_solo_lease.py tests/test_vrc_solo_bridge.py docs/vrc-solo-exclusive.md
git push origin main
```

This leaves unrelated staged changes out of the commit. GitHub-connected Render services deploy after the push if automatic deployment is enabled; otherwise use `Manual Deploy > Deploy latest commit` in the Render dashboard. These commands have not been run by the implementation task.

## Acceptance check

1. Instance A claims the room and plays. Instance B presses START: only `現在使用中です` is displayed.
2. Leave A thinking for more than 3 minutes: its heartbeat preserves the lease.
3. Complete a round in A and start the next: scores carry over and B remains blocked.
4. Confirm `利用終了` in A: B can START immediately with a fresh deal and zero score.
5. Disconnect/leave A without releasing: B can START after the server's 3-minute timeout.
6. Return an expired client to the world: it must not control B's game.

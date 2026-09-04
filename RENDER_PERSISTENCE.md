# Render persistent runtime data

The adaptive background-search model, editable room-management settings,
research kifu library, and privacy-limited usage analytics write runtime data
to the filesystem. Render's normal filesystem is ephemeral, so production
needs a persistent disk for these files to survive restarts and deploys.

Render persistent disks require a paid web service. A service with an attached
disk is limited to one service instance.

## Render dashboard setup

1. Open the web service and add a persistent disk.
2. Set its mount path to `/var/data`.
3. Add the environment variable `GOITA_PERSISTENT_DATA_DIR=/var/data`.
4. Deploy the service again.

The files are then stored at:

```text
/var/data/goita-ai/background_search_value.json
/var/data/goita-room-settings.json
/var/data/goita-research-kifu.sqlite3
/var/data/goita-analytics.sqlite3
/var/data/goita-members.sqlite3
```

The room settings file contains only editable room-management values: room
name, entry passphrase, AI profile, legal-action visibility, log visibility,
the configured room background path, and changed room-admin passwords as
salted PBKDF2 hashes. It never stores room-admin passwords in plaintext, and
does not contain hands, scores, occupied seats, or other live match state.

The analytics database stores anonymous browser/session IDs and an allow-listed
set of product events. Its schema has no fields for player names, chat, kifu,
hands, seats, partners, opponents, room IDs, or raw IP addresses. Users can
disable analytics in personal settings; doing so deletes the history associated
with that browser's analytics ID.

The existing `GOITA_AI_ADAPTIVE_VALUE_PATH` variable still has priority when
an exact file path is required. Without either setting, local development keeps
using `results/background_search_value.json`. On Render, that fallback is
reported as `render_persistent_data_directory_not_configured` in the runtime
diagnostics and service log.

`GOITA_ROOM_SETTINGS_PATH` can override only the room-settings file location.
Without it or `GOITA_PERSISTENT_DATA_DIR`, room setting changes remain
in-memory and reset on restart, matching the local development default.

`GOITA_ANALYTICS_DB_PATH` can override only the analytics database location.
The administrator dashboard is available directly at `/admin/` and is not
linked from the public top-page settings. Set `GOITA_ADMIN_SESSION_SECRET` to a
long random value when administrator login cookies should remain valid across
deploys; otherwise a secure random value is generated at each process start.

Member accounts use the same persistent directory in a separate database.
`GOITA_MEMBER_DB_PATH` can override its location. Member management requires
a non-default `LOBBY_ADMIN_PASSWORD`; see [MEMBER_ACCOUNTS.md](MEMBER_ACCOUNTS.md)
for issuance, expiry, reset, and session-security details.

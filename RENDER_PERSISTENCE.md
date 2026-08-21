# Render persistent runtime data

The adaptive background-search model and editable room-management settings
write runtime data to JSON files. Render's normal filesystem is ephemeral, so
production needs a persistent disk for these files to survive restarts and
deploys.

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
```

The room settings file contains only editable room-management values: room
name, entry passphrase, AI profile, legal-action visibility, log visibility,
and the configured room background path. It does not contain admin passwords,
hands, scores, occupied seats, or other live match state.

The existing `GOITA_AI_ADAPTIVE_VALUE_PATH` variable still has priority when
an exact file path is required. Without either setting, local development keeps
using `results/background_search_value.json`. On Render, that fallback is
reported as `render_persistent_data_directory_not_configured` in the runtime
diagnostics and service log.

`GOITA_ROOM_SETTINGS_PATH` can override only the room-settings file location.
Without it or `GOITA_PERSISTENT_DATA_DIR`, room setting changes remain
in-memory and reset on restart, matching the local development default.

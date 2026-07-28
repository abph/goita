from __future__ import annotations

import backend.app as app_module


def test_spectator_count_uses_unique_connected_clients_and_excludes_players() -> None:
    game_id = "spectator-count-test"
    game = app_module._create_game_obj()
    game["human_seats"] = {"A": "player-a"}
    app_module.GAMES[game_id] = game

    connection_keys = [
        (game_id, "player-a"),
        (game_id, "watcher-one"),
        (game_id, "watcher-two"),
    ]
    old_connections = {
        key: app_module.manager.client_connections.get(key)
        for key in connection_keys
    }

    try:
        app_module.manager.client_connections[(game_id, "player-a")] = {object()}
        app_module.manager.client_connections[(game_id, "watcher-one")] = {
            object(),
            object(),
        }
        app_module.manager.client_connections[(game_id, "watcher-two")] = {object()}

        state = app_module.get_state(
            game_id,
            viewer="W",
            client_id="watcher-one",
        )
        assert state["spectator_count"] == 2
    finally:
        for key, old_value in old_connections.items():
            if old_value is None:
                app_module.manager.client_connections.pop(key, None)
            else:
                app_module.manager.client_connections[key] = old_value
        app_module.GAMES.pop(game_id, None)


def test_frontend_has_spectator_button_without_old_seat_heading() -> None:
    html = (app_module.FRONTEND_DIR / "index.html").read_text(encoding="utf-8")

    assert '<button id="btnSpectator"' in html
    assert "`観戦${spectatorCount}人`" in html
    assert "async function selectSpectator()" in html
    assert "await leaveSeat(mySeat);" in html
    assert "socket.onopen = () => {" in html
    assert "<b>自分の席</b>" not in html


if __name__ == "__main__":
    test_spectator_count_uses_unique_connected_clients_and_excludes_players()
    test_frontend_has_spectator_button_without_old_seat_heading()
    print("SPECTATOR_COUNT_TEST_OK")

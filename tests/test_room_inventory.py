from __future__ import annotations

import asyncio

from fastapi import HTTPException

import backend.app as app_module


def test_private_b_uses_updated_entry_password() -> None:
    private_b = next(
        room for room in app_module.PRIVATE_ROOM_DEFINITIONS
        if room["gid"] == "room-silver-02"
    )
    assert private_b["pass"] == "1222"
    assert app_module.GAMES["room-silver-02"]["password"] == "1222"


def test_room_names_allow_twelve_characters_without_changing_player_name_limit() -> None:
    assert app_module._sanitize_room_name("1234567890123") == "123456789012"
    assert app_module._sanitize_player_name("1234567890") == "123456789"

    html = (app_module.FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    assert "ルーム名（最大12文字）" in html
    assert 'id="setRoomName" maxlength="12"' in html
    assert "名前（最大9文字）" in html
    assert "const ROOM_NAME_MAX_CHARACTERS = 12;" in html
    assert "function roomNameForDisplay(value)" in html
    assert "escapeLobbyHtml(roomNameForDisplay(room.owner_name))" in html
    assert ".slice(0, ROOM_NAME_MAX_CHARACTERS)" in html

    game_id = "room-copper-04"
    game = app_module.GAMES[game_id]
    original_owner_name = game["owner_name"]
    try:
        game["owner_name"] = "123456789012"
        listed_room = next(
            room
            for room in app_module.list_rooms()["rooms"]
            if room["game_id"] == game_id
        )
        assert listed_room["owner_name"] == "123456789012"
    finally:
        game["owner_name"] = original_owner_name


def test_public_rooms_default_to_three_people_rooms_and_one_ai_room() -> None:
    assert list(app_module.MAIN_ROOM_NAMES) == [
        "main",
        "main-b",
        "main-c",
        "main-e",
        "main-d",
        "main-f",
    ]
    assert app_module.LOBBY_ROOM_SETTINGS["main_room_count"] == 4
    assert app_module.LOBBY_MAIN_ROOM_IDS == (
        "main",
        "main-b",
        "main-c",
        "main-e",
    )
    assert list(app_module.MAIN_ROOM_NAMES.values()) == [
        "みんなでごいたA",
        "みんなでごいたB",
        "みんなでごいたC",
        "AIとごいたA",
        "埼玉的な集会室",
        "AIとごいたB",
    ]
    assert app_module.MAIN_ROOM_DEFAULT_AI_SEATS == {
        "main-e": ("B", "C", "D"),
        "main-f": ("B", "C", "D"),
    }
    assert app_module.GAMES["main-e"]["ai_seats"] == ["B", "C", "D"]
    assert app_module.GAMES["main-f"]["ai_seats"] == ["B", "C", "D"]
    assert app_module.GAMES["main-d"]["hidden_from_lobby"] is True
    assert app_module.GAMES["main-f"]["hidden_from_lobby"] is True
    visible_main_ids = [
        room["game_id"]
        for room in app_module.list_rooms()["rooms"]
        if room["is_main_room"]
    ]
    assert visible_main_ids == ["main", "main-b", "main-c", "main-e"]


def test_private_c_defaults_to_kanazawa_team_saitama_room() -> None:
    room_definition = app_module.PRIVATE_ROOM_DEFINITIONS[2]

    assert app_module.LOBBY_ROOM_SETTINGS["private_room_count"] == 4
    assert room_definition == {
        "gid": "room-bronze-03",
        "pass": "saitama1011",
        "admin": "1011made",
        "owner": "金沢大会チーム埼玉",
    }

    room = app_module.GAMES[room_definition["gid"]]
    assert room["password"] == "saitama1011"
    assert room["admin_password"] == "1011made"
    assert room["owner_name"] == "金沢大会チーム埼玉"
    assert room["hidden_from_lobby"] is False
    assert app_module.GAMES["room-copper-04"]["hidden_from_lobby"] is False
    assert app_module.GAMES["room-iron-05"]["hidden_from_lobby"] is True
    assert app_module.GAMES["room-platinum-06"]["hidden_from_lobby"] is True


def test_lobby_shows_configured_main_rooms_and_two_private_rooms() -> None:
    old_settings = dict(app_module.LOBBY_ROOM_SETTINGS)
    try:
        app_module.LOBBY_ROOM_SETTINGS.update(
            main_room_count=2,
            private_room_count=2,
        )
        app_module.setup_main_rooms()
        app_module.setup_supporter_rooms()
        rooms = app_module.list_rooms()["rooms"]

        main_rooms = [room for room in rooms if room["is_main_room"]]
        private_rooms = [room for room in rooms if not room["is_main_room"]]

        assert [room["game_id"] for room in main_rooms] == list(
            app_module.MAIN_ROOM_NAMES
        )[:2]
        assert [room["owner_name"] for room in main_rooms] == list(
            app_module.MAIN_ROOM_NAMES.values()
        )[:2]
        assert {room["game_id"] for room in private_rooms} == {
            "room-gold-01",
            "room-silver-02",
        }
        assert len(rooms) == 4
        assert app_module.DEBUG_GID not in {room["game_id"] for room in rooms}
    finally:
        app_module.LOBBY_ROOM_SETTINGS.update(old_settings)
        app_module.setup_main_rooms()
        app_module.setup_supporter_rooms()


def test_private_c_through_f_exist_but_are_hidden_when_only_two_are_shown() -> None:
    old_settings = dict(app_module.LOBBY_ROOM_SETTINGS)
    try:
        app_module.LOBBY_ROOM_SETTINGS["private_room_count"] = 2
        app_module.setup_supporter_rooms()
        room_ids = {room["game_id"] for room in app_module.list_rooms()["rooms"]}

        for game_id in (
            "room-bronze-03",
            "room-copper-04",
            "room-iron-05",
            "room-platinum-06",
        ):
            assert game_id in app_module.GAMES
            assert app_module.GAMES[game_id]["hidden_from_lobby"] is True
            assert game_id not in room_ids
    finally:
        app_module.LOBBY_ROOM_SETTINGS.update(old_settings)
        app_module.setup_supporter_rooms()


def test_every_main_room_disables_beginner_support() -> None:
    for game_id in app_module.MAIN_GIDS:
        try:
            app_module.get_beginner_recommendation(
                game_id,
                player="A",
                client_id="test-client",
            )
        except HTTPException as exc:
            assert exc.status_code == 403
        else:
            raise AssertionError(f"{game_id} must reject beginner support")


def test_public_room_reveal_requires_round_end_and_player_consent() -> None:
    game_id = next(iter(app_module.MAIN_ROOM_NAMES))
    old_game = app_module.GAMES[game_id]
    game = app_module._create_game_obj(dealer="A")
    game["is_started"] = True
    game["human_seats"] = {"A": "client-a", "B": "client-b"}
    game["ai_seats"] = ["C", "D"]
    app_module.GAMES[game_id] = game

    try:
        try:
            asyncio.run(
                app_module.reveal_hand(
                    game_id,
                    requester="B",
                    target="B",
                    client_id="client-b",
                )
            )
        except HTTPException as exc:
            assert exc.status_code == 409
        else:
            raise AssertionError("A public-room hand must stay hidden before the round ends")

        game["state"].finished = True
        result = asyncio.run(
            app_module.reveal_hand(
                game_id,
                requester="B",
                target="B",
                client_id="client-b",
            )
        )
        assert result == {"ok": True, "revealed_hand_seats": ["B"]}

        try:
            asyncio.run(
                app_module.reveal_hand(
                    game_id,
                    requester="A",
                    target="B",
                    client_id="client-a",
                )
            )
        except HTTPException as exc:
            assert exc.status_code == 403
        else:
            raise AssertionError("The host must not reveal another human player's hand")

        spectator_state = app_module.get_state(game_id, viewer="W")
        assert isinstance(spectator_state["hands"]["B"], list)
        assert isinstance(spectator_state["hands"]["C"], list)
        assert isinstance(spectator_state["hands"]["D"], list)
        assert spectator_state["hands"]["A"] == {"count": 8}
        assert spectator_state["revealed_hand_seats"] == ["B", "C", "D"]
    finally:
        app_module.GAMES[game_id] = old_game


def test_private_room_allows_ai_reveal_by_any_seated_player_after_round_end() -> None:
    game_id = "test-private-seat-reveal"
    game = app_module._create_game_obj(dealer="A")
    game["is_started"] = True
    game["human_seats"] = {"A": "client-a", "B": "client-b"}
    game["ai_seats"] = ["C", "D"]
    app_module.GAMES[game_id] = game

    try:
        own_result = asyncio.run(
            app_module.reveal_hand(
                game_id,
                requester="B",
                target="B",
                client_id="client-b",
            )
        )
        assert own_result["revealed_hand_seats"] == ["B"]

        ai_result = asyncio.run(
            app_module.reveal_hand(
                game_id,
                requester="A",
                target="C",
                client_id="client-a",
            )
        )
        assert ai_result["revealed_hand_seats"] == ["B", "C"]

        try:
            asyncio.run(
                app_module.reveal_hand(
                    game_id,
                    requester="B",
                    target="D",
                    client_id="client-b",
                )
            )
        except HTTPException as exc:
            assert exc.status_code == 403
        else:
            raise AssertionError("A non-host must not reveal an AI hand before the round ends")

        game["state"].finished = True
        finished_ai_result = asyncio.run(
            app_module.reveal_hand(
                game_id,
                requester="B",
                target="D",
                client_id="client-b",
            )
        )
        assert finished_ai_result["revealed_hand_seats"] == ["B", "C", "D"]

        try:
            asyncio.run(
                app_module.reveal_hand(
                    game_id,
                    requester="A",
                    target="B",
                    client_id="client-a",
                )
            )
        except HTTPException as exc:
            assert exc.status_code == 403
        else:
            raise AssertionError("The private-room host must not reveal another human hand")

        spectator_state = app_module.get_state(game_id, viewer="W")
        assert spectator_state["revealed_hand_seats"] == ["B", "C", "D"]
        assert isinstance(spectator_state["hands"]["B"], list)
        assert isinstance(spectator_state["hands"]["C"], list)
        assert spectator_state["hands"]["A"] == {"count": 8}
        assert isinstance(spectator_state["hands"]["D"], list)

        host_state = app_module.get_state(
            game_id,
            viewer="A",
            client_id="client-a",
            reveal_hands=1,
        )
        assert isinstance(host_state["hands"]["A"], list)
        assert isinstance(host_state["hands"]["D"], list)
    finally:
        app_module.GAMES.pop(game_id, None)


def test_next_round_reset_can_start_immediately_with_score_preserved() -> None:
    game_id = "test-auto-start-next-round"
    game = app_module._create_game_obj(dealer="A")
    game["is_started"] = True
    game["state"].finished = True
    game["human_seats"] = {"A": "client-a"}
    game["total_team_score"] = {"AC": 40, "BD": 30}
    game["round_count"] = 3
    game["revealed_hand_seats"] = ["A", "C"]
    app_module.GAMES[game_id] = game

    try:
        asyncio.run(
            app_module.reset_game(
                game_id,
                dealer="C",
                requester="A",
                client_id="client-a",
                keep_score=True,
                auto_start=True,
            )
        )
        next_game = app_module.GAMES[game_id]
        assert next_game["is_started"] is True
        assert next_game["dealer"] == "C"
        assert next_game["round_count"] == 4
        assert next_game["total_team_score"] == {"AC": 40, "BD": 30}
        assert next_game["revealed_hand_seats"] == []
        assert next_game["log"] == ["Game start. dealer=C"]

        next_game["state"].finished = True
        asyncio.run(
            app_module.reset_game_config(
                game_id,
                app_module.ResetConfigBody(
                    dealer="D",
                    requester="A",
                    client_id="client-a",
                    keep_score=True,
                    auto_start=True,
                ),
            )
        )
        configured_game = app_module.GAMES[game_id]
        assert configured_game["is_started"] is True
        assert configured_game["dealer"] == "D"
        assert configured_game["round_count"] == 5
        assert configured_game["total_team_score"] == {"AC": 40, "BD": 30}
        assert configured_game["log"] == ["Game start. dealer=D"]
    finally:
        app_module.GAMES.pop(game_id, None)


def test_frontend_recognizes_all_main_room_ids() -> None:
    html = (app_module.FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    frontend_ids = ["MAIN_GID", *[
        f'"{game_id}"'
        for game_id in app_module.MAIN_ROOM_NAMES
        if game_id != app_module.MAIN_GID
    ]]
    expected_set = f"const MAIN_ROOM_IDS = new Set([{', '.join(frontend_ids)}]);"
    assert expected_set in html
    assert "room.is_main_room === true" in html
    assert 'id="mainRoomPeopleCount"' not in html
    assert 'id="privateRoomPeopleCount"' not in html
    assert 'id="lobbySettingsModal"' in html
    assert 'id="lobbyAdminFields"' not in html
    assert 'id="lobbyMainPeopleCount"' not in html
    assert 'id="lobbyPrivatePeopleCount"' not in html
    assert 'id="lobbyAllPeopleCount"' not in html
    admin_html = (app_module.FRONTEND_DIR / "admin.html").read_text(encoding="utf-8")
    assert 'id="mainRoomCount"' in admin_html
    assert 'id="privateRoomCount"' in admin_html
    assert 'id="privateRoomPasswords"' in admin_html
    assert '"/admin/api/private-room-password"' in admin_html
    assert "function updateLobbyAdminPeopleCounts(roomTotals = null)" not in html
    assert 'id="handRevealConfirmModal"' in html
    assert "if(!await confirmHandReveal(target)) return;" in html
    assert "if(!window.confirm(message)) return;" not in html
    assert "data.room_totals" not in html
    assert "function openLobbySettings()" in html
    assert 'id="lobbyAdminSettingsPanel"' not in html
    assert '"/admin/api/login"' in admin_html
    assert '"/admin/api/settings"' in admin_html
    assert 'safeCount === 1 ? "person" : "people"' not in html
    assert 'id="handRevealPanel"' in html
    assert "function renderHandRevealControls(state)" in html
    assert 'id="lobbyCheckAutoRevealOwnHand"' in html
    assert 'id="lobbyCheckAutoRevealAiHands"' in html
    assert 'id="checkAutoRevealOwnHand"' in html
    assert 'id="checkAutoRevealAiHands"' in html
    assert "function maybeAutoRevealOwnAndAiHands(state)" in html
    assert "personalSettings.autoRevealOwnHand !== true && personalSettings.autoRevealAiHands === false" in html
    assert "/reveal_hand?${qs.toString()}" in html
    assert "toggle_reveal_hands" not in html
    assert "auto_start: autoStart" in html
    assert "auto_start: String(!!autoStart)" in html
    assert 'nextRoundButton.textContent = uiText("配牌中...")' in html


def test_lobby_admin_can_change_visible_room_counts() -> None:
    old_settings = dict(app_module.LOBBY_ROOM_SETTINGS)
    try:
        payload = app_module.verify_lobby_admin(app_module.LOBBY_ADMIN_PASSWORD)
        assert payload["main_room_max"] == len(app_module.LOBBY_MAIN_ROOM_IDS)
        assert payload["private_room_max"] == len(
            app_module.PRIVATE_ROOM_DEFINITIONS
        )
        assert payload["private_room_max"] == 6
        assert len(payload["private_rooms"]) == 6
        assert all("admin_password" not in room for room in payload["private_rooms"])

        expanded = asyncio.run(
            app_module.update_lobby_admin_settings(
                app_module.LobbySettingsUpdateRequest(
                    admin_password=app_module.LOBBY_ADMIN_PASSWORD,
                    main_room_count=4,
                    private_room_count=4,
                )
            )
        )
        expanded_rooms = app_module.list_rooms()["rooms"]
        assert expanded["main_room_count"] == 4
        assert expanded["private_room_count"] == 4
        assert len([room for room in expanded_rooms if room["is_main_room"]]) == 4
        assert len([room for room in expanded_rooms if not room["is_main_room"]]) == 4

        reduced = asyncio.run(
            app_module.update_lobby_admin_settings(
                app_module.LobbySettingsUpdateRequest(
                    admin_password=app_module.LOBBY_ADMIN_PASSWORD,
                    main_room_count=1,
                    private_room_count=0,
                )
            )
        )
        reduced_rooms = app_module.list_rooms()["rooms"]
        assert reduced["main_room_count"] == 1
        assert reduced["private_room_count"] == 0
        assert len([room for room in reduced_rooms if room["is_main_room"]]) == 1
        assert not [room for room in reduced_rooms if not room["is_main_room"]]

        try:
            app_module.verify_lobby_admin("wrong-password")
        except HTTPException as exc:
            assert exc.status_code == 401
        else:
            raise AssertionError("wrong lobby admin password must be rejected")
    finally:
        app_module.LOBBY_ROOM_SETTINGS.update(old_settings)
        app_module.setup_main_rooms()
        app_module.setup_supporter_rooms()


def test_room_list_counts_human_players_and_spectators_without_ai() -> None:
    app_module.setup_main_rooms()
    game_id = next(iter(app_module.MAIN_ROOM_NAMES))
    game = app_module.GAMES[game_id]
    old_human_seats = game.get("human_seats")
    old_ai_seats = game.get("ai_seats")
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
        game["human_seats"] = {"A": "player-a"}
        game["ai_seats"] = {"B", "C"}
        for key in connection_keys:
            app_module.manager.client_connections[key] = {object()}

        room = next(
            room
            for room in app_module.list_rooms()["rooms"]
            if room["game_id"] == game_id
        )

        assert room["player_count"] == 3
        assert room["human_count"] == 1
        assert room["spectator_count"] == 2
        assert room["people_count"] == 3
        expected_main_people = sum(
            len(app_module._human_seat_set(data))
            + app_module.manager.spectator_count(room_id, data)
            for room_id, data in app_module.GAMES.items()
            if app_module._is_main_game_id(room_id)
        )
        assert app_module.list_rooms()["room_totals"]["main_people_count"] == expected_main_people
    finally:
        game["human_seats"] = old_human_seats
        game["ai_seats"] = old_ai_seats
        for key, old_value in old_connections.items():
            if old_value is None:
                app_module.manager.client_connections.pop(key, None)
            else:
                app_module.manager.client_connections[key] = old_value


def test_private_total_includes_hidden_private_rooms_but_not_debug_room() -> None:
    app_module.setup_supporter_rooms()
    hidden_game_id = "room-iron-05"
    hidden_game = app_module.GAMES[hidden_game_id]
    old_human_seats = hidden_game.get("human_seats")

    try:
        hidden_game["human_seats"] = {"A": "hidden-private-player"}
        response = app_module.list_rooms()
        listed_ids = {room["game_id"] for room in response["rooms"]}
        expected_private_people = sum(
            len(app_module._human_seat_set(data))
            + app_module.manager.spectator_count(room_id, data)
            for room_id, data in app_module.GAMES.items()
            if not app_module._is_main_game_id(room_id)
            and room_id != app_module.DEBUG_GID
            and not data.get("is_debug_room", False)
        )

        assert hidden_game_id not in listed_ids
        assert response["room_totals"]["private_people_count"] == expected_private_people
        assert response["room_totals"]["private_people_count"] >= 1
    finally:
        hidden_game["human_seats"] = old_human_seats


def test_room_list_returns_named_site_presence_without_client_ids() -> None:
    app_module.setup_main_rooms()
    game_id = next(iter(app_module.MAIN_ROOM_NAMES))
    game = app_module.GAMES[game_id]
    old_human_seats = game.get("human_seats")
    old_player_names = game.get("player_names")
    connection_keys = [("lobby", "lobby-person"), (game_id, "room-person")]
    old_connections = {
        key: app_module.manager.client_connections.get(key)
        for key in connection_keys
    }
    old_names = {
        key: app_module.manager.client_names.get(key)
        for key in connection_keys
    }

    try:
        game["human_seats"] = {"B": "room-person"}
        game["player_names"] = {"A": "", "B": "山田", "C": "", "D": ""}
        for key in connection_keys:
            app_module.manager.client_connections[key] = {object()}
        app_module.manager.client_names[("lobby", "lobby-person")] = "鈴木"
        app_module.manager.client_names[(game_id, "room-person")] = "山田"

        people = app_module.list_rooms()["site_people"]

        assert {
            "name": "鈴木",
            "name_is_default": False,
            "tag": "",
            "location": "トップページ",
            "role": "lobby",
            "seat": "",
        } in people
        assert {
            "name": "山田",
            "name_is_default": False,
            "tag": "",
            "location": app_module.MAIN_ROOM_NAMES[game_id],
            "role": "player",
            "seat": "B",
        } in people
        assert "lobby-person" not in str(people)
        assert "room-person" not in str(people)
    finally:
        game["human_seats"] = old_human_seats
        game["player_names"] = old_player_names
        for key, old_value in old_connections.items():
            if old_value is None:
                app_module.manager.client_connections.pop(key, None)
            else:
                app_module.manager.client_connections[key] = old_value
        for key, old_value in old_names.items():
            if old_value is None:
                app_module.manager.client_names.pop(key, None)
            else:
                app_module.manager.client_names[key] = old_value


def test_site_presence_prioritizes_the_viewers_location_and_seat_order() -> None:
    app_module.setup_main_rooms()
    app_module.setup_supporter_rooms()
    main_room_ids = list(app_module.MAIN_ROOM_NAMES)
    viewer_room_id = main_room_ids[1]
    other_room_id = main_room_ids[0]
    private_room_id = app_module.PRIVATE_A_GID
    affected_games = {
        room_id: app_module.GAMES[room_id]
        for room_id in (viewer_room_id, other_room_id, private_room_id)
    }
    old_game_values = {
        room_id: (
            game.get("human_seats"),
            game.get("player_names"),
        )
        for room_id, game in affected_games.items()
    }
    old_connections = dict(app_module.manager.client_connections)
    old_names = dict(app_module.manager.client_names)

    try:
        app_module.manager.client_connections.clear()
        app_module.manager.client_names.clear()
        affected_games[viewer_room_id]["human_seats"] = {
            "A": "same-a",
            "B": "same-b",
        }
        affected_games[viewer_room_id]["player_names"] = {
            "A": "同室A",
            "B": "同室B",
            "C": "",
            "D": "",
        }
        affected_games[other_room_id]["human_seats"] = {"C": "other-c"}
        affected_games[other_room_id]["player_names"] = {
            "A": "",
            "B": "",
            "C": "別室C",
            "D": "",
        }
        affected_games[private_room_id]["human_seats"] = {"D": "private-d"}
        affected_games[private_room_id]["player_names"] = {
            "A": "",
            "B": "",
            "C": "",
            "D": "秘密D",
        }

        connection_names = {
            (viewer_room_id, "same-b"): "同室B",
            (viewer_room_id, "same-spectator"): "同室観戦",
            (viewer_room_id, "same-a"): "同室A",
            ("lobby", "lobby-person"): "ロビー",
            (other_room_id, "other-c"): "別室C",
            (private_room_id, "private-d"): "秘密D",
        }
        for key, name in connection_names.items():
            app_module.manager.client_connections[key] = {object()}
            app_module.manager.client_names[key] = name

        people = app_module.list_rooms(
            viewer_game_id=viewer_room_id,
            client_id="same-a",
        )["site_people"]

        assert [person["name"] for person in people] == [
            "同室A",
            "同室B",
            "同室観戦",
            "ロビー",
            "別室C",
            "＊＊＊＊",
        ]
        assert [person["seat"] for person in people[:3]] == ["A", "B", ""]
    finally:
        app_module.manager.client_connections.clear()
        app_module.manager.client_connections.update(old_connections)
        app_module.manager.client_names.clear()
        app_module.manager.client_names.update(old_names)
        for room_id, (human_seats, player_names) in old_game_values.items():
            affected_games[room_id]["human_seats"] = human_seats
            affected_games[room_id]["player_names"] = player_names


def test_debug_room_presence_is_completely_hidden() -> None:
    app_module.setup_debug_room()
    game = app_module.GAMES[app_module.DEBUG_GID]
    player_id = "debug-player"
    spectator_id = "debug-spectator"
    connection_keys = [
        ("lobby", player_id),
        (app_module.DEBUG_GID, player_id),
        (app_module.DEBUG_GID, spectator_id),
    ]
    old_human_seats = game.get("human_seats")
    old_player_names = game.get("player_names")
    old_connections = {
        key: app_module.manager.client_connections.get(key)
        for key in connection_keys
    }
    old_names = {
        key: app_module.manager.client_names.get(key)
        for key in connection_keys
    }

    try:
        game["human_seats"] = {"A": player_id}
        game["player_names"] = {"A": "デバッグ参加者", "B": "", "C": "", "D": ""}
        for key in connection_keys:
            app_module.manager.client_connections[key] = {object()}
        app_module.manager.client_names[("lobby", player_id)] = "デバッグ参加者"
        app_module.manager.client_names[(app_module.DEBUG_GID, player_id)] = "デバッグ参加者"
        app_module.manager.client_names[(app_module.DEBUG_GID, spectator_id)] = "デバッグ観戦者"

        people = app_module.list_rooms()["site_people"]

        assert "デバッグ参加者" not in str(people)
        assert "デバッグ観戦者" not in str(people)
        assert "デバッグルーム" not in str(people)
    finally:
        game["human_seats"] = old_human_seats
        game["player_names"] = old_player_names
        for key, old_value in old_connections.items():
            if old_value is None:
                app_module.manager.client_connections.pop(key, None)
            else:
                app_module.manager.client_connections[key] = old_value
        for key, old_value in old_names.items():
            if old_value is None:
                app_module.manager.client_names.pop(key, None)
            else:
                app_module.manager.client_names[key] = old_value


def test_private_room_presence_masks_names_outside_the_same_room() -> None:
    app_module.setup_supporter_rooms()
    game_id = app_module.PRIVATE_A_GID
    game = app_module.GAMES[game_id]
    client_id = "private-room-person"
    key = (game_id, client_id)
    old_human_seats = game.get("human_seats")
    old_player_names = game.get("player_names")
    old_connections = app_module.manager.client_connections.get(key)
    old_name = app_module.manager.client_names.get(key)

    try:
        game["human_seats"] = {"C": client_id}
        game["player_names"] = {"A": "", "B": "", "C": "秘密の名前", "D": ""}
        app_module.manager.client_connections[key] = {object()}
        app_module.manager.client_names[key] = "秘密の名前"

        lobby_response = app_module.list_rooms()
        lobby_people = lobby_response["site_people"]
        assert {
            "name": "＊＊＊＊",
            "name_is_default": False,
            "tag": "",
            "location": app_module.PRIVATE_ROOM_NAMES[game_id],
            "role": "player",
            "seat": "C",
        } in lobby_people
        assert "秘密の名前" not in str(lobby_people)
        lobby_room = next(
            room for room in lobby_response["rooms"] if room["game_id"] == game_id
        )
        assert lobby_room["seats"]["C"] == "＊＊＊＊"

        same_room_response = app_module.list_rooms(
            viewer_game_id=game_id,
            client_id=client_id,
        )
        same_room_people = same_room_response["site_people"]
        assert {
            "name": "秘密の名前",
            "name_is_default": False,
            "tag": "",
            "location": app_module.PRIVATE_ROOM_NAMES[game_id],
            "role": "player",
            "seat": "C",
        } in same_room_people
        same_room = next(
            room for room in same_room_response["rooms"] if room["game_id"] == game_id
        )
        assert same_room["seats"]["C"] == "秘密の名前"

        unrelated_people = app_module.list_rooms(
            viewer_game_id=game_id,
            client_id="not-connected",
        )["site_people"]
        assert "秘密の名前" not in str(unrelated_people)
    finally:
        game["human_seats"] = old_human_seats
        game["player_names"] = old_player_names
        if old_connections is None:
            app_module.manager.client_connections.pop(key, None)
        else:
            app_module.manager.client_connections[key] = old_connections
        if old_name is None:
            app_module.manager.client_names.pop(key, None)
        else:
            app_module.manager.client_names[key] = old_name


if __name__ == "__main__":
    test_private_b_uses_updated_entry_password()
    test_room_names_allow_twelve_characters_without_changing_player_name_limit()
    test_public_rooms_default_to_three_people_rooms_and_one_ai_room()
    test_private_c_defaults_to_kanazawa_team_saitama_room()
    test_lobby_shows_configured_main_rooms_and_two_private_rooms()
    test_private_c_through_f_exist_but_are_hidden_when_only_two_are_shown()
    test_every_main_room_disables_beginner_support()
    test_public_room_reveal_requires_round_end_and_player_consent()
    test_private_room_allows_ai_reveal_by_any_seated_player_after_round_end()
    test_next_round_reset_can_start_immediately_with_score_preserved()
    test_frontend_recognizes_all_main_room_ids()
    test_lobby_admin_can_change_visible_room_counts()
    test_room_list_counts_human_players_and_spectators_without_ai()
    test_private_total_includes_hidden_private_rooms_but_not_debug_room()
    test_room_list_returns_named_site_presence_without_client_ids()
    test_site_presence_prioritizes_the_viewers_location_and_seat_order()
    test_debug_room_presence_is_completely_hidden()
    test_private_room_presence_masks_names_outside_the_same_room()
    print("ROOM_INVENTORY_TEST_OK")

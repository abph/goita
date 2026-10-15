import copy
import asyncio
import time

import pytest
from fastapi import HTTPException

from backend import app as app_module


def participant(name: str, *, wait_offset: float = 0.0, pair_with: str = ""):
    return {
        "name": name,
        "tag": "",
        "status": "waiting",
        "wait_since": time.monotonic() - wait_offset,
        "games": 0,
        "pair_with": pair_with,
        "assigned_table_id": "",
    }


def test_rotation_replaces_exactly_two_people_when_two_waiters_exist(monkeypatch):
    players = {
        client_id: participant(client_id, wait_offset=index)
        for index, client_id in enumerate(("a", "b", "c", "d", "e", "f"), 1)
    }
    monkeypatch.setattr(app_module, "GATHERING_PARTICIPANTS", players)
    monkeypatch.setattr(app_module, "GATHERING_LAST_PLAYERS", ["a", "b", "c", "d"])

    selected = app_module._select_gathering_players(list(players), allow_ai=False)

    assert len(selected) == 4
    assert len(set(selected) - {"a", "b", "c", "d"}) == 2
    assert len(set(selected) & {"a", "b", "c", "d"}) == 2


def test_fixed_pair_is_never_split_by_matchmaking(monkeypatch):
    players = {
        "pair-a": participant("pair-a", pair_with="pair-b"),
        "pair-b": participant("pair-b", pair_with="pair-a"),
        "solo-c": participant("solo-c"),
        "solo-d": participant("solo-d"),
        "solo-e": participant("solo-e"),
    }
    monkeypatch.setattr(app_module, "GATHERING_PARTICIPANTS", players)
    monkeypatch.setattr(app_module, "GATHERING_LAST_PLAYERS", [])

    selected = set(app_module._select_gathering_players(list(players), allow_ai=False))

    assert ("pair-a" in selected) == ("pair-b" in selected)


def test_two_fixed_pairs_are_preferred_as_two_complete_teams(monkeypatch):
    players = {
        "pair-a": participant("pair-a", pair_with="pair-b"),
        "pair-b": participant("pair-b", pair_with="pair-a"),
        "pair-c": participant("pair-c", pair_with="pair-d"),
        "pair-d": participant("pair-d", pair_with="pair-c"),
        "solo-e": participant("solo-e"),
        "solo-f": participant("solo-f"),
    }
    monkeypatch.setattr(app_module, "GATHERING_PARTICIPANTS", players)
    monkeypatch.setattr(app_module, "GATHERING_LAST_PLAYERS", [])

    selected = set(app_module._select_gathering_players(list(players), allow_ai=False))

    assert selected == {"pair-a", "pair-b", "pair-c", "pair-d"}


def test_five_people_wait_for_a_sixth_before_one_person_fallback(monkeypatch):
    monkeypatch.setattr(app_module, "GATHERING_LAST_PLAYERS", ["a", "b", "c", "d"])

    assert app_module._gathering_waits_for_two_changes(
        ["a", "b", "c", "d", "e"], allow_fallback=False
    ) is True
    assert app_module._gathering_waits_for_two_changes(
        ["a", "b", "c", "d", "e"], allow_fallback=True
    ) is False
    assert app_module._gathering_waits_for_two_changes(
        ["a", "b", "c", "d", "e", "f"], allow_fallback=False
    ) is False


def test_fixed_pair_is_seated_as_partners_and_ai_fills_shortage(monkeypatch):
    old_game = app_module._create_game_obj(dealer="A")
    old_game["owner_name"] = "どこかの集会室"
    players = {
        "pair-a": participant("Aさん", pair_with="pair-b"),
        "pair-b": participant("Bさん", pair_with="pair-a"),
        "solo": participant("Cさん"),
    }
    games = dict(app_module.GAMES)
    games[app_module.MEETING_ROOM_GID] = old_game
    monkeypatch.setattr(app_module, "GAMES", games)
    monkeypatch.setattr(app_module, "GATHERING_PARTICIPANTS", players)
    monkeypatch.setattr(app_module, "GATHERING_LAST_PLAYERS", [])
    monkeypatch.setattr(app_module, "_arm_turn_timeout", lambda _game_id: None)

    app_module._gathering_replace_game(["pair-a", "pair-b", "solo"])

    game = games[app_module.MEETING_ROOM_GID]
    pair_seats = {
        seat for seat, owner in game["human_seats"].items()
        if owner in {"pair-a", "pair-b"}
    }
    assert pair_seats in ({"A", "C"}, {"B", "D"})
    assert len(game["human_seats"]) == 3
    assert len(game["ai_seats"]) == 1
    assert game["is_started"] is True
    assert game["gathering_match"] is True


def test_matchmaking_countdown_starts_game_automatically(monkeypatch):
    async def scenario():
        players = {
            client_id: participant(client_id)
            for client_id in ("a", "b", "c", "d")
        }
        games = dict(app_module.GAMES)
        monkeypatch.setattr(app_module, "GAMES", games)
        monkeypatch.setattr(app_module, "GATHERING_PARTICIPANTS", players)
        monkeypatch.setattr(app_module, "GATHERING_LAST_PLAYERS", [])
        monkeypatch.setattr(app_module, "GATHERING_TABLE_META", {})
        monkeypatch.setattr(app_module, "GATHERING_TABLE_TASKS", {})
        monkeypatch.setattr(app_module, "GATHERING_WAKE_TASK", None)
        monkeypatch.setattr(app_module, "GATHERING_WAIT_DEADLINE", time.monotonic() + 60)
        monkeypatch.setattr(app_module, "GATHERING_COUNTDOWN_SECONDS", 0.01)
        monkeypatch.setattr(app_module, "GATHERING_MATCH_TASK", None)
        monkeypatch.setitem(app_module.GATHERING_EVENT_SETTINGS, "enabled", True)
        monkeypatch.setitem(app_module.GATHERING_EVENT_SETTINGS, "ending", False)
        monkeypatch.setattr(
            app_module.manager,
            "has_client_connection",
            lambda game_id, client_id: game_id == app_module.MEETING_ROOM_GID and client_id in players,
        )

        async def no_broadcast(*_args, **_kwargs):
            return None

        monkeypatch.setattr(app_module.manager, "broadcast_update", no_broadcast)
        monkeypatch.setattr(app_module, "_arm_turn_timeout", lambda _game_id: None)

        await app_module._gathering_matchmaking_worker()
        await asyncio.gather(*list(app_module.GATHERING_TABLE_TASKS.values()))

        game = games[app_module.GATHERING_TABLE_IDS[0]]
        assert game["is_started"] is True
        assert set(game["human_seats"].values()) == set(players)
        assert all(item["status"] == "playing" for item in players.values())

    asyncio.run(scenario())


def test_matchmaking_opens_at_most_four_tables(monkeypatch):
    async def scenario():
        players = {
            f"p{index}": participant(f"p{index}", wait_offset=index)
            for index in range(20)
        }
        games = dict(app_module.GAMES)
        monkeypatch.setattr(app_module, "GAMES", games)
        monkeypatch.setattr(app_module, "GATHERING_PARTICIPANTS", players)
        monkeypatch.setattr(app_module, "GATHERING_LAST_PLAYERS", [])
        monkeypatch.setattr(app_module, "GATHERING_TABLE_META", {})
        monkeypatch.setattr(app_module, "GATHERING_TABLE_TASKS", {})
        monkeypatch.setattr(app_module, "GATHERING_WAKE_TASK", None)
        monkeypatch.setattr(app_module, "GATHERING_COUNTDOWN_SECONDS", 0.01)
        monkeypatch.setitem(app_module.GATHERING_EVENT_SETTINGS, "enabled", True)
        monkeypatch.setitem(app_module.GATHERING_EVENT_SETTINGS, "ending", False)
        monkeypatch.setattr(
            app_module.manager,
            "has_client_connection",
            lambda game_id, client_id: game_id == app_module.MEETING_ROOM_GID and client_id in players,
        )

        async def no_broadcast(*_args, **_kwargs):
            return None

        monkeypatch.setattr(app_module.manager, "broadcast_update", no_broadcast)
        monkeypatch.setattr(app_module, "_arm_turn_timeout", lambda _game_id: None)

        await app_module._gathering_matchmaking_worker()
        assert len(app_module.GATHERING_TABLE_TASKS) == 4
        await asyncio.gather(*list(app_module.GATHERING_TABLE_TASKS.values()))

        active_games = [
            games[table_id]
            for table_id in app_module.GATHERING_TABLE_IDS
            if table_id in games
        ]
        assert len(active_games) == 4
        assert sum(len(game["human_seats"]) for game in active_games) == 16
        assert sum(item["status"] == "waiting" for item in players.values()) == 4

    asyncio.run(scenario())


def test_ai_requires_button_after_five_minutes(monkeypatch):
    async def scenario():
        players = {
            client_id: participant(client_id, wait_offset=app_module.GATHERING_WAIT_SECONDS + 1)
            for client_id in ("a", "b", "c")
        }
        games = dict(app_module.GAMES)
        monkeypatch.setattr(app_module, "GAMES", games)
        monkeypatch.setattr(app_module, "GATHERING_PARTICIPANTS", players)
        monkeypatch.setattr(app_module, "GATHERING_TABLE_META", {})
        monkeypatch.setattr(app_module, "GATHERING_TABLE_TASKS", {})
        monkeypatch.setattr(app_module, "GATHERING_WAKE_TASK", None)
        monkeypatch.setattr(app_module, "GATHERING_COUNTDOWN_SECONDS", 0.01)
        monkeypatch.setitem(app_module.GATHERING_EVENT_SETTINGS, "enabled", True)
        monkeypatch.setitem(app_module.GATHERING_EVENT_SETTINGS, "ending", False)
        monkeypatch.setattr(
            app_module.manager,
            "has_client_connection",
            lambda game_id, client_id: game_id == app_module.MEETING_ROOM_GID and client_id in players,
        )

        async def no_broadcast(*_args, **_kwargs):
            return None

        monkeypatch.setattr(app_module.manager, "broadcast_update", no_broadcast)
        monkeypatch.setattr(app_module, "_arm_turn_timeout", lambda _game_id: None)

        await app_module._gathering_matchmaking_worker()
        assert not any(table_id in games for table_id in app_module.GATHERING_TABLE_IDS)
        assert app_module._gathering_event_public_payload("a")["can_start_with_ai"] is True

        table_id = await app_module._start_gathering_table_with_ai("a")
        await app_module.GATHERING_TABLE_TASKS[table_id]
        assert len(games[table_id]["human_seats"]) == 3
        assert len(games[table_id]["ai_seats"]) == 1

    asyncio.run(scenario())


def test_fourth_human_cancels_ai_countdown(monkeypatch):
    async def scenario():
        players = {
            client_id: participant(client_id, wait_offset=app_module.GATHERING_WAIT_SECONDS + 1)
            for client_id in ("a", "b", "c", "d")
        }
        for client_id in ("a", "b", "c"):
            players[client_id]["status"] = "selected"
            players[client_id]["assigned_table_id"] = app_module.GATHERING_TABLE_IDS[0]
        games = dict(app_module.GAMES)
        monkeypatch.setattr(app_module, "GAMES", games)
        monkeypatch.setattr(app_module, "GATHERING_PARTICIPANTS", players)
        monkeypatch.setattr(app_module, "GATHERING_TABLE_META", {
            app_module.GATHERING_TABLE_IDS[0]: {
                "status": "countdown",
                "players": ["a", "b", "c"],
                "ai_planned": True,
                "last_players": [],
            }
        })
        pending = asyncio.create_task(asyncio.sleep(60))
        monkeypatch.setattr(
            app_module,
            "GATHERING_TABLE_TASKS",
            {app_module.GATHERING_TABLE_IDS[0]: pending},
        )
        monkeypatch.setattr(app_module, "GATHERING_AI_REQUESTED_BY", "a")
        monkeypatch.setattr(
            app_module.manager,
            "has_client_connection",
            lambda _game_id, client_id: client_id in players,
        )

        app_module._cancel_ai_countdown_for_full_human_table()
        await asyncio.sleep(0)

        assert all(item["status"] == "waiting" for item in players.values())
        assert all(not item["assigned_table_id"] for item in players.values())
        assert app_module.GATHERING_TABLE_TASKS == {}
        assert pending.cancelled()
        assert app_module.GATHERING_AI_REQUESTED_BY == ""

    asyncio.run(scenario())


def test_ai_start_is_unavailable_while_human_table_is_active(monkeypatch):
    players = {
        client_id: participant(client_id, wait_offset=app_module.GATHERING_WAIT_SECONDS + 1)
        for client_id in ("a", "b", "c")
    }
    game = app_module._create_game_obj(dealer="A")
    game["is_started"] = True
    game["match_finished"] = False
    games = dict(app_module.GAMES)
    games[app_module.GATHERING_TABLE_IDS[0]] = game
    monkeypatch.setattr(app_module, "GAMES", games)
    monkeypatch.setattr(app_module, "GATHERING_PARTICIPANTS", players)
    monkeypatch.setattr(app_module, "GATHERING_TABLE_META", {
        app_module.GATHERING_TABLE_IDS[0]: {"status": "playing", "uses_ai": False}
    })
    monkeypatch.setattr(
        app_module.manager,
        "has_client_connection",
        lambda _game_id, client_id: client_id in players,
    )

    assert app_module._gathering_ai_start_state("a") == (False, 0)


def test_gathering_table_snapshot_hides_facedown_receive_piece(monkeypatch):
    table_id = app_module.GATHERING_TABLE_IDS[0]
    game = app_module._create_game_obj(dealer="A")
    game["board"]["A"]["receive"][0] = "角"
    game["board"]["A"]["receive_hidden"][0] = True
    game["board"]["A"]["attack"][0] = "銀"
    games = dict(app_module.GAMES)
    games[table_id] = game
    monkeypatch.setattr(app_module, "GAMES", games)
    monkeypatch.setattr(app_module, "GATHERING_TABLE_META", {
        table_id: {"status": "playing", "uses_ai": False}
    })

    snapshot = app_module._gathering_table_snapshot(table_id)

    assert snapshot is not None
    assert snapshot["board_public"]["A"]["receive"][0] == "■"
    assert snapshot["board_public"]["A"]["attack"][0] == "銀"
    assert snapshot["finished"] is False
    assert snapshot["attacker"] == ""
    assert snapshot["round_count"] == 1


def test_gathering_table_snapshot_names_waiting_room_and_selected_players(monkeypatch):
    table_id = app_module.GATHERING_TABLE_IDS[1]
    players = {
        "a": participant("Aさん"),
        "b": participant("Bさん"),
    }
    monkeypatch.setattr(app_module, "GATHERING_PARTICIPANTS", players)
    monkeypatch.setattr(app_module, "GATHERING_TABLE_META", {
        table_id: {
            "status": "countdown",
            "players": ["a", "b"],
            "countdown_until": time.monotonic() + 15,
        }
    })
    monkeypatch.setattr(app_module, "GAMES", {})

    snapshot = app_module._gathering_table_snapshot(table_id)

    assert app_module._gathering_table_label(table_id) == "第2卓"
    assert snapshot is not None
    assert snapshot["name"] == "第2卓"
    assert snapshot["selected_names"] == ["Aさん", "Bさん"]


def test_gathering_table_uses_public_room_visibility_rules():
    assert app_module._is_main_game_id(app_module.GATHERING_TABLE_IDS[0]) is True


def test_gathering_table_cannot_reveal_hands_during_play(monkeypatch):
    async def scenario():
        table_id = app_module.GATHERING_TABLE_IDS[0]
        game = app_module._create_game_obj(dealer="A")
        game["is_started"] = True
        game["human_seats"] = {"A": "player-a"}
        games = dict(app_module.GAMES)
        games[table_id] = game
        monkeypatch.setattr(app_module, "GAMES", games)

        with pytest.raises(HTTPException) as error:
            await app_module.reveal_hand(
                table_id,
                requester="A",
                target="A",
                client_id="player-a",
                visible=True,
            )
        assert error.value.status_code == 409

    asyncio.run(scenario())


def test_ending_event_waits_until_every_active_table_finishes(monkeypatch):
    async def scenario():
        first_table, second_table = app_module.GATHERING_TABLE_IDS[:2]
        players = {
            client_id: participant(client_id)
            for client_id in ("a", "b", "c", "d", "e", "f", "g", "h")
        }
        for client_id in ("a", "b", "c", "d"):
            players[client_id]["status"] = "playing"
            players[client_id]["assigned_table_id"] = first_table
        for client_id in ("e", "f", "g", "h"):
            players[client_id]["status"] = "playing"
            players[client_id]["assigned_table_id"] = second_table
        games = dict(app_module.GAMES)
        for table_id in (first_table, second_table):
            game = app_module._create_game_obj(dealer="A")
            game["is_started"] = True
            game["match_finished"] = False
            games[table_id] = game
        monkeypatch.setattr(app_module, "GAMES", games)
        monkeypatch.setattr(app_module, "GATHERING_PARTICIPANTS", players)
        monkeypatch.setattr(app_module, "GATHERING_PAIR_REQUESTS", {})
        monkeypatch.setattr(app_module, "GATHERING_LAST_PLAYERS", [])
        monkeypatch.setattr(app_module, "GATHERING_PAIR_HISTORY", app_module.Counter())
        monkeypatch.setattr(app_module, "GATHERING_TABLE_TASKS", {})
        monkeypatch.setattr(app_module, "GATHERING_TABLE_META", {
            first_table: {"status": "playing", "uses_ai": False},
            second_table: {"status": "playing", "uses_ai": False},
        })
        monkeypatch.setitem(app_module.GATHERING_EVENT_SETTINGS, "enabled", True)
        monkeypatch.setitem(app_module.GATHERING_EVENT_SETTINGS, "ending", True)
        monkeypatch.setitem(app_module.GATHERING_EVENT_SETTINGS, "event_id", "ending-test")
        monkeypatch.setattr(app_module, "ROOM_SETTINGS_PATH", None)

        async def no_broadcast(*_args, **_kwargs):
            return None

        monkeypatch.setattr(app_module, "_gathering_broadcast", no_broadcast)
        monkeypatch.setattr(app_module, "_cleanup_gathering_tables_after_delay", no_broadcast)

        games[first_table]["match_finished"] = True
        await app_module._gathering_match_completed(first_table)
        assert app_module.GATHERING_EVENT_SETTINGS["enabled"] is True
        assert app_module.GATHERING_EVENT_SETTINGS["ending"] is True

        games[second_table]["match_finished"] = True
        await app_module._gathering_match_completed(second_table)
        assert app_module.GATHERING_EVENT_SETTINGS["enabled"] is False
        assert app_module.GATHERING_EVENT_SETTINGS["ending"] is False

    asyncio.run(scenario())


def test_lobby_shows_gathering_and_hides_human_public_rooms(monkeypatch):
    event = copy.deepcopy(app_module.GATHERING_EVENT_SETTINGS)
    monkeypatch.setitem(app_module.GATHERING_EVENT_SETTINGS, "enabled", True)
    monkeypatch.setitem(app_module.GATHERING_EVENT_SETTINGS, "ending", False)
    monkeypatch.setitem(app_module.GATHERING_EVENT_SETTINGS, "event_id", "test-event")

    payload = app_module.list_rooms()

    room_ids = {room["game_id"] for room in payload["rooms"]}
    assert app_module.MEETING_ROOM_GID in room_ids
    assert not (app_module.GATHERING_HUMAN_PUBLIC_ROOM_IDS & room_ids)
    assert payload["gathering_event"]["enabled"] is True
    app_module.GATHERING_EVENT_SETTINGS.update(event)


def test_gathering_mode_blocks_new_public_seat_but_allows_reconnect(monkeypatch):
    async def scenario():
        games = dict(app_module.GAMES)
        game = copy.deepcopy(games[app_module.MAIN_GID])
        game["human_seats"] = {"A": "existing-player"}
        games[app_module.MAIN_GID] = game
        monkeypatch.setattr(app_module, "GAMES", games)
        monkeypatch.setitem(app_module.GATHERING_EVENT_SETTINGS, "enabled", True)

        with pytest.raises(HTTPException) as error:
            await app_module.claim_seat(
                app_module.MAIN_GID, "B", client_id="new-player"
            )
        assert error.value.status_code == 409
        assert "待合室" in str(error.value.detail)

        response = await app_module.claim_seat(
            app_module.MAIN_GID, "A", client_id="existing-player"
        )
        assert response["ok"] is True

    asyncio.run(scenario())


def test_gathering_assigned_seat_can_be_reclaimed_by_same_client(monkeypatch):
    async def scenario():
        table_id = app_module.GATHERING_TABLE_IDS[0]
        game = app_module._create_game_obj(dealer="A")
        game["human_seats"] = {"A": "assigned-player"}
        game["is_started"] = True
        game["state"].finished = True
        game["total_team_score"] = {"AC": 50, "BD": 20}
        games = dict(app_module.GAMES)
        games[table_id] = game
        monkeypatch.setattr(app_module, "GAMES", games)
        monkeypatch.setitem(app_module.GATHERING_EVENT_SETTINGS, "enabled", True)

        await app_module.reset_game(
            table_id,
            dealer="A",
            requester="A",
            client_id="assigned-player",
            keep_score=True,
            auto_start=True,
        )

        response = await app_module.claim_seat(
            table_id, "A", client_id="assigned-player"
        )
        assert response["ok"] is True
        assert response["human_seats"] == ["A"]
        assert games[table_id]["total_team_score"] == {"AC": 50, "BD": 20}
        assert games[table_id]["is_started"] is True

        with pytest.raises(HTTPException) as error:
            await app_module.claim_seat(
                table_id, "A", client_id="different-player"
            )
        assert error.value.status_code == 409
        assert "自動で決まります" in str(error.value.detail)

    asyncio.run(scenario())


def test_frontend_and_admin_include_gathering_controls():
    lobby = (app_module.BASE_DIR / "frontend" / "index.html").read_text(encoding="utf-8")
    admin = (app_module.BASE_DIR / "frontend" / "admin.html").read_text(encoding="utf-8")

    assert 'id="gatheringWelcomeModal"' in lobby
    assert 'id="gatheringRoomPanel"' in lobby
    assert "async function joinGatheringQueue()" in lobby
    assert "const keepGatheringSeat = (" in lobby
    assert "return MAIN_ROOM_IDS.has(roomId) || GATHERING_TABLE_IDS.has(roomId);" in lobby
    assert "今回は休憩する" in lobby
    assert "固定ペア" in lobby
    assert 'id="gatheringRoomTitle">待合室' in lobby
    assert 'id="gatheringExitButton"' in lobby
    assert "gathering-waiting-room" in lobby
    assert "function syncGatheringWaitingRoomReturn(state)" in lobby
    assert "await openGatheringHub();" in lobby
    assert "そろうごいた会　待合室" in lobby
    trace_results = (
        app_module.BASE_DIR / "frontend" / "trace-results.js"
    ).read_text(encoding="utf-8")
    assert "const gatheringRotationPending" in trace_results
    assert "&& !gatheringRotationPending" in trace_results
    assert 'id="gatheringAdminToggle"' in admin
    assert 'api("/admin/api/gathering"' in admin

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
        games[app_module.MEETING_ROOM_GID] = app_module._create_game_obj(dealer="A")
        monkeypatch.setattr(app_module, "GAMES", games)
        monkeypatch.setattr(app_module, "GATHERING_PARTICIPANTS", players)
        monkeypatch.setattr(app_module, "GATHERING_LAST_PLAYERS", [])
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

        game = games[app_module.MEETING_ROOM_GID]
        assert game["is_started"] is True
        assert set(game["human_seats"].values()) == set(players)
        assert all(item["status"] == "playing" for item in players.values())

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
        assert "集会室" in str(error.value.detail)

        response = await app_module.claim_seat(
            app_module.MAIN_GID, "A", client_id="existing-player"
        )
        assert response["ok"] is True

    asyncio.run(scenario())


def test_frontend_and_admin_include_gathering_controls():
    lobby = (app_module.BASE_DIR / "frontend" / "index.html").read_text(encoding="utf-8")
    admin = (app_module.BASE_DIR / "frontend" / "admin.html").read_text(encoding="utf-8")

    assert 'id="gatheringWelcomeModal"' in lobby
    assert 'id="gatheringRoomPanel"' in lobby
    assert "async function joinGatheringQueue()" in lobby
    assert "今回は休憩する" in lobby
    assert "固定ペア" in lobby
    assert 'id="gatheringAdminToggle"' in admin
    assert 'api("/admin/api/gathering"' in admin

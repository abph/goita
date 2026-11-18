"""Presence prioritizes public tables that people can join, without exposing private rooms."""

import pytest

import backend.app as app


@pytest.fixture
def presence(monkeypatch):
    app.setup_main_rooms()
    monkeypatch.setattr(app.manager, "client_connections", {})
    monkeypatch.setattr(app.manager, "client_names", {})
    monkeypatch.setattr(app.manager, "client_tags", {})
    monkeypatch.setattr(app.manager, "client_icons", {})
    monkeypatch.setitem(app.GATHERING_EVENT_SETTINGS, "enabled", False)

    def room(room_id, names, tags=None, *, connected=True):
        game = app.GAMES[room_id]
        monkeypatch.setitem(game, "human_seats", {seat: name for seat, name in names.items()})
        monkeypatch.setitem(game, "player_names", dict(names))
        monkeypatch.setitem(game, "player_tags", tags or {})
        if connected:
            for seat, name in names.items():
                connect(room_id, name)
        return game

    def connect(room_id, name, tag=""):
        key = (room_id, name)
        app.manager.client_connections[key] = {object()}
        app.manager.client_names[key] = name
        app.manager.client_tags[key] = tag

    return room, connect


def test_recruiting_players_then_observers_then_full_tables_then_low_priority_tags(presence):
    room, connect = presence
    room("main", {"A": "open-a", "C": "open-c"})
    room("main-b", dict(zip("ABCD", ["full-a", "full-b", "full-c", "full-d"])))
    room("main-e", {"A": "practice"}, {"A": "ai_practice"})
    connect("main", "watcher")
    connect("lobby", "lobby")
    connect("lobby", "watching-tag", "spectator")
    people = app.list_rooms(viewer_game_id="main-b", client_id="full-a")["site_people"]
    names = [p["name"] for p in people]
    assert names[:2] == ["open-a", "open-c"]
    assert set(names[2:4]) == {"watcher", "lobby"}
    assert names[4:8] == ["full-a", "full-b", "full-c", "full-d"]
    assert set(names[8:]) == {"practice", "watching-tag"}
    assert next(p for p in people if p["name"] == "watcher")["role"] == "spectator"


def test_ai_seats_are_joinable_and_status_updates_with_human_occupancy(presence, monkeypatch):
    room, connect = presence
    game = room("main-e", {"A": "host"})
    monkeypatch.setitem(game, "ai_seats", ["B", "C", "D"])
    monkeypatch.setitem(game, "is_started", True)
    connect("main-e", "observer")
    assert {p["public_room_status"] for p in app.list_rooms()["site_people"]} == {"recruiting"}
    monkeypatch.setitem(game, "human_seats", dict(zip("ABCD", ["host", "b", "c", "d"])))
    assert {p["public_room_status"] for p in app.list_rooms()["site_people"]} == {"full"}
    monkeypatch.setitem(game, "human_seats", {})
    assert {p["public_room_status"] for p in app.list_rooms()["site_people"]} == {"empty"}


def test_closed_public_rooms_are_not_marked_as_recruiting(presence, monkeypatch):
    room, _connect = presence
    room("main", {"A": "host"})
    monkeypatch.setitem(app.GATHERING_EVENT_SETTINGS, "enabled", True)
    assert app.list_rooms()["site_people"][0]["public_room_status"] == "closed"


def test_private_presence_does_not_publish_recruiting_or_room_identifiers(presence):
    room, _connect = presence
    room(app.PRIVATE_A_GID, {"A": "secret"}, {"A": "human_match"})
    people = app.list_rooms()["site_people"]
    assert len(people) == 1
    assert people[0]["name"] == "＊＊＊＊"
    assert people[0]["tag"] == ""
    assert "public_room_status" not in people[0]
    assert "game_id" not in people[0] and "client_id" not in people[0]

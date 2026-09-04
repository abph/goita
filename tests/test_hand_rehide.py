import asyncio

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend import app as app_module


@pytest.fixture(params=[False, True], ids=["private", "public"])
def room(request, monkeypatch):
    public = request.param
    gid = next(iter(app_module.MAIN_ROOM_NAMES)) if public else "test-hand-rehide"
    game = app_module._create_game_obj(dealer="A")
    game.update(is_started=True, human_seats={"A": "client-a", "B": "client-b"}, ai_seats=["C", "D"])
    game["state"].finished = public
    monkeypatch.setitem(app_module.GAMES, gid, game)
    return gid, game


def change(gid, target="B", requester="B", client_id="client-b", **kwargs):
    return asyncio.run(app_module.reveal_hand(gid, target=target, requester=requester, client_id=client_id, **kwargs))


def test_owner_can_rehide_and_manually_reveal_again(room):
    gid, game = room
    change(gid)
    assert isinstance(app_module.get_state(gid, viewer="W")["hands"]["B"], list)
    change(gid, visible=False)
    for viewer, client in [("W", ""), ("A", "client-a")]:
        state = app_module.get_state(gid, viewer=viewer, client_id=client)
        assert state["hands"]["B"] == {"count": 8}
        assert state["init_hands"]["B"] == {"count": 8}
        assert "B" not in state["revealed_hand_seats"]
        assert "B" in state["auto_reveal_blocked_seats"]
    assert isinstance(app_module.get_state(gid, viewer="B", client_id="client-b")["hands"]["B"], list)
    change(gid, automatic=True)
    assert "B" not in app_module.get_state(gid, viewer="W")["revealed_hand_seats"]
    change(gid)
    assert "B" in app_module.get_state(gid, viewer="W")["revealed_hand_seats"]
    assert "B" not in game["auto_reveal_blocked_seats"]


def test_ai_rehide_survives_other_clients_auto_reveal(room):
    gid, game = room
    game["state"].finished = True
    change(gid, target="C")
    change(gid, target="C", visible=False)
    change(gid, target="C", requester="A", client_id="client-a", automatic=True)
    assert app_module.get_state(gid, viewer="W")["hands"]["C"] == {"count": 8}
    change(gid, target="C")
    assert isinstance(app_module.get_state(gid, viewer="W")["hands"]["C"], list)


def test_rehide_preserves_ownership_permissions(room):
    gid, game = room
    for target, requester, client in [("B", "A", "client-a"), ("B", "B", "wrong-client"), ("C", "W", "")]:
        with pytest.raises(HTTPException) as error:
            change(gid, target=target, requester=requester, client_id=client, visible=False)
        assert error.value.status_code in (400, 403)
    assert game["auto_reveal_blocked_seats"] == []


def test_public_rehide_rejected_during_play(room):
    gid, game = room
    game["state"].finished = False
    if gid in app_module.MAIN_GIDS:
        with pytest.raises(HTTPException) as error:
            change(gid, visible=False)
        assert error.value.status_code == 409
    else:
        change(gid, target="C", requester="A", client_id="client-a", visible=False)
        with pytest.raises(HTTPException) as error:
            change(gid, target="D", visible=False)
        assert error.value.status_code == 403


def test_next_round_clears_auto_reveal_block(room):
    gid, game = room
    game["state"].finished = True
    change(gid, visible=False)
    asyncio.run(app_module.reset_game(gid, dealer="A", requester="A", client_id="client-a", keep_score=True))
    assert app_module.GAMES[gid]["auto_reveal_blocked_seats"] == []


def test_http_false_is_parsed_as_hide(room):
    gid, game = room
    game["state"].finished = True
    with TestClient(app_module.app) as client:
        response = client.post(f"/games/{gid}/reveal_hand", params={
            "requester": "B", "target": "C", "client_id": "client-b", "visible": "false",
        })
        assert response.status_code == 200
    assert "C" in game["auto_reveal_blocked_seats"]

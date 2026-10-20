import asyncio
from pathlib import Path

import pytest

from backend import app as game_app
from backend.vrc_solo_lease import SoloLeaseStore

OWNER = "a" * 32
OTHER = "b" * 32


def response_text(response) -> str:
    return response.body.decode("utf-8")


@pytest.fixture
def bridge(monkeypatch, tmp_path):
    monkeypatch.setattr(game_app, "VRC_SOLO_LEASES", SoloLeaseStore(tmp_path / "leases.sqlite3"))
    room_id = "pytest-solo"
    game_id = f"vrc-solo-{room_id}"
    game_app.GAMES.pop(game_id, None)
    game_app.GAME_TURN_LOCKS.pop(game_id, None)

    async def no_ai(_game):
        return None

    async def no_broadcast(_game_id):
        game = game_app.GAMES.get(_game_id)
        if game is not None:
            game["update_version"] = int(game.get("update_version", 0)) + 1

    monkeypatch.setattr(game_app, "_vrc_solo_run_ai", no_ai)
    monkeypatch.setattr(game_app.manager, "broadcast_update", no_broadcast)
    monkeypatch.setattr(game_app.random, "choice", lambda _items: "A")
    yield room_id, game_id
    game_app.GAMES.pop(game_id, None)
    game_app.GAME_TURN_LOCKS.pop(game_id, None)


def test_start_returns_compact_no_cache_snapshot(bridge):
    room_id, game_id = bridge
    response = asyncio.run(game_app.vrc_solo_claim(room_id, OWNER))
    body = response_text(response)

    assert body.startswith("G1|")
    assert "|turn=A|phase=attack|" in body
    assert "|ha=" in body
    assert "|board=" in body
    assert "|lb=" in body and "|la=" in body
    assert response.headers["cache-control"].startswith("no-store")
    assert game_app.GAMES[game_id]["human_seats"] == {"A": f"vrc-solo:{room_id}"}
    assert game_app.GAMES[game_id]["ai_seats"] == ["B", "C", "D"]


def test_block_then_attack_is_one_server_transaction(bridge):
    room_id, game_id = bridge
    asyncio.run(game_app.vrc_solo_claim(room_id, OWNER))
    game = game_app.GAMES[game_id]
    legal = [action for action in game["state"].legal_actions("A") if action[0] == "attack_after_block"]
    assert legal
    _kind, block, attack = legal[0]

    ready = asyncio.run(game_app.vrc_solo_block(room_id, int(block), OWNER))
    assert response_text(ready) == f"R1|kind=block|piece={block}"
    assert game["kifu_moves"] == []

    result = asyncio.run(game_app.vrc_solo_attack(room_id, int(attack), OWNER))
    assert response_text(result).startswith("G1|")
    assert game["vrc_pending"] is None
    assert len(game["kifu_moves"]) == 1
    assert game["kifu_moves"][0][0] == "0"
    assert game["state"].turn == "B"


def test_attack_without_first_selection_is_rejected_without_mutation(bridge):
    room_id, game_id = bridge
    asyncio.run(game_app.vrc_solo_claim(room_id, OWNER))
    game = game_app.GAMES[game_id]
    before = {seat: list(hand) for seat, hand in game["state"].hands.items()}

    response = asyncio.run(game_app.vrc_solo_attack(room_id, 1, OWNER))

    assert response_text(response).startswith("E1|code=missing_selection|")
    assert game["state"].hands == before
    assert game["kifu_moves"] == []


def test_browser_supports_direct_vrc_solo_spectator_link():
    html = (Path(__file__).parents[1] / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "function vrcSoloRoomRequestFromUrl()" in html
    assert "vrc-solo-[a-z0-9]" in html
    assert "await enterRoom(vrcSoloRoom.roomId, vrcSoloRoom.roomName)" in html


def test_second_instance_and_identical_claim_are_rejected(bridge, monkeypatch):
    room, gid = bridge

    async def slow_ai(_game):
        await asyncio.sleep(0.01)

    monkeypatch.setattr(game_app, "_vrc_solo_run_ai", slow_ai)

    async def race():
        return await asyncio.gather(game_app.vrc_solo_claim(room, OWNER), game_app.vrc_solo_claim(room, OTHER))

    responses = asyncio.run(race())
    assert sum(response_text(item).startswith("G1|") for item in responses) == 1
    assert sum("code=busy" in response_text(item) for item in responses) == 1
    game = game_app.GAMES[gid]
    assert "code=busy" in response_text(asyncio.run(game_app.vrc_solo_claim(room, OWNER)))
    for call in (
        game_app.vrc_solo_start(room, OTHER),
        game_app.vrc_solo_state(room, OTHER),
        game_app.vrc_solo_block(room, 1, OTHER),
        game_app.vrc_solo_receive(room, 1, OTHER),
        game_app.vrc_solo_attack(room, 1, OTHER),
        game_app.vrc_solo_pass(room, OTHER),
        game_app.vrc_solo_heartbeat(room, OTHER),
        game_app.vrc_solo_release(room, OTHER),
    ):
        assert "code=busy" in response_text(asyncio.run(call))
    assert game_app.GAMES[gid] is game
    assert game["kifu_moves"] == []
    assert game["vrc_pending"] is None


def test_old_world_without_session_cannot_bypass_lock(bridge):
    room, _gid = bridge
    assert "code=upgrade_required" in response_text(asyncio.run(game_app.vrc_solo_start(room)))
    asyncio.run(game_app.vrc_solo_claim(room, OWNER))
    assert "code=upgrade_required" in response_text(asyncio.run(game_app.vrc_solo_pass(room)))


def test_release_opens_room_and_rejects_stale_requests(bridge):
    room, gid = bridge
    asyncio.run(game_app.vrc_solo_claim(room, OWNER))
    assert response_text(asyncio.run(game_app.vrc_solo_release(room, OWNER))).startswith("X1|")
    assert game_app.GAMES[gid]["is_started"] is False
    assert response_text(asyncio.run(game_app.vrc_solo_claim(room, OTHER))).startswith("G1|")
    assert "code=busy" in response_text(asyncio.run(game_app.vrc_solo_heartbeat(room, OWNER)))
    assert "code=busy" in response_text(asyncio.run(game_app.vrc_solo_release(room, OWNER)))
    assert game_app.GAMES[gid]["total_team_score"] == {"AC": 0, "BD": 0}


def test_heartbeat_preserves_thinking_time_and_timeout_allows_new_instance(bridge, monkeypatch):
    room, gid = bridge
    now = [1000.0]
    monkeypatch.setattr(game_app.VRC_SOLO_LEASES, "clock", lambda: now[0])
    asyncio.run(game_app.vrc_solo_claim(room, OWNER))
    now[0] += 120
    assert response_text(asyncio.run(game_app.vrc_solo_heartbeat(room, OWNER))).startswith("H1|")
    now[0] += 120
    assert "code=busy" in response_text(asyncio.run(game_app.vrc_solo_claim(room, OTHER)))
    now[0] += 61
    assert "code=connection_retired" in response_text(asyncio.run(game_app.vrc_solo_claim(room, OWNER)))
    assert response_text(asyncio.run(game_app.vrc_solo_claim(room, OTHER))).startswith("G1|")
    assert "code=busy" in response_text(asyncio.run(game_app.vrc_solo_attack(room, 1, OWNER)))
    assert game_app.GAMES[gid]["kifu_moves"] == []


def test_finished_round_keeps_lease_and_next_round_keeps_score(bridge):
    room, gid = bridge
    asyncio.run(game_app.vrc_solo_claim(room, OWNER))
    previous = game_app.GAMES[gid]
    previous["state"].finished = True
    previous["state"].winner = "C"
    previous["total_team_score"] = {"AC": 40, "BD": 20}
    previous["vrc_last_request"] = None
    assert "code=busy" in response_text(asyncio.run(game_app.vrc_solo_claim(room, OTHER)))
    assert response_text(asyncio.run(game_app.vrc_solo_start(room, OWNER))).startswith("G1|")
    assert game_app.GAMES[gid]["dealer"] == "C"
    assert game_app.GAMES[gid]["total_team_score"] == {"AC": 40, "BD": 20}


def test_server_restart_invalidates_old_lease_and_retires_connection(tmp_path):
    path = tmp_path / "leases.sqlite3"
    first = SoloLeaseStore(path)
    first.claim("table1", OWNER)
    restarted = SoloLeaseStore(path)
    from backend.vrc_solo_lease import LeaseError
    with pytest.raises(LeaseError, match="lease_lost"):
        restarted.touch("table1", OWNER)
    with pytest.raises(LeaseError, match="connection_retired"):
        restarted.claim("table1", OWNER)
    restarted.claim("table1", OTHER)


def test_http_routes_bind_session_query_and_preserve_lease_for_spectators(bridge):
    import httpx
    room, gid = bridge

    async def request_flow():
        transport = httpx.ASGITransport(app=game_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            first = await client.get(f"/vrc/solo/{room}/claim", params={"session": OWNER})
            assert first.text.startswith("G1|")
            assert first.headers["cache-control"].startswith("no-store")
            denied = await client.get(f"/vrc/solo/{room}/claim", params={"session": OTHER})
            assert "code=busy" in denied.text
            legacy = await client.get(f"/vrc/solo/{room}/start")
            assert "code=upgrade_required" in legacy.text
            spectator = await client.get(f"/games/{gid}/state", params={"viewer": "W"})
            assert spectator.status_code == 200
            assert spectator.json()["hands"]["A"] == {"count": 8}
            heartbeat = await client.get(f"/vrc/solo/{room}/heartbeat", params={"session": OWNER})
            assert heartbeat.text == "H1|seconds=180"
            released = await client.get(f"/vrc/solo/{room}/release", params={"session": OWNER})
            assert released.text.startswith("X1|")

    asyncio.run(request_flow())

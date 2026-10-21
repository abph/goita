import asyncio
import copy

import httpx
import pytest

from backend import app as game_app
from backend.vrc_solo_lease import SoloLeaseStore

OWNER = "c" * 32
OTHER = "d" * 32
CLIENT = "browser-partner"
REAL_RUN_AI = game_app._vrc_solo_run_ai


@pytest.fixture
def coop(monkeypatch, tmp_path):
    room = "pytest-coop"
    gid = "vrc-solo-" + room
    lease_clock, browser_clock = [1000.0], [1000.0]
    monkeypatch.setattr(game_app, "VRC_SOLO_LEASES", SoloLeaseStore(tmp_path / "lease.db", clock=lambda: lease_clock[0]))
    monkeypatch.setattr(game_app.VRC_WEB, "clock", lambda: browser_clock[0])
    monkeypatch.setattr(game_app.random, "choice", lambda _: "A")
    monkeypatch.setattr(game_app, "_schedule_ai_background_search", lambda *args: None)
    monkeypatch.setattr(game_app, "_arm_turn_timeout", lambda *args: None)

    async def broadcast(game_id):
        game = game_app.GAMES.get(game_id)
        if game:
            game["update_version"] = game.get("update_version", 0) + 1

    async def run_ai(game):
        # Exercise real rules and board mutation, without costly AI searches.
        state = game["state"]
        for _ in range(64):
            if state.finished or state.turn not in game_app._ai_seat_set(game):
                return
            legal = state.legal_actions(state.turn)
            action = next((a for a in legal if a[0] == "pass"), legal[0])
            game_app._vrc_solo_apply_action(game, state.turn, action)
        raise AssertionError("AI failed to stop at a human turn")

    monkeypatch.setattr(game_app.manager, "broadcast_update", broadcast)
    monkeypatch.setattr(game_app, "_vrc_solo_run_ai", run_ai)
    game_app.GAMES.pop(gid, None)
    game_app.GAME_TURN_LOCKS.pop(gid, None)
    yield room, gid, lease_clock, browser_clock
    game_app.GAMES.pop(gid, None)
    game_app.GAME_TURN_LOCKS.pop(gid, None)


async def client_flow():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=game_app.app), base_url="http://test")


async def claim(client, room):
    response = await client.get(f"/vrc/solo/{room}/claim", params={"session": OWNER, "mode": "coop"})
    assert response.text.startswith("G1|")
    assert "|waiting=1|" in response.text
    assert "|ha=#0|" in response.text
    return response


async def join(client, gid, who=CLIENT):
    return await client.post(f"/games/{gid}/claim", params={"seat": "C", "client_id": who})


def test_wait_join_start_hidden_hands_and_reserved_seats(coop):
    room, gid, *_ = coop

    async def flow():
        async with await client_flow() as client:
            await claim(client, room)
            assert (await client.get(f"/vrc/solo/{room}/start", params={"session": OWNER})).text.find("waiting=1") >= 0
            for seat in ("A", "B", "D"):
                denied = await client.post(f"/games/{gid}/claim", params={"seat": seat, "client_id": CLIENT})
                assert denied.status_code == 403
            assert (await join(client, gid)).status_code == 200
            assert (await join(client, gid, "spectator")).status_code == 409
            started = await client.get(f"/vrc/solo/{room}/start", params={"session": OWNER})
            assert "|waiting=0|" in started.text
            assert "|cweb=1|" in started.text
            assert "|hc=#8|" in started.text
            state = (await client.get(f"/games/{gid}/state", params={"viewer": "C", "client_id": CLIENT})).json()
            assert isinstance(state["hands"]["C"], list)
            assert state["hands"]["A"] == {"count": 8}
            assert state["hands"]["B"] == {"count": 8}
            assert state["owned_human_seats"] == ["C"]
            assert "session" not in state["vrc_web"]
            spectator = (await client.get(f"/games/{gid}/state", params={"viewer": "A", "client_id": "vrc-solo:" + room})).json()
            assert spectator["hands"]["A"] == {"count": 8}
            assert spectator["hands"]["C"] == {"count": 8}
            assert spectator["owned_human_seats"] == []
            assert "code=busy" in (await client.get(f"/vrc/solo/{room}/claim", params={"session": OTHER, "mode": "coop"})).text
    asyncio.run(flow())


def test_browser_move_advances_ai_and_vrc_state_without_manual_sync(coop):
    room, gid, *_ = coop

    async def flow():
        async with await client_flow() as client:
            await claim(client, room)
            await join(client, gid)
            await client.get(f"/vrc/solo/{room}/start", params={"session": OWNER})
            game = game_app.GAMES[gid]
            first = game["state"].legal_actions("A")[0]
            await client.get(f"/vrc/solo/{room}/block/{first[1]}", params={"session": OWNER})
            await client.get(f"/vrc/solo/{room}/attack/{first[2]}", params={"session": OWNER})
            assert game["state"].turn == "C"
            version = game.get("update_version", 0)
            legal = (await client.get(f"/games/{gid}/legal_actions", params={"player": "C", "client_id": CLIENT})).json()
            result = await client.post(f"/games/{gid}/step", json={"player": "C", "client_id": CLIENT, "action": legal[0]})
            assert result.status_code == 200, result.text
            # A receive is a first half: finish the attack before expecting AI D.
            if game["state"].turn == "C":
                legal = (await client.get(f"/games/{gid}/legal_actions", params={"player": "C", "client_id": CLIENT})).json()
                assert (await client.post(f"/games/{gid}/step", json={"player": "C", "client_id": CLIENT, "action": legal[0]})).status_code == 200
            assert game["state"].finished or game["state"].turn == "A"
            assert game["update_version"] > version
            snapshot = await client.get(f"/vrc/solo/{room}/state", params={"session": OWNER})
            assert snapshot.text.startswith("G1|")
            assert f"|v={game['update_version']}|" in snapshot.text
            assert "|hc=#" in snapshot.text or game["state"].finished
            denied = await client.post(f"/games/{gid}/step", json={"player": "C", "client_id": "spectator", "action": legal[0]})
            assert denied.status_code == 403
    asyncio.run(flow())


def test_next_round_and_new_match_keep_browser_identity(coop):
    room, gid, *_ = coop

    async def flow():
        async with await client_flow() as client:
            await claim(client, room)
            await join(client, gid)
            await client.get(f"/vrc/solo/{room}/start", params={"session": OWNER})
            for match_finished in (False, True):
                previous = game_app.GAMES[gid]
                previous["player_names"]["C"] = "サイトの友達"
                previous["state"].finished = True
                previous["state"].winner = "C"
                previous["total_team_score"] = {"AC": 40, "BD": 20}
                previous["match_finished"] = match_finished
                old_deal = previous["action_state_nonce"]
                result = await client.get(f"/vrc/solo/{room}/start", params={"session": OWNER})
                assert result.text.startswith("G1|")
                new = game_app.GAMES[gid]
                assert new["human_seats"]["C"] == CLIENT
                assert new["player_names"]["C"] == "サイトの友達"
                assert new["ai_seats"] == ["B", "D"]
                assert new["action_state_nonce"] != old_deal
                assert new["total_team_score"] == ({"AC": 0, "BD": 0} if match_finished else {"AC": 40, "BD": 20})
    asyncio.run(flow())


def test_disconnect_reconnect_and_explicit_takeover(coop):
    room, gid, _lease_clock, clock = coop

    async def flow():
        async with await client_flow() as client:
            await claim(client, room)
            await join(client, gid)
            await client.get(f"/vrc/solo/{room}/start", params={"session": OWNER})
            clock[0] += 60
            assert await game_app._release_disconnected_client_seats(gid, CLIENT) is False
            denied = await client.get(f"/vrc/solo/{room}/takeover", params={"session": OWNER})
            assert "partner_online" in denied.text
            assert (await join(client, gid)).status_code == 200
            clock[0] += 121
            snapshot = await client.get(f"/vrc/solo/{room}/state", params={"session": OWNER})
            assert "|ctake=1|" in snapshot.text
            assert (await join(client, gid, "other-browser")).status_code == 409
            taken = await client.get(f"/vrc/solo/{room}/takeover", params={"session": OWNER})
            assert taken.text.startswith("G1|")
            assert "C" not in game_app.GAMES[gid]["human_seats"]
            assert set(game_app.GAMES[gid]["ai_seats"]) == {"B", "C", "D"}
            assert (await join(client, gid)).status_code == 409
            before = copy.deepcopy(game_app.GAMES[gid]["kifu_moves"])
            await client.get(f"/vrc/solo/{room}/takeover", params={"session": OWNER})
            assert game_app.GAMES[gid]["kifu_moves"] == before
    asyncio.run(flow())


def test_browser_cannot_extend_lease_or_control_table_settings(coop):
    room, gid, clock, _browser_clock = coop

    async def flow():
        async with await client_flow() as client:
            await claim(client, room)
            await join(client, gid)
            for action in ("start", "reset", "set_ai", "cpu_step", "auto_step", "seat_vacate", "reveal_hand", "toggle_reveal_hands", "reset_all_seats"):
                denied = await client.post(f"/games/{gid}/{action}", params={"seat": "C", "client_id": CLIENT})
                assert denied.status_code == 403, (action, denied.text)
            clock[0] += 150
            heartbeat = await client.post(f"/vrc/solo/{room}/web/heartbeat", json={"client_id": CLIENT})
            assert heartbeat.json()["active"] is True
            clock[0] += 31
            heartbeat = await client.post(f"/vrc/solo/{room}/web/heartbeat", json={"client_id": CLIENT})
            assert heartbeat.json()["active"] is False
            assert game_app.GAMES[gid]["is_started"] is False
            assert (await join(client, gid)).status_code == 409
            new = await client.get(f"/vrc/solo/{room}/claim", params={"session": OTHER, "mode": "coop"})
            assert new.text.startswith("G1|")
            assert game_app.GAMES[gid]["vrc_web"]["client_id"] == ""
    asyncio.run(flow())


def test_leave_and_release_notify_browser_and_clear_ownership(coop):
    room, gid, *_ = coop

    async def flow():
        async with await client_flow() as client:
            await claim(client, room)
            await join(client, gid)
            left = await client.post(f"/games/{gid}/release", params={"seat": "C", "client_id": CLIENT})
            assert left.status_code == 200
            state = (await client.get(f"/games/{gid}/state", params={"client_id": CLIENT})).json()
            assert state["owned_human_seats"] == []
            assert state["vrc_web"]["can_takeover"] is True
            assert (await join(client, gid)).status_code == 200
            await client.get(f"/vrc/solo/{room}/release", params={"session": OWNER})
            status = (await client.post(f"/vrc/solo/{room}/web/heartbeat", json={"client_id": CLIENT})).json()
            assert status["active"] is False
            assert status["owned"] is False
            state = (await client.get(f"/games/{gid}/state", params={"client_id": CLIENT})).json()
            assert state["owned_human_seats"] == []
            assert state["hands"]["C"] == {"count": 0}
    asyncio.run(flow())


def test_complete_mixed_round_uses_server_ai_loop_and_reveals_only_at_finish(coop, monkeypatch):
    room, gid, *_ = coop
    monkeypatch.setattr(game_app, "_vrc_solo_run_ai", REAL_RUN_AI)

    def fast_ai_turn(game, seat):
        legal = game["state"].legal_actions(seat)
        action = next((a for a in legal if a[0] == "pass"), legal[0])
        game_app._vrc_solo_apply_action(game, seat, action)
        return {"status": "ok"}

    monkeypatch.setattr(game_app, "_apply_agent_turn", fast_ai_turn)

    async def flow():
        async with await client_flow() as client:
            await claim(client, room)
            await join(client, gid)
            await client.get(f"/vrc/solo/{room}/start", params={"session": OWNER})
            game = game_app.GAMES[gid]
            for _ in range(200):
                state = game["state"]
                if state.finished:
                    break
                assert state.turn in {"A", "C"}, "server must run AI without a browser CPU request"
                game["vrc_last_request"] = None  # No real Udon request delay in this simulation.
                legal = state.legal_actions(state.turn)
                action = next((a for a in legal if a[0] != "pass"), legal[0])
                if state.turn == "C":
                    result = await client.post(f"/games/{gid}/step", json={
                        "player": "C", "client_id": CLIENT,
                        "expected_action_token": game_app._action_state_token(game, state),
                        "action": game_app._actions_to_json([action])[0],
                    })
                    assert result.status_code == 200, result.text
                elif action[0] == "pass":
                    result = await client.get(f"/vrc/solo/{room}/pass", params={"session": OWNER})
                    assert result.text.startswith("G1|"), result.text
                else:
                    kind, block, attack = action
                    if kind == "receive":
                        shadow = copy.deepcopy(state)
                        game_app._apply_action(shadow, "A", action)
                        attack = next((a[2] for a in shadow.legal_actions("A") if a[0] == "attack"), "1")
                    select = "receive" if kind == "receive" else "block"
                    result = await client.get(f"/vrc/solo/{room}/{select}/{block}", params={"session": OWNER})
                    assert result.text.startswith("R1|"), result.text
                    # Simulated turns have no 5-second Udon delay; discard retry cache.
                    game["vrc_last_request"] = None
                    result = await client.get(f"/vrc/solo/{room}/attack/{attack}", params={"session": OWNER})
                    assert result.text.startswith("G1|"), result.text
                if not state.finished:
                    spectator = (await client.get(f"/games/{gid}/state")).json()
                    assert all(isinstance(value, dict) for value in spectator["hands"].values())
            assert state.finished, "mixed human/AI round did not finish"
            spectator = (await client.get(f"/games/{gid}/state")).json()
            assert all(isinstance(value, list) for value in spectator["hands"].values())
            assert any(row[0] == "2" for row in game["kifu_moves"]), "C must actually play"
    asyncio.run(flow())


def test_concurrent_browser_joins_only_one_wins(coop):
    room, gid, *_ = coop

    async def flow():
        async with await client_flow() as client:
            await claim(client, room)
            results = await asyncio.gather(join(client, gid, "first"), join(client, gid, "second"))
            assert sorted(response.status_code for response in results) == [200, 409]
            owner = game_app.GAMES[gid]["vrc_web"]["client_id"]
            assert owner in {"first", "second"}
            assert game_app.GAMES[gid]["human_seats"]["C"] == owner
    asyncio.run(flow())


def test_hidden_c_block_is_not_sent_in_board_or_text_log(coop):
    room, gid, *_ = coop

    async def flow():
        async with await client_flow() as client:
            await claim(client, room)
            await join(client, gid)
            await client.get(f"/vrc/solo/{room}/start", params={"session": OWNER})
            game = game_app.GAMES[gid]
            game["board"]["C"]["receive"][0] = "7"
            game["board"]["C"]["receive_hidden"][0] = True
            game["log"] = ["C: block 7 -> attack 1 (hidden)"]
            snapshot = (await client.get(f"/vrc/solo/{room}/state", params={"session": OWNER})).text
            board = dict(part.split("=", 1) for part in snapshot.split("|")[1:])["board"]
            assert board[16] == "1", "VRChat must receive an anonymous hidden placeholder"
            spectator = (await client.get(f"/games/{gid}/state")).json()
            assert spectator["board_public"]["C"]["receive"][0] == "■"
            assert spectator["log"] == []
            owned = (await client.get(f"/games/{gid}/state", params={"client_id": CLIENT})).json()
            assert owned["board_public"]["C"]["receive"][0] == "7"
            game["state"].finished = True
            spectator = (await client.get(f"/games/{gid}/state")).json()
            assert spectator["board_public"]["C"]["receive"][0] == "7"
            assert spectator["log"] == game["log"]
    asyncio.run(flow())


def test_waiting_disconnected_partner_does_not_start_and_solo_fallback_works(coop):
    room, gid, _lease, clock = coop

    async def flow():
        async with await client_flow() as client:
            await claim(client, room)
            await join(client, gid)
            clock[0] += 31
            result = await client.get(f"/vrc/solo/{room}/start", params={"session": OWNER})
            assert "|waiting=1|" in result.text
            assert game_app.GAMES[gid]["is_started"] is False
            clock[0] += 90
            result = await client.get(f"/vrc/solo/{room}/takeover", params={"session": OWNER})
            assert "|waiting=0|" in result.text
            assert "|cweb=0|" in result.text
            assert game_app.GAMES[gid]["is_started"] is True
    asyncio.run(flow())

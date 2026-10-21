import asyncio
import copy

import pytest

from backend import app as game_app
from test_vrc_solo_coop import coop, OWNER, OTHER, client_flow


async def claim_multi(client, room):
    result = await client.get(f"/vrc/solo/{room}/claim", params={"session": OWNER, "mode": "multi"})
    assert result.text.startswith("G1|"), result.text
    assert "|waiting=1|" in result.text
    assert "|webmulti=1|" in result.text


async def join_seat(client, gid, seat, who=None):
    return await client.post(f"/games/{gid}/claim", params={"seat": seat, "client_id": who or "browser-" + seat})


@pytest.mark.parametrize("human_seats", [(), ("B",), ("C",), ("D",), ("B", "C"), ("B", "D"), ("C", "D"), ("B", "C", "D")])
def test_human_combinations_and_empty_seats_become_ai(coop, human_seats):
    room, gid, *_ = coop

    async def flow():
        async with await client_flow() as client:
            await claim_multi(client, room)
            assert game_app.GAMES[gid]["ai_seats"] == []
            for seat in human_seats:
                assert (await join_seat(client, gid, seat)).status_code == 200
            result = await client.get(f"/vrc/solo/{room}/start", params={"session": OWNER})
            assert "|waiting=0|" in result.text, result.text
            game = game_app.GAMES[gid]
            assert set(game["human_seats"]) == {"A", *human_seats}
            assert set(game["ai_seats"]) == set("BCD") - set(human_seats)
            for seat in human_seats:
                state = (await client.get(f"/games/{gid}/state", params={"viewer": seat, "client_id": "browser-" + seat})).json()
                assert state["owned_human_seats"] == [seat]
                for other in "ABCD":
                    assert isinstance(state["hands"][other], list if seat == other else dict)
                assert state["vrc_web"]["seats"][seat]["owned"] is True
                assert "client_id" not in state["vrc_web"]["seats"][seat]
            spectator = (await client.get(f"/games/{gid}/state")).json()
            assert all(isinstance(hand, dict) for hand in spectator["hands"].values())
            assert all(not row["can_join"] for row in spectator["vrc_web"]["seats"].values())
    asyncio.run(flow())


def test_single_client_one_seat_and_prestart_switching(coop):
    room, gid, *_ = coop

    async def flow():
        async with await client_flow() as client:
            await claim_multi(client, room)
            assert (await join_seat(client, gid, "B", "same-browser")).status_code == 200
            assert (await join_seat(client, gid, "D", "same-browser")).status_code == 409
            assert (await join_seat(client, gid, "B", "other-browser")).status_code == 409
            assert (await join_seat(client, gid, "A", "same-browser")).status_code == 403
            left = await client.post(f"/games/{gid}/release", params={"seat": "B", "client_id": "same-browser"})
            assert left.status_code == 200
            assert (await join_seat(client, gid, "D", "same-browser")).status_code == 200
            assert (await join_seat(client, gid, "B", "other-browser")).status_code == 200
            results = await asyncio.gather(join_seat(client, gid, "C", "first"), join_seat(client, gid, "C", "second"))
            assert sorted(result.status_code for result in results) == [200, 409]
    asyncio.run(flow())


@pytest.mark.parametrize("seat", ["B", "C", "D"])
def test_browser_moves_each_seat_and_only_refreshes_own_presence(coop, seat):
    room, gid, _lease, clock = coop

    async def flow():
        async with await client_flow() as client:
            await claim_multi(client, room)
            for human in "BCD":
                await join_seat(client, gid, human)
            await client.get(f"/vrc/solo/{room}/start", params={"session": OWNER})
            game = game_app.GAMES[gid]
            # Move through the real game until the requested human has a turn.
            for _ in range(100):
                if game["state"].turn == seat:
                    break
                actor = game["state"].turn
                legal = game["state"].legal_actions(actor)
                game_app._vrc_solo_apply_action(game, actor, next((a for a in legal if a[0] == "pass"), legal[0]))
            assert game["state"].turn == seat
            clock[0] += 15
            legal = (await client.get(f"/games/{gid}/legal_actions", params={"player": seat, "client_id": "browser-" + seat})).json()
            assert legal
            denied = await client.post(f"/games/{gid}/step", json={"player": seat, "client_id": "browser-" + ("D" if seat != "D" else "B"), "action": legal[0]})
            assert denied.status_code == 403
            good = await client.post(f"/games/{gid}/step", json={"player": seat, "client_id": "browser-" + seat, "action": legal[0]})
            assert good.status_code == 200, good.text
            web = game["vrc_web"]
            for other in "BCD":
                assert game_app.VRC_WEB.slot(web, other)["last_seen"] == (1015 if other == seat else 1000)
    asyncio.run(flow())


@pytest.mark.parametrize("seat", ["B", "C", "D"])
def test_reconnect_and_ai_takeover_are_per_seat(coop, seat):
    room, gid, _lease, clock = coop

    async def flow():
        async with await client_flow() as client:
            await claim_multi(client, room)
            for human in "BCD":
                await join_seat(client, gid, human)
            await client.get(f"/vrc/solo/{room}/start", params={"session": OWNER})
            clock[0] += 121
            # Other seats remain online and must not be replaced.
            for other in set("BCD") - {seat}:
                await client.post(f"/vrc/solo/{room}/web/heartbeat", json={"client_id": "browser-" + other})
            meta = (await client.post(f"/vrc/solo/{room}/web/heartbeat", json={"client_id": "spectator"})).json()
            assert meta["seats"][seat]["can_takeover"] is True
            assert all(not meta["seats"][other]["can_takeover"] for other in set("BCD") - {seat})
            assert (await join_seat(client, gid, seat, "stranger")).status_code == 409
            # Same browser reconnects and stops a queued/stale takeover attempt.
            assert (await join_seat(client, gid, seat)).status_code == 200
            denied = await client.get(f"/vrc/solo/{room}/takeover", params={"session": OWNER, "seat": seat})
            assert "partner_online" in denied.text
            left = await client.post(f"/games/{gid}/release", params={"seat": seat, "client_id": "browser-" + seat})
            assert left.status_code == 200
            def reservation(other):
                slot = game_app.VRC_WEB.slot(game_app.GAMES[gid]["vrc_web"], other)
                return {key: slot[key] for key in ("client_id", "last_seen", "departed", "takeover")}
            before = {other: reservation(other) for other in set("BCD") - {seat}}
            taken = await client.get(f"/vrc/solo/{room}/takeover", params={"session": OWNER, "seat": seat})
            assert taken.text.startswith("G1|"), taken.text
            game = game_app.GAMES[gid]
            assert set(game["ai_seats"]) == {seat}
            assert seat not in game["human_seats"]
            assert all(reservation(other) == record for other, record in before.items())
            assert (await join_seat(client, gid, seat)).status_code == 409
    asyncio.run(flow())


def test_next_round_preserves_all_three_players_and_ai_assignments(coop):
    room, gid, *_ = coop

    async def flow():
        async with await client_flow() as client:
            await claim_multi(client, room)
            for seat in "BCD":
                await join_seat(client, gid, seat)
                game_app.GAMES[gid]["player_names"][seat] = "友達" + seat
            await client.get(f"/vrc/solo/{room}/start", params={"session": OWNER})
            previous = game_app.GAMES[gid]
            previous["state"].finished = True
            previous["state"].winner = "B"
            previous["total_team_score"] = {"AC": 40, "BD": 50}
            await client.get(f"/vrc/solo/{room}/start", params={"session": OWNER})
            game = game_app.GAMES[gid]
            assert game["total_team_score"] == {"AC": 40, "BD": 50}
            assert game["ai_seats"] == []
            for seat in "BCD":
                assert game["human_seats"][seat] == "browser-" + seat
                assert game["player_names"][seat] == "友達" + seat
            assert "code=busy" in (await client.get(f"/vrc/solo/{room}/claim", params={"session": OTHER, "mode": "multi"})).text
    asyncio.run(flow())


def test_offline_reservation_blocks_start_until_reconnected_or_replaced(coop):
    room, gid, _lease, clock = coop

    async def flow():
        async with await client_flow() as client:
            await claim_multi(client, room)
            await join_seat(client, gid, "D")
            clock[0] += 31
            assert "|waiting=1|" in (await client.get(f"/vrc/solo/{room}/start", params={"session": OWNER})).text
            clock[0] += 90
            result = await client.get(f"/vrc/solo/{room}/takeover", params={"session": OWNER, "seat": "D"})
            assert "|waiting=1|" in result.text, "Multi-seat mode must still wait for explicit START"
            assert game_app.GAMES[gid]["ai_seats"] == ["D"]
            assert (await join_seat(client, gid, "B")).status_code == 200
            assert "|waiting=0|" in (await client.get(f"/vrc/solo/{room}/start", params={"session": OWNER})).text
            assert game_app.GAMES[gid]["ai_seats"] == ["C", "D"]
    asyncio.run(flow())


@pytest.mark.parametrize("humans", [("B",), ("D",), ("B", "D"), ("B", "C", "D")])
def test_complete_round_with_browser_opponents_and_four_humans(coop, humans):
    room, gid, *_ = coop

    async def flow():
        async with await client_flow() as client:
            await claim_multi(client, room)
            for seat in humans:
                await join_seat(client, gid, seat)
            await client.get(f"/vrc/solo/{room}/start", params={"session": OWNER})
            game = game_app.GAMES[gid]
            for _ in range(200):
                state = game["state"]
                if state.finished:
                    break
                actor = state.turn
                assert actor in {"A", *humans}
                legal = state.legal_actions(actor)
                action = next((move for move in legal if move[0] != "pass"), legal[0])
                game["vrc_last_request"] = None
                if actor != "A":
                    result = await client.post(f"/games/{gid}/step", json={
                        "player": actor, "client_id": "browser-" + actor,
                        "expected_action_token": game_app._action_state_token(game, state),
                        "action": game_app._actions_to_json([action])[0],
                    })
                    assert result.status_code == 200, result.text
                elif action[0] == "pass":
                    assert (await client.get(f"/vrc/solo/{room}/pass", params={"session": OWNER})).text.startswith("G1|")
                else:
                    kind, first, attack = action
                    if kind == "receive":
                        shadow = copy.deepcopy(state)
                        game_app._apply_action(shadow, "A", action)
                        attack = next((move[2] for move in shadow.legal_actions("A") if move[0] == "attack"), "1")
                    select = "receive" if kind == "receive" else "block"
                    assert (await client.get(f"/vrc/solo/{room}/{select}/{first}", params={"session": OWNER})).text.startswith("R1|")
                    result = await client.get(f"/vrc/solo/{room}/attack/{attack}", params={"session": OWNER})
                    assert result.text.startswith("G1|"), result.text
            assert state.finished, "multi-seat round did not reach a finish"
            public = (await client.get(f"/games/{gid}/state")).json()
            assert all(isinstance(hand, list) for hand in public["hands"].values())
            await client.get(f"/vrc/solo/{room}/start", params={"session": OWNER})
            assert set(game_app.GAMES[gid]["human_seats"]) == {"A", *humans}
    asyncio.run(flow())


def test_start_join_race_never_assigns_same_seat_to_ai_and_human(coop):
    room, gid, *_ = coop

    async def flow():
        async with await client_flow() as client:
            await claim_multi(client, room)
            started, joined = await asyncio.gather(
                client.get(f"/vrc/solo/{room}/start", params={"session": OWNER}),
                join_seat(client, gid, "B"),
            )
            assert started.text.startswith("G1|")
            assert joined.status_code in {200, 409}
            game = game_app.GAMES[gid]
            assert ("B" in game["human_seats"]) != ("B" in game["ai_seats"])
            assert ("B" in game["human_seats"]) == (joined.status_code == 200)
    asyncio.run(flow())

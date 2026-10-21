import asyncio
import copy
import pytest
from backend import app as game_app
from backend import vrc_solo_playback
from test_vrc_solo_coop import coop, OWNER, OTHER, client_flow
from test_vrc_solo_playback import visual_game


def fields(response):
    assert response.text.startswith("G1|"), response.text
    return dict(item.split("=", 1) for item in response.text.split("|")[1:])


async def get(client, room, path, **params):
    return await client.get(f"/vrc/solo/{room}/{path}", params={"session": OWNER, **params})


@pytest.mark.parametrize("vrc,web", [(vrc, web) for vrc in ("B", "C", "D", "BC", "BD", "CD")
                                    for web in ("", *[s for s in "BCD" if s not in vrc])])
def test_mixed_combinations_ownership_and_start(coop, vrc, web):
    room, gid, *_ = coop
    async def flow():
        async with await client_flow() as client:
            assert fields(await get(client, room, "claim", mode="mixed"))["waiting"] == "1"
            for seat in vrc:
                assert fields(await get(client, room, f"player/{seat}/join"))["vrcmask"] == str(1 | sum(1 << "ABCD".index(s) for s in vrc[:vrc.index(seat)+1]))
                # Fixed URL retry is idempotent.
                fields(await get(client, room, f"player/{seat}/join"))
                assert (await client.post(f"/games/{gid}/claim", params={"seat": seat, "client_id": "intruder"})).status_code == 409
                assert (await client.post(f"/games/{gid}/claim", params={"seat": seat, "client_id": f"vrc-solo:{room}:{seat}"})).status_code == 403
            for seat in web:
                assert (await client.post(f"/games/{gid}/claim", params={"seat": seat, "client_id": "browser"})).status_code == 200
                assert "seat_busy" in (await get(client, room, f"player/{seat}/join")).text
            snap = fields(await get(client, room, "start"))
            game = game_app.GAMES[gid]
            assert set(game["human_seats"]) == {"A", *vrc, *web}
            assert set(game["ai_seats"]) == set("BCD") - set(vrc + web)
            for seat in "ABCD":
                assert snap["h"+seat.lower()].startswith("#") == (seat not in "A" + vrc)
            spectator = (await client.get(f"/games/{gid}/state")).json()
            assert not spectator["owned_human_seats"]
            assert all(isinstance(h, dict) for h in spectator["hands"].values())
            for seat in vrc:
                row = spectator["vrc_web"]["seats"][seat]
                assert row["source"] == "vrchat" and not row["can_join"] and not row["owned"]
            assert snap["chatshow"] == str(int(bool(web)))
    asyncio.run(flow())


def test_cap_race_release_and_late_join(coop):
    room, gid, *_ = coop
    async def flow():
        async with await client_flow() as client:
            await get(client, room, "claim", mode="mixed")
            responses = await asyncio.gather(*(get(client, room, f"player/{s}/join") for s in "BCD"))
            assert sum(r.text.startswith("G1|") for r in responses) == 2
            assert sum("vrc_full" in r.text for r in responses) == 1
            game = game_app.GAMES[gid]
            seat = next(s for s in "BCD" if s in game["human_seats"])
            fields(await get(client, room, f"player/{seat}/leave"))
            fields(await get(client, room, f"player/{seat}/leave"))
            assert seat not in game["human_seats"]
            # A site/VRChat race has exactly one winner under the same turn lock.
            results = await asyncio.gather(get(client, room, f"player/{seat}/join"), client.post(f"/games/{gid}/claim", params={"seat":seat,"client_id":"browser"}))
            assert sum(r.text.startswith("G1|") or r.status_code == 200 and r.text.startswith("{") for r in results) == 1
            open_seat = next(s for s in "BCD" if s not in game["human_seats"])
            await get(client, room, "start")
            assert "already_started" in (await get(client, room, f"player/{open_seat}/join")).text
            assert "code=busy" in (await client.get(f"/vrc/solo/{room}/player/B/leave", params={"session":OTHER})).text
    asyncio.run(flow())


@pytest.mark.parametrize("seat", list("BCD"))
def test_real_game_moves_all_vrc_seats_replay_and_stamps(coop, seat):
    room, gid, *_ = coop
    async def flow():
        async with await client_flow() as client:
            await get(client, room, "claim", mode="mixed")
            await get(client, room, f"player/{seat}/join")
            website = next(s for s in "BCD" if s != seat)
            await client.post(f"/games/{gid}/claim", params={"seat":website,"client_id":"browser"})
            await get(client, room, "start")
            game = game_app.GAMES[gid]
            state = game["state"]
            vrc_solo_playback.start(game)
            seen = set()
            for _ in range(100):
                if state.finished: break
                actor = state.turn
                action = next((a for a in state.legal_actions(actor) if a[0] != "pass"), state.legal_actions(actor)[0])
                if actor not in {"A", seat}:
                    game_app._vrc_solo_apply_action(game, actor, action)
                    continue
                seen.add(actor)
                game["vrc_last_request"] = None
                before = copy.deepcopy(state)
                snap = fields(await get(client, room, "state"))
                assert snap["lb"] or snap["lr"] or snap["la"] or snap["pass"] == "1"
                wrong = website
                assert "not_your_turn" in (await get(client, room, "block/1", seat=wrong)).text
                if action[0] == "pass":
                    result = await get(client, room, "pass", seat=actor)
                elif action[0] == "attack_after_block":
                    first = await get(client, room, "block/" + action[1], seat=actor)
                    assert first.text.startswith("R1|")
                    assert game["state"].hands == before.hands  # no half-applied move
                    result = await get(client, room, "attack/" + action[2], seat=actor)
                elif action[0] == "receive":
                    shadow = copy.deepcopy(state)
                    game_app._apply_action(shadow, actor, action)
                    follow = shadow.legal_actions(actor)[0]
                    assert (await get(client, room, "receive/" + action[1], seat=actor)).text.startswith("R1|")
                    result = await get(client, room, "attack/" + follow[2], seat=actor)
                else:
                    raise AssertionError(action)
                fields(result)
            assert seat in seen
            history = game["vrc_playback"]
            for frame in [history["base"], *history["steps"]]:
                assert len(frame) == 14
                for other in "BCD":
                    assert (frame[10 + "ABCD".index(other)] == "#") == (other != seat)
            # Site human can never obtain VRC hands, even by guessing bridge IDs.
            if not state.finished:
                web = (await client.get(f"/games/{gid}/state", params={"viewer":seat,"client_id":f"vrc-solo:{room}:{seat}"})).json()
                assert not web["owned_human_seats"] and all(isinstance(h,dict) for h in web["hands"].values())
            sent = await get(client, room, "stamp/sorry", seat=seat)
            assert sent.text.startswith("S1|")
            assert game["chat_messages"][-1]["seat"] == seat
            assert "invalid_seat" in (await get(client, room, "stamp/nice", seat=website)).text
    asyncio.run(flow())


def test_departure_takeover_and_next_deal_preserves_seats(coop):
    room, gid, _lease, clock = coop
    async def flow():
        async with await client_flow() as client:
            await get(client, room, "claim", mode="mixed")
            await get(client, room, "player/C/join")
            await client.post(f"/games/{gid}/claim", params={"seat":"B","client_id":"browser"})
            await get(client, room, "start")
            game = game_app.GAMES[gid]
            clock[0] += 121
            snap = fields(await get(client, room, "state"))
            assert snap["conline"] == "1" and snap["ctake"] == "0"  # bridge refreshes VRC presence only
            assert snap["bonline"] == "0" and snap["btake"] == "1"
            game["state"].finished = True
            game["state"].winner = "A"
            fields(await get(client, room, "start"))
            assert game_app.GAMES[gid]["vrc_web"]["transport"] == "vrc"
            snap = fields(await get(client, room, "player/C/leave"))
            assert snap["ctake"] == "1" and snap["conline"] == "0"
            assert "already_started" in (await get(client, room, "player/C/join")).text
            snap = fields(await get(client, room, "takeover", seat="C"))
            assert int(snap["vrcmask"]) == 1 and "C" in game_app.GAMES[gid]["ai_seats"]
            await get(client, room, "release")
            assert "lease_lost" in (await get(client, room, "player/D/join")).text
    asyncio.run(flow())


@pytest.mark.parametrize("seat", list("BCD"))
def test_vrc_identical_pair_replay_preserves_one_copy_and_masks_other_seats(seat):
    game = visual_game()
    slot = {"client_id":"vrc-solo:test:" + seat, "transport":"vrc"}
    game["vrc_web"] = {"enabled":True,"multi":True,"mixed":True,"seats":{}}
    if seat == "C": game["vrc_web"].update(slot)
    else: game["vrc_web"]["seats"][seat] = slot
    vrc_solo_playback.start(game)
    game["state"].hands[seat] = list("234569")
    game["board"][seat]["receive"][0] = "1"
    game["board"][seat]["receive_hidden"][0] = True
    game["board"][seat]["attack"][0] = "1"
    vrc_solo_playback.record(game, seat, ("attack_after_block","1","1"))
    first,second = game["vrc_playback"]["steps"]
    index = "ABCD".index(seat)
    assert first[10+index].count("1") == 1
    assert "1" not in second[10+index]
    assert first[6][index] == "7" and second[6][index] == "6"
    assert first[3][index*8+4] == "0" and second[3][index*8+4] == "1"

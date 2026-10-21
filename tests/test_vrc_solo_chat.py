import asyncio
import copy

import pytest

from backend import app as game_app
from backend.vrc_solo_chat import STAMPS
from test_vrc_solo_coop import coop, OWNER, OTHER, client_flow


def fields(text):
    return dict(item.split("=", 1) for item in text.split("|")[1:] if "=" in item)


async def setup(client, room, gid, seat="C"):
    assert (await client.get(f"/vrc/solo/{room}/claim", params={"session": OWNER, "mode": "multi"})).text.startswith("G1|")
    if seat:
        assert (await client.post(f"/games/{gid}/claim", params={"seat": seat, "client_id": "browser"})).status_code == 200


def test_bridge_stamps_match_the_six_free_site_stamps():
    assert STAMPS == ("greeting", "thanks", "thinking", "nice", "sorry", "surprised")
    assert set(STAMPS) <= game_app.PUBLIC_CHAT_STAMP_IDS


@pytest.mark.parametrize("seat", ["B", "C", "D"])
def test_panel_appears_for_any_human_and_stays_during_disconnect(coop, seat):
    room, gid, _, clock = coop
    async def flow():
        async with await client_flow() as client:
            await setup(client, room, gid, seat)
            clock[0] += 200
            state = await client.get(f"/vrc/solo/{room}/state", params={"session": OWNER})
            assert fields(state.text)["chatshow"] == "1"
            assert "|waiting=1|" in state.text
            await client.post(f"/games/{gid}/release", params={"seat": seat, "client_id": "browser"})
            result = await client.get(f"/vrc/solo/{room}/chat", params={"session": OWNER})
            assert result.text == "C1|chatshow=0|chat="
    asyncio.run(flow())


def test_spectators_and_ai_only_never_enable_panel(coop):
    room, gid, *_ = coop
    async def flow():
        async with await client_flow() as client:
            await setup(client, room, gid, None)
            await client.get(f"/games/{gid}/state", params={"client_id": "spectator"})
            await client.get(f"/vrc/solo/{room}/start", params={"session": OWNER})
            before = copy.deepcopy(game_app.GAMES[gid]["state"].hands)
            assert (await client.get(f"/vrc/solo/{room}/chat", params={"session": OWNER})).text == "C1|chatshow=0|chat="
            result = await client.get(f"/vrc/solo/{room}/stamp/greeting", params={"session": OWNER})
            assert "code=chat_unavailable" in result.text
            assert game_app.GAMES[gid]["chat_messages"] == []
            assert game_app.GAMES[gid]["state"].hands == before
    asyncio.run(flow())


@pytest.mark.parametrize("stamp", STAMPS)
def test_six_public_stamps_are_a_room_messages_without_game_moves(coop, stamp):
    room, gid, *_ = coop
    async def flow():
        async with await client_flow() as client:
            await setup(client, room, gid)
            game = game_app.GAMES[gid]
            before = copy.deepcopy(game["board"]), copy.deepcopy(game["state"].hands), list(game["kifu_moves"])
            response = await client.get(f"/vrc/solo/{room}/stamp/{stamp}", params={"session": OWNER})
            assert response.text.startswith("S1|stamp=" + stamp)
            item = game["chat_messages"][-1]
            assert item["seat"] == "A" and item["stamp_id"] == stamp and item["message_type"] == "stamp"
            assert item["room_id"] == gid and item["origin"] == "room"
            assert (game["board"], game["state"].hands, game["kifu_moves"]) == before
            state = (await client.get(f"/games/{gid}/state", params={"viewer": "C", "client_id": "browser"})).json()
            assert state["chat_messages"][-1]["stamp_id"] == stamp
            assert isinstance(state["hands"]["A"], dict)
    asyncio.run(flow())


@pytest.mark.parametrize("stamp", ["greeting", "sorry", "surprised"])
def test_stamps_are_serialized_deduplicated_and_rate_limited(coop, monkeypatch, stamp):
    room, gid, *_ = coop
    now = [100.0]
    monkeypatch.setattr(game_app.VRC_CHAT, "clock", lambda: now[0])
    async def flow():
        async with await client_flow() as client:
            await setup(client, room, gid)
            url = f"/vrc/solo/{room}/stamp/{stamp}"
            a, b = await asyncio.gather(client.get(url, params={"session": OWNER}), client.get(url, params={"session": OWNER}))
            assert a.text == b.text and len(game_app.GAMES[gid]["chat_messages"]) == 1
            other = await client.get(f"/vrc/solo/{room}/stamp/thanks", params={"session": OWNER})
            assert "code=stamp_cooldown" in other.text
            now[0] += 10.1
            await client.get(url, params={"session": OWNER})
            assert len(game_app.GAMES[gid]["chat_messages"]) == 2
    asyncio.run(flow())


@pytest.mark.parametrize("session", ["", OTHER])
def test_non_holder_cannot_read_chat_or_send_stamp(coop, session):
    room, gid, *_ = coop
    async def flow():
        async with await client_flow() as client:
            await setup(client, room, gid)
            for endpoint in ("chat", "stamp/greeting"):
                result = await client.get(f"/vrc/solo/{room}/{endpoint}", params={"session": session})
                assert result.text.startswith("E1|")
                assert "chat=" not in result.text
            assert not game_app.GAMES[gid]["chat_messages"]
    asyncio.run(flow())


@pytest.mark.parametrize("stamp", ["happy", "weekly_champion", "unknown"])
def test_restricted_or_unknown_stamps_are_not_unlocked(coop, stamp):
    room, gid, *_ = coop
    async def flow():
        async with await client_flow() as client:
            await setup(client, room, gid)
            result = await client.get(f"/vrc/solo/{room}/stamp/{stamp}", params={"session": OWNER})
            assert "code=invalid_stamp" in result.text
            assert not game_app.GAMES[gid]["chat_messages"]
    asyncio.run(flow())


def test_chat_is_bounded_room_only_and_protocol_safe(coop):
    room, gid, *_ = coop
    async def flow():
        async with await client_flow() as client:
            await setup(client, room, gid)
            messages = game_app.GAMES[gid]["chat_messages"]
            for i in range(20):
                messages.append({"seat": "C", "message": f"{i}: 日本語|chatshow=0\n\u202e<size=99>" + "長" * 200})
            response = await client.get(f"/vrc/solo/{room}/chat", params={"session": OWNER})
            result = fields(response.text)
            assert result["chatshow"] == "1" and len(result) == 2
            text = bytes.fromhex(result["chat"]).decode("utf-8")
            assert len(text.splitlines()) == 8 and "C: 19:" in text
            assert all(len(line) <= 93 for line in text.splitlines())
            assert "\u202e" not in text and len(result["chat"]) < 12000
            assert "ha=" not in response.text and "session=" not in response.text and "browser" not in text
    asyncio.run(flow())


def test_all_taken_over_hides_panel_and_release_rejects_stale_messages(coop):
    room, gid, *_ = coop
    async def flow():
        async with await client_flow() as client:
            await setup(client, room, gid)
            await client.get(f"/vrc/solo/{room}/start", params={"session": OWNER})
            await client.post(f"/games/{gid}/release", params={"seat": "C", "client_id": "browser"})
            assert fields((await client.get(f"/vrc/solo/{room}/chat", params={"session": OWNER})).text)["chatshow"] == "1"
            await client.get(f"/vrc/solo/{room}/takeover", params={"session": OWNER, "seat": "C"})
            assert (await client.get(f"/vrc/solo/{room}/chat", params={"session": OWNER})).text == "C1|chatshow=0|chat="
            await client.get(f"/vrc/solo/{room}/release", params={"session": OWNER})
            stale = await client.get(f"/vrc/solo/{room}/stamp/greeting", params={"session": OWNER})
            assert stale.text.startswith("E1|")
    asyncio.run(flow())


def test_chat_and_cooldown_survive_round_change_but_not_new_lease(coop):
    room, gid, *_ = coop
    async def flow():
        async with await client_flow() as client:
            await setup(client, room, gid)
            await client.get(f"/vrc/solo/{room}/start", params={"session": OWNER})
            first = await client.get(f"/vrc/solo/{room}/stamp/thanks", params={"session": OWNER})
            game_app.GAMES[gid]["state"].finished = True
            game_app.GAMES[gid]["state"].winner = "A"
            await client.get(f"/vrc/solo/{room}/start", params={"session": OWNER})
            assert len(game_app.GAMES[gid]["chat_messages"]) == 1
            retry = await client.get(f"/vrc/solo/{room}/stamp/thanks", params={"session": OWNER})
            assert first.text == retry.text
            await client.get(f"/vrc/solo/{room}/release", params={"session": OWNER})
            await client.get(f"/vrc/solo/{room}/claim", params={"session": OTHER, "mode": "multi"})
            assert game_app.GAMES[gid]["chat_messages"] == []
            assert "vrc_stamp_last" not in game_app.GAMES[gid]
    asyncio.run(flow())

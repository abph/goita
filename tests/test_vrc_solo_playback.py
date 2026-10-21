import asyncio
import copy
from types import SimpleNamespace

import httpx

from backend import app as game_app
from backend import vrc_solo_playback as playback
from test_vrc_solo_bridge import bridge, OWNER, response_text

REAL_RUN_AI = game_app._vrc_solo_run_ai
REAL_AGENT_TURN = game_app._apply_agent_turn


def visual_game():
    return {
        "vrc_solo_room": "test", "is_started": True,
        "state": SimpleNamespace(hands={s: list("11234569") for s in "ABCD"},
                                 turn="A", phase="attack", attacker=None, current_attack=None),
        "board": game_app._new_board_snapshot(),
    }


def frames(body):
    field = dict(item.split("=", 1) for item in body.split("|")[1:] if "=" in item)
    return field, [frame.split(",") for frame in field["replay"].split(";")]


def test_initial_frame_and_non_bridge_rooms():
    game = visual_game()
    playback.start(game)
    base = game["vrc_playback"]["base"]
    assert base == ["0", "-", "-", "0" * 32, "0" * 16, "11234569", "8888", "A", "a", "-", "0"]
    del game["vrc_solo_room"]
    game.pop("vrc_playback")
    playback.start(game)
    playback.record(game, "B", ("pass", None, None))
    assert playback.fields(game) == []


def test_atomic_pair_is_two_frames_and_never_changes_actual_game():
    game = visual_game()
    playback.start(game)
    game["state"].hands["B"] = list("123456")
    game["board"]["B"]["receive"][0] = "9"
    game["board"]["B"]["receive_hidden"][0] = True
    game["board"]["B"]["attack"][0] = "2"
    game["state"].turn, game["state"].attacker, game["state"].current_attack = "C", "B", "2"
    before = copy.deepcopy(game["state"]), copy.deepcopy(game["board"])
    playback.record(game, "B", ("attack_after_block", "9", "2"))
    first, second = game["vrc_playback"]["steps"]
    assert first[0:3] == ["1", "B", "H"]
    assert first[3][8:16] == "10000000"  # hidden 9 is never sent
    assert first[6] == "8788" and first[7:11] == ["B", "a", "-", "0"]
    assert second[0:3] == ["2", "B", "A"]
    assert second[3][8:16] == "10002000" and second[6] == "8688"
    assert game["state"].__dict__ == before[0].__dict__ and game["board"] == before[1]


def test_identical_a_pair_keeps_one_copy_in_middle_hand():
    game = visual_game()
    playback.start(game)
    game["state"].hands["A"] = list("234569")
    game["board"]["A"]["receive"][0] = "1"
    game["board"]["A"]["receive_hidden"][0] = True
    game["board"]["A"]["attack"][0] = "1"
    playback.record(game, "A", ("attack_after_block", "1", "1"))
    first, second = game["vrc_playback"]["steps"]
    assert first[5].count("1") == 1 and first[6] == "7888"
    assert "1" not in second[5] and second[6] == "6888"


def test_receive_attack_pass_order_and_history_bound():
    game = visual_game()
    playback.start(game)
    for action in (("receive", "1", None), ("attack", None, "2"), ("pass", None, None)):
        playback.record(game, "C", action)
    assert [row[2] for row in game["vrc_playback"]["steps"]] == ["R", "A", "P"]
    for _ in range(200):
        playback.record(game, "D", ("pass", None, None))
    history = game["vrc_playback"]
    assert len(history["steps"]) == playback.MAX_STEPS
    assert history["number"] == 203 and history["base"][0] == "0"
    assert history["steps"][0][0] == "76"


def test_history_masks_hidden_values_even_after_finish():
    game = visual_game()
    playback.start(game)
    game["state"].finished = True
    for seat in "BCD":
        game["board"][seat]["receive"][0] = "9"
        game["board"][seat]["receive_hidden"][0] = True
    playback.record(game, "D", ("attack", None, "2"))
    record = game["vrc_playback"]["steps"][-1]
    assert all(record[3]["ABCD".index(seat) * 8] == "1" for seat in "BCD")
    assert record[6] == "8888" and len(record) == 11
    assert "11234569" not in ",".join(record[6:])
    assert game["state"].hands["B"] == list("11234569")


def test_vrc_http_pair_records_steps_and_retry_is_idempotent(bridge):
    room, gid = bridge
    start = response_text(asyncio.run(game_app.vrc_solo_claim(room, OWNER)))
    assert frames(start)[0]["ply"] == "0"
    game = game_app.GAMES[gid]
    action = next(a for a in game["state"].legal_actions("A") if a[0] == "attack_after_block")
    asyncio.run(game_app.vrc_solo_block(room, int(action[1]), OWNER))
    first = response_text(asyncio.run(game_app.vrc_solo_attack(room, int(action[2]), OWNER)))
    second = response_text(asyncio.run(game_app.vrc_solo_attack(room, int(action[2]), OWNER)))
    assert first == second
    fields, steps = frames(first)
    assert fields["ply"] == "2" and [row[2] for row in steps] == ["-", "H", "A"]
    assert len(game["kifu_moves"]) == 1
    assert all(len(row[3]) == 32 and len(row[4]) == 16 for row in steps)


def test_waiting_start_and_next_deal_have_fresh_history(bridge):
    room, gid = bridge
    waiting = response_text(asyncio.run(game_app.vrc_solo_claim(room, OWNER, mode="multi")))
    assert "|replay=" not in waiting and "vrc_playback" not in game_app.GAMES[gid]
    game = game_app.GAMES[gid]
    # The real runner initializes a baseline before the first AI/human move.
    assert game_app.VRC_WEB.start_waiting(game)
    playback.start(game)
    old_nonce = game["action_state_nonce"]
    game["state"].finished = True
    game["state"].winner = "A"
    game["vrc_last_request"] = None
    fresh = response_text(asyncio.run(game_app.vrc_solo_start(room, OWNER)))
    fields, steps = frames(fresh)
    assert fields["deal"] != old_nonce and fields["ply"] == "0" and len(steps) == 1


def test_real_ai_runner_records_complete_round_without_changing_rules(bridge, monkeypatch):
    room, gid = bridge
    monkeypatch.setattr(game_app, "_schedule_ai_background_search", lambda *args: None)

    def cheap_agent(game, seat):
        legal = game["state"].legal_actions(seat)
        action = next((a for a in legal if a[0] == "pass"), legal[0])
        # Use the production mutation path, but skip expensive strategic search.
        return REAL_AGENT_TURN(game, seat, forced_action=action)

    monkeypatch.setattr(game_app, "_apply_agent_turn", cheap_agent)
    monkeypatch.setattr(game_app, "_vrc_solo_run_ai", REAL_RUN_AI)
    asyncio.run(game_app.vrc_solo_claim(room, OWNER))
    game = game_app.GAMES[gid]
    for _ in range(100):
        if game["state"].finished:
            break
        assert game["state"].turn == "A"
        legal = game["state"].legal_actions("A")
        game_app._vrc_solo_apply_action(game, "A", next((a for a in legal if a[0] != "pass"), legal[0]))
        asyncio.run(REAL_RUN_AI(game))
    assert game["state"].finished
    fields, steps = frames(game_app._vrc_solo_snapshot(game))
    assert fields["finished"] == "1" and len(steps) > 7
    assert {f[1] for f in steps[1:]} == set("ABCD")
    assert [int(f[0]) for f in steps] == list(range(len(steps)))
    previous = steps[0]
    for row in steps[1:]:
        changed = [i for i in range(32) if row[3][i] != previous[3][i]]
        assert len(changed) == (0 if row[2] == "P" else 1)
        if changed:
            assert changed[0] // 8 == "ABCD".index(row[1])
            assert previous[3][changed[0]] == "0"
        assert len(row[5]) == int(row[6][0])
        for seat in range(4):
            assert int(row[6][seat]) + sum(c != "0" for c in row[3][seat * 8:seat * 8 + 8]) == 8
        previous = row


def test_ai_dealer_opening_has_full_baseline_and_sequential_steps(bridge, monkeypatch):
    room, gid = bridge
    monkeypatch.setattr(game_app.random, "choice", lambda _items: "B")
    monkeypatch.setattr(game_app, "_schedule_ai_background_search", lambda *args: None)
    monkeypatch.setattr(game_app, "_vrc_solo_run_ai", REAL_RUN_AI)

    def cheap_agent(game, seat):
        action = game["state"].legal_actions(seat)[0]
        return REAL_AGENT_TURN(game, seat, forced_action=action)

    monkeypatch.setattr(game_app, "_apply_agent_turn", cheap_agent)
    body = response_text(asyncio.run(game_app.vrc_solo_claim(room, OWNER)))
    fields, steps = frames(body)
    assert steps[0][3] == "0" * 32 and steps[0][6] == "8888"
    assert steps[1][1:3] == ["B", "H"] and steps[2][1:3] == ["B", "A"]
    assert fields["turn"] == "A" or fields["finished"] == "1"


def test_browser_move_uses_same_history_but_never_receives_a_hand(bridge, monkeypatch):
    room, gid = bridge
    monkeypatch.setattr(game_app, "_vrc_solo_run_ai", REAL_RUN_AI)

    async def flow():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=game_app.app), base_url="http://test") as client:
            await client.get(f"/vrc/solo/{room}/claim", params={"session": OWNER, "mode": "multi"})
            for seat in "BCD":
                await client.post(f"/games/{gid}/claim", params={"seat": seat, "client_id": "browser-" + seat})
            await client.get(f"/vrc/solo/{room}/start", params={"session": OWNER})
            game = game_app.GAMES[gid]
            action = next(a for a in game["state"].legal_actions("A") if a[0] == "attack_after_block")
            await client.get(f"/vrc/solo/{room}/block/{action[1]}", params={"session": OWNER})
            await client.get(f"/vrc/solo/{room}/attack/{action[2]}", params={"session": OWNER})
            legal = game["state"].legal_actions("B")
            action = next((a for a in legal if a[0] == "pass"), legal[0])
            result = await client.post(f"/games/{gid}/step", json={"player": "B", "client_id": "browser-B",
                "action": {"action_type": action[0], "block": action[1], "attack": action[2]}})
            assert result.status_code == 200, result.text
            state = result.json()["state"]
            assert isinstance(state["hands"]["A"], dict)
            assert "vrc_playback" not in state and "replay" not in state
            body = (await client.get(f"/vrc/solo/{room}/state", params={"session": OWNER})).text
            _, steps = frames(body)
            assert steps[-1][1] == "B" and int(steps[-1][0]) > 2
    asyncio.run(flow())

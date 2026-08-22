from __future__ import annotations

import asyncio
from collections import Counter

from fastapi import HTTPException

from backend import app as app_module
from backend.frequent_deal import (
    TOP_100_HAND_STRUCTURES,
    hand_structure,
    is_frequent_deal,
    is_frequent_hand,
)
from goita_ai2.constants import PIECE_TOTALS


def test_top_100_structure_filter() -> None:
    assert len(TOP_100_HAND_STRUCTURES) == 100
    assert is_frequent_hand(list("11123447"))
    assert hand_structure(list("11123447")) == (3, 1, 1, 1, 0, 0, 1, 0, 0)
    assert not is_frequent_hand(list("11111111"))


def test_frequent_deals_keep_the_full_deck_and_structure_limit() -> None:
    expected = Counter({str(piece): count for piece, count in PIECE_TOTALS.items()})
    for _ in range(20):
        hands = app_module.create_hands_for_deal_mode("frequent")
        assert set(hands) == set(app_module.ALL_SEATS)
        assert all(len(hand) == 8 for hand in hands.values())
        assert Counter(piece for hand in hands.values() for piece in hand) == expected
        assert is_frequent_deal(hands)
        assert all(hand.count("1") <= 4 for hand in hands.values())


def test_frequent_deal_mode_updates_now_and_persists_to_next_round() -> None:
    async def scenario() -> None:
        game_id = app_module.PRIVATE_A_GID
        client_id = "frequent-deal-host"
        previous = app_module.GAMES.get(game_id)
        game = app_module._create_game_obj(dealer="A")
        game["human_seats"] = {"A": client_id}
        app_module.GAMES[game_id] = game
        try:
            saved = await app_module.update_deal_mode(
                game_id,
                app_module.DealModeUpdateRequest(
                    requester="A",
                    client_id=client_id,
                    mode="frequent",
                ),
            )
            assert saved["applies_next_round"] is False
            assert game["deal_mode"] == "frequent"
            assert game["next_deal_mode"] == "frequent"
            assert is_frequent_deal(game["init_hands"])

            await app_module.start_game(game_id, requester="A", client_id=client_id)
            queued = await app_module.update_deal_mode(
                game_id,
                app_module.DealModeUpdateRequest(
                    requester="A",
                    client_id=client_id,
                    mode="normal",
                ),
            )
            assert queued["applies_next_round"] is True
            assert game["deal_mode"] == "frequent"
            assert game["next_deal_mode"] == "normal"

            await app_module.reset_game(
                game_id,
                dealer="A",
                requester="A",
                client_id=client_id,
            )
            next_game = app_module.GAMES[game_id]
            assert next_game["deal_mode"] == "normal"
            assert next_game["next_deal_mode"] == "normal"
        finally:
            app_module._cancel_turn_timeout_task(game_id)
            if previous is None:
                app_module.GAMES.pop(game_id, None)
            else:
                app_module.GAMES[game_id] = previous
            app_module.GAME_TURN_LOCKS.pop(game_id, None)

    asyncio.run(scenario())


def test_public_rooms_reject_frequent_deal_setting() -> None:
    async def scenario() -> None:
        try:
            await app_module.update_deal_mode(
                app_module.MAIN_GID,
                app_module.DealModeUpdateRequest(
                    requester="A",
                    client_id="unused",
                    mode="frequent",
                ),
            )
        except HTTPException as exc:
            assert exc.status_code == 403
        else:
            raise AssertionError("Public rooms must reject high-frequency deals")

    asyncio.run(scenario())


def test_preset_hands_take_priority_without_disabling_the_mode() -> None:
    async def scenario() -> None:
        game_id = app_module.PRIVATE_A_GID
        client_id = "frequent-preset-host"
        previous = app_module.GAMES.get(game_id)
        game = app_module._create_game_obj(dealer="A", deal_mode="frequent")
        game["human_seats"] = {"A": client_id}
        app_module.GAMES[game_id] = game
        hands = {
            "A": list("11112345"),
            "B": list("11112345"),
            "C": list("11234567"),
            "D": list("23456789"),
        }
        preset_counts = {
            seat: dict(Counter(hand)) for seat, hand in hands.items()
        }
        try:
            await app_module.reset_game_config(
                game_id,
                app_module.ResetConfigBody(
                    dealer="A",
                    preset_counts=preset_counts,
                    requester="A",
                    client_id=client_id,
                ),
            )
            preset_game = app_module.GAMES[game_id]
            assert preset_game["init_hands"] == hands
            assert preset_game["deal_mode"] == "frequent"
            assert preset_game["next_deal_mode"] == "frequent"
        finally:
            app_module._cancel_turn_timeout_task(game_id)
            if previous is None:
                app_module.GAMES.pop(game_id, None)
            else:
                app_module.GAMES[game_id] = previous
            app_module.GAME_TURN_LOCKS.pop(game_id, None)

    asyncio.run(scenario())


def test_frontend_contains_private_room_deal_setting() -> None:
    html = app_module.FRONTEND_DIR.joinpath("index.html").read_text(encoding="utf-8")
    assert 'id="dealModeSelect"' in html
    assert '<option value="normal" selected>通常配牌</option>' in html
    assert '<option value="frequent">高頻度配牌（練習用）</option>' in html
    assert "PRIVATE_ROOM_IDS.has(settingTargetGid || gid)" in html
    assert "/deal_mode" in html


if __name__ == "__main__":
    test_top_100_structure_filter()
    test_frequent_deals_keep_the_full_deck_and_structure_limit()
    test_frequent_deal_mode_updates_now_and_persists_to_next_round()
    test_public_rooms_reject_frequent_deal_setting()
    test_preset_hands_take_priority_without_disabling_the_mode()
    test_frontend_contains_private_room_deal_setting()
    print("FREQUENT_DEAL_TEST_OK")

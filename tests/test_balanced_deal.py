from __future__ import annotations

import asyncio
from collections import Counter

from backend import app as app_module
from backend.balanced_deal import (
    BALANCED_ABSOLUTE_RANKS,
    absolute_hand_rank,
    is_balanced_rank_deal,
)
from goita_ai2.constants import PIECE_TOTALS


def test_absolute_rank_filter_uses_the_existing_hand_rank_definition() -> None:
    examples = {
        "S": "13457789",
        "A": "34445679",
        "B": "11233369",
        "C": "11123458",
        "D": "11244567",
        "E": "11123556",
        "F": "11113456",
        "X": "11133556",
    }
    assert {rank: absolute_hand_rank(list(hand)) for rank, hand in examples.items()} == {
        rank: rank for rank in examples
    }
    assert BALANCED_ABSOLUTE_RANKS == {"B", "C", "D", "E", "F"}


def test_balanced_mode_deals_full_decks_without_s_a_or_x_hands() -> None:
    expected = Counter({str(piece): count for piece, count in PIECE_TOTALS.items()})
    for _ in range(40):
        hands = app_module.create_hands_for_deal_mode("balanced")
        assert set(hands) == set(app_module.ALL_SEATS)
        assert all(len(hand) == 8 for hand in hands.values())
        assert Counter(piece for hand in hands.values() for piece in hand) == expected
        assert all(hand.count("1") <= 4 for hand in hands.values())
        assert is_balanced_rank_deal(hands)
        assert all(absolute_hand_rank(hand) in BALANCED_ABSOLUTE_RANKS for hand in hands.values())


def test_balanced_mode_is_normalized_as_a_supported_deal_mode() -> None:
    assert app_module._normalize_deal_mode("balanced") == "balanced"
    assert app_module._normalize_deal_mode("unknown") == "normal"


def test_balanced_mode_updates_current_deal_and_persists_to_next_round() -> None:
    async def scenario() -> None:
        game_id = app_module.PRIVATE_A_GID
        client_id = "balanced-deal-host"
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
                    mode="balanced",
                ),
            )
            assert saved["applies_next_round"] is False
            assert game["deal_mode"] == "balanced"
            assert game["next_deal_mode"] == "balanced"
            assert is_balanced_rank_deal(game["init_hands"])

            await app_module.start_game(game_id, requester="A", client_id=client_id)
            await app_module.reset_game(
                game_id,
                dealer="A",
                requester="A",
                client_id=client_id,
            )
            next_game = app_module.GAMES[game_id]
            assert next_game["deal_mode"] == "balanced"
            assert next_game["next_deal_mode"] == "balanced"
            assert is_balanced_rank_deal(next_game["init_hands"])
        finally:
            app_module._cancel_turn_timeout_task(game_id)
            if previous is None:
                app_module.GAMES.pop(game_id, None)
            else:
                app_module.GAMES[game_id] = previous
            app_module.GAME_TURN_LOCKS.pop(game_id, None)

    asyncio.run(scenario())


def test_hand_presets_bypass_balanced_filter_without_disabling_it() -> None:
    async def scenario() -> None:
        game_id = app_module.PRIVATE_A_GID
        client_id = "balanced-preset-host"
        previous = app_module.GAMES.get(game_id)
        game = app_module._create_game_obj(dealer="A", deal_mode="balanced")
        game["human_seats"] = {"A": client_id}
        app_module.GAMES[game_id] = game
        hands = {
            "A": list("11112345"),
            "B": list("11112345"),
            "C": list("11234567"),
            "D": list("23456789"),
        }
        try:
            await app_module.reset_game_config(
                game_id,
                app_module.ResetConfigBody(
                    dealer="A",
                    preset_counts={seat: dict(Counter(hand)) for seat, hand in hands.items()},
                    requester="A",
                    client_id=client_id,
                ),
            )
            preset_game = app_module.GAMES[game_id]
            assert preset_game["init_hands"] == hands
            assert preset_game["deal_mode"] == "balanced"
            assert preset_game["next_deal_mode"] == "balanced"
        finally:
            app_module._cancel_turn_timeout_task(game_id)
            if previous is None:
                app_module.GAMES.pop(game_id, None)
            else:
                app_module.GAMES[game_id] = previous
            app_module.GAME_TURN_LOCKS.pop(game_id, None)

    asyncio.run(scenario())


def test_balanced_label_is_translated() -> None:
    root = app_module.FRONTEND_DIR
    assert '"均衡配牌（S・A・Xなし）":' in root.joinpath("i18n.js").read_text(encoding="utf-8")
    assert '"均衡配牌（S・A・Xなし）":' in root.joinpath("i18n-en.js").read_text(encoding="utf-8")

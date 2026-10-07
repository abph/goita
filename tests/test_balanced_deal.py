from __future__ import annotations

import asyncio
from collections import Counter

from backend import app as app_module
from backend.balanced_deal import (
    BALANCED_ABSOLUTE_RANKS,
    absolute_hand_rank,
    dealer_unanswerable_attack_count,
    has_wardna_forced_opening,
    is_balanced_rank_deal,
    is_wardna_balanced_deal,
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
    assert app_module._normalize_deal_mode("balanced_wardna") == "balanced_wardna"
    assert app_module._normalize_deal_mode("unknown") == "normal"


def test_wardna_mode_rejects_three_unanswerable_gold_attacks() -> None:
    hands = {
        "A": list("11355579"),
        "B": list("11122347"),
        "C": list("11123568"),
        "D": list("11234446"),
    }
    assert is_balanced_rank_deal(hands)
    assert dealer_unanswerable_attack_count(hands, "A") == 3
    assert has_wardna_forced_opening(hands, "A")
    assert not is_wardna_balanced_deal(hands, "A")


def test_wardna_mode_rejects_three_unanswerable_kyosha_attacks() -> None:
    hands = {
        "A": list("11122235"),
        "B": list("11344579"),
        "C": list("11123468"),
        "D": list("11345567"),
    }
    assert is_balanced_rank_deal(hands)
    assert dealer_unanswerable_attack_count(hands, "A") == 3
    assert not is_wardna_balanced_deal(hands, "A")


def test_wardna_mode_counts_mixed_hisha_and_kaku_attacks() -> None:
    hands = {
        "A": list("11136779"),
        "B": list("11122355"),
        "C": list("11234468"),
        "D": list("11234455"),
    }
    assert is_balanced_rank_deal(hands)
    assert dealer_unanswerable_attack_count(hands, "A") == 3
    assert not is_wardna_balanced_deal(hands, "A")
    assert is_wardna_balanced_deal(hands, "B")


def test_wardna_mode_keeps_deal_when_opponent_royal_can_receive() -> None:
    hands = {
        "A": list("11355579"),
        "B": list("11122348"),
        "C": list("11123567"),
        "D": list("11234446"),
    }
    assert is_balanced_rank_deal(hands)
    assert dealer_unanswerable_attack_count(hands, "A") == 0
    assert is_wardna_balanced_deal(hands, "A")


def test_wardna_mode_generates_balanced_non_forced_deals() -> None:
    for dealer in app_module.ALL_SEATS:
        for _ in range(10):
            hands = app_module.create_hands_for_deal_mode(
                "balanced_wardna",
                dealer=dealer,
            )
            assert is_wardna_balanced_deal(hands, dealer)


def test_wardna_mode_updates_current_deal_for_the_configured_dealer() -> None:
    async def scenario() -> None:
        game_id = app_module.PRIVATE_A_GID
        client_id = "wardna-deal-host"
        previous = app_module.GAMES.get(game_id)
        game = app_module._create_game_obj(dealer="C")
        game["human_seats"] = {"A": client_id}
        app_module.GAMES[game_id] = game
        try:
            saved = await app_module.update_deal_mode(
                game_id,
                app_module.DealModeUpdateRequest(
                    requester="A",
                    client_id=client_id,
                    mode="balanced_wardna",
                ),
            )
            assert saved["applies_next_round"] is False
            assert game["deal_mode"] == "balanced_wardna"
            assert game["next_deal_mode"] == "balanced_wardna"
            assert is_wardna_balanced_deal(game["init_hands"], "C")
        finally:
            app_module._cancel_turn_timeout_task(game_id)
            if previous is None:
                app_module.GAMES.pop(game_id, None)
            else:
                app_module.GAMES[game_id] = previous
            app_module.GAME_TURN_LOCKS.pop(game_id, None)

    asyncio.run(scenario())


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
    assert '"均衡配碑（wardna式）":' in root.joinpath("i18n.js").read_text(encoding="utf-8")
    assert '"均衡配碑（wardna式）":' in root.joinpath("i18n-en.js").read_text(encoding="utf-8")
    assert '"wardna式は、親が敵方に受けられない攻めを3回続けられる配牌も除外します。":' in root.joinpath("i18n.js").read_text(encoding="utf-8")
    assert '"wardna式は、親が敵方に受けられない攻めを3回続けられる配牌も除外します。":' in root.joinpath("i18n-en.js").read_text(encoding="utf-8")

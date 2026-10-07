import asyncio

import pytest
from fastapi import HTTPException

from backend import app as app_module


def make_game(human_seats, ai_seats=()):
    game = app_module._create_game_obj(dealer="A")
    game["human_seats"] = dict(human_seats)
    game["ai_seats"] = list(ai_seats)
    game["player_names"] = {
        seat: (f"name-{owner}" if owner else "")
        for seat, owner in {
            "A": human_seats.get("A", ""),
            "B": human_seats.get("B", ""),
            "C": human_seats.get("C", ""),
            "D": human_seats.get("D", ""),
        }.items()
    }
    game["player_tags"] = {
        seat: (f"tag-{owner}" if owner else "")
        for seat, owner in {
            "A": human_seats.get("A", ""),
            "B": human_seats.get("B", ""),
            "C": human_seats.get("C", ""),
            "D": human_seats.get("D", ""),
        }.items()
    }
    return game


def shuffle(monkeypatch, game, client_id="client-a"):
    game_id = "test-seat-shuffle"
    monkeypatch.setitem(app_module.GAMES, game_id, game)
    return asyncio.run(
        app_module.shuffle_waiting_room_seats(
            game_id,
            requester="A",
            client_id=client_id,
        )
    )


def assert_human_identity_follows_seat(game):
    for seat, owner in game["human_seats"].items():
        assert game["player_names"][seat] == f"name-{owner}"
        assert game["player_tags"][seat] == f"tag-{owner}"


def test_four_humans_get_a_different_partner_while_a_stays_fixed(monkeypatch):
    game = make_game({"A": "client-a", "B": "client-b", "C": "client-c", "D": "client-d"})

    result = shuffle(monkeypatch, game)

    assert game["human_seats"]["A"] == "client-a"
    assert game["human_seats"]["C"] in {"client-b", "client-d"}
    assert set(game["human_seats"].values()) == {"client-a", "client-b", "client-c", "client-d"}
    assert result["human_seats"] == ["A", "B", "C", "D"]
    assert_human_identity_follows_seat(game)
    assert game["chat_messages"][-1]["message"] == "Aが席をシャッフルしました。"


@pytest.mark.parametrize(
    ("other_seat", "expected_partner_is_human"),
    [("B", True), ("C", False)],
)
def test_two_humans_toggle_between_partner_and_opponent(
    monkeypatch,
    other_seat,
    expected_partner_is_human,
):
    ai_seats = [seat for seat in ("B", "C", "D") if seat != other_seat]
    game = make_game({"A": "client-a", other_seat: "client-other"}, ai_seats)

    shuffle(monkeypatch, game)

    assert game["human_seats"]["A"] == "client-a"
    assert (game["human_seats"].get("C") == "client-other") is expected_partner_is_human
    assert len(game["human_seats"]) == 2
    assert len(game["ai_seats"]) == 2
    assert_human_identity_follows_seat(game)


@pytest.mark.parametrize("current_partner", ["human", "ai"])
def test_three_humans_change_the_host_partner(monkeypatch, current_partner):
    if current_partner == "human":
        humans = {"A": "client-a", "B": "client-b", "C": "client-c"}
        ai_seats = ["D"]
        previous_partner = "client-c"
    else:
        humans = {"A": "client-a", "B": "client-b", "D": "client-d"}
        ai_seats = ["C"]
        previous_partner = None
    game = make_game(humans, ai_seats)

    shuffle(monkeypatch, game)

    assert game["human_seats"].get("C") in {"client-b", "client-c", "client-d"}
    assert game["human_seats"].get("C") != previous_partner
    assert game["human_seats"]["A"] == "client-a"
    assert_human_identity_follows_seat(game)


@pytest.mark.parametrize("invalid_state", ["one_human", "started", "later_round", "scored"])
def test_shuffle_rejects_invalid_waiting_states(monkeypatch, invalid_state):
    game = make_game({"A": "client-a", "B": "client-b"}, ["C", "D"])
    if invalid_state == "one_human":
        game = make_game({"A": "client-a"}, ["B", "C", "D"])
    elif invalid_state == "started":
        game["is_started"] = True
    elif invalid_state == "later_round":
        game["round_count"] = 2
    elif invalid_state == "scored":
        game["total_team_score"]["AC"] = 10

    with pytest.raises(HTTPException) as error:
        shuffle(monkeypatch, game)

    assert error.value.status_code == 409


def test_shuffle_requires_the_a_seat_owner(monkeypatch):
    game = make_game({"A": "client-a", "B": "client-b"}, ["C", "D"])

    with pytest.raises(HTTPException) as error:
        shuffle(monkeypatch, game, client_id="different-client")

    assert error.value.status_code == 403


def test_frontend_shows_shuffle_only_to_waiting_host_with_two_or_more_humans():
    html = (app_module.BASE_DIR / "frontend" / "index.html").read_text(encoding="utf-8")

    assert 'id="shuffleSeatsBtn" onclick="shuffleSeats()"' in html
    assert "async function shuffleSeats()" in html
    assert "/shuffle_seats?${query.toString()}" in html
    assert "humanSeatCount >= 2" in html
    assert "humanSeatCount <= 4" in html
    assert "Number(state.round_count || 1) === 1" in html
    assert "!state.is_started" in html
    assert "if (shuffleSeatsRequestInFlight) return;" in html
    assert "if (shuffleSeatsRequestInFlight || startRequestInFlight" in html
    assert "const movedSeat = seatOwnershipLost && ownedHumanSeats.length === 1" in html


def test_shuffle_labels_are_translated():
    english = (app_module.BASE_DIR / "frontend" / "i18n-en.js").read_text(encoding="utf-8")
    chinese = (app_module.BASE_DIR / "frontend" / "i18n.js").read_text(encoding="utf-8")

    for source in [
        "席をシャッフル",
        "席をシャッフルしています...",
        "席をシャッフルできませんでした。",
        "Aが席をシャッフルしました。",
    ]:
        assert source in english
        assert source in chinese

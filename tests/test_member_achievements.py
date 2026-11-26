from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi import Request

from backend import app as app_module
from backend.member_achievements import CATALOG, round_items, score_progress
from backend.member_api import MEMBER_COOKIE, create_member_router
from backend.member_store import MemberError, MemberStore
from backend.trace_results import TraceStore


def request_for(token=""):
    return Request({"type": "http", "scheme": "https", "path": "/", "headers": [
        (b"cookie", f"{MEMBER_COOKIE}={token}".encode()),
    ]})


def test_achievement_catalog_awards_each_item_once_even_concurrently(tmp_path):
    store = MemberStore(tmp_path / "members.sqlite3")
    _, token, _ = store.register("player", "member-test-password")
    with ThreadPoolExecutor(max_workers=4) as workers:
        outcomes = list(workers.map(lambda _: store.record_achievements("player", ["score_day_100", "first_win"]), range(4)))
    assert sum(len(outcome) for outcome in outcomes) == 2
    assert store.kifu_limit(store.authenticate(token)) == 26
    assert store.achievement_titles("player") == {"score_day_100": "ごいた命"}
    assert len(CATALOG) == 65
    assert "tutorial_animals" not in CATALOG
    assert sum(bool(item["title"]) for item in CATALOG.values()) == 5


def test_member_api_privacy_and_title_picker_permissions(tmp_path, monkeypatch):
    store = MemberStore(tmp_path / "members.sqlite3")
    _, token, _ = store.register("player", "member-test-password")
    _, other_token, _ = store.register("player-other", "member-test-password")
    router = create_member_router(store, lambda _request: None)
    endpoints = {route.endpoint.__name__: route.endpoint for route in router.routes}
    with pytest.raises(MemberError) as guest:
        endpoints["achievements"](request_for())
    assert guest.value.status == 401
    request = request_for(token)
    with pytest.raises(MemberError) as forged:
        endpoints["browser_achievement"]("score_weekly_first", request)
    assert forged.value.status == 404
    assert endpoints["browser_achievement"]("image_kifu", request)["kifu_limit"] == 21
    assert endpoints["browser_achievement"]("image_kifu", request)["granted"] == []
    store.record_achievements("player", ["score_streak_10"])
    monkeypatch.setattr(app_module, "MEMBER_STORE", store)
    assert app_module._sanitize_player_tag("score_streak_10", token) == "score_streak_10"
    assert app_module._sanitize_player_tag("score_streak_10", other_token) == ""
    assert app_module._sanitize_player_tag("score_streak_10", "") == ""
    assert app_module._sanitize_player_tag("ai_practice", "") == "ai_practice"
    data = endpoints["achievements"](request)
    assert data["completed"] == 2 and data["total"] == 65 and data["kifu_limit"] == 26
    assert endpoints["session"](request)["member"]["achievement_titles"] == {"score_streak_10": "ごいた無双"}
    assert endpoints["achievements"](request_for(other_token))["completed"] == 0


def test_streak_distinct_kifu_and_daily_counts_are_independent():
    rows = [dict(day="2026-11-25", exact_match_bonus=0, challenge=str(i % 3), improvement=10,
                 winner_seat="A" if i % 2 else "C", selection_mode="random") for i in range(99)]
    rows.extend([dict(rows[0], improvement=0, day="2026-11-26"), dict(rows[0], improvement=-10, day="2026-11-26")])
    values = score_progress(rows)
    assert values["score_day_50"] == values["score_day_100"] == 99
    assert values["score_streak_10"] == 99
    assert values["score_three_kifu"] == 3 and values["score_both_winners"] == 2
    assert score_progress([rows[0], rows[-2], rows[1]])["score_streak_3"] == 1


def test_trace_summary_uses_jst_and_excludes_practice(tmp_path):
    now = [datetime.fromisoformat("2026-11-25T23:59:59+09:00").timestamp()]
    store = TraceStore(tmp_path / "trace.sqlite3", clock=lambda: now[0])
    payload = {"hands": {}, "dealer": "A", "moves": [], "score_before": {"AC": 0, "BD": 0},
               "score_after": {"AC": 0, "BD": 10}}
    for i in range(2):
        attempt = store.start("member:player", False, "Player", payload)
        assert store.finish(attempt, {"AC": 20, "BD": 0}, winner_seat="A" if i == 0 else "C") == "player"
        now[0] += 2
    practice = store.start("member:player", False, "Player", payload, practice=True)
    store.finish(practice, {"AC": 30, "BD": 0}, winner_seat="A")
    assert store.member_achievement_progress("player")["score_streak_3"] == 2
    assert store.member_achievement_progress("player")["score_day_50"] == 1
    assert store.member_achievement_progress("player")["score_both_winners"] == 2


def test_settled_rank_rewards_stack_with_achievement_rewards(tmp_path, monkeypatch):
    now = [datetime.fromisoformat("2026-11-22T12:00:00+09:00").timestamp()]
    trace = TraceStore(tmp_path / "trace.sqlite3", clock=lambda: now[0])
    store = MemberStore(tmp_path / "members.sqlite3", clock=lambda: now[0])
    _, token, _ = store.register("player", "member-test-password")
    _, runner_up, _ = store.register("runner", "member-test-password")
    payload = {"hands": {}, "dealer": "A", "moves": [], "score_before": {"AC": 0, "BD": 0},
               "score_after": {"AC": 0, "BD": 10}}
    for member, points in (("player", 30), ("runner", 10)):
        attempt = trace.start("member:" + member, False, member, payload)
        trace.finish(attempt, {"AC": points, "BD": 0})
    now[0] += 86400
    monkeypatch.setattr(app_module, "MEMBER_STORE", store)
    monkeypatch.setattr(app_module, "get_trace_store", lambda: trace)
    monkeypatch.setattr(app_module, "_SCORE_REWARD_SETTLED_DAY", "")
    monkeypatch.setattr(app_module, "_SCORE_REWARD_SETTLED_WEEK", "")
    app_module._settle_score_rewards()
    app_module._settle_score_rewards()
    assert store.achievement_titles("player") == {"score_weekly_top3": "週間入賞", "score_weekly_first": "週間王者"}
    assert store.achievement_titles("runner") == {"score_weekly_top3": "週間入賞"}
    assert store.kifu_limit(store.authenticate(token)) == 20 + 10 + 1 + 5 + 1 + 1
    assert store.kifu_limit(store.authenticate(runner_up)) == 20 + 5 + 1


def test_kakarigotae_and_royal_order_are_inferred_from_the_actual_moves():
    game = {"kifu_moves": [["2", "し", "馬"], ["0", "馬", "馬"], ["0", "し", "玉"], ["0", "し", "王"]],
            "state": SimpleNamespace(winner="D"), "init_hands": {}}
    items = round_items(game, "A", ai=False, other_humans=True, public=True, support=False)
    assert {"public_human", "kakarigotae", "royal_second_third", "different_three"} <= items
    assert "first_win" not in items


def test_finished_round_grants_only_to_verified_participant(tmp_path, monkeypatch):
    store = MemberStore(tmp_path / "members.sqlite3")
    _, token, _ = store.register("player", "member-test-password")
    _, stranger, _ = store.register("visitor", "member-test-password")
    game = app_module._create_game_obj(dealer="A")
    game_id, client = "achievement-test-room", "card-player"
    game["state"].finished = True
    game["state"].winner = "A"
    game.update(human_seats={"A": client}, ai_seats=["B", "C", "D"], achievement_actors={"A"},
                achievement_round_roster={"A": client}, match_finished=True, match_winner="AC",
                total_team_score={"AC": 150, "BD": 0}, last_round_score=50,
                kifu_moves=[["0", "し", "銀"], ["0", "し", "玉"], ["0", "し", "王"], ["0", "し", "王"]])
    conn = SimpleNamespace(url=SimpleNamespace(scheme="wss", netloc="testserver"),
                           headers={"origin": "https://testserver"}, cookies={MEMBER_COOKIE: token})
    monkeypatch.setattr(app_module, "MEMBER_STORE", store)
    monkeypatch.setitem(app_module.GAMES, game_id, game)
    monkeypatch.setitem(app_module.manager.client_connections, (game_id, client), [conn])
    app_module._record_game_achievements(game)
    items = {item["id"] for item in game["achievement_updates"][client]["items"]}
    assert {"first_win", "first_match", "shutout", "ai_round", "royal_second_third"} <= items
    result = lambda member_token: app_module._state_public_view(
        game["state"], game_id=game_id, viewer="A", game_obj=game, client_id=client, member_token=member_token)
    assert result(token)["achievement_updates"]["member_id"] == "player"
    assert not result(stranger)["achievement_updates"]
    assert not result("")["achievement_updates"]

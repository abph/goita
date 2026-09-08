from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from backend import app as game_app
from backend.trace_results import TraceStore
from test_trace_results import payload, trace_client


def stamp(date):
    return datetime.fromisoformat(date + "+09:00").timestamp()


def test_daily_and_weekly_floor_daily_total_and_recalculate_today(tmp_path, payload):
    now = [stamp("2026-09-09T12:00:00")]
    store = TraceStore(tmp_path / "period.sqlite", clock=lambda: now[0])
    payload["score_before"] = {"AC": 0, "BD": 0}
    payload["score_after"] = {"AC": 70, "BD": 0}

    def add(owner, date, points, name="Player"):
        now[0] = stamp(date)
        attempt = store.start(owner, owner.startswith("guest:"), name, payload)
        store.finish(attempt, {"AC": 70 + points, "BD": 0})
        return attempt

    add("member:one", "2026-09-06T23:59:59", 100)  # Previous week, in Japan.
    add("member:one", "2026-09-07T00:00:00", 50)
    add("member:one", "2026-09-08T12:00:00", -70)
    add("member:two", "2026-09-08T12:00:00", 80, "Other")
    add("guest:three", "2026-09-08T12:00:00", -10, "Guest")
    add("member:one", "2026-09-09T12:00:00", 30)
    week = store.period_ranking("weekly", owner="member:one")
    assert (week["start_date"], week["end_date"]) == ("2026-09-07", "2026-09-13")
    assert [row["rank"] for row in week["ranking"]] == [1, 1, 3]
    own = next(row for row in week["ranking"] if row["self"])
    assert own["score"] == 80 and own["games"] == 3
    add("member:one", "2026-09-09T13:00:00", -50, "Renamed")
    day = store.period_ranking("daily", owner="member:one")
    assert day["ranking"][0]["score"] == -20
    assert day["ranking"][0]["name"] == "Renamed" and day["ranking"][0]["games"] == 2
    week = store.period_ranking("weekly", owner="member:one")
    own = next(row for row in week["ranking"] if row["self"])
    assert own["score"] == 50 and own["games"] == 4  # Clamp +30-50 to zero, not each game.
    store.start("member:one", False, "Renamed", payload)  # Unfinished never counted.
    assert store.period_ranking("daily")["ranking"][0]["games"] == 2
    now[0] = stamp("2026-09-10T00:00:00")
    assert store.period_ranking("daily")["ranking"] == []
    now[0] = stamp("2026-09-14T00:00:00")
    assert store.period_ranking("weekly")["ranking"] == []


def test_period_ranking_limit_and_no_private_identifiers(tmp_path, payload):
    store = TraceStore(tmp_path / "many.sqlite", clock=lambda: stamp("2026-09-09T12:00:00"))
    for i in range(101):
        attempt = store.start(f"member:secret-{i}", False, f"Player {i:03}", payload)
        store.finish(attempt, payload["score_after"])
    result = store.period_ranking()
    assert result["total"] == 101 and len(result["ranking"]) == 100
    assert all(row["rank"] == 1 for row in result["ranking"])
    assert "secret-" not in str(result) and "attempt_id" not in str(result)
    with pytest.raises(game_app.HTTPException):
        store.period_ranking("monthly")


def test_lobby_rankings_allow_visitors_without_room_or_new_identity(trace_client):
    before = set(game_app.GAMES)
    visitor = TestClient(game_app.app, headers={"X-Goita-Member":"1"})
    response = visitor.get("/api/score-attack/rankings?period=weekly")
    assert response.status_code == 200 and response.json()["ranking"] == []
    assert response.headers["Cache-Control"] == "no-store"
    assert not visitor.cookies and set(game_app.GAMES) == before
    assert visitor.get("/api/score-attack/rankings?period=bad").status_code == 400
    assert visitor.get("/api/score-attack/rankings", headers={"Origin":"https://evil.test"}).status_code == 403
    assert visitor.get("/api/score-attack/rankings", headers={"X-Goita-Member":""}).status_code == 403

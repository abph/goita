import pytest

from backend import app as game_app
from backend.trace_results import RETENTION, TraceStore
from test_trace_results import payload, trace_client
from test_score_rooms import enter, finish


def record(store, payload, owner, improvement, *, guest=False, **kwargs):
    attempt = store.start(owner, guest, "Player", payload, **kwargs)
    before, after = payload["score_before"], payload["score_after"]
    margin = after["AC"] - before["AC"] - after["BD"] + before["BD"] + improvement
    store.finish(attempt, {"AC": before["AC"] + max(0, margin),
                           "BD": before["BD"] + max(0, -margin)})
    return attempt


@pytest.mark.parametrize("scores,eligible", [
    ([0, 60], False), ([0, 10, 60], True), ([-20, 0, 30], True),
    ([-10, 0, 30], False), ([-50, 0, 0, 0], False),
    ([-50, -50, 0, 0], True), ([-10, -10, -10, -10, 60], False),
])
def test_divergence_thresholds(tmp_path, payload, scores, eligible):
    store = TraceStore(tmp_path / "trace.sqlite")
    for n, score in enumerate(scores):
        record(store, payload, f"member:{n}", score)
    assert (store.challenge_id(payload) in store.divergent_challenges()) is eligible


def test_retries_practice_unfinished_and_expired_results_do_not_qualify(tmp_path, payload):
    now = [1000]
    store = TraceStore(tmp_path / "trace.sqlite", clock=lambda: now[0])
    record(store, payload, "member:one", 0)
    record(store, payload, "member:one", 80)  # Retry, not a second player.
    record(store, payload, "member:two", 60)
    record(store, payload, "member:practice", 40, practice=True)
    store.start("member:unfinished", False, "Player", payload)
    assert not store.divergent_challenges()
    record(store, payload, "guest:three", 10, guest=True)
    assert store.divergent_challenges() == {store.challenge_id(payload)}
    now[0] += RETENTION + 1
    assert not store.divergent_challenges()


def test_comparison_is_anonymous_excludes_self_and_survives_history(tmp_path, payload):
    store = TraceStore(tmp_path / "trace.sqlite")
    for n, score in enumerate([-50, -50, 10, 30]):
        record(store, payload, f"member:{n}", score)
        record(store, payload, f"member:{n}", 80)
    attempt = record(store, payload, "member:me", 20, selection_mode="divergent")
    result = store.read("member:me", attempt)
    assert result["ranked"] is True
    assert result["improvement"] == 20
    assert result["selection_mode"] == "divergent"
    assert result["comparison"]["players"] == 4
    assert result["comparison"]["distribution"] == [
        {"improvement": -50, "players": 2}, {"improvement": 10, "players": 1},
        {"improvement": 30, "players": 1}]
    retry = record(store, payload, "member:me", 30, selection_mode="divergent")
    assert not store.read("member:me", retry)["ranked"]
    assert store.read("member:me", retry)["comparison"] == result["comparison"]
    normal = record(store, payload, "member:normal", 0)
    assert store.read("member:normal", normal)["comparison"] is None


def test_divergent_start_uses_current_eligible_pool_and_existing_scoring(trace_client, payload, monkeypatch):
    client = trace_client
    room = enter(client)
    store = game_app.get_trace_store()
    for n, score in enumerate([0, 10, 60]):
        record(store, payload, f"member:{n}", score)
    challenge = store.challenge_id(payload)
    # Historical results alone cannot restore a removed/flagged archive game.
    monkeypatch.setattr(game_app, "_random_trace_candidates", lambda: [])
    path = f"/games/{room}/trace_random_start"
    body = {"client_id": "owner", "selection_mode": "divergent"}
    assert client.post(path, json=body).status_code == 404
    assert not game_app.GAMES[room].get("trace_mode")
    candidate = {"payload": payload, "source": {}, "challenge_id": challenge, "candidate_id": "approved"}
    monkeypatch.setattr(game_app, "_random_trace_candidates", lambda: [candidate])
    assert client.post(path, json={**body, "selection_mode": "bad"}).status_code == 422
    assert client.post(path, json={**body, "client_id": "other"}).status_code == 403
    response = client.post(path, json=body)
    assert response.status_code == 200, response.text
    attempt = response.json()["state"]["trace_attempt_id"]
    assert "comparison" not in response.json()  # No results before finishing.
    assert client.get(f"/games/{room}/trace_results/{attempt}").status_code == 409
    finish(client, room)
    result = client.get(f"/games/{room}/trace_results/{attempt}").json()
    assert result["ranked"] and result["improvement"] == 0
    assert result["comparison"]["players"] == 3
    assert result["comparison"]["distribution"] == [
        {"improvement": 0, "players": 1}, {"improvement": 10, "players": 1},
        {"improvement": 60, "players": 1}]
    with store.db() as db:
        assert db.execute("SELECT audit_candidate_id FROM trace_attempts WHERE id=?", (attempt,)).fetchone()[0] == "approved"


def test_regular_start_still_works_without_divergent_results(trace_client):
    room = enter(trace_client)
    response = trace_client.post(f"/games/{room}/trace_random_start", json={"client_id": "owner"})
    assert response.status_code == 200, response.text
    assert game_app.GAMES[room]["trace_selection_mode"] == "random"

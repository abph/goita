import json
import time

import pytest
from fastapi.testclient import TestClient

from backend import app as game_app
from backend.private_kifu_archive import archive_path, save_archive


@pytest.fixture
def archive_client(tmp_path, monkeypatch):
    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.setenv("GOITA_PERSISTENT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(game_app, "LOBBY_ADMIN_PASSWORD", "unique-test-password")
    monkeypatch.setattr(game_app, "_KIFU_ARCHIVE_CACHE", None)
    monkeypatch.setattr(game_app, "_KIFU_RANDOM_TRACE_CACHE", None)
    # Avoid startup/background jobs; exercise the real routes and static mount.
    return TestClient(game_app.app)


@pytest.fixture
def archive_bytes():
    return json.dumps({"matches": [{"id": "001", "players": {"p0": "private name"},
        "rounds": [{"round_index": 1, "hand": {"p0": "sample hand"},
                    "game": [["private moves"]]}]}]}).encode()


def admin_login(client):
    client.cookies.set(game_app.ADMIN_SESSION_COOKIE, game_app._admin_session_token(int(time.time()) + 60))


def test_raw_static_urls_are_blocked_even_with_legacy_files(archive_client, tmp_path, monkeypatch):
    directory = tmp_path / "static"
    directory.mkdir()
    for name in ("kifu_data.json", "kifu_data_raw.json"):
        (directory / name).write_text("private content")
    mount = next(route for route in game_app.app.routes if route.path == "/static")
    monkeypatch.setattr(mount.app, "all_directories", [str(directory)])
    for url in ("/static/kifu_data.json", "/static/kifu_data_raw.json", "/static/%6bifu_data.json"):
        response = archive_client.get(url)
        assert response.status_code == 404
        assert "private content" not in response.text
    assert archive_client.get("/private_data/kifu_data.json").status_code == 404


def test_upload_and_read_require_authentication_and_same_origin(archive_client, archive_bytes):
    client = archive_client
    headers = {"X-Goita-Member": "1"}
    assert client.put("/admin/api/private-kifu", content=archive_bytes, headers=headers).status_code == 401
    assert client.get("/api/private-kifu/preset?match_id=1&round_index=1", headers=headers).status_code == 403
    assert client.post("/games/debug/trace_random_start", json={"client_id": "any"}, headers=headers).status_code == 403
    admin_login(client)
    assert client.put("/admin/api/private-kifu", content=archive_bytes).status_code == 403
    assert client.put("/admin/api/private-kifu", content=archive_bytes,
                      headers={**headers, "Origin": "https://other.example"}).status_code == 403
    result = client.put("/admin/api/private-kifu", content=archive_bytes, headers=headers)
    assert result.status_code == 200
    assert result.json() == {"ok": True, "match_count": 1, "round_count": 1}
    assert archive_path(game_app.BASE_DIR).read_bytes() == archive_bytes
    assert client.get("/admin/api/private-kifu", headers=headers).json()["registered"] is True
    response = client.get("/api/private-kifu/preset?match_id=1&round_index=1", headers=headers)
    assert response.status_code == 200
    assert response.json() == {"match_id": "001", "round": {"hand": {"p0": "sample hand"}}}
    assert response.headers["cache-control"] == "no-store"
    assert "private name" not in response.text and "private moves" not in response.text
    assert client.get("/admin/api/private-kifu/download", headers=headers).status_code == 404


def test_invalid_upload_preserves_existing_archive(archive_client, archive_bytes):
    client = archive_client
    admin_login(client)
    headers = {"X-Goita-Member": "1"}
    assert client.put("/admin/api/private-kifu", content=archive_bytes, headers=headers).status_code == 200
    for bad in (b"not json", b'{"matches":[null]}', b'{"matches":[]}'):
        assert client.put("/admin/api/private-kifu", content=bad, headers=headers).status_code == 400
        assert archive_path(game_app.BASE_DIR).read_bytes() == archive_bytes


def test_archive_path_cannot_be_public_or_ephemeral_on_render(monkeypatch, tmp_path):
    monkeypatch.setenv("GOITA_PERSISTENT_DATA_DIR", str(tmp_path / "frontend"))
    with pytest.raises(ValueError):
        archive_path(tmp_path)
    monkeypatch.delenv("GOITA_PERSISTENT_DATA_DIR")
    monkeypatch.setenv("RENDER", "true")
    with pytest.raises(ValueError):
        archive_path(tmp_path)
    monkeypatch.delenv("RENDER")
    assert archive_path(tmp_path) == tmp_path / "private_data" / "kifu_data.json"


def test_random_trace_anonymizes_archive_metadata(archive_client, monkeypatch):
    client = archive_client
    monkeypatch.setitem(game_app.GAMES, "debug", {"is_debug_room": True, "human_seats": {"A": "owner"}})
    candidate = {"payload": {"player_names": {"A": "private name"}},
                 "source": {"match_id": "private-id", "played_at": "private-date"}}
    monkeypatch.setattr(game_app, "_random_trace_candidates", lambda: [candidate])
    async def start(game_id, payload, *, client_id, source):
        assert source == {}
        assert payload["player_names"] == {seat: f"プレイヤー{seat}" for seat in "ABCD"}
        return {"ok": True, "source": source}
    monkeypatch.setattr(game_app, "_start_debug_trace_payload", start)
    monkeypatch.setattr(game_app.MEMBER_STORE, "is_operator_session", lambda token: token == "operator-test")
    client.cookies.set(game_app.MEMBER_COOKIE, "operator-test")
    headers = {"X-Goita-Member": "1"}
    assert client.post("/games/debug/trace_random_start", json={"client_id": "other"}, headers=headers).status_code == 403
    result = client.post("/games/debug/trace_random_start", json={"client_id": "owner"}, headers=headers)
    assert result.status_code == 200
    assert result.headers["cache-control"] == "no-store"
    assert "private" not in result.text
    assert candidate["payload"]["player_names"]["A"] == "private name"


def test_archive_cache_tracks_file_replacement(archive_client, archive_bytes):
    path = archive_path(game_app.BASE_DIR)
    save_archive(path, archive_bytes)
    assert game_app._load_kifu_archive()["matches"][0]["id"] == "001"
    admin_login(archive_client)
    response = archive_client.put("/admin/api/private-kifu", content=archive_bytes.replace(b'001', b'002'),
                                  headers={"X-Goita-Member": "1"})
    assert response.status_code == 200
    assert game_app._load_kifu_archive()["matches"][0]["id"] == "002"

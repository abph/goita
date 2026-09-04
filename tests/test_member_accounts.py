from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import sqlite3

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from backend.member_api import MEMBER_COOKIE, create_member_router
from backend.member_store import (
    MemberError, MemberStore, SESSION_SECONDS, TEMP_PASSWORD_SECONDS,
    TEMP_SESSION_SECONDS, hash_password, resolve_member_path, verify_password,
)


PASSWORD = "a different member password 2026"
HEADERS = {"X-Goita-Member": "1", "Origin": "https://testserver"}


@pytest.fixture
def store(tmp_path):
    return MemberStore(tmp_path / "members.sqlite3")


@pytest.fixture
def client(store):
    app = FastAPI()

    def admin(request):
        if request.cookies.get("test_admin") != "authorized":
            raise HTTPException(401, "Admin required")

    app.include_router(create_member_router(store, admin, persistent=True))
    with TestClient(app, base_url="https://testserver", headers=HEADERS) as client:
        yield client


def ready(store, member_id="tester"):
    issued = store.create(member_id)
    _, old_token, _ = store.login(member_id, issued["temporary_password"])
    member, token, _ = store.change_password(old_token, issued["temporary_password"], PASSWORD)
    return member, token


def test_hash_is_salted_and_not_plaintext():
    first = hash_password(PASSWORD)
    second = hash_password(PASSWORD)
    assert first != second and PASSWORD not in first
    assert verify_password(PASSWORD, first)
    assert not verify_password("wrong", first)
    assert not verify_password(PASSWORD, "invalid")


def test_store_path_precedence(tmp_path):
    fallback = tmp_path / "fallback.sqlite3"
    assert resolve_member_path({}, fallback) == fallback
    assert resolve_member_path({"GOITA_PERSISTENT_DATA_DIR": str(tmp_path)}, fallback) == tmp_path / "goita-members.sqlite3"
    assert resolve_member_path({"GOITA_MEMBER_DB_PATH": str(fallback), "GOITA_PERSISTENT_DATA_DIR": "/var/data"}, tmp_path) == fallback


@pytest.mark.parametrize("member_id", ["a", "UPPER space", "日本語", "../test", "x" * 33, "_tester"])
def test_invalid_id(store, member_id):
    with pytest.raises(MemberError):
        store.create(member_id)


def test_ids_case_insensitive_and_unique(store):
    issued = store.create("Test-User")
    assert issued["member"]["member_id"] == "test-user"
    assert store.login(" TEST-USER ", issued["temporary_password"])[0]["member_id"] == "test-user"
    with pytest.raises(MemberError) as error:
        store.create("test-user")
    assert error.value.status == 409


def test_no_password_or_token_exposed_or_stored_plaintext(store):
    issued = store.create("tester")
    _, token, seconds = store.login("tester", issued["temporary_password"])
    assert seconds == TEMP_SESSION_SECONDS
    public = store.list_members()
    assert "password_hash" not in str(public)
    assert "temporary_password" not in str(public)
    with sqlite3.connect(store.path) as db:
        content = "\n".join(db.iterdump())
    assert issued["temporary_password"] not in content
    assert token not in content
    assert "client_id" not in content and "analytics_id" not in content


def test_first_change_is_mandatory_and_rotates_sessions(store):
    issued = store.create("tester")
    _, first, _ = store.login("tester", issued["temporary_password"])
    _, second, _ = store.login("tester", issued["temporary_password"])
    with pytest.raises(MemberError) as error:
        store.authenticate(first)
    assert error.value.status == 403
    member, new_token, seconds = store.change_password(first, issued["temporary_password"], PASSWORD)
    assert not member["must_change_password"] and seconds == SESSION_SECONDS
    for old in (first, second):
        with pytest.raises(MemberError):
            store.authenticate(old)
    assert store.authenticate(new_token)["member_id"] == "tester"
    with pytest.raises(MemberError):
        store.login("tester", issued["temporary_password"])


@pytest.mark.parametrize("password", ["short", " " * 20, "x" * 129])
def test_password_policy(store, password):
    issued = store.create("tester")
    _, token, _ = store.login("tester", issued["temporary_password"])
    with pytest.raises(MemberError):
        store.change_password(token, issued["temporary_password"], password)


def test_password_change_checks_current_and_difference(store):
    issued = store.create("tester")
    _, token, _ = store.login("tester", issued["temporary_password"])
    with pytest.raises(MemberError):
        store.change_password(token, "wrong", PASSWORD)
    with pytest.raises(MemberError):
        store.change_password(token, issued["temporary_password"], issued["temporary_password"])


def test_logout_and_session_expiration(store):
    _, first = ready(store)
    _, other, _ = store.login("tester", PASSWORD)
    store.logout(first)
    with pytest.raises(MemberError):
        store.authenticate(first)
    assert store.authenticate(other)
    now = store.clock()
    store.clock = lambda: now + SESSION_SECONDS + 1
    with pytest.raises(MemberError):
        store.authenticate(other)


def test_temp_credentials_and_temp_sessions_expire(store):
    issued = store.create("tester")
    now = store.clock()
    _, token, _ = store.login("tester", issued["temporary_password"])
    store.clock = lambda: now + TEMP_SESSION_SECONDS + 1
    with pytest.raises(MemberError):
        store.authenticate(token, allow_temporary=True)
    store.clock = lambda: now + TEMP_PASSWORD_SECONDS + 1
    with pytest.raises(MemberError):
        store.login("tester", issued["temporary_password"])


def test_suspension_and_reset_revoke_all_devices(store):
    _, first = ready(store)
    _, second, _ = store.login("tester", PASSWORD)
    store.update("tester", enabled=False, paid_enabled=True, paid_until=None)
    for token in (first, second):
        with pytest.raises(MemberError):
            store.authenticate(token)
    with pytest.raises(MemberError):
        store.login("tester", PASSWORD)
    store.update("tester", enabled=True, paid_enabled=True, paid_until=None)
    with pytest.raises(MemberError):
        store.authenticate(first)
    _, third, _ = store.login("tester", PASSWORD)
    issued = store.reset_password("tester")
    with pytest.raises(MemberError):
        store.authenticate(third)
    with pytest.raises(MemberError):
        store.login("tester", PASSWORD)
    assert store.login("tester", issued["temporary_password"])[0]["must_change_password"]


def test_paid_expiry_is_japan_end_of_day_and_not_login_suspension(store):
    _, token = ready(store)
    store.clock = lambda: datetime(2026, 9, 4, 14, 59, 59, tzinfo=timezone.utc).timestamp()
    # Make session time independent of the actual execution date.
    _, token, _ = store.login("tester", PASSWORD)
    store.update("tester", enabled=True, paid_enabled=True, paid_until="2026-09-04")
    assert store.authenticate(token, require_paid=True)["paid_active"]
    store.clock = lambda: datetime(2026, 9, 4, 15, 0, 0, tzinfo=timezone.utc).timestamp()
    assert not store.authenticate(token)["paid_active"]
    with pytest.raises(MemberError) as error:
        store.authenticate(token, require_paid=True)
    assert error.value.status == 403
    store.update("tester", enabled=True, paid_enabled=True, paid_until=None)
    assert store.authenticate(token, require_paid=True)
    store.update("tester", enabled=True, paid_enabled=False, paid_until=None)
    with pytest.raises(MemberError):
        store.authenticate(token, require_paid=True)


@pytest.mark.parametrize("expiry", ["2026-02-30", "20260904", "2026-9-4", "9999-01-01"])
def test_invalid_expiry(store, expiry):
    with pytest.raises(MemberError):
        store.create("tester", paid_until=expiry)


def test_login_rate_limit_unknown_and_existing_accounts_survives_restart(store):
    store.create("tester")
    for member_id in ("tester", "unknown"):
        for _ in range(5):
            with pytest.raises(MemberError) as error:
                store.login(member_id, "wrong")
            assert error.value.status == 401
        restarted = MemberStore(store.path)
        with pytest.raises(MemberError) as error:
            restarted.login(member_id, "wrong")
        assert error.value.status == 429
    now = store.clock()
    store.clock = lambda: now + 901
    with pytest.raises(MemberError) as error:
        store.login("tester", "wrong")
    assert error.value.status == 401


def test_global_limit_prevents_many_account_guesses(store):
    for i in range(10):
        store._attempt(f"id{i}")
    with pytest.raises(MemberError) as error:
        store._attempt("another")
    assert error.value.status == 429


def test_rate_limit_is_atomic_across_store_instances(store):
    def attempt(_):
        try:
            MemberStore(store.path)._attempt("same")
            return True
        except MemberError:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(attempt, range(10))) == 5


def test_persistence_and_member_isolation(store):
    _, token = ready(store, "first")
    _, other = ready(store, "second")
    restarted = MemberStore(store.path)
    assert restarted.authenticate(token)["member_id"] == "first"
    restarted.reset_password("second")
    assert restarted.authenticate(token)
    with pytest.raises(MemberError):
        restarted.authenticate(other)


def test_full_http_flow_and_cookie_protection(client, store):
    assert client.get("/api/member/session").json()["authenticated"] is False
    assert client.post("/admin/api/members", json={"member_id": "tester"}).status_code == 401
    client.cookies.set("test_admin", "authorized")
    response = client.post("/admin/api/members", json={"member_id": "tester"})
    assert response.status_code == 200
    temporary = response.json()["temporary_password"]
    response = client.post("/api/member/login", json={"member_id": "tester", "password": temporary})
    assert response.status_code == 200
    assert response.json()["member"]["must_change_password"]
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "secure" in cookie and "samesite=strict" in cookie
    assert response.headers["cache-control"] == "no-store"
    assert client.get("/api/member/me").status_code == 403
    response = client.post("/api/member/password", json={"current_password": temporary, "new_password": PASSWORD})
    assert response.status_code == 200 and not response.json()["member"]["must_change_password"]
    assert client.get("/api/member/me").status_code == 200
    client.cookies.delete("test_admin")
    assert client.get("/admin/api/members").status_code == 401
    assert client.post("/admin/api/members/tester/reset-password", json={}).status_code == 401
    assert client.post("/api/member/logout", json={}).status_code == 200
    assert client.get("/api/member/me").status_code == 401


def test_http_rejects_csrf_including_reads(client, store):
    _, token = ready(store)
    client.cookies.set(MEMBER_COOKIE, token)
    for headers in ({"Origin": "https://evil.example"}, {"X-Goita-Member": ""}, {"Sec-Fetch-Site": "cross-site"}, {"Origin": "https://sub.testserver"}):
        assert client.get("/api/member/me", headers=headers).status_code == 403
        assert client.post("/api/member/logout", json={}, headers=headers).status_code == 403
    assert client.get("/api/member/me").status_code == 200


def test_admin_update_and_reset_http(client, store):
    _, token = ready(store)
    client.cookies.set(MEMBER_COOKIE, token)
    client.cookies.set("test_admin", "authorized")
    response = client.put("/admin/api/members/tester", json={"enabled": True, "paid_enabled": False, "paid_until": None})
    assert response.status_code == 200 and not response.json()["member"]["paid_active"]
    assert not client.get("/api/member/me").json()["member"]["paid_active"]
    response = client.post("/admin/api/members/tester/reset-password", json={})
    assert response.status_code == 200
    assert client.get("/api/member/me").status_code == 401
    assert "temporary_password" not in client.get("/admin/api/members").text


def test_validation_does_not_echo_password_or_allow_self_grant(client):
    secret = "my-private-value" * 30
    response = client.post("/api/member/login", json={"member_id": "tester", "password": secret})
    assert response.status_code == 422 and secret not in response.text
    assert response.headers["cache-control"] == "no-store"
    response = client.post("/api/member/login", json={"member_id": "tester", "password": "valid", "paid_enabled": True})
    assert response.status_code == 422


def test_anonymous_visit_does_not_create_member_database(client, store):
    assert not store.path.exists()
    assert client.get("/api/member/session").json() == {"authenticated": False, "member": None}
    assert client.post("/api/member/logout", json={}).status_code == 200
    assert not store.path.exists()


def test_member_ids_never_grant_site_or_room_admin(monkeypatch, store):
    from backend import app as app_module
    from starlette.requests import Request

    _, token = ready(store)
    scope = {"type": "http", "headers": [(b"cookie", f"{MEMBER_COOKIE}={token}".encode())]}
    with pytest.raises(HTTPException) as error:
        app_module._require_site_admin(Request(scope))
    assert error.value.status_code == 401
    monkeypatch.setattr(app_module, "LOBBY_ADMIN_PASSWORD", app_module.DEFAULT_LOBBY_ADMIN_PASSWORD)
    monkeypatch.setattr(app_module, "_require_site_admin", lambda request: None)
    with pytest.raises(HTTPException) as error:
        app_module._require_member_admin(Request(scope))
    assert error.value.status_code == 503
    monkeypatch.setattr(app_module, "LOBBY_ADMIN_PASSWORD", "unique-local-test-admin")
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setattr(app_module, "MEMBER_PERSISTENT", False)
    with pytest.raises(HTTPException) as error:
        app_module._require_member_admin(Request(scope))
    assert error.value.status_code == 503


def test_missing_member_and_account_expiry_checks_do_not_mutate_others(store):
    _, token = ready(store)
    with pytest.raises(MemberError) as error:
        store.update("missing", enabled=False, paid_enabled=False, paid_until=None)
    assert error.value.status == 404
    with pytest.raises(MemberError) as error:
        store.reset_password("missing")
    assert error.value.status == 404
    assert store.authenticate(token)


def test_session_tokens_are_not_accepted_as_other_credentials(client, store):
    _, token = ready(store)
    client.cookies.set(MEMBER_COOKIE, token + "forged")
    assert client.get("/api/member/me").status_code == 401
    assert client.get("/api/member/session").json()["authenticated"] is False


def test_password_changes_are_throttled_and_reset_clears_lock(store):
    _, token = ready(store)
    for _ in range(5):
        with pytest.raises(MemberError) as error:
            store.change_password(token, "wrong", PASSWORD + "next")
        assert error.value.status == 400
    with pytest.raises(MemberError) as error:
        store.change_password(token, PASSWORD, PASSWORD + "next")
    assert error.value.status == 429
    issued = store.reset_password("tester")
    assert store.login("tester", issued["temporary_password"])


def test_site_admin_login_is_throttled_before_member_management(monkeypatch):
    from backend import app as app_module

    app_module.ADMIN_LOGIN_FAILURES.clear()
    monkeypatch.setattr(app_module, "LOBBY_ADMIN_PASSWORD", "a-site-admin-test-password")
    try:
        for _ in range(app_module.ADMIN_LOGIN_MAX_FAILURES):
            app_module._record_site_admin_login(False)
        with pytest.raises(HTTPException) as error:
            app_module._check_site_admin_login_limit()
        assert error.value.status_code == 429
        app_module._record_site_admin_login(True)
        app_module._check_site_admin_login_limit()
    finally:
        app_module.ADMIN_LOGIN_FAILURES.clear()

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_member_page_is_available_from_lobby_and_room_settings():
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert html.count("data-member-entry") == 2
    assert html.count("data-member-panel") == 2
    assert 'id="lobbyMemberTab"' in html
    assert 'id="memberSettingsTab"' in html
    assert "function openMemberPage()" in html
    assert "function showLobbySettingsTab(tab)" in html
    assert "const isMember = tabName === \"member\"" in html
    assert 'src="/static/member.js?v=20260904a"' in html
    assert 'href="/static/member.css?v=20260904a"' in html


def test_member_ui_never_uses_browser_credential_storage():
    script = (ROOT / "frontend" / "member.js").read_text(encoding="utf-8")
    assert 'credentials: "same-origin"' in script
    assert '"X-Goita-Member": "1"' in script
    assert "localStorage" not in script
    assert "sessionStorage" not in script
    assert "document.cookie" not in script
    assert "paid_enabled:" not in script
    assert "paid_until:" not in script


def test_admin_has_manual_member_issuance_and_no_delete_action():
    html = (ROOT / "frontend" / "admin.html").read_text(encoding="utf-8")
    script = (ROOT / "frontend" / "admin-members.js").read_text(encoding="utf-8")
    assert 'data-tab="members"' in html
    assert 'id="memberCreateForm"' in html
    assert "会員を発行" in html
    assert "仮パスワード再発行" in script
    assert 'src="/static/admin-members.js?v=20260904a"' in html
    assert 'method: "DELETE"' not in script
    assert 'credentials: "same-origin"' in script
    assert '"X-Goita-Member": "1"' in script


def test_privacy_policy_discloses_member_storage_and_analytics_separation():
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "会員機能では、会員ID、暗号学的に保護したパスワード、ログイン情報、有料権限と有効期限を保存します。" in html
    assert "これらは利用状況の分析用IDとは結び付けません。" in html


def test_initial_release_does_not_unlock_stamps_or_kifu_by_client_side_membership():
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "PRIVATE_ROOM_IDS.has(gid) || gid === DEBUG_GID" in html
    assert "goitaMembers?.paid" not in html
    assert "研究用棋譜ライブラリはプライベートルーム専用です" in (
        ROOT / "backend" / "app.py"
    ).read_text(encoding="utf-8")

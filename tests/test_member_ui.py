from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_member_page_is_separate_from_lobby_and_room_settings():
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert html.count("data-member-entry") == 2
    assert html.count("data-member-panel") == 2
    assert 'id="lobbyMemberTab"' not in html
    assert 'id="memberSettingsTab"' not in html
    assert "function openMemberPage()" in html
    assert "function showLobbySettingsTab(tab)" in html
    assert "const isMember = tabName === \"member\"" in html
    assert 'src="/static/member.js?v=20260905g"' in html
    assert 'href="/static/member.css?v=20260905f"' in html


def test_member_ui_never_uses_browser_credential_storage():
    script = (ROOT / "frontend" / "member.js").read_text(encoding="utf-8")
    assert 'credentials: "same-origin"' in script
    assert '"X-Goita-Member": "1"' in script
    assert "localStorage" not in script
    assert "sessionStorage" not in script
    assert "document.cookie" not in script
    assert "paid_enabled:" not in script
    assert "paid_until:" not in script


def test_admin_has_manual_member_issuance_and_confirmed_delete_action():
    html = (ROOT / "frontend" / "admin.html").read_text(encoding="utf-8")
    script = (ROOT / "frontend" / "admin-members.js").read_text(encoding="utf-8")
    assert 'data-tab="members"' in html
    assert 'id="memberCreateForm"' in html
    assert "会員を発行" in html
    assert "仮パスワード再発行" in script
    assert 'src="/static/admin-members.js?v=20260905f"' in html
    assert '"DELETE"' in script
    assert 'data-delete' in script
    assert '元に戻せません。' in script
    assert 'credentials: "same-origin"' in script
    assert '"X-Goita-Member": "1"' in script


def test_privacy_policy_discloses_member_storage_and_analytics_separation():
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "会員機能では、会員ID、暗号学的に保護したパスワード、ログイン情報、有料権限と有効期限を保存します。" in html
    assert "これらは利用状況の分析用IDとは結び付けません。" in html


def test_paid_stamps_and_member_library_are_separate_from_room_admin():
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "PRIVATE_ROOM_IDS.has(gid) || gid === DEBUG_GID" in html
    assert "MAIN_ROOM_IDS.has(gid) && window.goitaMembers?.canUseAllStamps()" in html
    assert '"X-Goita-Member": "1"' in html
    assert "researchKifuAdminPassword" not in html
    assert '/api/member/kifu' in html
    assert 'function resetMemberKifuLibrary()' in html


def test_member_password_inputs_accept_eight_characters():
    script = (ROOT / "frontend" / "member.js").read_text(encoding="utf-8")
    assert script.count('minlength="8"') == 2
    assert 'minlength="15"' not in script


def test_login_explains_supporter_benefits_with_separate_support_page():
    script = (ROOT / "frontend" / "member.js").read_text(encoding="utf-8")
    login = script.split("if (!member) {")[1].split("} else if")[0]
    assert "支援者向けの会員機能です。" in login
    assert "公開部屋でも全スタンプ" in login
    assert 'href="https://vrcgoita.com/support/" target="_blank" rel="noopener noreferrer"' in login
    assert login.index("支援について") < login.index('name="member_id"')


def test_member_page_hides_limit_and_reissue_notices():
    script = (ROOT / "frontend" / "member.js").read_text(encoding="utf-8")
    assert "保存上限：" not in script
    assert "保存した棋譜は本人だけが閲覧できます。" not in script
    assert "会員発行・パスワード再発行は運営へお問い合わせください。" not in script


def test_lobby_stamp_submission_sends_member_auth_header():
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    submit = html.split('fetch(`${API}/lobby/chat`, {')[1].split("body:")[0]
    assert 'credentials: "same-origin"' in submit
    assert '"X-Goita-Member": "1"' in submit

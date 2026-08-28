from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_admin_dashboard_is_separate_and_not_linked_from_lobby_settings() -> None:
    lobby = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    admin = (ROOT / "frontend" / "admin.html").read_text(encoding="utf-8")

    assert 'id="lobbyAdminSettingsPanel"' not in lobby
    assert 'href="/admin/"' not in lobby
    assert "そろうごいた 管理者ページ" in admin
    assert 'data-tab="settings"' in admin
    assert 'data-tab="analytics"' in admin
    assert 'id="privateRoomPasswords"' in admin
    assert 'id="sessionList"' in admin
    assert 'id="responseMetricGrid"' in admin
    assert "AI応答辞書の計測" in admin
    assert 'data.ai_conditional_response || {}' in admin

    backend = (ROOT / "backend" / "app.py").read_text(encoding="utf-8")
    assert '"ai_conditional_response"' in backend
    assert "conditional_response_runtime_snapshot" in backend


def test_privacy_policy_leads_with_kifu_guarantee_and_explains_analytics() -> None:
    lobby = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    first = "本人から明示的な依頼または同意がある場合を除き、運営側が特定の利用者の棋譜や打ち方を収集・分析することはありません。"
    second = "利用者が自ら保存を選択しない限り、棋譜がサーバーに保存されることもありません。"
    assert first in lobby
    assert second in lobby
    assert lobby.index(first) < lobby.index(second)
    assert "分析用の記録には、名前、チャット内容、棋譜、手駒、ペア相手、対戦相手の情報を含めません。" in lobby
    assert "デバッグルームのボイスチャットは" not in lobby

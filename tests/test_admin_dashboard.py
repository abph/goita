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
    assert 'id="genericPatternMetricGrid"' in admin
    assert "AI応答辞書の計測" in admin
    assert "汎用パターンの収集" in admin
    assert "十分な支持がある型は探索順の優先候補として使用します" in admin
    assert "中分類パターン" in admin
    assert "中分類から優先" in admin
    assert "影響なし比較" in admin
    assert "現在AIと不一致" in admin
    assert "優先候補に使用" in admin
    assert "辞書効果を比較" in admin
    assert "最終判断が変化" in admin
    assert "辞書あり平均深さ" in admin
    assert "短縮時間（比較値）" in admin
    assert "仮想絞り込みを比較" in admin
    assert "絞り込み判断一致率" in admin
    assert "絞り込み推定短縮時間" in admin
    assert "実際に絞り込み" in admin
    assert "安全条件で見送り" in admin
    assert "デバッグルームでは、安全条件を満たす場合に限り" in admin
    assert 'data.ai_conditional_response || {}' in admin
    assert 'data.ai_generic_response_patterns || {}' in admin

    backend = (ROOT / "backend" / "app.py").read_text(encoding="utf-8")
    assert '"ai_conditional_response"' in backend
    assert "conditional_response_runtime_snapshot" in backend
    assert '"ai_generic_response_patterns"' in backend
    assert "checkpoint_generic_response_patterns" in backend


def test_privacy_policy_leads_with_kifu_guarantee_and_explains_analytics() -> None:
    lobby = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    first = "本人から明示的な依頼または同意がある場合を除き、運営側が特定の利用者の棋譜や打ち方を収集・分析することはありません。"
    second = "利用者が自ら保存を選択しない限り、棋譜がサーバーに保存されることもありません。"
    assert first in lobby
    assert second in lobby
    assert lobby.index(first) < lobby.index(second)
    assert "分析用の記録には、名前、チャット内容、棋譜、手駒、ペア相手、対戦相手の情報を含めません。" in lobby
    assert "デバッグルームのボイスチャットは" not in lobby

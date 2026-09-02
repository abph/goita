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
    assert 'data-tab="aiMetrics"' in admin
    assert 'id="aiMetricsView"' in admin
    assert 'id="aiMetricsDownload"' in admin
    assert 'id="aiMetricsStatus"' in admin
    assert '"/admin/api/ai-metrics/export"' in admin
    assert "goita-ai-metrics-" in admin
    assert 'id="privateRoomPasswords"' in admin
    assert 'id="publicRoomNames"' in admin
    assert "公開部屋の名前" in admin
    assert 'class="public-room-name"' in admin
    assert "main_room_names: mainRoomNames" in admin
    assert 'id="privateAdRoomSelect"' in admin
    assert 'id="privateAdSummary"' in admin
    assert 'id="sessionList"' in admin
    assert 'id="regionRows"' in admin
    assert "地域（推定）" in admin
    assert 'id="responseMetricGrid"' in admin
    assert 'id="genericPatternMetricGrid"' in admin
    assert "AI応答辞書の計測" in admin
    assert "汎用パターンの収集" in admin
    assert "十分な支持がある型は探索順の優先候補として使用します" in admin
    assert "中分類パターン" in admin
    assert "戦術分類パターン" in admin
    assert "戦術分類・影響なし比較" in admin
    assert "現在AIがパスし、人間棋譜が同じ駒または王・玉で受ける場合" in admin
    assert "人間棋譜・影響なし比較" in admin
    assert "人間棋譜・現在AIと不一致" in admin
    assert "初手固定ルートを比較" in admin
    assert "重点比較の対象" in admin
    assert "対象外として比較を省略" in admin
    assert "人間ルートで勝ちを確認" in admin
    assert "両方が到達した深さ5以上の最深地点だけを比較" in admin
    assert "比較に使った平均共通深さ" in admin
    assert "人間ルート−AIルートの平均評価差" in admin
    assert "詳細診断・比較対象" in admin
    assert "深さ不足・人間側の探索量上限" in admin
    assert "深さ不足・AI側の時間上限" in admin
    assert "内訳：確定勝敗部分の差" in admin
    assert "内訳：途中局面評価の差" in admin
    assert "判断別：人間" in admin
    assert "人間棋譜ルートの型別比較" in admin
    assert "評価差は「人間ルート − AIルート」" in admin
    assert "人間側が3勝以上・勝率60%以上でAI側より多く勝ち" in admin
    assert "人間棋譜・採用候補" in admin
    assert "集中再検証を通過した3型" in admin
    assert "人間候補優先・候補に使用" in admin
    assert "人間候補優先・最終採用" in admin
    assert "人間候補あり・なしを比較" in admin
    assert "人間候補比較・判断一致率" in admin
    assert "人間候補比較・平均評価差" in admin
    assert "候補外・重点対象ではない" in admin
    assert "候補外・人間3勝未満" in admin
    assert "候補外・人間勝率60%未満" in admin
    assert "候補外・確定負け増加" in admin
    assert 'id="humanRootPatternRows"' in admin
    assert "型別比較の型" in admin
    assert "戦術分類の不一致内訳" in admin
    assert "人間棋譜辞書の不一致内訳" in admin
    assert "匿名の局面" in admin
    assert "受け後の見通し" in admin
    assert "再参加困難" in admin
    assert "戦術優先・候補に使用" in admin
    assert "戦術優先・最終採用" in admin
    assert "戦術あり・なしを比較" in admin
    assert "戦術比較・判断一致率" in admin
    assert 'id="tacticalMismatchRows"' in admin
    assert 'id="humanMismatchRows"' in admin
    assert "棋譜観測・支持" in admin
    assert "観測・支持" in admin
    assert "戦術分類はデバッグルームだけで" in admin
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
    assert "見送り・推定信頼度不足" in admin
    assert "見送り・辞書候補が上位2つ外" in admin
    assert "見送り・3番手との差不足" in admin
    assert "深掘り時間を追加" in admin
    assert "追加した深掘り時間" in admin
    assert "デバッグルームでは、安全条件を満たす場合に限り" in admin
    assert 'data.ai_conditional_response || {}' in admin
    assert 'data.ai_generic_response_patterns || {}' in admin

    backend = (ROOT / "backend" / "app.py").read_text(encoding="utf-8")
    assert '"ai_conditional_response"' in backend
    assert "conditional_response_runtime_snapshot" in backend
    assert '"ai_generic_response_patterns"' in backend
    assert "checkpoint_generic_response_patterns" in backend
    assert '@app.get("/admin/api/ai-metrics/export")' in backend
    assert '"passwords_included": False' in backend


def test_privacy_policy_leads_with_kifu_guarantee_and_explains_analytics() -> None:
    lobby = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    first = "本人から明示的な依頼または同意がある場合を除き、運営側が特定の利用者の棋譜や打ち方を収集・分析することはありません。"
    second = "利用者が自ら保存を選択しない限り、棋譜がサーバーに保存されることもありません。"
    assert first in lobby
    assert second in lobby
    assert lobby.index(first) < lobby.index(second)
    assert "分析用の記録には、名前、チャット内容、棋譜、手駒、ペア相手、対戦相手の情報を含めません。" in lobby
    assert "推定した都道府県を記録する場合があります。" in lobby
    assert "デバッグルームのボイスチャットは" not in lobby

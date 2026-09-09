# AIの判断レポート

デバッグルームのA席で、ログ欄の「この手を報告」または「AIの判断についてレポートを作る」から作成する。

1. 番号付きの手番を選ぶ。受けと直後の攻めは同じ番号、パスも1手番として数える。番号は局ごとに1から始まる。
2. 受け・攻めのどちらの判断かを選び、判断直前の手駒と盤面を確認する。
3. 推奨手と理由を入力する。推奨手が不明ならチェックを入れ、疑問点を理由に記入する。継続手順と例外は任意。
4. 確認後にJSONを保存し、開発チャットに添付する。文章と再現データをまとめてコピーすることもできる。

作成開始時の棋譜とログをブラウザー内に保持する。対局が進んでも内容は変化しない。閉じた後にログ下部の作成ボタンを押すと下書きを再開する。別の「この手を報告」を押すと新規作成になる。ページを再読み込みすると未保存の下書きは消える。

## ファイル形式

- `format`: `sorou-goita-ai-review`、`schema_version`: `1`。
- `initial_hands` と `dealer`: 局の初期状態。駒は既存の1〜9表記。
- `kifu_moves`: 受け・攻め・パスを省略していない既存棋譜行。
- `log`: 取得時点の局全体の原ログ。
- `decisions`: 各行動、手番番号、判断直前の局面、原ログに残った理由と候補評価。
- `target.decision_log_index`: 対象判断を示す `log` 配列の0始まりの添字。
- `target.turn_log_indices`: 受けと攻めをまとめた手番の原ログ添字。
- `human_review`: 人間の提案。`status: proposed` は実装方針がまだ合意されていないことを示す。
- `versions` / `settings_at_capture`: 出力用スナップショット取得時のコード識別情報・設定。対局時に記録されていない乱数状態や設定は未記録と明記する。

調査では、初期状態から対象行動の直前までを再現する。推奨手の検討に未来の手順や他家の非公開手駒を使わない。元の判断理由は再計算した理由で置き換えない。AI修正は人間との方針合意後に行う。

## 検証

Pythonの依存関係にpytest・FastAPI・PyYAML、UI検証にはPlaywrightとEdgeが必要。

```text
python -m pytest tests/test_ai_review_report.py tests/test_localized_game_log.py -q
python tests/test_ai_review_report.py
node tests/ai_review_report_ui_behavior.cjs
```

UI検証用棋譜・スクリーンショットは `results/ai_review_report/` に生成する。テスト時は `GOITA_PERSISTENT_DATA_DIR` を専用の一時フォルダーへ設定する。

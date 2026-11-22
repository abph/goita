# 現AIの修正とニューラル教材の登録

合意済みの戦略修正では、回帰テスト・AI判断チェックの正解登録・ニューラル教材の登録を一組で行います。今後の作業方針はこのディレクトリの `AGENTS.md` に記録しています。

## 作業の流れ

1. 提出されたレポートをAI判断チェックに取り込む。
2. ユーザーが確定した最善手・許容手・避けたい手・理由を登録する。
3. 汎用的なAI修正と回帰テストを行う。
4. 教材を同期し、登録状態を確認する。

```bat
python -m goita_ai2.neural_review_materials sync
python -m goita_ai2.neural_review_materials check
python -m goita_ai2.neural_review_materials status
```

`sync` は `private_data/ai_decision_check/suite.json` の確定回答を `experimental_ai2/data/review_corrections.jsonl` に反映します。受け→攻めは別の判断データに分けます。同じ局面に複数の許容手がある場合は、既存の学習形式に合わせて複数の正解として保持します。最善手と許容手の区別も記録しますが、現行学習器では両方を許容例として学習します。

避けたい手は正解例にせず、検証用の情報として残します。「ある受けの後の攻めが悪い」という判断を「その受け自体が悪い」という教材には変えません。正解の変更・確定解除・除外は、この同期機能で管理する教材に反映します。以前からある手動教材は維持し、矛盾を検出した場合は書き込み前に停止します。

自分の手駒と公開情報だけをモデル用に出力します。他人の手駒・伏せ駒の種類・未来の手順は入力に含めません。誰かが5し以上の局と未確認の判断は除外します。`risk` は記録できますが、安全思考の学習から除外する既存処理を維持します。

`status` と `check` は読み取り専用です。`check` は登録漏れや回答変更の未反映があれば終了コード1、同期済みなら0を返します。未確認の判断は一覧に表示しますが、勝手に正解にしません。同期時は元教材を `private_data/ai_decision_check/neural_materials_backups/` に退避し、登録状況を `neural_materials_status.json` に保存します。

## 再学習は別工程

教材同期ではAIの手・モデル・学習の重みを変更しません。モデルに記録した教材ハッシュと現在の安全思考教材を照合し、未反映なら `retraining_pending` と表示します。教材ハッシュ未記録の古いモデルも、反映済みとは断定しません。

再学習を依頼された場合は、現行モデルに上書きせず候補を作ります。

```bat
python -m goita_ai2.train_neural_policy --model private_data/neural_training/candidate_policy.json --report private_data/neural_training/candidate_report.json
python -m goita_ai2.neural_review_materials status --model private_data/neural_training/candidate_policy.json
```

教材に使った問題での一致率は、未知の局面での強さとは分けて評価します。AI判断チェックではハイブリッドAIの最終手、ニューラル単体の比較では候補順位を確認します。モデルの採否と本番への反映は、この比較の後に判断します。

# AI判断チェック

提出されたAI思考修正レポートを問題集にし、**最新の強化中AI2**の選択をまとめて確認するローカルツールです。
AIの戦略・重みは変更しません。保存版の中級者（中3）にも変更を加えません。

## 最初の使い方（Windowsのコマンドプロンプト）

プロジェクトで使っているPython環境で実行してください。

```bat
cd /d C:\Users\kosaka\Documents\Codex\project_clean_2
python -m goita_ai2.decision_check import "C:\Users\kosaka\Downloads\goita_ai_review_20261111055909_round1_D_turn4.json"
start "" "private_data\ai_decision_check\review.html"
```

1. 画面で対象の局面と安全思考／リスク思考を選びます。
2. 各手順を「最善手」「許容できる手」「避けたい手」に分け、理由を入力します。受ける手は、その直後の攻めまで一組で選びます。
3. **「この回答を確定」**を押します。選択や入力だけでは下書きです。未分類の手は、自動で不正解にはなりません。
4. **「確定済み回答を書き出す」**を押し、ダウンロードしたJSONを取り込みます。

```bat
python -m goita_ai2.decision_check answers "C:\Users\kosaka\Downloads\ai_decision_check_answers.json"
python -m goita_ai2.decision_check run
start "" "private_data\ai_decision_check\result.html"
```

こちらにレポートを提出していただけば、合意済みの方針をこちらで正解として整理して、実行する使い方もできます。
実行した場合は、**「AI判断チェックを使ったこと・対象AIと思考スタイル・件数・結果・未完了やエラー」**をお伝えします。

## 複数レポート・繰り返し確認

`import`には複数ファイルを指定できます。フォルダ指定では直下の`goita_ai_review_*.json`だけを取り込みます。
既に登録した局面を再度取り込んでも、確定した正解を維持します。元レポートは個別に保存します。

```bat
python -m goita_ai2.decision_check import "C:\Users\kosaka\Downloads\report1.json" "C:\Users\kosaka\Downloads\report2.json"
python -m goita_ai2.decision_check review
python -m goita_ai2.decision_check run
```

AIを修正した後に再び`run`を実行すると、前回からの「改善・維持・後退・要確認」や手順／理由の変化を表示します。
初回は比較基準を作るだけなので、「改善した」とは判定しません。最善手と許容手の区別も比較します。
人間の正解・理由、乱数seed、探索設定などが変わった場合は「条件・正解が変更」とし、改善・後退の判定を保留します。
途中エラー・中断・実行中のソース変更がある回は結果を残しますが、前回の成功基準を上書きしません。
特定局面だけの実行でも、それ以外の局面の基準は維持します。

```bat
python -m goita_ai2.decision_check run --case D-局面ID
python -m goita_ai2.decision_check run --style risk
python -m goita_ai2.decision_check run --seconds 1 --seed 1222
```

## 結果の意味

- **最善手／許容できる手**：確定した正解に一致。
- **避けたい手**：確定した避けたい手に一致。修正候補です。
- **未登録の手・要確認**：正解にも避けたい手にも登録されていない手順。人間による追加判断が必要です。
- **正解未確認**：その思考スタイルの正解がまだ未確定。AIの判断は表示しますが採点しません。
- **除外**：誰かの初期手駒が5し以上。実行しません。
- **実行エラー**：局面の再現やAIの実行に失敗。他の局面は続けます。

当時のログと最新の実行ログを分けて表示します。受けと攻めそれぞれの実際の理由、候補評価、探索・ニューラル記録、実行時間を保存します。
理由が記録されなかった場合、後付けの説明は生成しません。

## 再現条件

- 強化中AI2（`experimental_ai2`）を局面ごとに新しく作り、対象行動の**直前まで**公開手順を通知します。未来の手順をAIには渡しません。
- 元レポートの探索設定に、サーバーと同じ戦術辞書設定を反映します。バックグラウンドの先読みは無効です。
- 配布済みの人間棋譜辞書は使います。稼働中に蓄積される応答パターンは局面ごとに空の一時領域とし、既存の蓄積データを更新したり、前の問題の評価を次の問題に持ち越したりしません。
- デフォルトでは本番に配置しないローカル生成の相対ランク表を使用しません。必要なら`--rank-table local`で`results/relative_hand_all_4745_500_combined.csv`を使えます。
- コード・モデル・辞書のハッシュ、実際の探索設定、Python環境、seedを結果に記録します。
- 時間制限付き探索のため、同じseedでもCPU負荷などによって結果が変わる場合があります。当時の乱数や先読みキャッシュは未記録なので、当時の完全再現ではありません。
- `--seconds`は各探索の基本時間であり、受けと攻めを含む局面全体の制限ではありません。`--no-search`は短い動作確認用で、探索ありの結果とは別条件になります。
- 旧レポートに初期得点がない場合はAC/BDとも0点として再現し、その条件を表示します。
- `risk`は、現在実装済みの**終盤し攻めのリスク選択**を有効にします。全戦略がリスク思考に変わるわけではありません。

## 保存場所・対応範囲

デフォルトではGit管理外の`private_data/ai_decision_check/`に保存します。

- `suite.json`：局面・確定した正解
- `sources/`：取り込んだ元レポート
- `review.html`：正解の確認画面
- `runs/`：実行結果JSONと成功基準の履歴
- `result.html`：直近の実行結果

保存先を変える場合は、サブコマンドより**前**に`--workdir`を指定します。

```bat
python -m goita_ai2.decision_check --workdir "private_data\another_check" import "C:\path\report.json"
```

画面は外部通信なしで開けます。回答の下書きはブラウザ内に一時保存しますが、正式な登録にはJSONの書き出しと`answers`による取り込みが必要です。
HTMLにも棋譜や理由が含まれるため、公開用ファイルとして扱わないでください。
明示的に提出されたレポートだけを使い、他のプレイヤーの棋譜を自動収集する処理はありません。

対応入力はデバッグルームの`format: sorou-goita-ai-review`のJSONです。
旧「棋譜・AI理解度確認」の回答JSONは、初期手駒と全手順が含まれないため、そのままでは取り込めません。
既存の棋譜確認画面・管理者ページのベンチマークには変更を加えていません。
重い評価を本番サーバーで動かさないため、この機能の利用にデプロイは不要です。

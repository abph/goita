# 中級者（中3）

2026年10月7日時点の「強化中AI2」を固定保存した安定版です。

- ルール層: `goita_ai2/current_ai` の同日時点のスナップショット
- 選択層: `goita_ai2/experimental_ai2/agent.py` の同日時点のスナップショット
- ニューラル推論: `goita_ai2/neural_policy.py` の同日時点のスナップショット
- 学習済みモデル: `data/neural_policy.json`
- 人間棋譜辞書: `data/human-response-patterns.json`

このパッケージから開発中AIのパッケージを参照しないでください。実行時に生成する
適応値、汎用戦術辞書、計測データも、中級者（中3）専用の保存先を使用します。

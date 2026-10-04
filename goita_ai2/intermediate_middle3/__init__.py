"""中級者（中3）AIパッケージの公開入口です。

2026年10月5日時点の強化中AI2を、ルール層・ニューラル推論・学習済み
モデルごと独立保存しています。開発中AIの後続変更から影響を受けません。
"""

from goita_ai2.intermediate_middle3.agent import RuleBasedAgent

__all__ = ["RuleBasedAgent"]

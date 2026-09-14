"""中級者（中2）AIパッケージの公開入口です。
2026年9月14日時点の強化中AIを固定保存し、今後の強化から分離します。
外部コードから利用するRuleBasedAgentを、このファイルで公開します。
"""

from goita_ai2.intermediate_middle2.agent import RuleBasedAgent

__all__ = ["RuleBasedAgent"]

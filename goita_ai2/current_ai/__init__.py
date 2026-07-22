"""強化中AIパッケージの公開入口です。
外部コードから利用するRuleBasedAgentを、このファイルで一つにまとめて公開します。
AI本体の実装はagent.pyにあり、利用側が内部構成を意識せず呼び出せるようにします。
"""

from goita_ai2.current_ai.agent import RuleBasedAgent

__all__ = ["RuleBasedAgent"]

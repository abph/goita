"""従来のインポート方法を維持するための互換入口です。
実際の強化中AIはgoita_ai2.current_aiにあり、このファイルから同じRuleBasedAgentを公開します。
バックエンドや既存ツールは、参照先を変更せず新しい分割構成を利用できます。
"""

from goita_ai2.current_ai import RuleBasedAgent

__all__ = ["RuleBasedAgent"]

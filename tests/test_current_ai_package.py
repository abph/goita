from __future__ import annotations

from goita_ai2.current_ai import RuleBasedAgent as PackagedRuleBasedAgent
from goita_ai2.rule_based import RuleBasedAgent as CompatibilityRuleBasedAgent


def test_compatibility_entry_point_exports_packaged_agent() -> None:
    assert CompatibilityRuleBasedAgent is PackagedRuleBasedAgent
    assert PackagedRuleBasedAgent.__module__ == "goita_ai2.current_ai.agent"


if __name__ == "__main__":
    test_compatibility_entry_point_exports_packaged_agent()
    print("CURRENT_AI_PACKAGE_TEST_OK")

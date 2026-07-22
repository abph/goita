from __future__ import annotations

from pathlib import Path

import backend.app as app_module
from goita_ai2.rule_based import RuleBasedAgent as CurrentRuleBasedAgent
from goita_ai2.rule_based_beginner_upper import RuleBasedAgent as BeginnerUpperRuleBasedAgent
from goita_ai2.rule_based_intermediate_lower import RuleBasedAgent as IntermediateLowerRuleBasedAgent


ROOT = Path(__file__).resolve().parents[1]


def test_three_ai_profiles_are_available() -> None:
    assert set(app_module.AI_PROFILES) == {
        "current",
        "intermediate_lower",
        "beginner_upper",
    }
    assert app_module.AI_PROFILES["current"]["class"] is CurrentRuleBasedAgent
    assert app_module.AI_PROFILES["intermediate_lower"]["class"] is IntermediateLowerRuleBasedAgent
    assert app_module.AI_PROFILES["beginner_upper"]["class"] is BeginnerUpperRuleBasedAgent


def test_intermediate_lower_profile_creates_frozen_agents() -> None:
    agents = app_module._create_agents("intermediate_lower")
    assert set(agents) == {"A", "B", "C", "D"}
    assert all(isinstance(agent, IntermediateLowerRuleBasedAgent) for agent in agents.values())
    assert all(agent.me == seat for seat, agent in agents.items())


def test_settings_fallback_contains_all_profiles() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert '<option value="current">強化中AI</option>' in html
    assert '<option value="intermediate_lower">中級者（下）</option>' in html
    assert '<option value="beginner_upper">初級者（上）</option>' in html


if __name__ == "__main__":
    test_three_ai_profiles_are_available()
    test_intermediate_lower_profile_creates_frozen_agents()
    test_settings_fallback_contains_all_profiles()
    print("AI_PROFILES_TEST_OK")

from __future__ import annotations

from pathlib import Path

import backend.app as app_module
from goita_ai2.rule_based import RuleBasedAgent as CurrentRuleBasedAgent
from goita_ai2.rule_based_beginner_upper import RuleBasedAgent as BeginnerUpperRuleBasedAgent
from goita_ai2.rule_based_intermediate_lower import RuleBasedAgent as IntermediateLowerRuleBasedAgent
from goita_ai2.rule_based_intermediate_middle import RuleBasedAgent as IntermediateMiddleRuleBasedAgent
from goita_ai2.rule_based_intermediate_middle2 import RuleBasedAgent as IntermediateMiddle2RuleBasedAgent


ROOT = Path(__file__).resolve().parents[1]


def test_five_ai_profiles_are_available() -> None:
    assert app_module.DEFAULT_AI_PROFILE == "intermediate_middle2"
    assert set(app_module.AI_PROFILES) == {
        "current",
        "intermediate_middle2",
        "intermediate_middle",
        "intermediate_lower",
        "beginner_upper",
    }
    assert app_module.AI_PROFILES["current"]["class"] is CurrentRuleBasedAgent
    assert app_module.AI_PROFILES["intermediate_middle2"]["class"] is IntermediateMiddle2RuleBasedAgent
    assert app_module.AI_PROFILES["intermediate_middle"]["class"] is IntermediateMiddleRuleBasedAgent
    assert app_module.AI_PROFILES["intermediate_lower"]["class"] is IntermediateLowerRuleBasedAgent
    assert app_module.AI_PROFILES["beginner_upper"]["class"] is BeginnerUpperRuleBasedAgent


def test_profile_defaults_keep_development_only_surfaces_on_current_ai() -> None:
    assert app_module._normalize_ai_profile(None) == "intermediate_middle2"
    assert app_module._normalize_ai_profile("unknown-profile") == "intermediate_middle2"
    assert app_module._create_game_obj(dealer="A")["ai_profile"] == "intermediate_middle2"

    backend = (ROOT / "backend" / "app.py").read_text(encoding="utf-8")
    assert '_create_game_obj(dealer="A", ai_profile="current")' in backend
    assert 'trace_game["ai_profile"] = "current"' in backend
    assert '_create_game_obj(dealer="A", ai_profile="intermediate_middle2")' in backend


def test_intermediate_lower_profile_creates_frozen_agents() -> None:
    agents = app_module._create_agents("intermediate_lower")
    assert set(agents) == {"A", "B", "C", "D"}
    assert all(isinstance(agent, IntermediateLowerRuleBasedAgent) for agent in agents.values())
    assert all(agent.me == seat for seat, agent in agents.items())


def test_intermediate_middle_profile_is_isolated_from_current_ai() -> None:
    agents = app_module._create_agents("intermediate_middle")
    assert set(agents) == {"A", "B", "C", "D"}
    assert all(isinstance(agent, IntermediateMiddleRuleBasedAgent) for agent in agents.values())
    assert all(not isinstance(agent, CurrentRuleBasedAgent) for agent in agents.values())
    assert all(agent.__class__.__module__ == "goita_ai2.intermediate_middle.agent" for agent in agents.values())
    package_files = (ROOT / "goita_ai2" / "intermediate_middle").glob("*.py")
    assert all("goita_ai2.current_ai" not in path.read_text(encoding="utf-8") for path in package_files)


def test_intermediate_middle2_profile_is_a_frozen_current_ai_snapshot() -> None:
    agents = app_module._create_agents("intermediate_middle2")
    assert set(agents) == {"A", "B", "C", "D"}
    assert all(not isinstance(agent, CurrentRuleBasedAgent) for agent in agents.values())
    assert all(agent.__class__.__module__ == "goita_ai2.intermediate_middle2.agent" for agent in agents.values())
    package_files = (ROOT / "goita_ai2" / "intermediate_middle2").glob("*.py")
    assert all("goita_ai2.current_ai" not in path.read_text(encoding="utf-8") for path in package_files)


def test_settings_fallback_contains_all_profiles() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    zh = (ROOT / "frontend" / "i18n.js").read_text(encoding="utf-8")
    en = (ROOT / "frontend" / "i18n-en.js").read_text(encoding="utf-8")
    assert '<option value="current">強化中AI</option>' in html
    assert '<option value="intermediate_middle2" selected>中級者（中2）</option>' in html
    assert '<option value="intermediate_middle">中級者（中）</option>' in html
    assert '<option value="intermediate_lower">中級者（下）</option>' in html
    assert '<option value="beginner_upper">初級者（上）</option>' in html
    assert "opt.textContent = uiText(label)" in html
    assert '"中級者（中）": "中级（中阶）"' in zh
    assert '"中級者（中2）": "中级（中阶2）"' in zh
    assert '"中級者（中）": "Intermediate (Middle)"' in en
    assert '"中級者（中2）": "Intermediate (Middle 2)"' in en


if __name__ == "__main__":
    test_five_ai_profiles_are_available()
    test_profile_defaults_keep_development_only_surfaces_on_current_ai()
    test_intermediate_lower_profile_creates_frozen_agents()
    test_intermediate_middle_profile_is_isolated_from_current_ai()
    test_intermediate_middle2_profile_is_a_frozen_current_ai_snapshot()
    test_settings_fallback_contains_all_profiles()
    print("AI_PROFILES_TEST_OK")

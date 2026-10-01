from __future__ import annotations

from pathlib import Path

import backend.app as app_module
from goita_ai2.rule_based import RuleBasedAgent as CurrentRuleBasedAgent
from goita_ai2.rule_based_beginner_upper import RuleBasedAgent as BeginnerUpperRuleBasedAgent
from goita_ai2.rule_based_intermediate_lower import RuleBasedAgent as IntermediateLowerRuleBasedAgent
from goita_ai2.rule_based_intermediate_middle import RuleBasedAgent as IntermediateMiddleRuleBasedAgent
from goita_ai2.rule_based_intermediate_middle2 import RuleBasedAgent as IntermediateMiddle2RuleBasedAgent
from goita_ai2.experimental_ai2 import RuleBasedAgent as ExperimentalAI2RuleBasedAgent


ROOT = Path(__file__).resolve().parents[1]


def test_six_ai_profiles_are_available() -> None:
    assert app_module.DEFAULT_AI_PROFILE == "intermediate_middle2"
    assert set(app_module.AI_PROFILES) == {
        "current",
        "experimental_ai2",
        "intermediate_middle2",
        "intermediate_middle",
        "intermediate_lower",
        "beginner_upper",
    }
    assert app_module.AI_PROFILES["current"]["class"] is CurrentRuleBasedAgent
    assert app_module.AI_PROFILES["experimental_ai2"]["class"] is ExperimentalAI2RuleBasedAgent
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
    assert '<option value="experimental_ai2">強化中AI2</option>' in html
    assert '<option value="intermediate_middle2" selected>中級者（中2）</option>' in html
    assert '<option value="intermediate_middle">中級者（中）</option>' in html
    assert '<option value="intermediate_lower">中級者（下）</option>' in html
    assert '<option value="beginner_upper">初級者（上）</option>' in html
    assert "opt.textContent = uiText(label)" in html
    assert '"中級者（中）": "中级（中阶）"' in zh
    assert '"中級者（中2）": "中级（中阶2）"' in zh
    assert '"強化中AI2": "强化中AI2"' in zh
    assert '"中級者（中）": "Intermediate (Middle)"' in en
    assert '"中級者（中2）": "Intermediate (Middle 2)"' in en
    assert '"強化中AI2": "AI in Development 2"' in en


def test_neural_primary_log_distinguishes_selected_and_rule_actions() -> None:
    class Agent:
        last_neural_shadow = {
            "available": True,
            "mode": "primary",
            "recommended_action": ["pass", None, None],
            "selected_action": ["pass", None, None],
            "rule_action": ["receive", "1", None],
            "match": False,
            "applied": True,
            "safety_locked": False,
            "margin": 5.523,
            "elapsed_ms": 3.6,
        }

    log = app_module._format_neural_shadow(Agent())

    assert "NEURAL-PRIMARY" in log
    assert "採用=パス" in log
    assert "ニューラル候補=パス" in log
    assert "現AI=しで受ける" in log
    assert "ニューラル候補を採用" in log


def test_neural_primary_log_explains_low_confidence_strong_rule_fallback() -> None:
    class Agent:
        last_neural_shadow = {
            "available": True,
            "mode": "primary",
            "recommended_action": ["attack", None, "3"],
            "selected_action": ["attack", None, "1"],
            "rule_action": ["attack", None, "1"],
            "match": False,
            "applied": False,
            "safety_locked": False,
            "confidence_deferred": True,
            "margin": 1.282,
            "elapsed_ms": 3.0,
        }

    log = app_module._format_neural_shadow(Agent())

    assert "強い戦術を覆す確信が不足したため現AIを採用" in log


def test_neural_primary_log_marks_hidden_block_only_disagreement() -> None:
    class Agent:
        last_neural_shadow = {
            "available": True,
            "mode": "primary",
            "recommended_action": ["attack_after_block", "2", "2"],
            "selected_action": ["attack_after_block", "1", "2"],
            "rule_action": ["attack_after_block", "1", "2"],
            "match": False,
            "applied": False,
            "safety_locked": False,
            "confidence_deferred": True,
            "block_only_disagreement": True,
            "protect_attack_reserve": True,
            "margin": 0.160,
            "elapsed_ms": 3.0,
        }

    log = app_module._format_neural_shadow(Agent())

    assert "伏せ駒のみ不一致" in log
    assert "連続攻めの駒を残すため現AIを採用" in log
    assert "香を伏せて" not in log


def test_neural_primary_log_explains_unstoppable_finish_preservation() -> None:
    class Agent:
        last_neural_shadow = {
            "available": True,
            "mode": "primary",
            "recommended_action": ["receive", "2", None],
            "selected_action": ["pass", None, None],
            "rule_action": ["pass", None, None],
            "match": False,
            "applied": False,
            "safety_locked": False,
            "confidence_deferred": True,
            "protect_unstoppable_finish": True,
            "margin": 5.018,
            "elapsed_ms": 3.0,
        }

    log = app_module._format_neural_shadow(Agent())

    assert "一巡確定の攻め駒を残すため現AIを採用" in log


if __name__ == "__main__":
    test_six_ai_profiles_are_available()
    test_profile_defaults_keep_development_only_surfaces_on_current_ai()
    test_intermediate_lower_profile_creates_frozen_agents()
    test_intermediate_middle_profile_is_isolated_from_current_ai()
    test_intermediate_middle2_profile_is_a_frozen_current_ai_snapshot()
    test_settings_fallback_contains_all_profiles()
    print("AI_PROFILES_TEST_OK")

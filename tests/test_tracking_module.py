from __future__ import annotations

from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.current_ai.tracking import TrackingMixin


def test_rule_based_agent_uses_tracking_mixin() -> None:
    assert issubclass(RuleBasedAgent, TrackingMixin)


def test_tracking_methods_are_owned_by_mixin() -> None:
    for method_name in ("_get_my_initial_hand", "_ensure_trackers", "on_public_action"):
        assert method_name in TrackingMixin.__dict__
        assert method_name not in RuleBasedAgent.__dict__


if __name__ == "__main__":
    test_rule_based_agent_uses_tracking_mixin()
    test_tracking_methods_are_owned_by_mixin()
    print("TRACKING_MODULE_TEST_OK")

from __future__ import annotations

from goita_ai2.current_ai.endgame import EndgameMixin
from goita_ai2.rule_based import RuleBasedAgent


def test_rule_based_agent_uses_endgame_mixin() -> None:
    assert issubclass(RuleBasedAgent, EndgameMixin)


def test_endgame_methods_are_owned_by_mixin() -> None:
    for method_name in (
        "_max_tsume_score",
        "_high_score_tsume_action",
        "_give_way_to_ally_guaranteed_win_action",
    ):
        assert method_name in EndgameMixin.__dict__
        assert method_name not in RuleBasedAgent.__dict__


if __name__ == "__main__":
    test_rule_based_agent_uses_endgame_mixin()
    test_endgame_methods_are_owned_by_mixin()
    print("ENDGAME_MODULE_TEST_OK")

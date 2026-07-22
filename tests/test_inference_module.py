from __future__ import annotations

from goita_ai2.current_ai.inference import PublicInferenceMixin
from goita_ai2.rule_based import RuleBasedAgent


def test_rule_based_agent_uses_inference_mixin() -> None:
    assert issubclass(RuleBasedAgent, PublicInferenceMixin)
    assert "_refresh_public_piece_inference" in PublicInferenceMixin.__dict__
    assert "_public_hand_rank_estimate" in PublicInferenceMixin.__dict__
    assert "_refresh_public_piece_inference" not in RuleBasedAgent.__dict__
    assert "_public_hand_rank_estimate" not in RuleBasedAgent.__dict__


if __name__ == "__main__":
    test_rule_based_agent_uses_inference_mixin()
    print("INFERENCE_MODULE_TEST_OK")

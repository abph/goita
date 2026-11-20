"""Opt-in royal decisions by the existing human policy, above unproved rules."""
from __future__ import annotations

import copy


class RoyalNeuralPriorityMixin:
    @staticmethod
    def _royal_action_usage(action):
        return tuple(piece if piece in ("8", "9") else None for piece in action[1:])

    def _royal_neural_review(self, state, actions, rule_action, snapshot, *,
                             rule_authority, rule_reason, rule_detail,
                             immediate_finish, unstoppable_finish):
        if not self.NEURAL_ROYAL_PRIORITY_ENABLED or not snapshot.get("available") or len(actions) < 2:
            return None
        neural = tuple(snapshot.get("global_recommended_action", ()))
        if neural not in actions:
            return None
        if (state.phase == "receive"
                and any(a[0] == "receive" and a[1] in ("8", "9") for a in actions)):
            context = "receive_use_or_reserve"
        elif self._royal_action_usage(rule_action) != self._royal_action_usage(neural):
            context = "attack_or_block_use_or_reserve"
        else:
            return None
        guard = ("enemy_immediate_finish" if immediate_finish else
                 "public_unstoppable_finish" if unstoppable_finish else
                 "proven_decision" if rule_authority == "proven"
                 or self._rule_search_authority(rule_reason, rule_detail) == "proven" else "")
        return {"context": context, "guard": guard, "proven": bool(guard),
                "source": "neural_policy_global_first_choice"}

    @staticmethod
    def _promote_global_neural_candidate(snapshot):
        snapshot["recommended_action"] = list(snapshot["global_recommended_action"])
        snapshot["match"] = snapshot["recommended_action"] == snapshot["rule_action"]
        snapshot["margin"] = snapshot["global_margin"]
        snapshot["current_score_gap"] = snapshot["global_current_score_gap"]

    def _record_generic_response_evidence(self, state, player, actions, baseline_action,
                                          selected_action, **evidence):
        deferred = getattr(self, "_royal_neural_deferred_records", None)
        if isinstance(deferred, list):
            deferred.append((state, player, list(actions), baseline_action, selected_action, evidence))
            return False
        return super()._record_generic_response_evidence(
            state, player, actions, baseline_action, selected_action, **evidence)

    def _flush_royal_neural_evidence(self, state, player, chosen, records):
        for recorded_state, actor, legal, baseline, proposed, evidence in records:
            if recorded_state is state and actor == player and proposed == chosen:
                super()._record_generic_response_evidence(state, player, legal, baseline, chosen, **evidence)

    def _finalize_attack_candidate_snapshot(self, actions, chosen):
        super()._finalize_attack_candidate_snapshot(actions, chosen)
        if self.last_decision_reason == "neural_royal":
            self.last_attack_candidate_snapshot["decision_reason"] = self.last_decision_reason
            self.last_attack_candidate_snapshot["decision_detail"] = self.last_score_fallback_detail
            self.last_attack_candidate_snapshot["neural_royal_decision"] = {
                "selected_action": list(chosen),
                "rule_action": copy.deepcopy(self.last_neural_shadow.get("rule_action")),
                "source": "neural_policy_global_first_choice", "proven": False,
            }

"""Preserve the team's last lance stop against a separately signalled threat.

This is a safety preference from public evidence, not a forced-loss proof.
It never replaces a proven finish or a response to an immediate finishing hand.
"""
from __future__ import annotations

import copy

from goita_ai2.current_ai.endgame import ForcedWinStatus


class LanceReserveMixin:
    def _last_team_lance_context(self, state, player, actions):
        tracker = self._track.get(id(state))
        attacker = state.attacker
        if (not tracker or state.phase != "receive" or state.current_attack != "2"
                or attacker is None or self._same_team(attacker, player)
                or len(state.hands[player]) not in (4, 6)
                or state.hands[player].count("2") != 1
                or ("pass", None, None) not in actions or ("receive", "2", None) not in actions
                or len(state.hands[attacker]) < 6
                or int(tracker.get("enemy_attack_counts", {}).get(attacker, 0)) != 1
                or any(len(state.hands[seat]) <= 2 for seat in "ABCD" if seat != player)):
            return None
        ally = self._ally_of(player)
        if self._estimate_remaining_range(tracker, ally, "2")[1] > 0:
            return None
        if self._estimate_remaining_range(tracker, attacker, "2")[1] > 0:
            return None
        pool = self._public_unknown_piece_pool(state, player, tracker)
        if pool is None or pool.get("2", 0) != 1:
            return None
        enemy = self._ally_of(attacker)
        model = tracker.get("public_hand_models", {}).get(enemy, {})
        estimate = self._estimated_current_piece(tracker, enemy, "2") or {}
        if (len(state.hands[enemy]) != 4 or model.get("first_attack") != "2"
                or model.get("strategy_broken") or model.get("first_attack_role") == "ally_echo"
                or int(model.get("attack_count", 0)) < 2
                or int(estimate.get("max", 0)) < 1
                or float(estimate.get("expected", 0)) < 0.5):
            return None
        return {"source": "own_hand_and_public_middle_signals", "proven": False,
                "enemy": enemy, "enemy_hand_count": 4,
                "enemy_lance_expected": float(estimate["expected"]),
                "enemy_lance_range": [int(estimate.get("min", 0)), int(estimate["max"])],
                "attacker": attacker, "attacker_attack_number": 1,
                "attacker_hand_count": len(state.hands[attacker]),
                "own_lances_before": 1, "own_lances_after_receive": 0,
                "ally_lance_max": 0, "royal_can_replace_lance_receive": False}

    def _last_team_lance_review(self, state, player, context):
        receive = ("receive", "2", None)
        if self._forced_win_result_after_receive_action(state, player, receive).status == ForcedWinStatus.PROVEN:
            return None
        after = self._timed_search_apply(state, player, receive)
        reviews = []
        for attack in after.legal_actions(player):
            review = self._kyosha_continuation_review(state, player, attack[2])
            if review is None:
                return None  # Incomplete public information cannot install this preference.
            reviews.append({"attack": attack[2], "strength": review["strength"],
                            "uncertainty_penalty": review["uncertainty_penalty"],
                            "conditional_finish_score": review["outline"]["conditional_finish_score"]})
        if not reviews or any(r["strength"] >= 2.5 and r["uncertainty_penalty"] < 120 for r in reviews):
            return None
        return dict(context, followups=reviews, selection_basis="preserve_last_team_lance_stop")

    def _select_action_with_measurement(self, state, player, actions):
        context = self._last_team_lance_context(state, player, actions)
        before = copy.deepcopy(self._track[id(state)]) if context else None
        chosen = super()._select_action_with_measurement(state, player, actions)
        if (context is None or chosen != ("receive", "2", None)
                or self.last_rule_search_authority == "proven"
                or self._rule_search_authority(self.last_decision_reason, self.last_score_fallback_detail) == "proven"):
            return chosen
        with self._measure_performance("rule_based"):
            review = self._last_team_lance_review(state, player, context)
        if review is None:
            return chosen
        tracker = self._track[id(state)]
        diagnostics = {key: value for key, value in tracker.items() if key.startswith("last_")}
        tracker.clear()
        tracker.update(before)
        tracker.update(diagnostics)
        tracker["pending_kyosha_receive_attack_piece"] = None
        tracker["last_lance_reserve_review"] = review
        comparison = tracker.get("last_round_route_comparison")
        if isinstance(comparison, dict):
            comparison["proposed_chosen"] = copy.deepcopy(comparison.get("chosen"))
            comparison["chosen"] = (("pass", None, None),)
            comparison["last_lance_reserve_changed_response"] = True
            comparison["lance_reserve_review"] = copy.deepcopy(review)
        search = tracker.get("last_time_limited_search")
        if isinstance(search, dict):
            search["final_response_action"] = ("pass", None, None)
            search["selection_stage"] = "last_team_lance_reserve"
        self._set_decision_reason("score_fallback")
        self._set_score_fallback_detail(f"pass_preserve_last_team_lance_enemy_{context['enemy']}")
        self.last_rule_search_authority = "strong"
        return ("pass", None, None)

    def _rule_search_authority(self, reason, detail):
        if str(detail or "").startswith("pass_preserve_last_team_lance_enemy_"):
            return "strong"
        return super()._rule_search_authority(reason, detail)

    def _finalize_attack_candidate_snapshot(self, actions, chosen):
        super()._finalize_attack_candidate_snapshot(actions, chosen)
        if not self.last_score_fallback_detail.startswith("pass_preserve_last_team_lance_enemy_"):
            return
        tracker = self._track.get(getattr(self, "_live_plan_review_state_id", None), {})
        review = tracker.get("last_lance_reserve_review")
        if review:
            self.last_attack_candidate_snapshot["lance_reserve"] = copy.deepcopy(review)

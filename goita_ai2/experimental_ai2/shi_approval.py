"""Communicate a first shi approval after showing another attack axis.

Three initial shi must not pretend to be an opening four-shi signal. After a
different opening attack, an unused three-shi reserve can instead acknowledge
an ally's repeated shi attack, spending two and retaining one for defense.
"""
from __future__ import annotations

import copy
from collections import Counter


class FirstShiApprovalMixin:
    def _first_three_shi_approval_evidence(self, state, player, actions):
        tracker = self._track.get(id(state))
        if (tracker is None or state.phase != "receive"
                or state.current_attack != "1" or state.attacker != tracker.get("ally")
                or ("receive", "1", None) not in actions
                or tracker.get("my_shi_approval_sent")
                or (tracker.get("my_shi_approval_pending")
                    and not tracker.get("first_three_shi_approval_plan"))
                or int(tracker["my_init_count"].get("1", 0)) != 3
                or state.hands[player].count("1") < 3
                or not tracker.get("my_attack_history")
                or "1" in tracker["my_attack_history"]
                or tracker.get("shi_attack_mode")
                or tracker.get("inherit_ally_shi_attack")
                or self._has_strong_repeat_attack(Counter(state.hands[player]))):
            return None
        ally = tracker["ally"]
        # A third shi is still a future plan. Once the ally is already ready
        # to finish, the established endgame and handoff rules take priority.
        if (tuple(tracker.get("ally_attack_history", ())) != ("1", "1")
                or len(state.hands[ally]) < 4):
            return None
        model = tracker.get("public_hand_models", {}).get(ally, {})
        if model.get("strategy_broken") or model.get("inferred_attack_strategy_active") is False:
            return None
        minimum, maximum = self._estimate_remaining_range(tracker, ally, "1")
        estimate = self._estimated_current_piece(tracker, ally, "1") or {}
        expected = float(estimate.get("expected", 0.0))
        map_count = int(estimate.get("map_count", 0))
        if maximum < 1 or (minimum < 1 and (expected < 0.75 or map_count < 1)):
            return None
        after_receive = list(state.hands[player])
        after_receive.remove("1")
        return {"source": "own_hand_and_public_inference", "ally": ally,
                "ally_remaining_shi_min": minimum, "ally_remaining_shi_max": maximum,
                "ally_remaining_shi_expected": round(expected, 3),
                "ally_remaining_shi_map": map_count,
                "own_shi_after_return": state.hands[player].count("1") - 2,
                "expected_hand_after_receive": sorted(after_receive),
                "attack_count_before": int(tracker.get("my_attack_count", 0)),
                "received": False}

    def _select_rule_based_action(self, state, player, actions):
        self._ensure_trackers(state)
        tracker = self._track[id(state)]
        pending = tracker.get("first_three_shi_approval_plan")
        before = copy.deepcopy(tracker) if pending is not None else None
        chosen = super()._select_rule_based_action(state, player, actions)
        if pending is not None and state.phase == "attack":
            valid = (pending.get("received")
                     and sorted(state.hands[player]) == pending["expected_hand_after_receive"]
                     and int(before.get("my_attack_count", 0)) == pending["attack_count_before"]
                     and not before.get("my_shi_approval_sent")
                     and ("attack", None, "1") in actions)
            protected = (self._rule_search_authority(
                self.last_decision_reason, self.last_score_fallback_detail) == "proven"
                or self.last_decision_reason in ("win_now", "tsume", "conditional_tsume", "inferred_endgame"))
            if valid and not protected:
                if chosen != ("attack", None, "1"):
                    tracker.clear()
                    tracker.update(before)
                tracker["my_attack_count"] = pending["attack_count_before"] + 1
                self.last_attack_candidate_scores = []
                self._set_decision_reason("shi_signal")
                self._set_score_fallback_detail("attack_first_three_shi_approval")
                return ("attack", None, "1")
            tracker.pop("first_three_shi_approval_plan", None)
            tracker["my_shi_approval_pending"] = False

        # Other pass reasons include already-sent approval, a self-led shi
        # reserve and proven finishes. Only replace the old generic 3-shi pass.
        if (chosen == ("pass", None, None) and self.last_decision_reason == "shi_signal"
                and not self.last_score_fallback_detail):
            evidence = self._first_three_shi_approval_evidence(state, player, actions)
            if evidence is not None:
                tracker["first_three_shi_approval_plan"] = evidence
                tracker["last_first_three_shi_approval"] = copy.deepcopy(evidence)
                tracker["my_shi_approval_pending"] = True
                self._set_score_fallback_detail("receive_first_three_shi_approval")
                return ("receive", "1", None)
        return chosen

    def on_public_action(self, state, player, action):
        tracker = self._track.get(id(state))
        plan = tracker.get("first_three_shi_approval_plan") if tracker else None
        sent_before = bool(tracker.get("my_shi_approval_sent")) if tracker else False
        super().on_public_action(state, player, action)
        if plan is None:
            return
        if (player == self.me and tuple(action) == ("receive", "1", None)
                and sorted(state.hands[player]) == plan["expected_hand_after_receive"]):
            plan["received"] = True
            return
        tracker.pop("first_three_shi_approval_plan", None)
        tracker["my_shi_approval_pending"] = False
        if player == self.me and tuple(action) == ("attack", None, "1") and plan.get("received"):
            # The shared tracker has already marked the actually played
            # approval as sent. Keep its source distinct from a self-led plan.
            tracker["shi_attack_mode_source"] = "ally_signal"
        else:
            tracker["my_shi_approval_sent"] = sent_before

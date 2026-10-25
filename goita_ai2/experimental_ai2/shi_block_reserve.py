"""Keep a sufficient shi attack reserve without discarding useful receivers."""
from __future__ import annotations


class ShiBlockReserveMixin:
    SHI_ADEQUATE_ATTACK_RESERVE_BONUS = 60.0

    def _shi_block_reserve_context(self, state, player, action_type, attack):
        if (state.phase != "attack" or action_type != "attack_after_block" or attack != "1"
                or self._weak_shi_attack_plan_value(state, player) <= 0
                or self._is_dealer_opening_attack(state, player)):
            return None
        # Compare against the actual legal alternative of folding and attacking
        # with shi. This is not a promise that future shi receives are free.
        alternative = ("attack_after_block", "1", "1")
        if alternative not in state.legal_actions(player):
            return None
        after = self._hand_after_attack_action(state.hands[player], alternative)
        if after is None or len(after) < 4:
            # Keep the existing mixed-pair/endgame logic for the last two cards.
            return None
        future_attacks = (len(after) + 1) // 2
        remaining_shi = after.count("1")
        if remaining_shi < future_attacks:
            return None
        return {"source": "own_hand_and_remaining_attack_slots",
                "future_attack_slots": future_attacks,
                "shi_after_block_and_attack": remaining_shi,
                "remaining_hand": sorted(after),
                "other_receive_types": sorted(set(after) - {"1"}),
                "additional_shi_receives_reserved": max(0, remaining_shi - future_attacks),
                "proven": False}

    def _shi_attack_hidden_block_adjustment(self, state, player, action_type, block, attack):
        original = super()._shi_attack_hidden_block_adjustment(
            state, player, action_type, block, attack)
        context = self._shi_block_reserve_context(state, player, action_type, attack)
        if context is None:
            return original
        if block == "1":
            return max(original, self.SHI_ADEQUATE_ATTACK_RESERVE_BONUS)
        if block in ("3", "4", "5", "6", "7") and original > 0:
            # Once the alternative already retains one shi per future attack,
            # do not award a flat bonus for sacrificing another receive type
            # just to stockpile more shi. All ordinary tactical scores remain.
            return 0.0
        return original

    def _classify_score_fallback(self, state, player, action, *, has_non_king_attack_option):
        detail = super()._classify_score_fallback(
            state, player, action, has_non_king_attack_option=has_non_king_attack_option)
        if action != ("attack_after_block", "1", "1"):
            return detail
        context = self._shi_block_reserve_context(state, player, action[0], action[2])
        scored = self.last_attack_candidate_scores
        if context is None or not scored:
            return detail

        def original_score(row):
            candidate = tuple(row["action"])
            old = super(ShiBlockReserveMixin, self)._shi_attack_hidden_block_adjustment(
                state, player, *candidate)
            new = self._shi_attack_hidden_block_adjustment(state, player, *candidate)
            return float(row["score"]) - new + old

        original_best = max(scored, key=original_score)
        if tuple(original_best["action"]) == action:
            return detail
        # Explain this correction only when removing it changes the winner of
        # the actual candidate comparison, not merely when the shape matches.
        self.last_shi_block_reserve_evaluation = dict(
            context, selected_action=action, previous_action=tuple(original_best["action"]))
        return (f"block_shi_keep_receivers_reserve_{context['shi_after_block_and_attack']}"
                f"_needed_{context['future_attack_slots']}")

    def _finalize_attack_candidate_snapshot(self, actions, chosen):
        super()._finalize_attack_candidate_snapshot(actions, chosen)
        if self.last_score_fallback_detail.startswith("block_shi_keep_receivers_"):
            evidence = getattr(self, "last_shi_block_reserve_evaluation", None)
            if evidence is not None and tuple(evidence["selected_action"]) == chosen:
                self.last_attack_candidate_snapshot["shi_block_reserve"] = dict(evidence)

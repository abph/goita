"""Preserve receivers when a planned repeat attack has a redundant copy."""
from __future__ import annotations


class PlannedBlockReserveMixin:
    PLANNED_BLOCK_RESERVE_ENABLED = True

    def _unconfirmed_first_attack_piece_for_next_action(self, state, player=None):
        piece = super()._unconfirmed_first_attack_piece_for_next_action(state, player)
        tracker = self._track.get(id(state), {})
        owner = player or self.me
        if (self.PLANNED_BLOCK_RESERVE_ENABLED and piece in ("2", "3", "4", "5", "6", "7")
                and tracker.get("unknown_piece_pool", {}).get(piece) == 0
                and state.hands.get(owner, ()).count(piece) >= 3):
            # Folding and attacking with one copy each still leaves the third
            # attack available. The single-copy support guard remains intact.
            return None
        return piece

    def _planned_block_dominates(self, state, player, better, original, attack_number):
        kind, block, attack = better.action
        if (not self.PLANNED_BLOCK_RESERVE_ENABLED or kind != "attack_after_block"
                or attack not in ("2", "3", "4", "5", "6", "7") or block != attack
                or original.action[0] != kind or original.action[2] != attack
                or original.action[1] == attack):
            return False
        tracker = self._track.get(id(state), {})
        # Physical public accounting, not an inference maximum: no outside
        # normal copy can be returned for us to receive. Royals can still stop
        # the attack, equally for both candidates.
        if tracker.get("unknown_piece_pool", {}).get(attack) != 0:
            return False
        required_attacks = min(max(0, self.BRANCHED_ATTACK_MAX_ATTACKS - attack_number),
                               (len(better.remaining_hand) + 1) // 2)
        return (better.remaining_hand.count(attack) >= required_attacks
                and better.finish_score >= original.finish_score
                and self._branched_receive_width(better.remaining_hand)
                > self._branched_receive_width(original.remaining_hand))

    def _branched_continuation_candidates(self, state, player, hand, *, previous_attack,
                                         attack_number, requires_block, preferred_attack=None,
                                         future_preferred_attacks=(), permanently_protected_pieces=(),
                                         limit=None):
        tracker = self._track.get(id(state), {})
        has_surplus_candidate = any(
            hand.count(piece) >= 2 and tracker.get("unknown_piece_pool", {}).get(piece) == 0
            for piece in ("2", "3", "4", "5", "6", "7"))
        if not self.PLANNED_BLOCK_RESERVE_ENABLED or not requires_block or not has_surplus_candidate:
            return super()._branched_continuation_candidates(
                state, player, hand, previous_attack=previous_attack, attack_number=attack_number,
                requires_block=requires_block, preferred_attack=preferred_attack,
                future_preferred_attacks=future_preferred_attacks,
                permanently_protected_pieces=permanently_protected_pieces, limit=limit)
        # Review before the beam of one discards a useful block alternative.
        # The parent still enforces legal consumption and reserved future cards.
        candidates = super()._branched_continuation_candidates(
            state, player, hand, previous_attack=previous_attack, attack_number=attack_number,
            requires_block=True, preferred_attack=preferred_attack,
            future_preferred_attacks=future_preferred_attacks,
            permanently_protected_pieces=permanently_protected_pieces,
            limit=max(1, len(set(hand)) ** 2))
        ordered = []
        visited = set()
        by_action = {candidate.action: candidate for candidate in candidates}
        for candidate in candidates:
            # Only the same-attack/same-block alternative can dominate. Index it
            # once instead of comparing every pair of candidates at every node.
            attack = candidate.action[2]
            alternative = by_action.get(("attack_after_block", attack, attack))
            if (alternative is not None and alternative.action not in visited
                    and self._planned_block_dominates(state, player, alternative, candidate, attack_number)):
                ordered.append(alternative)
                visited.add(alternative.action)
            if candidate.action not in visited:
                ordered.append(candidate)
                visited.add(candidate.action)
        beam = self.BRANCHED_ATTACK_CONTINUATION_BEAM if limit is None else limit
        return tuple(ordered[:max(0, int(beam))])

    def _live_planned_block_review(self, state, player, action):
        if (not self.PLANNED_BLOCK_RESERVE_ENABLED or action[0] != "attack_after_block"
                or action[2] not in ("2", "3", "4", "5", "6", "7")):
            return None
        tracker = self._track.get(id(state), {})
        if tracker.get("unknown_piece_pool", {}).get(action[2]) != 0:
            return None
        if self._unconfirmed_first_attack_piece_for_next_action(state, player) == action[2]:
            return None
        attack_number = max(int(tracker.get("my_attack_count", 0)),
                            len(tracker.get("my_attack_history", ()))) + 1
        # The same bounded list is used to create future nodes and to check a
        # live/cached node. No opposing actual hand is consulted.
        candidates = self._branched_continuation_candidates(
            state, player, state.hands[player], previous_attack=tracker.get("my_last_attack"),
            attack_number=attack_number, requires_block=True, preferred_attack=action[2],
            limit=max(1, len(set(state.hands[player])) ** 2))
        chosen = next((item for item in candidates if item.action == action), None)
        if chosen is None:
            return None
        if action[1] == action[2]:
            better = chosen
            originals = [item for item in candidates if self._planned_block_dominates(
                state, player, better, item, attack_number)]
            if not originals:
                return None
            original = max(originals, key=lambda item: item.projected_score)
        else:
            original = chosen
            better = next((item for item in candidates if self._planned_block_dominates(
                state, player, item, original, attack_number)), None)
            if better is None:
                return None
        return {"source": "own_hand_public_pool_and_projected_lap",
                "selected_action": better.action, "previous_action": original.action,
                "attack_number": attack_number,
                "required_future_attacks": min(max(0, self.BRANCHED_ATTACK_MAX_ATTACKS - attack_number),
                                                (len(better.remaining_hand) + 1) // 2),
                "remaining_attack_copies": better.remaining_hand.count(action[2]),
                "remaining_hand": list(better.remaining_hand),
                "receive_width": self._branched_receive_width(better.remaining_hand),
                "previous_receive_width": self._branched_receive_width(original.remaining_hand),
                "projected_finish_score": better.finish_score,
                "previous_projected_finish_score": original.finish_score,
                "continuation_assumption": "attacks_return_after_a_full_lap", "proven": False}

    def _branched_planned_action(self, state, player, actions):
        action = super()._branched_planned_action(state, player, actions)
        active = self._active_branched_attack_plan(state)
        if action is None or (active is not None and active.plan.evaluation.guaranteed_win):
            return action
        review = self._live_planned_block_review(state, player, action)
        if review is not None and tuple(review["selected_action"]) != action:
            # Never execute a different block with an unchanged old plan graph:
            # its following nodes reserve a different set of cards.
            self._invalidate_branched_attack_plan(state, "planned_block_loses_receive_width")
            return None
        return action

    def _production_branched_attack_action(self, state, player, actions):
        choice = super()._production_branched_attack_action(state, player, actions)
        if choice is None or choice.active.plan.evaluation.guaranteed_win:
            return choice
        review = self._live_planned_block_review(state, player, choice.action)
        if review is not None and tuple(review["selected_action"]) != choice.action:
            self._invalidate_branched_attack_plan(state, "planned_root_block_loses_receive_width")
            return None
        return choice

    def _classify_score_fallback(self, state, player, action, *, has_non_king_attack_option):
        detail = super()._classify_score_fallback(
            state, player, action, has_non_king_attack_option=has_non_king_attack_option)
        review = self._live_planned_block_review(state, player, action)
        if review is not None and tuple(review["selected_action"]) == action:
            self.last_planned_block_reserve = review
            return (f"plan_block_surplus_{action[2]}_reserve_{review['remaining_attack_copies']}"
                    f"_needed_{review['required_future_attacks']}_width_{int(review['receive_width'])}")
        return detail

    def _select_rule_based_action(self, state, player, actions):
        self.last_planned_block_reserve = None
        chosen = super()._select_rule_based_action(state, player, actions)
        review = self.last_planned_block_reserve
        if (review is not None and tuple(review["selected_action"]) == chosen
                and self.last_decision_reason in ("score_fallback", "branched_plan")):
            self._set_score_fallback_detail(
                f"plan_block_surplus_{chosen[2]}_reserve_{review['remaining_attack_copies']}"
                f"_needed_{review['required_future_attacks']}_width_{int(review['receive_width'])}")
        return chosen

    def _finalize_attack_candidate_snapshot(self, actions, chosen):
        super()._finalize_attack_candidate_snapshot(actions, chosen)
        review = getattr(self, "last_planned_block_reserve", None)
        if (self.last_score_fallback_detail.startswith("plan_block_surplus_")
                and review is not None and tuple(review["selected_action"]) == chosen):
            self.last_attack_candidate_snapshot["planned_block_reserve"] = dict(review)

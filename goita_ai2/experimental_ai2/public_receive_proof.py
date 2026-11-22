"""Bridge an eight-card receive into the existing six-card public win proof."""
from __future__ import annotations

import copy

from goita_ai2.current_ai.endgame import ForcedWinStatus


class PublicReceiveProofMixin:
    def _forced_win_result_after_receive_action(self, state, player, action):
        result = super()._forced_win_result_after_receive_action(state, player, action)
        limit = int(self.EXACT_FORCED_WIN_MAX_HAND)
        if (not self.LARGE_HAND_RECEIVE_PUBLIC_PROOF_ENABLED
                or result.status != ForcedWinStatus.UNKNOWN
                or state.phase != "receive" or len(state.hands[player]) != 8
                or len(state.hands[player]) != limit + 2
                or action[0] != "receive" or action not in state.legal_actions(player)):
            return result
        tracker = self._track.get(id(state))
        if tracker is None:
            return result
        tracker.get("large_hand_receive_proofs", {}).pop(action[1], None)
        prepared = self._prepared_opening_receive_attacks(state, player, tracker, action[1])
        pool = self._public_unknown_piece_pool(state, player, tracker)
        # Moving our receive piece from hand to public leaves this pool
        # unchanged. Reject ordinary positions before copying a large tracker.
        if pool is None or not any(
                state.hands[player].count(piece) > int(piece == action[1])
                and (piece in prepared or not self._forced_win_external_receivers(pool, piece))
                for piece in set(state.hands[player])):
            return result

        # Receiving removes one card; the immediate attack removes another.
        # Use the existing six-card proof. Include the prepared attack for a
        # natural receive, provided it also survives the current public proof.
        after = self._apply_action_on_copy(state, player, action)
        after_tracker = copy.deepcopy(tracker)
        seen = after_tracker["public_seen_counts"]
        seen[action[1]] = int(seen.get(action[1], 0)) + 1
        self._track[id(after)] = after_tracker
        proven = []
        try:
            pool = self._public_unknown_piece_pool(after, player, after_tracker)
            if pool is None:
                return result
            for followup in after.legal_actions(player):
                if followup[0] != "attack":
                    continue
                receivable = bool(self._forced_win_external_receivers(pool, followup[2]))
                if receivable and followup[2] not in prepared:
                    continue
                proof = self._forced_win_result_after_attack_action(after, player, followup)
                if proof.status == ForcedWinStatus.PROVEN and proof.minimum_score is not None:
                    proven.append((followup, proof, receivable))
        finally:
            self._track.pop(id(after), None)
        if not proven:
            return result
        best = max((proof for _, proof, _ in proven), key=self._forced_win_preference_key)
        tracker.setdefault("large_hand_receive_proofs", {})[action[1]] = {
            "source": "public_information", "proven": True,
            "receive": action[1], "hand_size_before": len(state.hands[player]),
            "minimum_score": best.minimum_score,
            "followups": [{"action": list(move), "minimum_score": proof.minimum_score,
                           "expected_score": proof.expected_score, "maximum_score": proof.maximum_score}
                          for move, proof, _ in proven],
            "prepared_attack_proven": any(receivable for _, _, receivable in proven),
        }
        return best

    def _large_receive_followup_key(self, state, player):
        tracker = self._track[id(state)]
        return (tuple(sorted(state.hands[player])),
                tuple(sorted(tracker["public_seen_counts"].items())),
                state.king_block_used, tuple(len(state.hands[s]) for s in "ABCD"))

    def _guaranteed_finish_receive_action(self, state, player, actions):
        chosen = super()._guaranteed_finish_receive_action(state, player, actions)
        tracker = self._track.get(id(state), {})
        proof = tracker.get("large_hand_receive_proofs", {}).get(chosen[1]) if chosen else None
        if not proof or len(state.hands[player]) != 8:
            return chosen
        # Keep the proven first attack coupled to the receive. At an identical
        # guaranteed/expected/maximum score, spend an already-unreceivable royal
        # first rather than arbitrarily switching to the repeated middle piece.
        score_key = lambda row: (row["minimum_score"], row["expected_score"], row["maximum_score"])
        best_key = max(map(score_key, proof["followups"]))
        best = [row for row in proof["followups"] if score_key(row) == best_key]
        first = max(best, key=lambda row: row["action"][2] in ("8", "9"))
        after = self._apply_action_on_copy(state, player, chosen)
        after_tracker = copy.deepcopy(tracker)
        after_tracker["public_seen_counts"][chosen[1]] += 1
        self._track[id(after)] = after_tracker
        try:
            key = self._large_receive_followup_key(after, player)
        finally:
            self._track.pop(id(after), None)
        proof["reserved_followup"] = list(first["action"])
        tracker["pending_large_hand_receive_proof"] = {"key": key, "action": tuple(first["action"])}
        return chosen

    def _select_rule_based_action(self, state, player, actions):
        tracker = self._track.get(id(state), {})
        pending = tracker.pop("pending_large_hand_receive_proof", None)
        if (pending and self.LARGE_HAND_RECEIVE_PUBLIC_PROOF_ENABLED
                and state.phase == "attack" and state.attacker == player
                and state.current_attack is None and pending["action"] in actions
                and pending["key"] == self._large_receive_followup_key(state, player)):
            proof = self._forced_win_result_after_attack_action(state, player, pending["action"])
            if proof.status == ForcedWinStatus.PROVEN and proof.minimum_score is not None:
                self.last_attack_intent_comparison = None
                self.last_attack_candidate_scores = []
                self.last_rule_search_authority = "proven"
                self._commit_timed_search_action(state, player, pending["action"])
                self._set_decision_reason("tsume")
                self._set_score_fallback_detail(f"high_score_{int(proof.minimum_score)}")
                return pending["action"]
        return super()._select_rule_based_action(state, player, actions)

    def _finalize_attack_candidate_snapshot(self, actions, chosen):
        super()._finalize_attack_candidate_snapshot(actions, chosen)
        if chosen[0] != "receive" or self.last_score_fallback_detail != "receive_tsume_after":
            return
        tracker = self._track.get(getattr(self, "_live_plan_review_state_id", None), {})
        proof = tracker.get("large_hand_receive_proofs", {}).get(chosen[1])
        if proof:
            self.last_attack_candidate_snapshot["public_receive_proof"] = copy.deepcopy(proof)
            pending = tracker.get("pending_large_hand_receive_proof")
            if pending:
                self.last_attack_candidate_snapshot["public_receive_proof"]["reserved_followup"] = list(pending["action"])
            self.last_attack_candidate_snapshot["decision_reason"] = self.last_decision_reason
            self.last_attack_candidate_snapshot["decision_detail"] = self.last_score_fallback_detail

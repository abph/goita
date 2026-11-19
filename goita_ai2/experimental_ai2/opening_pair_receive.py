"""Evaluate an opening middle-piece receipt when both sides appear to hold two.

The attacker's pair is a conventional first-attack inference, not a fact about
their hidden hand. Remove the blanket weak-hand preference, not other tactics.
"""
from __future__ import annotations

import copy


class OpeningPairReceiveMixin:
    def _opening_shared_pair_context(self, state, player):
        tracker = self._track.get(id(state))
        piece, attacker = state.current_attack, state.attacker
        if (not tracker or state.phase != "receive" or piece not in ("2", "3", "4", "5")
                or attacker is None or self._same_team(attacker, player)
                or player != state.next_player(attacker)
                or len(state.hands[player]) != 8
                or int(tracker.get("my_attack_count", 0)) != 0
                or int(tracker.get("enemy_attack_counts", {}).get(attacker, 0)) != 1
                or state.hands[player].count(piece) != 2
                or int(tracker.get("my_init_count", {}).get(piece, 0)) != 2
                or int(tracker.get("public_seen_counts", {}).get(piece, 0)) != 1
                or tracker.get("ally_open_shi_attack_pending")
                or tracker.get("shi_attack_mode") or tracker.get("inherit_ally_shi_attack")
                or any(len(state.hands[seat]) <= 2 for seat in "ABCD"
                       if not self._same_team(seat, player))):
            return None
        model = tracker.get("public_hand_models", {}).get(attacker, {})
        if model.get("first_attack") != piece or model.get("strategy_broken"):
            return None
        axes = self._initial_hand_axes_for_state(state, player)
        if self._strategy_rank_from_axes(axes) not in ("C", "D", "E", "F", "X"):
            return None  # Preserve the existing strong-hand receive policies.
        if not self._has_meaningful_attack_after_same_piece_receive(state, player, piece):
            return None
        return {"piece": piece, "attacker": attacker, "own_initial_count": 2,
                "assumed_attacker_initial_count": 2, "own_remaining_after_receive": 1,
                "basis": "first_attack_repeat_signal", "proven": False}

    def _enemy_first_receive_strength_bonus(self, state, player, action_type, block):
        context = self._opening_shared_pair_context(state, player)
        if context and (action_type == "pass"
                        or (action_type == "receive" and block == context["piece"])):
            # Compare ordinary receive evidence and continuations without the
            # initial-rank-only bias (e.g. +1450 pass versus +414.5 receive).
            return 0.0
        return super()._enemy_first_receive_strength_bonus(state, player, action_type, block)

    def _enemy_first_same_piece_rank_policy_action(self, state, player, actions):
        action = super()._enemy_first_same_piece_rank_policy_action(state, player, actions)
        if action == ("pass", None, None) and self._opening_shared_pair_context(state, player):
            # A weak first-lance rank policy must also reach the comparison.
            # Later receivers and weak-hand shi signals retain their policies.
            return None
        return action

    def _lookup_conditional_response_plan(self, state, player, actions, baseline_action):
        if (baseline_action[0] == "receive"
                and self._opening_shared_pair_context(state, player)):
            # Cached responses produced by the former weak-hand preference
            # must not bypass the newly evaluated opening response.
            self.last_conditional_response_hit = False
            self.last_conditional_response_key = None
            return None
        return super()._lookup_conditional_response_plan(state, player, actions, baseline_action)

    def _classify_score_fallback(self, state, player, action, *, has_non_king_attack_option):
        context = self._opening_shared_pair_context(state, player)
        if context and (action[0] == "pass"
                        or (action[0] == "receive" and action[1] == context["piece"])):
            return f"{action[0]}_enemy_first_shared_pair_{context['piece']}"
        return super()._classify_score_fallback(
            state, player, action, has_non_king_attack_option=has_non_king_attack_option)

    def _finalize_attack_candidate_snapshot(self, actions, chosen):
        super()._finalize_attack_candidate_snapshot(actions, chosen)
        if not self.last_score_fallback_detail.startswith(
                ("receive_enemy_first_shared_pair_", "pass_enemy_first_shared_pair_")):
            return
        # Store only the inference and the scores actually used for this turn.
        piece = self.last_score_fallback_detail.rsplit("_", 1)[-1]
        self.last_attack_candidate_snapshot["opening_shared_pair"] = {
            "piece": piece, "own_initial_count": 2,
            "assumed_attacker_initial_count": 2, "proven": False,
            "weak_initial_hand_bias_removed": True,
            "scores": copy.deepcopy(self.last_attack_candidate_scores),
        }

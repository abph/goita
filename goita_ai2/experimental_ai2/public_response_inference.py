"""Soft evidence from declined receives; never exclude a physically possible deal."""
from __future__ import annotations

import math
from collections import Counter


class PublicResponseInferenceMixin:
    # Behavioural model parameters, not measured frequencies or certainty.
    ROYAL_INSTEAD_OF_SAME_PRESENT_LIKELIHOOD = 0.12
    DECLINE_REACH_PRESENT_LIKELIHOOD = 0.08

    def _record_piece_pass_evidence(self, state, tr, player):
        super()._record_piece_pass_evidence(state, tr, player)
        context = tr.get("active_attack_context") or {}
        attacker, piece = context.get("attacker"), context.get("piece")
        if (player == self.me or attacker is None or piece not in ("3", "4", "5", "6", "7")
                or self._same_team(player, attacker)
                or not 2 <= len(state.hands[attacker]) <= 4
                or len(state.hands[player]) <= 2):
            return
        events = tr.get("piece_pass_evidence", {}).get(player, {}).get(piece, [])
        if events:
            events[-1]["pass_if_held"] = min(
                events[-1]["pass_if_held"], self.DECLINE_REACH_PRESENT_LIKELIHOOD)
            events[-1]["response_context"] = "enemy_near_reach"

    def _update_public_hand_model(self, state, tr, player, action_type, block, attack):
        context = tr.get("active_attack_context") or {}
        piece, attacker = context.get("piece"), context.get("attacker")
        # At two remaining pairs a royal can deliberately keep a scoring pair.
        # Do not infer absence from that finishing choice, or from partner play.
        if (player != self.me and action_type == "receive" and block in ("8", "9")
                and piece in ("3", "4", "5", "6", "7") and attacker is not None
                and not self._same_team(player, attacker) and len(state.hands[player]) >= 5):
            event = {
                "piece": piece, "attacker": attacker,
                "observed_used": self._observed_piece_count_for_player(tr, player, piece),
                "hand_slots": len(state.hands[player]),
                "hidden_slots": int(tr.get("hidden_block_counts", {}).get(player, 0)),
            }
            model = tr.get("public_hand_models", {}).get(player, {})
            model.setdefault("royal_instead_of_same_evidence", []).append(event)
        return super()._update_public_hand_model(state, tr, player, action_type, block, attack)

    def _probabilistic_candidate_action_evidence(self, state, player, tr, prediction):
        result = super()._probabilistic_candidate_action_evidence(state, player, tr, prediction)
        if result is None:
            return None
        log_weight, confidence, evidence = result
        extra, labels = self._public_response_evidence(tr, prediction)
        return log_weight + extra, confidence, tuple(evidence) + labels

    @staticmethod
    def _response_likelihood(event, original_count, present_likelihood):
        count = original_count - event["observed_used"]
        concealed = event["hidden_slots"]
        total = event["hand_slots"] + concealed
        no_current = (1.0 if count <= 0 else
                      math.comb(concealed, count) / math.comb(total, count)
                      if count <= concealed and count <= total else 0.0)
        return no_current + (1.0 - no_current) * present_likelihood

    def _public_response_evidence(self, tr, prediction, *, include_declines=False):
        log_weight, evidence = 0.0, []
        hidden = dict(prediction.opponent_hidden)
        for other, hand in prediction.opponent_hands:
            original = Counter(hand) + Counter(hidden.get(other, ()))
            model = tr.get("public_hand_models", {}).get(other, {})
            for event in model.get("royal_instead_of_same_evidence", ()):
                piece = event["piece"]
                count = original[piece] + self._observed_piece_count_for_player(tr, other, piece)
                factor = self._response_likelihood(
                    event, count, self.ROYAL_INSTEAD_OF_SAME_PRESENT_LIKELIHOOD)
                log_weight += math.log(max(1e-9, factor))
                evidence.append(f"{other}:royal_instead_of_same_{piece}")
            if include_declines:
                # The shared search's legacy weighting uses count estimates,
                # unlike the probabilistic posterior's explicit pass likelihood.
                # Apply the new threat-context evidence once in that pipeline.
                for piece, events in tr.get("piece_pass_evidence", {}).get(other, {}).items():
                    for event in events:
                        if event.get("response_context") != "enemy_near_reach":
                            continue
                        count = original[piece] + self._observed_piece_count_for_player(tr, other, piece)
                        factor = self._response_likelihood(event, count, event["pass_if_held"])
                        log_weight += math.log(max(1e-9, factor))
                        evidence.append(f"{other}:declined_near_reach_{piece}")
        return log_weight, tuple(evidence)

    def _information_set_candidate_evidence(self, prediction, player, tr):
        result = super()._information_set_candidate_evidence(prediction, player, tr)
        if result is None:
            return None
        log_weight, confidence = result
        extra, _ = self._public_response_evidence(tr, prediction, include_declines=True)
        return log_weight + extra, confidence

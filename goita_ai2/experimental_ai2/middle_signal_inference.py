"""Keep opening claims, partner echoes and public counts as distinct evidence.

Legacy ranges describe a preferred allocation. Middle-piece behaviour is not
a physical constraint on the sampled deals used to choose an action.
"""
from __future__ import annotations

import math
import copy
from collections import Counter


class MiddleSignalInferenceMixin:
    def _update_public_hand_model(self, state, tr, player, action_type, block, attack):
        own_first = (player == self.me and attack in ("2", "3", "4", "5")
                     and action_type in ("attack", "attack_after_block")
                     and not tr.get("public_hand_models", {}).get(player, {}).get("attack_count"))
        partner = self._ally_of(player)
        echo = own_first and bool(tr.get("public_hand_models", {}).get(partner, {}).get("attacks", {}).get(attack, 0))
        result = super()._update_public_hand_model(state, tr, player, action_type, block, attack)
        if own_first:
            signals = tr.setdefault("middle_first_attack_signals", {})
            signals[player] = {"piece": attack, "role": "ally_echo" if echo else "independent_claim",
                               "partner": partner, "order": len(signals), "proven": False}
            tr["public_hand_models"][player]["first_attack_role"] = signals[player]["role"]
        return result

    def _record_first_attack_count_estimate(self, state, tr, player, attack, strategy):
        if player == self.me or attack not in ("2", "3", "4", "5"):
            return super()._record_first_attack_count_estimate(state, tr, player, attack, strategy)
        ally = self._ally_of(player)
        echo = bool(tr.get("public_hand_models", {}).get(ally, {}).get("attacks", {}).get(attack, 0))
        signals = tr.setdefault("middle_first_attack_signals", {})
        signals[player] = {"piece": attack, "role": "ally_echo" if echo else "independent_claim",
                           "partner": ally, "order": len(signals), "proven": False}
        tr["public_hand_models"][player]["first_attack_role"] = signals[player]["role"]
        if echo:
            self._update_observed_piece_count_estimate(tr, player, attack)
            self._reconcile_piece_count_estimates(state, tr, attack)
        else:
            super()._record_first_attack_count_estimate(state, tr, player, attack, strategy)

    def _reconcile_piece_count_estimates(self, state, tr, piece):
        signals = tr.get("middle_first_attack_signals", {})
        if piece not in ("2", "3", "4", "5") or not any(
                s["piece"] == piece and s["role"] == "ally_echo" for s in signals.values()):
            return super()._reconcile_piece_count_estimates(state, tr, piece)
        players = [seat for seat in "ABCD" if seat != self.me]
        total = self._piece_total(piece) - int(tr.get("my_init_count", {}).get(piece, 0))
        observed = {p: self._observed_piece_count_for_player(tr, p, piece) for p in players}
        lower = dict(observed)
        for p in players:
            signal = signals.get(p, {})
            model = tr.get("public_hand_models", {}).get(p, {})
            if signal.get("piece") == piece and signal.get("role") == "independent_claim" and not model.get("strategy_broken"):
                lower[p] = max(observed[p], min(2, total))
        # Public consumption wins. Echoes never establish a new two-copy lower
        # bound; if claims conflict, weaken the later, unconfirmed claim first.
        while sum(lower.values()) > total:
            candidates = [p for p in players if lower[p] > observed[p]]
            if not candidates:
                break
            p = max(candidates, key=lambda seat: signals.get(seat, {}).get("order", -1))
            lower[p] -= 1
        for p in players:
            estimate = self._ensure_piece_count_estimate(tr, p, piece)
            signal = signals.get(p, {})
            source = (signal["role"] if signal.get("piece") == piece else "observed")
            estimate.update(min=lower[p], max=max(lower[p], total - sum(lower[q] for q in players if q != p)),
                            source=source, bounds_are="strategic_preference")

    def _refresh_public_piece_inference_impl(self, state, tr, *, reason):
        if tr and tr.get("estimated_current_hands"):
            # Comparison with the previous estimates belongs to the base model.
            # Remove our explanatory annotations/cap before that comparison, so
            # unchanged evidence does not repeatedly invalidate prediction caches.
            clean = {}
            for seat, hand in tr["estimated_current_hands"].items():
                clean[seat] = {}
                for piece, item in hand.items():
                    if item.get("bounds_are") == "strategic_preference":
                        raw = {k: v for k, v in item.items() if k not in ("bounds_are", "signal_prior_confidence")}
                        raw["confidence"] = item.get("signal_prior_confidence", item.get("confidence", 0))
                        clean[seat][piece] = raw
                    else:
                        clean[seat][piece] = item
            tr["estimated_current_hands"] = clean
        result = super()._refresh_public_piece_inference_impl(state, tr, reason=reason)
        if not tr:
            return result
        pieces = {s["piece"] for s in tr.get("middle_first_attack_signals", {}).values() if s["role"] == "ally_echo"}
        for hand in tr.get("estimated_current_hands", {}).values():
            for piece in pieces:
                if piece in hand:
                    hand[piece]["bounds_are"] = "strategic_preference"
                    hand[piece]["signal_prior_confidence"] = hand[piece].get("confidence", 0)
                    hand[piece]["confidence"] = min(0.85, float(hand[piece].get("confidence", 0)))
        return result

    @staticmethod
    def _middle_signal_sampling_tracker(tr):
        pieces = {s["piece"] for s in tr.get("middle_first_attack_signals", {}).values() if s["role"] == "ally_echo"}
        if not pieces:
            return tr
        relaxed = dict(tr)
        relaxed["estimated_current_hands"] = {
            p: {piece: dict(item) for piece, item in hand.items()}
            for p, hand in tr.get("estimated_current_hands", {}).items()}
        for hand in relaxed["estimated_current_hands"].values():
            for piece in pieces:
                if piece in hand:
                    # Pool counts and visible hand/hidden-slot sizes enforce
                    # physical limits in the shared sampler and consistency check.
                    hand[piece].update(min=0, max=int(tr.get("unknown_piece_pool", {}).get(piece, 0)))
        return relaxed

    def _timed_search_sample_state(self, state, player, tr, rng):
        return super()._timed_search_sample_state(
            state, player, self._middle_signal_sampling_tracker(tr), rng)

    def _information_set_candidate_evidence(self, prediction, player, tr):
        result = super()._information_set_candidate_evidence(
            prediction, player, self._middle_signal_sampling_tracker(tr))
        if result is None:
            return None
        likelihood, confidence = result
        hands, hidden = dict(prediction.opponent_hands), dict(prediction.opponent_hidden)
        echo_pieces = {s["piece"] for s in tr.get("middle_first_attack_signals", {}).values() if s["role"] == "ally_echo"}
        for seat, signal in tr.get("middle_first_attack_signals", {}).items():
            piece = signal["piece"]
            if signal["role"] != "independent_claim" or seat == player or piece not in echo_pieces:
                continue
            count = (Counter(hands.get(seat, ()))[piece] + Counter(hidden.get(seat, ()))[piece]
                     + self._observed_piece_count_for_player(tr, seat, piece))
            if count < 2:
                likelihood += math.log(0.2)
        return likelihood, confidence

    def _probabilistic_candidate_action_evidence(self, state, player, tr, prediction):
        result = super()._probabilistic_candidate_action_evidence(state, player, tr, prediction)
        if result is None:
            return None
        likelihood, confidence, evidence = result
        hands, hidden = dict(prediction.opponent_hands), dict(prediction.opponent_hidden)
        for seat, signal in tr.get("middle_first_attack_signals", {}).items():
            if signal["role"] != "ally_echo" or seat == player:
                continue
            if f"{seat}:opening_partner_response" in evidence:
                continue  # The compact endgame model already applied this evidence.
            piece = signal["piece"]
            count = (Counter(hands.get(seat, ()))[piece] + Counter(hidden.get(seat, ()))[piece]
                     + self._observed_piece_count_for_player(tr, seat, piece))
            old, _ = self._probabilistic_first_attack_likelihood(
                piece, count, strategy_broken=bool(tr.get("public_hand_models", {}).get(seat, {}).get("strategy_broken")),
                my_initial_count=tr.get("my_init_count", Counter()))
            likelihood -= math.log(max(1e-9, old))
            evidence = tuple(evidence) + (f"{seat}:ally_middle_echo_{piece}",)
        return likelihood, confidence, evidence

    def _middle_signal_summary(self, tracker):
        return copy.deepcopy({"signals": tracker.get("middle_first_attack_signals", {}), "proven": False,
                "observer": self.me, "own_initial_counts": dict(tracker.get("my_init_count", {})),
                "preferred_initial_counts": tracker.get("joint_hand_inference", {}).get("map_original_counts", {}),
                "preferred_current_counts": tracker.get("joint_hand_inference", {}).get("map_current_counts", {}),
                "preferred_hidden_counts": tracker.get("joint_hand_inference", {}).get("map_hidden_counts", {})})

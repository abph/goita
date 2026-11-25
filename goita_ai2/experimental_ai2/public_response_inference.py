"""Soft evidence from declined receives; never exclude a physically possible deal."""
from __future__ import annotations

import math
from collections import Counter


class PublicResponseInferenceMixin:
    # Behavioural model parameters, not measured frequencies or certainty.
    ROYAL_INSTEAD_OF_SAME_PRESENT_LIKELIHOOD = 0.12
    DECLINE_REACH_PRESENT_LIKELIHOOD = 0.08
    DECLINE_OWN_FINISH_PRESENT_LIKELIHOOD = 0.12
    SWITCH_REPEAT_ATTACK_PRESENT_LIKELIHOOD = 0.30
    REACH_ATTACK_IMMEDIATE_LOSS_PENALTY = 240.0

    def _record_piece_pass_evidence(self, state, tr, player):
        super()._record_piece_pass_evidence(state, tr, player)
        context = tr.get("active_attack_context") or {}
        attacker, piece = context.get("attacker"), context.get("piece")
        if player == self.me or attacker is None or piece not in ("2", "3", "4", "5", "6", "7"):
            return
        events = tr.get("piece_pass_evidence", {}).get(player, {}).get(piece, [])
        if not events:
            return
        same_team = self._same_team(player, attacker)
        if len(state.hands[player]) == 2:
            # A second pass with the same two cards is not independent evidence.
            # Passing a ready partner's attack can deliberately let them finish.
            if (not same_team or len(state.hands[attacker]) > 2) and not any(
                event.get("response_context") == "declined_own_finish" for event in events[:-1]
            ):
                events[-1]["pass_if_held"] = self.DECLINE_OWN_FINISH_PRESENT_LIKELIHOOD
                events[-1]["response_context"] = "declined_own_finish"
        elif (piece != "2" and not same_team and len(state.hands[player]) > 2
              and 2 <= len(state.hands[attacker]) <= 4):
            events[-1]["pass_if_held"] = min(
                events[-1]["pass_if_held"], self.DECLINE_REACH_PRESENT_LIKELIHOOD)
            events[-1]["response_context"] = "enemy_near_reach"

    def _update_public_hand_model(self, state, tr, player, action_type, block, attack):
        context = tr.get("active_attack_context") or {}
        piece, attacker = context.get("piece"), context.get("attacker")
        model = tr.get("public_hand_models", {}).get(player, {})
        if (player != self.me and action_type == "attack_after_block"
                and player == attacker and context.get("attack_no") == 2
                and piece in ("2", "3", "4", "5") and model.get("first_attack") == piece
                and attack != piece and len(state.hands[player]) == 2):
            model["repeat_attack_switch"] = {
                "piece": piece,
                "observed_used": self._observed_piece_count_for_player(tr, player, piece),
                # The different attack is known. The just-played block could
                # still have been used in the attack before it went face-down.
                "hand_slots": len(state.hands[player]) + 1,
                "hidden_slots": int(tr.get("hidden_block_counts", {}).get(player, 0)),
            }
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
            switch = model.get("repeat_attack_switch")
            if switch:
                piece = switch["piece"]
                count = original[piece] + self._observed_piece_count_for_player(tr, other, piece)
                factor = self._response_likelihood(switch, count, self.SWITCH_REPEAT_ATTACK_PRESENT_LIKELIHOOD)
                log_weight += math.log(max(1e-9, factor))
                evidence.append(f"{other}:repeat_attack_switch_{piece}")
            if include_declines:
                # The shared search's legacy weighting uses count estimates,
                # unlike the probabilistic posterior's explicit pass likelihood.
                # Apply the new threat-context evidence once in that pipeline.
                for piece, events in tr.get("piece_pass_evidence", {}).get(other, {}).items():
                    for event in events:
                        if event.get("response_context") not in ("enemy_near_reach", "declined_own_finish"):
                            continue
                        count = original[piece] + self._observed_piece_count_for_player(tr, other, piece)
                        factor = self._response_likelihood(event, count, event["pass_if_held"])
                        log_weight += math.log(max(1e-9, factor))
                        evidence.append(f"{other}:{event['response_context']}_{piece}")
        return log_weight, tuple(evidence)

    def _information_set_candidate_evidence(self, prediction, player, tr):
        result = super()._information_set_candidate_evidence(prediction, player, tr)
        if result is None:
            return None
        log_weight, confidence = result
        extra, _ = self._public_response_evidence(tr, prediction, include_declines=True)
        return log_weight + extra, confidence

    def _reach_attack_response_review(self, state, player, actions):
        if state.phase != "attack" or len(state.hands[player]) not in (3, 4):
            return None
        next_enemy = state.next_player(player)
        enemies = (next_enemy, self._ally_of(next_enemy))
        ready_enemies = [enemy for enemy in enemies if len(state.hands[enemy]) == 2]
        if not ready_enemies:
            return None
        reach_kind = "attack" if len(state.hands[player]) == 3 else "attack_after_block"
        attacks = sorted({attack for kind, _block, attack in actions if kind == reach_kind})
        if len(attacks) < 2:
            return None
        tr = self._track.get(id(state))
        if tr is None:
            return None
        evidence = {}
        for enemy in enemies:
            switch = tr["public_hand_models"].get(enemy, {}).get("repeat_attack_switch")
            if switch and switch["piece"] in attacks:
                evidence.setdefault(switch["piece"], []).append({"seat": enemy, "reason": "repeat_attack_switch"})
            for piece, events in tr.get("piece_pass_evidence", {}).get(enemy, {}).items():
                if piece in attacks and any(event.get("response_context") == "declined_own_finish" for event in events):
                    evidence.setdefault(piece, []).append({"seat": enemy, "reason": "declined_own_finish"})
        if not evidence:
            return None

        samples = self._timed_search_generate_sample_states(state, player, tr, 80)
        try:
            _, worlds = self._information_set_search_worlds(state, player, tr, samples)
        except ValueError:
            return None
        squared_mass = math.fsum(world.probability ** 2 for world in worlds)
        # Reject estimates where a handful of deals would dominate the score.
        if len(worlds) < 24 or not 0 < squared_mass <= 1 / 10:
            return None

        candidates = {}
        for piece in attacks:
            holding = {enemy: math.fsum(world.probability for world in worlds
                                        if piece in world.state.hands[enemy]) for enemy in enemies}
            any_holds = math.fsum(world.probability for world in worlds
                                  if any(piece in world.state.hands[enemy] for enemy in enemies))
            # The second opponent still needs the attack to reach them. Count
            # their ability to win as a risk rather than assuming a rescue.
            finish = math.fsum(world.probability for world in worlds if any(
                self._forced_win_receive_options(world.state.hands[enemy], piece) for enemy in ready_enemies
            ))
            reuse = self._enemy_used_attack_reuse_penalty(state, player, piece)
            # Past use carries less weight once the later play supports a low
            # chance of still holding that piece. Royal receives count for loss risk.
            credit = reuse * (1.0 - any_holds) if piece in evidence else 0.0
            candidates[piece] = {"holding": holding, "ready_enemy_receive_mass": finish,
                                 "adjustment": credit - self.REACH_ATTACK_IMMEDIATE_LOSS_PENALTY * finish,
                                 "evidence": evidence.get(piece, [])}
        return {"ready_enemies": ready_enemies, "samples": len(worlds), "attack_kind": reach_kind,
                "candidates": candidates}

    def _select_rule_based_action(self, state, player, actions):
        tr = self._track.get(id(state), {})
        tr.pop("last_reach_attack_response_review", None)
        review = self._reach_attack_response_review(state, player, actions)
        previous = getattr(self, "_active_reach_attack_review", None)
        self._active_reach_attack_review = (id(state), player, review) if review else None
        try:
            chosen = super()._select_rule_based_action(state, player, actions)
        finally:
            self._active_reach_attack_review = previous
        if (review and chosen[0] == review["attack_kind"] and self.last_decision_reason == "score_fallback"
                and len(self.last_attack_candidate_scores) == len(actions)):
            candidates = review["candidates"]
            baseline = max(self.last_attack_candidate_scores, key=lambda item: item["score"]
                           - candidates.get(item["action"][2], {}).get("adjustment", 0.0))
            if baseline["action"] != chosen:
                review.update(previous_attack=baseline["action"][2], selected_attack=chosen[2])
                tr["last_reach_attack_response_review"] = review
                self._set_score_fallback_detail("attack_reach_public_receive_risk")
        return chosen

    def _score_attack_phase(self, state, player, action_type, block, attack, *, has_non_king_attack_option):
        score = super()._score_attack_phase(state, player, action_type, block, attack,
                                            has_non_king_attack_option=has_non_king_attack_option)
        active = getattr(self, "_active_reach_attack_review", None)
        if active and active[:2] == (id(state), player) and action_type == active[2]["attack_kind"]:
            return score + active[2]["candidates"].get(attack, {}).get("adjustment", 0.0)
        return score

    def _finalize_attack_candidate_snapshot(self, actions, chosen):
        super()._finalize_attack_candidate_snapshot(actions, chosen)
        tracker = self._track.get(getattr(self, "_live_plan_review_state_id", None), {})
        review = tracker.get("last_reach_attack_response_review")
        if (review and self.last_score_fallback_detail == "attack_reach_public_receive_risk"
                and review["selected_attack"] == chosen[2]):
            self.last_attack_candidate_snapshot["reach_attack_response_review"] = review

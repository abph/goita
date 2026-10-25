"""Public behavioural evidence is a likelihood, never a revealed hand."""
from __future__ import annotations

import math
import random
import time
from collections import Counter

from goita_ai2.current_ai.endgame import ForcedWinStatus


class _BeliefLimit(Exception):
    pass


class BeliefEndgameMixin:
    """Bounded endgame comparison; samples are estimates, never proof sources."""

    BELIEF_ENDGAME_DRAWS = 256
    BELIEF_ENDGAME_WORLDS = 48
    BELIEF_ENDGAME_SECONDS = 0.45
    BELIEF_ENDGAME_NODES = 40000
    # Initial behavioural weights, not empirically calibrated win probabilities.
    UNRETURNED_MIDDLE_PRESENT_LIKELIHOOD = 0.30
    LOWER_MIDDLE_OPENING_GOLD_PAIR_LIKELIHOOD = 0.65
    THIRD_NONROYAL_RETAINS_ROYAL_LIKELIHOOD = 0.75
    BELIEF_TEAM_WIN_VALUE = 1000
    BELIEF_IMMEDIATE_LOSS_PENALTY = 300

    def _infer_missing_kings_from_third_visible_attack(self, state, tr, player, action_type, attack):
        if (player != self.me and action_type == "attack" and attack not in ("8", "9")
                and tr.get("public_hand_models", {}).get(player, {}).get("attack_count") == 3):
            tr["public_hand_models"][player]["soft_third_nonroyal"] = True

    def _update_public_hand_model(self, state, tr, player, action_type, block, attack):
        model = tr.get("public_hand_models", {}).get(player, {})
        if (player != self.me and action_type in ("attack", "attack_after_block")
                and not model.get("attack_count")):
            partner = self._ally_of(player)
            first = tr.get("public_hand_models", {}).get(partner, {}).get("first_attack")
            if first in ("3", "4", "5", "6", "7") and attack == first:
                model["soft_opening_response_piece"] = first
            if first in ("3", "4", "5") and attack != first:
                model["soft_unreturned_opening"] = {
                    "piece": first,
                    "observed_used": self._observed_piece_count_for_player(tr, player, first),
                    "hidden_slots": int(tr.get("hidden_block_counts", {}).get(player, 0))
                        + int(action_type == "attack_after_block"),
                    "hand_slots": len(state.hands[player]),
                }
        return super()._update_public_hand_model(state, tr, player, action_type, block, attack)

    def _behaviour_piece_likelihood(self, tr, player, piece, original_count):
        model = tr.get("public_hand_models", {}).get(player, {})
        first = model.get("first_attack")
        weight = 1.0
        # A lower-ranked middle opening weakens, but cannot exclude, a gold pair.
        if piece == "5" and first in ("3", "4") and original_count >= 2:
            weight *= self.LOWER_MIDDLE_OPENING_GOLD_PAIR_LIKELIHOOD
        reply = model.get("soft_unreturned_opening", {})
        if reply.get("piece") == piece and not model.get("attacks", {}).get(piece, 0):
            remaining = max(0, original_count - int(reply["observed_used"]))
            hidden, slots = int(reply["hidden_slots"]), int(reply["hand_slots"])
            total = hidden + slots
            no_current = (1.0 if remaining == 0 else
                          math.comb(hidden, remaining) / math.comb(total, remaining)
                          if remaining <= hidden and remaining <= total else 0.0)
            weight *= no_current + (1.0 - no_current) * self.UNRETURNED_MIDDLE_PRESENT_LIKELIHOOD
        return max(1e-9, weight)

    def _probabilistic_candidate_action_evidence(self, state, player, tr, prediction):
        # This posterior validates physical public facts, independently of the
        # conventional count ranges retained by the ordinary opening tactics.
        result = super()._probabilistic_candidate_action_evidence(state, player, tr, prediction)
        if result is None:
            return None
        if (max(len(hand) for hand in state.hands.values()) > 4
                or min(len(state.hands[seat]) for seat in "ABCD"
                       if not self._same_team(seat, player)) > 2):
            return result
        log_weight, confidence, evidence = result
        evidence = list(evidence)
        hidden = dict(prediction.opponent_hidden)
        for other, hand in prediction.opponent_hands:
            current = Counter(hand)
            original = current + Counter(hidden[other])
            model = tr.get("public_hand_models", {}).get(other, {})
            for piece in map(str, range(1, 10)):
                original[piece] += self._observed_piece_count_for_player(tr, other, piece)
                factor = self._behaviour_piece_likelihood(tr, other, piece, original[piece])
                log_weight += math.log(factor)
                if factor < 1:
                    evidence.append(f"{other}:soft_opening_or_unreturned_{piece}")
            response = model.get("soft_opening_response_piece")
            if response is not None:
                # A returned partner signal can be a singleton kakarigotae,
                # not another independent claim of an opening pair.
                prior_factor = self._probabilistic_first_attack_likelihood(
                    response, original[response], strategy_broken=bool(model.get("strategy_broken")),
                    my_initial_count=tr.get("my_init_count", Counter()),
                )[0]
                log_weight -= math.log(max(1e-9, prior_factor))
                evidence.append(f"{other}:opening_partner_response")
            if tr.get("public_hand_models", {}).get(other, {}).get("soft_third_nonroyal"):
                if current["8"] or current["9"]:
                    log_weight += math.log(self.THIRD_NONROYAL_RETAINS_ROYAL_LIKELIHOOD)
                    evidence.append(f"{other}:soft_third_nonroyal")
        return log_weight, confidence, tuple(evidence)

    def _solve_inferred_endgame(self, state, root_player, baseline_scores, depth, memo):
        budget = getattr(self, "_belief_solver_budget", None)
        if budget is not None:
            budget["nodes"] += 1
            if budget["nodes"] > self.BELIEF_ENDGAME_NODES or time.perf_counter() >= budget["deadline"]:
                raise _BeliefLimit()
        return super()._solve_inferred_endgame(state, root_player, baseline_scores, depth, memo)

    def _belief_immediate_enemy_win(self, state, player):
        actor = state.turn
        if state.finished or self._same_team(actor, player) or len(state.hands[actor]) != 2:
            return False
        for action in state.legal_actions(actor):
            after = self._apply_action_on_copy(state, actor, action)
            if after.finished:
                return True
            if action[0] == "receive" and any(
                self._apply_action_on_copy(after, actor, attack).finished
                for attack in after.legal_actions(actor)
            ):
                return True
        return False

    def _belief_endgame_comparison(self, state, player, actions):
        tr = self._track.get(id(state))
        if (tr is None or state.turn != player or len(actions) < 2
                or max(len(hand) for hand in state.hands.values()) > 4
                or min(len(state.hands[seat]) for seat in "ABCD" if not self._same_team(seat, player)) > 2):
            return None
        key = (self._information_set_key(state, player, tr).digest, tuple(actions))
        cache = tr.setdefault("belief_endgame_cache", {})
        if key in cache:
            tr["last_belief_endgame"] = cache[key]
            return cache[key]
        started = time.perf_counter()
        deadline = started + self.BELIEF_ENDGAME_SECONDS
        rng = random.Random(self._probabilistic_prior_seed(state, player, tr))
        grouped = Counter()
        for _ in range(self.BELIEF_ENDGAME_DRAWS):
            if time.perf_counter() >= deadline:
                return None
            prediction = self._probabilistic_uniform_prediction(state, player, tr, rng)
            if prediction is not None:
                grouped[prediction] += 1
        weighted = []
        for prediction, count in grouped.items():
            if time.perf_counter() >= deadline:
                return None
            result = self._probabilistic_candidate_action_evidence(state, player, tr, prediction)
            if result is not None:
                weighted.append((prediction, math.log(count) + result[0]))
        if not weighted:
            return None
        peak = max(value for _, value in weighted)
        weighted = [(pred, math.exp(value - peak)) for pred, value in weighted]
        weighted.sort(key=lambda item: (-item[1], repr(item[0])))
        # Keep a bounded, stratified selection over the whole posterior instead
        # of retaining only its easiest/highest-mass individual worlds.
        total = sum(weight for _, weight in weighted)
        selected = Counter()
        cursor = 0
        cumulative = weighted[0][1] / total
        count = min(self.BELIEF_ENDGAME_WORLDS, len(weighted))
        for index in range(count):
            point = (index + 0.5) / count
            while point > cumulative and cursor + 1 < len(weighted):
                cursor += 1
                cumulative += weighted[cursor][1] / total
            selected[weighted[cursor][0]] += 1
        plans = None
        rows = []
        completed = 0
        unique_worlds = 0
        previous = getattr(self, "_belief_solver_budget", None)
        self._belief_solver_budget = {"nodes": 0, "deadline": deadline}
        try:
            for prediction, mass in selected.items():
                sampled = self._timed_search_materialize_prediction(state, player, prediction)
                baseline = dict(sampled.team_score)
                if plans is None:
                    plans = []
                    for action in actions:
                        after = self._apply_action_on_copy(sampled, player, action)
                        if action[0] == "receive" and not after.finished:
                            plans.extend((action, attack) for attack in after.legal_actions(player))
                        else:
                            plans.append((action, None))
                    rows = [dict(action=action, followup=followup, wins=0.0, immediate_loss=0.0,
                                 score=0.0, outcomes=Counter()) for action, followup in plans]
                outcomes = []
                memo = {}
                for action, followup in plans:
                    after = self._apply_action_on_copy(sampled, player, action)
                    if followup is not None:
                        after = self._apply_action_on_copy(after, player, followup)
                    immediate = self._belief_immediate_enemy_win(after, player)
                    outcome = self._solve_inferred_endgame(after, player, baseline, 48, memo)
                    if outcome is None:
                        raise _BeliefLimit()
                    outcomes.append((outcome, immediate))
                for row, (outcome, immediate) in zip(rows, outcomes):
                    _, winner, score, _ = outcome
                    won = self._same_team(winner, player)
                    row["wins"] += mass * won
                    row["immediate_loss"] += mass * immediate
                    row["score"] += mass * (score if won else -score)
                    row["outcomes"][(winner, score)] += mass
                completed += mass
                unique_worlds += 1
        except _BeliefLimit:
            pass
        finally:
            nodes = self._belief_solver_budget["nodes"]
            self._belief_solver_budget = previous
        if count < 8 or unique_worlds < 2 or completed != count:
            return None
        for row in rows:
            for field in ("wins", "immediate_loss", "score"):
                row[field] /= completed
            row["value"] = (row["wins"] * self.BELIEF_TEAM_WIN_VALUE
                            - row["immediate_loss"] * self.BELIEF_IMMEDIATE_LOSS_PENALTY
                            + row["score"])
        best = max(rows, key=lambda row: row["value"])
        winner, score = best["outcomes"].most_common(1)[0][0]
        result = dict(action=best["action"], followup=best["followup"], winner=winner, score=score,
                      worlds=completed, wins=best["wins"], immediate_loss=best["immediate_loss"],
                      candidates=[{k: v for k, v in row.items() if k != "outcomes"} for row in rows],
                      source="public_action_weighted_worlds", proven=False, nodes=nodes,
                      unique_worlds=unique_worlds,
                      elapsed_ms=round((time.perf_counter() - started) * 1000, 2))
        if len(cache) >= 4:
            cache.clear()
        cache[key] = result
        tr["last_belief_endgame"] = result
        return result

    def _inferred_endgame_team_result_action(self, state, player, actions):
        if (state.phase != "receive" or state.attacker is None
                or self._same_team(state.attacker, player)):
            return None
        # The caller also consults inferred endgames to improve a guaranteed
        # receive's score. Sample estimates cannot replace that public proof.
        if any(self._forced_win_result_after_receive_action(state, player, action).status
               == ForcedWinStatus.PROVEN for action in actions if action[0] == "receive"):
            return None
        result = self._belief_endgame_comparison(state, player, actions)
        if result is None:
            # An uncertain one-world result must not become a strong finish
            # claim when this bounded comparison cannot complete.
            tr = self._track.get(id(state), {})
            compact = all(len(hand) <= 4 for hand in state.hands.values())
            if compact:
                tr["pending_inferred_endgame_attack"] = None
                return None
            return super()._inferred_endgame_team_result_action(state, player, actions)
        tr = self._track[id(state)]
        tr["pending_inferred_endgame_attack"] = result["followup"]
        tr["belief_endgame_log_pending"] = result
        return result["action"], result["winner"], result["score"]

    def _select_rule_based_action(self, state, player, actions):
        tr = self._track.get(id(state), {})
        tr.pop("belief_endgame_log_pending", None)
        if state.phase == "attack" and tr.get("pending_inferred_endgame_attack") is not None:
            result = self._belief_endgame_comparison(state, player, actions)
            tr["pending_inferred_endgame_attack"] = None if result is None else result["action"]
            if result is not None:
                tr["belief_endgame_log_pending"] = result
        chosen = super()._select_rule_based_action(state, player, actions)
        if self.last_score_fallback_detail.startswith("belief_endgame_") and chosen[0] == "attack":
            result = tr.get("last_belief_endgame", {})
            self.last_attack_candidate_scores = [
                {"action": row["action"], "score": row["value"], "candidate_role": "public_belief_estimate"}
                for row in result.get("candidates", ()) if row["followup"] is None
            ]
        return chosen

    def _finalize_attack_candidate_snapshot(self, actions, chosen):
        super()._finalize_attack_candidate_snapshot(actions, chosen)
        if self.last_score_fallback_detail.startswith("belief_endgame_") and chosen[0] == "attack":
            result = self._track.get(getattr(self, "_belief_active_state_id", None), {}).get("last_belief_endgame")
            if result is not None:
                self.last_attack_candidate_snapshot["belief_endgame"] = result

    def _set_score_fallback_detail(self, detail):
        tr = self._track.get(getattr(self, "_belief_active_state_id", None), {})
        result = tr.get("belief_endgame_log_pending")
        if result is not None and detail.startswith("inferred_endgame_"):
            detail = (f"belief_endgame_worlds_{result['worlds']}_win_{round(result['wins'] * 100)}"
                      f"_danger_{round(result['immediate_loss'] * 100)}")
            tr.pop("belief_endgame_log_pending", None)
        return super()._set_score_fallback_detail(detail)

    def select_action(self, state, player, actions):
        self._belief_active_state_id = id(state)
        return super().select_action(state, player, actions)

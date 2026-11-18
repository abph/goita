"""Bounded public-event reviews of the current attack outline.

An outline assumes future attacks return. It is a hypothesis, never a win
proof. Only our cards and tracked public information are used. Reviews do
not run rollouts, exact proofs, neural inference, or background jobs.
"""
from __future__ import annotations

import copy
import time


class LivePlanReviewMixin:
    LIVE_PLAN_REVIEW_MAX_CANDIDATES = 3
    LIVE_PLAN_REVIEW_MIN_GAIN = 20.0

    def on_public_action(self, state, player, action):
        super().on_public_action(state, player, action)
        if self.me is not None and id(state) in self._track:
            self._review_live_attack_plan(state, player, action)

    def _light_attack_outline(self, state, hand, first_number, preferred=()):
        """Greedily check at most four own-card pairs, without opponent search."""
        cards = list(hand)
        steps = []
        total = 0.0
        previous = next(iter(reversed(self._track[id(state)].get("my_attack_history", []))), None)
        king_used = state.king_block_used > 0
        both = bool(state.had_both_kings.get(self.me, False))
        for offset in range(4):
            if not cards:
                break
            need_block = len(cards) % 2 == 0
            attack_no = first_number + offset
            frozen = tuple(sorted(cards))
            attacks = sorted(set(cards))
            if offset < len(preferred):
                attacks = [preferred[offset]] if preferred[offset] in cards else []
            attack_values = {p: self._planned_future_attack_value(
                state, self.me, p, attack_no, previous, frozen,
            ) for p in attacks}
            blocks = sorted(set(cards)) if need_block else [None]
            block_values = {p: self._planned_future_block_value(state, self.me, p, frozen)
                            if p is not None else 0.0 for p in blocks}
            choices = []
            for attack in attacks:
                for block in blocks:
                    if not self._forced_win_attack_is_legal(
                        frozen, attack, block=block, king_used=king_used, had_both_kings=both,
                    ):
                        continue
                    rest = list(cards)
                    if block is not None:
                        rest.remove(block)
                    rest.remove(attack)
                    if not rest:
                        # A single card after a natural receive has no hidden block.
                        finish = (self._planned_finish_score(block, attack) if block is not None
                                  else self._forced_win_finish_score(None, attack))
                        score = finish * self.GENERAL_ATTACK_PLAN_FINISH_WEIGHT
                        reserve = 0.0
                    else:
                        score = attack_values[attack] + block_values[block]
                        reserve = float(self._best_pair_finish_score_from_hand(rest))
                    choices.append(((score + reserve * self.GENERAL_ATTACK_PLAN_FINISH_WEIGHT,
                                     score, attack, block or ""), rest, block, attack, score))
            if not choices:
                return None
            _, cards, block, attack, score = max(choices, key=lambda item: item[0])
            steps.append({"block": block, "attack": attack})
            total += score
            previous = attack
        if cards:
            return None
        return {"attacks": [s["attack"] for s in steps], "steps": steps,
                "score": round(total, 3), "assumption": "our_attacks_return", "proven": False}

    def _review_live_attack_plan(self, state, actor, action):
        started = time.perf_counter()
        tr = self._track[id(state)]
        strategy = tr.setdefault("round_strategy", {})
        old = strategy.get("live_review") or {}
        first_number = len(tr.get("my_attack_history", [])) + 1
        previous = list(old.get("attacks", strategy.get("next_attacks", strategy.get("opening_attacks", ()))))
        if actor == self.me and action[0] in ("attack", "attack_after_block"):
            if previous and previous[0] == action[2]:
                previous = previous[1:]
            else:
                previous = []
        hand = tuple(sorted(state.hands[self.me]))
        visible_block = None if actor != self.me and action[0] == "attack_after_block" else action[1]
        enemies = [p for p in "ABCD" if not self._same_team(p, self.me)]
        unseen = {p: max(0, self._piece_total(p) - tr["public_seen_counts"].get(p, 0) - hand.count(p))
                  for p in map(str, range(1, 10))}
        threats = {p: len(state.hands[p]) for p in enemies}  # Public sizes, not their cards.
        active = self._active_branched_attack_plan(state)
        protected = bool(tr.get("active_forced_win_plan") or
                         (active is not None and active.plan.evaluation.guaranteed_win))
        review = {
            "source": "light_public_event_review", "revision": int(old.get("revision", 0)) + 1,
            "event": {"actor": actor, "action": [action[0], visible_block, action[2]]},
            "hand": list(hand), "first_attack_number": first_number,
            "public_unseen": unseen, "enemy_hand_sizes": threats,
            "protected_plan": protected, "proven": False, "candidates": [],
            "reasons": ["own_hand_changed" if actor == self.me and action[0] != "pass"
                        else "ally_public_action" if self._same_team(actor, self.me)
                        else "enemy_public_action"],
        }
        if state.finished or not hand:
            review.update(status="finished", attacks=[], needs_deep_review=False)
        else:
            # Scores use cached public-count inference, not hidden-hand samples.
            values = {p: self._planned_future_attack_value(state, self.me, p, first_number, None, hand)
                      for p in sorted(set(hand))}
            heads = sorted(values, key=lambda p: (values[p], p), reverse=True)[:2]
            candidates = []
            retained = self._light_attack_outline(state, hand, first_number, previous) if previous else None
            if retained is not None:
                candidates.append(retained)
            for piece in heads:
                if len(candidates) >= self.LIVE_PLAN_REVIEW_MAX_CANDIDATES:
                    break
                candidate = self._light_attack_outline(state, hand, first_number, (piece,))
                if candidate is not None and all(candidate["steps"] != c["steps"] for c in candidates):
                    candidates.append(candidate)
            best = max(candidates, key=lambda c: c["score"], default=None)
            selected = retained
            if best is not None and (selected is None or
                                     best["score"] > selected["score"] + self.LIVE_PLAN_REVIEW_MIN_GAIN):
                selected = best
            # Never replace an already proven plan with a heuristic outline.
            if protected:
                attacks = previous
                status = "protected"
            elif selected is None:
                attacks = []
                status = "needs_replan"
            else:
                attacks = selected["attacks"]
                status = "continued" if attacks == previous else "updated"
            changed = attacks != previous
            review.update(status=status, attacks=attacks, candidates=candidates,
                          attack_values={p: round(v, 3) for p, v in values.items()},
                          needs_deep_review=not protected and (changed or not candidates
                                                              or min(threats.values()) <= 2))
            if status == "updated":
                review["reasons"].append("current_outline_updated")
            if min(threats.values()) <= 2:
                review["reasons"].append("enemy_can_finish_on_next_attack")
        review["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 4)
        strategy["live_review"] = review
        # Separate from the immutable opening outline and the last installed
        # exact/branched plan. This remains a tentative, public-information hint.
        strategy["current_attacks"] = list(review["attacks"][:3])
        return review

    def _build_branched_attack_plan(self, state, player, root_action, **kwargs):
        """Offer the updated outline to normal candidate evaluation, never force it."""
        label = str(kwargs.get("candidate_label", ""))
        if (kwargs.get("source", "public_branch_generation") == "public_branch_generation"
                and label.startswith("c") and label[1:].isdigit()):
            review = (self._track.get(id(state), {}).get("round_strategy") or {}).get("live_review")
            if (review and not review["protected_plan"] and review["status"] != "finished"
                    and review["hand"] == sorted(state.hands[player])
                    and review["first_attack_number"] == kwargs.get("attack_number")):
                attacks = review["attacks"]
                kwargs["preferred_attacks"] = tuple(attacks) if attacks and attacks[0] == root_action[2] else ()
        return super()._build_branched_attack_plan(state, player, root_action, **kwargs)

    def _finalize_attack_candidate_snapshot(self, actions, chosen):
        super()._finalize_attack_candidate_snapshot(actions, chosen)
        # Record a hint as a hint; the existing final decision reason stays intact.
        review = (self._track.get(getattr(self, "_live_plan_review_state_id", None), {})
                  .get("round_strategy", {}).get("live_review"))
        if review and isinstance(self.last_attack_candidate_snapshot, dict):
            self.last_attack_candidate_snapshot["live_plan_review"] = {
                "revision": review["revision"], "status": review["status"],
                "source": review["source"], "reasons": list(review["reasons"]),
                "attacks": list(review["attacks"]), "event": copy.deepcopy(review["event"]),
                "needs_deep_review": review["needs_deep_review"],
                "chosen_attack_matches_hint": (chosen[2] == review["attacks"][0]
                                                if chosen[2] and review["attacks"] else None),
            }

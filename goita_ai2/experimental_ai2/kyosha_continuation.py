"""Review lance receipts through three own attacks, using public facts only.

Outlines assume all-pass returns; interception branches separately spend a
matching receive or a royal and rebuild the remaining outline. They are
strategic hypotheses, never forced-win proofs or fixed future instructions.
"""
from __future__ import annotations

import copy
from collections import Counter


class KyoshaContinuationMixin:
    def _kyosha_own_attack_outlines(self, hand, pool, tracker, *, first_attack,
                                    attack_number, king_used, had_both_kings,
                                    receive_block=None):
        """Enumerate at most four own attacks, accounting for every hidden card."""
        ally_pieces = set(tracker.get("ally_past_attacks", ()))
        ally_first = tracker.get("ally_first_attack")
        if ally_first:
            ally_pieces.add(ally_first)
        ally_model = tracker.get("public_hand_models", {}).get(tracker.get("ally"), {})
        if ally_model.get("strategy_broken"):
            ally_pieces.clear()

        def paths(cards, number, steps, forced=None):
            attacks = [forced] if forced is not None else sorted(set(cards))
            for attack in attacks:
                blocks = [None] if forced is not None else sorted(set(cards))
                for block in blocks:
                    if not self._forced_win_attack_is_legal(
                        tuple(cards), attack, block=block, king_used=king_used,
                        had_both_kings=had_both_kings,
                    ):
                        continue
                    remaining = list(cards)
                    if block is not None:
                        remaining.remove(block)
                    remaining.remove(attack)
                    step = {"number": number, "block": block, "attack": attack,
                            "remaining_hand": sorted(remaining)}
                    next_steps = steps + [step]
                    if not remaining:
                        yield next_steps
                    elif number < 4:
                        yield from paths(remaining, number + 1, next_steps)

        candidates = []
        for steps in paths(sorted(hand), attack_number, [], first_attack):
            previous_attacks = Counter()
            repeated = set()
            ally_support = set()
            strength = 0.0
            for step in steps:
                piece = step["attack"]
                purposes = []
                finishing = not step["remaining_hand"]
                if step["number"] <= 3 and not finishing:
                    receivers = self._forced_win_external_receivers(pool, piece)
                    if not receivers:
                        purposes.append("retain_initiative")
                        strength += 2.0
                    elif piece in "34567" and previous_attacks[piece] >= pool[piece]:
                        purposes.append("force_royal_or_keep_lead")
                        strength += 1.5
                    if previous_attacks[piece] and piece not in repeated:
                        purposes.append("continue_repeat_attack")
                        repeated.add(piece)
                        strength += 0.75
                    if piece == "2" and pool[piece] <= 1:
                        purposes.append("limited_lance_receivers")
                        strength += 1.0
                    if piece in ally_pieces and piece not in ally_support:
                        purposes.append("support_public_ally_attack")
                        ally_support.add(piece)
                        strength += 1.5
                if finishing:
                    purposes.append("finish_if_route_holds")
                step["purposes"] = purposes or ["probe"]
                previous_attacks[piece] += 1
            final = steps[-1]
            finish_score = self._forced_win_finish_score(
                final["block"] if final["block"] is not None else receive_block,
                final["attack"],
            )
            candidates.append({"steps": steps, "strength": strength,
                               "conditional_finish_score": finish_score})
        return sorted(candidates, key=lambda plan: (
            plan["strength"], plan["conditional_finish_score"],
            tuple(step["attack"] for step in plan["steps"]),
        ), reverse=True)

    def _kyosha_continuation_review(self, state, player, attack):
        tracker = self._track.get(id(state))
        if not tracker:
            return None
        pool = self._public_unknown_piece_pool(state, player, tracker)
        if pool is None:
            return None
        hand = list(state.hands[player])
        if "2" not in hand:
            return None
        hand.remove("2")
        if ("attack", None, attack) not in self._timed_search_apply(
            state, player, ("receive", "2", None)
        ).legal_actions(player):
            return None
        number = int(tracker.get("my_attack_count", 0)) + 1
        kwargs = {"attack_number": number,
                  "king_used": bool(state.king_block_used),
                  "had_both_kings": bool(state.had_both_kings[player])}
        outlines = self._kyosha_own_attack_outlines(
            hand, pool, tracker, first_attack=attack, receive_block="2", **kwargs,
        )
        if not outlines:
            return None
        best = outlines[0]
        remaining = list(hand)
        remaining.remove(attack)
        branches = []
        # A publicly demonstrated enemy attack is a possible counterattack,
        # not a claim about that enemy's actual remaining hidden hand.
        counterattacks = {str(state.current_attack)}
        for enemy, model in tracker.get("public_hand_models", {}).items():
            if not self._same_team(enemy, player):
                counterattacks.update(model.get("attacks", {}))
        receivers = self._forced_win_external_receivers(pool, attack)
        for counter in sorted(counterattacks):
            if pool.get(counter, 0) <= 0 or not remaining:
                continue
            for receiver in receivers:
                branch_pool = pool.copy()
                branch_pool[receiver] -= 1
                if branch_pool.get(counter, 0) <= 0:
                    continue
                branch_pool[counter] -= 1
                responses = []
                for block in self._forced_win_receive_options(tuple(remaining), counter):
                    after = list(remaining)
                    after.remove(block)
                    for followup in sorted(set(after)):
                        options = self._kyosha_own_attack_outlines(
                            after, branch_pool, tracker, first_attack=followup,
                            attack_number=number + 1,
                            king_used=bool(state.king_block_used)
                            or receiver in "89" or block in "89",
                            had_both_kings=bool(state.had_both_kings[player]),
                            receive_block=block,
                        )
                        if options:
                            responses.append({"receive": block, **options[0]})
                continuation = max(responses, key=lambda plan: (
                    plan["strength"], plan["conditional_finish_score"],
                ), default=None)
                branches.append({"enemy_receive": receiver,
                                 "counterattack": counter,
                                 "continuation": continuation})
        strength = float(best["strength"])
        recoverable = bool(branches) and all(branch["continuation"] for branch in branches)
        if recoverable:
            strength += 0.75
        penalty = 0.0 if strength >= 2.5 else 40.0 if strength >= 1.5 else 120.0
        if branches and not recoverable:
            penalty += 40.0
        # An immediate finish does not require a later attack outline.
        if not remaining:
            penalty = 0.0
        return {"source": "own_hand_and_public_information", "attack": attack,
                "assumption": "all_pass_outline_with_interception_branches",
                "outline": best, "interception_branches": branches,
                "strength": strength, "uncertainty_penalty": penalty,
                "proven": False}

    def _compare_kyosha_round_routes(self, state, player):
        comparison = super()._compare_kyosha_round_routes(state, player)
        if comparison is None:
            return None
        reviews = {piece: self._kyosha_continuation_review(state, player, piece)
                   for piece in comparison["attack_values"]}
        comparison["three_attack_reviews"] = reviews
        comparison["raw_receive_value"] = comparison["receive_value"]
        comparison["raw_attack_values"] = dict(comparison["attack_values"])
        if any(review is None for review in reviews.values()):
            # Incomplete public information cannot justify a new preference.
            return comparison
        values = {piece: value - reviews[piece]["uncertainty_penalty"]
                  for piece, value in comparison["attack_values"].items()}
        eligible = set(values) - set(comparison["blocked_early_royal_attacks"])
        if not eligible:
            return comparison
        piece = comparison["best_receive_attack"]
        if not comparison["followup_override_blocked"]:
            piece = max(sorted(eligible), key=lambda candidate: values[candidate])
        # Preserve a publicly proved continuation regardless of this heuristic.
        protected = comparison["followup_rule_authority"] == "proven"
        if protected:
            return comparison
        comparison["attack_values"] = {p: round(v, 2) for p, v in values.items()}
        comparison["receive_value"] = round(values[piece], 2)
        comparison["best_receive_attack"] = piece
        comparison["search_best_attack"] = max(sorted(eligible), key=lambda p: values[p])
        margin = max(float(self.ROUND_ROUTE_MIN_MARGIN),
                     float(self.KYOSHA_PASS_COMPARE_LATER_MIN_MARGIN)
                     if self._kyosha_pass_compare_receiver_position(
                         state, player, state.legal_actions(player)) == "later" else 0.0)
        raw_receive_chosen = len(comparison["chosen"]) == 2
        receive = values[piece] > comparison["pass_value"] + margin
        comparison["chosen"] = ((("receive", "2", None), ("attack", None, piece))
                                if receive else (("pass", None, None),))
        comparison["three_attack_plan_changed_response"] = raw_receive_chosen != receive
        return comparison

    def _select_action_with_measurement(self, state, player, actions):
        tracker = self._track.get(id(state))
        before = copy.deepcopy(tracker) if tracker is not None else None
        if tracker is not None:
            tracker.pop("last_round_route_comparison", None)
        chosen = super()._select_action_with_measurement(state, player, actions)
        tracker = self._track.get(id(state)) or {}
        comparison = tracker.get("last_round_route_comparison") or {}
        # Cached responses, later lances and incomplete searches also need an
        # outline. Proven finishes remain above this strategic uncertainty.
        if (chosen == ("receive", "2", None) and not comparison
                and state.attacker is not None
                and not self._same_team(state.attacker, player)
                and ("pass", None, None) in actions
                and self._rule_search_authority(self.last_decision_reason,
                                                self.last_score_fallback_detail) != "proven"
                and self.last_decision_reason not in ("tsume", "conditional_tsume", "inferred_endgame")):
            after = self._timed_search_apply(state, player, chosen)
            attacks = after.legal_actions(player)
            followup = self._kyosha_round_followup_rule(state, after, player, attacks)
            review = (self._kyosha_continuation_review(state, player, followup["action"][2])
                      if followup and followup["action"][2] else None)
            tracker["last_kyosha_continuation_review"] = review
            if (review and review["uncertainty_penalty"] >= 120.0
                    and followup["authority"] != "proven"):
                diagnostics = {key: value for key, value in tracker.items()
                               if key.startswith("last_")}
                if before is not None:
                    tracker.clear()
                    tracker.update(before)
                    tracker.update(diagnostics)
                chosen = ("pass", None, None)
                self._set_decision_reason("score_fallback")
                self.last_rule_search_authority = "strong"
                self._set_kyosha_plan_pass_detail(review)
                search = tracker.get("last_time_limited_search")
                if isinstance(search, dict):
                    search["final_response_action"] = chosen
                    search["override_accepted"] = False
                    search["selection_stage"] = "three_attack_continuation"
                    search["three_attack_review"] = review
        if (state.phase == "receive" and state.current_attack == "2"
                and comparison.get("three_attack_plan_changed_response")
                and chosen == ("pass", None, None)
                and self.last_decision_reason == "time_search"):
            self.last_rule_search_authority = "strong"
            self._set_kyosha_plan_pass_detail(
                comparison["three_attack_reviews"][comparison["best_receive_attack"]])
        return chosen

    def _set_kyosha_plan_pass_detail(self, review):
        sequence = "".join(step["attack"] for step in review["outline"]["steps"]
                           if step["number"] <= 3) or review["attack"]
        self._set_score_fallback_detail(
            f"pass_kyosha_three_attack_plan_{sequence}_"
            f"penalty_{int(review['uncertainty_penalty'])}"
        )

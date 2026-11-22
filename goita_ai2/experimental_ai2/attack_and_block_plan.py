"""Plan the use of spare attackers while retaining useful receiving pieces."""

from collections import Counter

from goita_ai2.constants import POINTS


class AttackAndBlockPlanMixin:
    _RECEIVE_RESERVES = ("1", "6", "7")

    def _attack_and_block_steps(self, hand, attacks):
        """Allocate the planned attacks first, then choose from spare blocks."""
        cards = Counter(hand)
        steps = []
        for index, attack in enumerate(attacks):
            needs_block = cards.total() % 2 == 0
            if not cards[attack]:
                return []
            cards[attack] -= 1
            block = None
            if needs_block:
                future = Counter(attacks[index + 1:])
                spares = [piece for piece in cards if cards[piece] > future[piece]]
                if not spares:
                    return []
                block = min(spares, key=lambda piece: (
                    cards[piece] == 1 and piece in self._RECEIVE_RESERVES,
                    cards[piece] == 1,
                    POINTS[piece],
                    piece,
                ))
                cards[block] -= 1
            steps.append((block, attack))
        remaining = sorted(cards.elements())
        if len(remaining) != 2:
            return []
        finish = max(set(remaining), key=lambda piece: (POINTS[piece], piece))
        remaining.remove(finish)
        return [*steps, (remaining[0], finish)]

    def _reserve_attack_outline(self, state, player, hand, repeat, tracker):
        """Review the next attack(s) using the hand and current public signals."""
        if len(hand) not in (3, 4, 5, 6):
            return []
        count = (len(hand) - 1) // 2
        big_pieces = sorted(set(hand) & {"6", "7"})
        ally_attack = tracker.get("ally_last_attack")
        enemy_attacks = tracker.get("enemy_past_attacks", ())
        if repeat not in hand:
            # Once the repeat has been spent, retain the ordinary replan.
            return []
        candidates = []
        if count == 1:
            orderings = [(piece,) for piece in [repeat, *big_pieces]]
        else:
            orderings = [(piece, repeat) for piece in big_pieces]
            orderings += [(repeat, piece) for piece in big_pieces]
        for order in orderings:
            steps = self._attack_and_block_steps(hand, order)
            if not steps:
                continue
            block, head = steps[0]
            remaining = list(hand)
            remaining.remove(head)
            if block is not None:
                remaining.remove(block)
            key = (
                head not in enemy_attacks,
                head == repeat == ally_attack,
                head != repeat if count == 2 else head == repeat,
                sum(piece in remaining for piece in self._RECEIVE_RESERVES),
                self._planned_receive_width(remaining),
                self._piece_count_attack_adjustment(state, player, head),
                POINTS[head] if count == 2 else 0,
                head,
            )
            candidates.append((key, steps))
        return max(candidates, key=lambda row: row[0])[1] if candidates else []

    def _apply_attack_and_block_plan(self, state, player, actions, baseline):
        tracker = self._track.get(id(state))
        if tracker is None or state.phase != "attack":
            return baseline
        if baseline is None:
            # Normal tactics and proven finishes are tried before this fallback.
            review = tracker.get("round_strategy", {}).get("live_review", {})
            plan = review.get("attack_and_block_plan")
            if (plan is None or review.get("protected_plan")
                    or review["hand"] != sorted(state.hands[player])):
                return None
            block, attack = plan["steps"][0]
            action = ("attack" if block is None else "attack_after_block", block, attack)
            return (action, {"attack_and_block_plan": plan}) if action in actions else None

        original, original_plan = baseline
        if original_plan.get("opening_continuation_proof"):
            return baseline
        hand = state.hands[player]
        own_counts = Counter(hand)
        middle_pairs = {piece for piece in ("2", "3", "4", "5") if own_counts[piece] == 2}
        if (len(hand) != 8 or len(middle_pairs) < 2 or own_counts["1"] != 1
                or not (own_counts["6"] or own_counts["7"]) or own_counts["8"] or own_counts["9"]):
            return baseline

        candidates = []
        for action in actions:
            kind, block, attack = action
            if kind != "attack_after_block" or block not in middle_pairs or attack not in middle_pairs:
                continue
            if block == attack:
                continue
            remaining = list(hand)
            remaining.remove(block)
            remaining.remove(attack)
            continuation = self._reserve_attack_outline(state, player, remaining, attack, tracker)
            if not continuation:
                continue
            width = self._planned_receive_width(remaining)
            candidates.append(((width, POINTS[attack], -POINTS[block], action == original, attack, block),
                               action, [(block, attack), *continuation], remaining))
        if not candidates:
            return baseline
        _, action, steps, remaining = max(candidates, key=lambda row: row[0])
        outline = {
            "stage": "opening", "repeat": action[2], "steps": steps,
            "receive_reserves": [piece for piece in self._RECEIVE_RESERVES if piece in remaining],
        }
        plan = {
            **original_plan,
            "steps": steps, "attacks": [attack for _, attack in steps],
            "final_pair": steps[-1], "finish_score": self._planned_finish_score(*steps[-1]),
            "receive_width_after_opening": self._planned_receive_width(remaining),
            "projected_score": None,
            "attack_and_block_plan": outline,
        }
        tracker["shallow_eight_card_plan"] = plan
        return action, plan

    def _review_live_attack_plan(self, state, actor, action):
        tracker = self._track[id(state)]
        previous = tracker.get("round_strategy", {}).get("live_review", {}).get("attack_and_block_plan")
        review = super()._review_live_attack_plan(state, actor, action)
        opening_plan = tracker.get("shallow_eight_card_plan") or {}
        opening = opening_plan.get("attack_and_block_plan")
        if opening is None or state.finished or review["protected_plan"]:
            return review
        if review["first_attack_number"] == 1:
            return review
        if (actor == self.me and action[0] in ("attack", "attack_after_block")
                and review["first_attack_number"] == 2
                and tuple(action[1:]) != tuple(opening["steps"][0])):
            opening_plan.pop("attack_and_block_plan")
            return review
        steps = self._reserve_attack_outline(state, self.me, review["hand"], opening["repeat"], tracker)
        if steps:
            block, attack = steps[0]
            reserves = Counter(review["hand"])
            for piece in (block, attack):
                if piece is not None:
                    reserves[piece] -= 1
            outline = {
                "stage": "review", "repeat": opening["repeat"], "steps": steps,
                "receive_reserves": [piece for piece in self._RECEIVE_RESERVES if reserves[piece] > 0],
            }
            attacks = [attack for _, attack in steps]
            if previous is None or previous["steps"] != steps:
                review.update(status="updated", needs_deep_review=True)
                if "current_outline_updated" not in review["reasons"]:
                    review["reasons"].append("current_outline_updated")
            review.update(attack_and_block_plan=outline, attacks=attacks)
            tracker["round_strategy"]["current_attacks"] = attacks[:3]
        return review

    def _finalize_attack_candidate_snapshot(self, actions, chosen):
        super()._finalize_attack_candidate_snapshot(actions, chosen)
        detail = self.last_score_fallback_detail or ""
        if detail not in ("attack_and_block_plan_opening", "attack_and_block_plan_review"):
            return
        tracker = self._track.get(getattr(self, "_live_plan_review_state_id", None), {})
        if detail.endswith("_opening"):
            plan = tracker.get("shallow_eight_card_plan") or {}
        else:
            plan = tracker.get("round_strategy", {}).get("live_review", {})
        outline = plan.get("attack_and_block_plan")
        if outline is not None:
            self.last_attack_candidate_snapshot["attack_and_block_plan"] = outline

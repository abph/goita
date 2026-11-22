"""Prepare a few score-ordered receive and attack plans from the initial hand."""

from collections import Counter
from functools import lru_cache

from goita_ai2.constants import PIECE_TOTALS, POINTS
from goita_ai2.current_ai.endgame import ForcedWinStatus


class OpeningReceivePlansMixin:
    def _initial_round_strategy(self, state, player):
        strategy = super()._initial_round_strategy(state, player)
        hand = tuple(sorted(self._get_my_initial_hand(state)))
        if len(hand) != 8:
            return strategy
        rank = self._classify_hand_axes(list(hand))["rank"]
        if rank not in ("S", "A"):
            return strategy
        plans = self._initial_receive_plan_options(hand)
        if plans:
            strategy["opening_receive_plans"] = {"absolute_rank": rank, "plans": plans}
        return strategy

    def _initial_receive_plan_options(self, hand):
        pool = Counter(PIECE_TOTALS)
        pool.subtract(hand)
        if min(pool.values()) < 0:
            return []
        both_royals = "8" in hand and "9" in hand
        # At least one controlled attack is needed to re-enter safely. This
        # keeps the more expensive conditional proof focused.
        if not any(not self._forced_win_external_receivers(pool, piece) for piece in set(hand)):
            return self._provisional_receive_plan_options(hand, pool, both_royals)
        groups = {}
        memo = {}
        for incoming in sorted(piece for piece in pool if pool[piece] > 0):
            rest_pool = pool.copy()
            rest_pool[incoming] -= 1
            for receive in self._forced_win_receive_options(hand, incoming):
                after_receive = list(hand)
                after_receive.remove(receive)
                frozen = tuple(after_receive)
                king_used = receive in ("8", "9")
                choices = []
                for attack in sorted(set(frozen)):
                    if not self._forced_win_attack_is_legal(
                        frozen, attack, block=None, king_used=king_used, had_both_kings=both_royals,
                    ):
                        continue
                    remaining = list(frozen)
                    remaining.remove(attack)
                    proof = self._forced_win_resolve_attack(
                        tuple(remaining), rest_pool, attack=attack, king_used=king_used,
                        had_both_kings=both_royals, depth=10, external_cards_used=0,
                        minimum_enemy_hand=4, memo=memo,
                    )
                    if proof.status == ForcedWinStatus.PROVEN:
                        minimum, _, maximum = self._forced_win_preference_key(proof)
                        choices.append(((minimum, maximum, attack not in ("8", "9"), attack),
                                        tuple(remaining)))
                if not choices:
                    continue
                (minimum, maximum, _, attack), remaining = max(choices)
                group = (minimum, maximum, not king_used)
                route = {
                    "incoming": incoming, "receive": receive, "attack": attack,
                    "all_pass_steps": self._opening_receive_all_pass_steps(
                        remaining, rest_pool, both_royals, king_used,
                    ),
                }
                groups.setdefault(group, []).append(route)
        plans = [
            {"priority": priority, "minimum_score": minimum, "maximum_score": maximum,
             "routes": groups[(minimum, maximum, natural)]}
            for priority, (minimum, maximum, natural) in enumerate(sorted(groups, reverse=True)[:3], 1)
        ]
        return plans or self._provisional_receive_plan_options(hand, pool, both_royals)

    def _provisional_receive_plan_options(self, hand, pool, both_royals):
        """Draft alternatives for hands whose initial score floor is unknown."""
        natural = []
        royal_routes = []
        for receive in sorted(set(hand)):
            king_used = receive in ("8", "9")
            incoming = ([piece for piece in sorted(pool) if piece not in ("1", "2") and pool[piece] > 0]
                        if king_used else [receive] if pool[receive] > 0 else [])
            if not incoming:
                continue
            cards = list(hand)
            cards.remove(receive)
            attacks = [piece for piece in set(cards) if self._forced_win_attack_is_legal(
                tuple(cards), piece, block=None, king_used=king_used, had_both_kings=both_royals,
            )]
            if not attacks:
                continue
            attack = max(attacks, key=lambda piece: (
                not self._forced_win_external_receivers(pool, piece),
                piece not in ("8", "9"), cards.count(piece), POINTS[piece], piece,
            ))
            cards.remove(attack)
            steps = self._opening_receive_all_pass_steps(
                tuple(cards), pool, both_royals, king_used, require_unreceivable=False,
            )
            if not steps:
                continue
            score = self._forced_win_finish_score(*steps[-1])
            routes = [{"incoming": piece, "receive": receive, "attack": attack,
                       "all_pass_steps": steps} for piece in incoming]
            if king_used:
                royal_routes.extend(routes)
            else:
                natural.append((score, routes))
        natural.sort(key=lambda candidate: candidate[0], reverse=True)
        chosen = [routes for _, routes in natural[:2 if royal_routes else 3]]
        if royal_routes:
            chosen.append(royal_routes)
        return [
            {"priority": priority, "minimum_score": None, "maximum_score": None, "routes": routes}
            for priority, routes in enumerate(chosen, 1)
        ]

    def _opening_receive_all_pass_steps(self, hand, pool, both_royals, king_used, *, require_unreceivable=True):
        """Show the follow-up when the first planned attack comes around."""
        @lru_cache(maxsize=None)
        def search(cards):
            choices = []
            for attack in sorted(set(cards)):
                if (require_unreceivable and len(cards) > 2
                        and self._forced_win_external_receivers(pool, attack)):
                    continue
                rest = list(cards)
                rest.remove(attack)
                for block in sorted(set(rest)):
                    if not self._forced_win_attack_is_legal(
                        cards, attack, block=block, king_used=king_used, had_both_kings=both_royals,
                    ):
                        continue
                    next_hand = rest.copy()
                    next_hand.remove(block)
                    if next_hand:
                        followup = search(tuple(next_hand))
                        if followup is None:
                            continue
                        score, tail = followup
                    else:
                        score, tail = self._forced_win_finish_score(block, attack), ()
                    choices.append((score, ((block, attack), *tail)))
            return max(choices, key=lambda row: row[0], default=None)

        result = search(hand)
        return list(result[1]) if result else []

    @staticmethod
    def _active_opening_receive_plans(state, player, tracker):
        if len(state.hands[player]) != 8 or tracker.get("my_attack_count", 0) != 0:
            return None
        return tracker.get("round_strategy", {}).get("opening_receive_plans")

    def _prepared_opening_receive_attacks(self, state, player, tracker, receive):
        opening = self._active_opening_receive_plans(state, player, tracker)
        if receive in ("8", "9") or not opening:
            return set()
        return {
            route["attack"]
            for plan in opening["plans"]
            for route in plan["routes"]
            if (route["incoming"], route["receive"]) == (state.current_attack, receive)
        }

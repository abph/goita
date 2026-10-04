"""Compare an exhausted attack passed to a ready partner with receiving it.

This is a conservative, one-orbit opportunity comparison, not a forced-win proof.
Every root uses the same public-information candidate deals. An opponent able
to intercept is assumed to do so; no partner win is credited in that branch.
"""

from __future__ import annotations

import copy
import random
import time

from goita_ai2.constants import ALL_SEATS


class AllyReachMixin:
    @staticmethod
    def _reach_hand_can_receive(hand, piece):
        return piece in hand or (piece not in ("1", "2") and any(p in hand for p in ("8", "9")))

    def _ally_reach_offer_outcome(self, hands, player, attacker, piece):
        """Enumerate interception, partner receive, and both-pass outcomes.

        Hands here belong to a sampled world, never to the live hidden deal.
        Treating every possible interception as denial is a lower bound on
        the immediate partner opportunity, without clairvoyant partner play.
        """
        enemy = ALL_SEATS[(ALL_SEATS.index(player) + 1) % 4]
        ally = self._ally_of(player)
        intercepted = self._reach_hand_can_receive(hands[enemy], piece)
        ally_can_receive = self._reach_hand_can_receive(hands[ally], piece)
        reaches_ally = not intercepted and ally_can_receive
        both_pass = not intercepted and not ally_can_receive
        return {
            "ally_finish": float(reaches_ally),
            "ally_royal_receive": float(reaches_ally and piece not in hands[ally]),
            "interception": float(intercepted),
            "both_pass": float(both_pass),
            "enemy_finish": float(
                (intercepted and len(hands[enemy]) == 2)
                or (both_pass and attacker != player and len(hands[attacker]) == 2)
            ),
        }

    def _ally_reach_handoff_action(self, state, player, actions):
        self.last_ally_reach_comparison = None
        tr = self._track.get(id(state))
        if (
            not self.ALLY_REACH_HANDOFF_ENABLED
            or tr is None or state.phase != "receive" or state.turn != player
            or state.current_attack not in ("3", "4", "5", "6", "7")
            or state.attacker is None or self._same_team(state.attacker, player)
            or state.next_player(state.attacker) != player
            or ("pass", None, None) not in actions
        ):
            return None
        ally = self._ally_of(player)
        enemy = state.next_player(player)
        hand = list(state.hands[player])
        piece = state.current_attack
        if (
            8 - int(tr.get("ally_consumed_count", 0)) != 2
            or len(state.hands[ally]) != 2
            or not any(action[0] == "receive" for action in actions)
            # Do not rely on a finite sample missing an immediate enemy win.
            or len(state.hands[enemy]) <= 2 or len(state.hands[state.attacker]) <= 2
            or int(tr.get("public_seen_counts", {}).get(piece, 0)) + hand.count(piece) < self._piece_total(piece)
        ):
            return None
        # The matching piece is exhausted outside our hand. A remaining royal
        # is the only way the partner (or intervening enemy) can receive it.
        if not any((self._estimated_piece_hold_risk(tr, ally, royal) or 0) > 0 for royal in ("8", "9")):
            return None
        unknown = tr.get("unknown_piece_pool", {})
        future_pieces = [p for p, count in unknown.items() if int(count) > 0]
        if not future_pieces or not all(self._reach_hand_can_receive(hand, p) for p in future_pieces):
            return None

        # Enumerate ALL legal receive + attack pairs, not only the rank rule's
        # preferred receive. No policy/AI is called on a hidden world.
        routes = []
        for action in actions:
            if action[0] != "receive":
                continue
            after = copy.deepcopy(state)
            after.apply_receive(player, action[1])
            for followup in after.legal_actions(player):
                if followup[0] == "attack":
                    routes.append((action, followup))
        if not routes:
            return None

        rng = random.Random(self._timed_search_public_seed(state, player, tr))
        samples = []
        deadline = time.monotonic() + self.ALLY_REACH_HANDOFF_MAX_SECONDS
        for _ in range(self.ALLY_REACH_HANDOFF_SAMPLE_COUNT):
            if time.monotonic() >= deadline:
                break
            sampled = self._timed_search_sample_state(state, player, tr, rng)
            if sampled is not None:
                samples.append(sampled)
        if len(samples) < self.ALLY_REACH_HANDOFF_MIN_SAMPLES:
            return None
        try:
            information = self._build_information_set(state, player, tr, samples)
        except ValueError:
            return None
        if information.effective_candidate_count < self.ALLY_REACH_HANDOFF_MIN_EFFECTIVE_SAMPLES:
            return None

        offers = [(None, None)] + routes
        comparisons = []
        for receive, attack in offers:
            remaining = list(hand)
            if receive is not None:
                remaining.remove(receive[1])
                remaining.remove(attack[2])
            keeps_cover = all(self._reach_hand_can_receive(remaining, p) for p in future_pieces)
            values = dict.fromkeys(("ally_finish", "ally_royal_receive", "interception", "both_pass", "enemy_finish"), 0.0)
            for candidate in information.candidates:
                hands = dict(candidate.prediction.opponent_hands)
                hands[player] = hand
                outcome = self._ally_reach_offer_outcome(
                    hands, player, state.attacker if receive is None else player,
                    piece if attack is None else attack[2],
                )
                for key, value in outcome.items():
                    values[key] += candidate.probability * value
            comparisons.append({"receive": receive, "attack": attack, "keeps_cover": keeps_cover, **values})
        passed = comparisons[0]
        covered_receives = [row for row in comparisons[1:] if row["keeps_cover"]]
        best_covered_chance = max((row["ally_finish"] for row in covered_receives), default=0.0)
        best_any_chance = max(row["ally_finish"] for row in comparisons[1:])
        advantage = passed["ally_finish"] - best_covered_chance
        adopted = (
            passed["enemy_finish"] <= 1e-9
            and passed["ally_finish"] >= self.ALLY_REACH_HANDOFF_MIN_CHANCE
            and advantage >= self.ALLY_REACH_HANDOFF_MIN_ADVANTAGE
            # Receiving with our royal and returning the same exhausted piece
            # can offer exactly the same chance, but sacrifices defensive cover.
            # Do not pass if any receive route offers a greater opportunity.
            and passed["ally_finish"] + 1e-9 >= best_any_chance
        )
        self.last_ally_reach_comparison = {
            "ally": ally, "piece": piece, "samples": len(samples),
            "effective_samples": information.effective_candidate_count,
            "source": "public_information_candidates",
            "royal_known": any(
                int((self._estimated_current_piece(tr, ally, royal) or {}).get("min", 0)) > 0
                for royal in ("8", "9")
            ),
            "defensive_cover_preserved": True, "adopted": adopted,
            "comparisons": comparisons,
        }
        if not adopted:
            return None
        self._set_decision_reason("score_fallback")
        self._set_score_fallback_detail(
            f"pass_ally_reach_royal_{ally}_piece_{piece}_"
            f"chance_{round(passed['ally_finish'] * 100)}_"
            f"receive_{round(best_covered_chance * 100)}"
        )
        return ("pass", None, None)

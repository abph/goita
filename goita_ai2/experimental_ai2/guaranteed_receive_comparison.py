"""Compare publicly proven receives and waits in the same sampled deals."""

from dataclasses import dataclass
from time import perf_counter

from goita_ai2.constants import ALL_SEATS, POINTS
from goita_ai2.current_ai.endgame import ForcedWinStatus


@dataclass(frozen=True)
class GuaranteedRoute:
    actions: tuple
    minimum: float
    maximum: float


class GuaranteedReceiveComparisonMixin:
    GUARANTEED_RECEIVE_SAMPLES = 24
    GUARANTEED_RECEIVE_SECONDS = 0.20

    def _proven_receive_routes(self, state, player, actions, tracker):
        large_hand = len(state.hands[player]) > self.EXACT_FORCED_WIN_MAX_HAND
        for receive in actions:
            if receive[0] != "receive":
                continue
            if large_hand and receive[1] not in tracker.get("large_hand_receive_proofs", {}):
                continue
            after = self._timed_search_apply(state, player, receive)
            seen = dict(tracker["public_seen_counts"])
            seen[receive[1]] += 1
            self._track[id(after)] = {"public_seen_counts": seen}
            try:
                pool = self._public_unknown_piece_pool(after, player, self._track[id(after)])
                for attack in after.legal_actions(player):
                    if large_hand and (
                        pool is None or self._forced_win_external_receivers(pool, attack[2])
                    ):
                        continue
                    proof = self._forced_win_result_after_attack_action(after, player, attack)
                    if proof.status == ForcedWinStatus.PROVEN:
                        minimum, _, maximum = self._forced_win_preference_key(proof)
                        yield GuaranteedRoute((receive, attack), minimum, maximum)
            finally:
                self._track.pop(id(after), None)

    def _proven_wait_branches(self, state, player, tracker):
        # Use the existing one-cycle coverage proof. No other seat may go
        # out before our next turn, and we must cover every public possibility.
        hand = tuple(sorted(state.hands[player]))
        if (state.attacker is None or self._same_team(state.attacker, player)
                or len(hand) != 4 or tracker.get("my_attack_count") != 2
                or not any(piece in ("8", "9") for piece in hand)):
            return []
        other_sizes = [len(state.hands[seat]) for seat in ALL_SEATS if seat != player]
        if min(other_sizes) <= 2:
            return []
        pool = self._public_unknown_piece_pool(state, player, tracker)
        if pool is None or not any(
            piece not in ("8", "9") and not self._forced_win_external_receivers(pool, piece)
            for piece in hand
        ):
            return []
        branches = []
        memo = {}
        for piece in sorted(pool):
            if pool[piece] <= 0:
                continue
            remaining = pool.copy()
            remaining[piece] -= 1
            proof = self._forced_win_resolve_external_return(
                hand, remaining, returned_attack=piece,
                king_used=state.king_block_used > 0,
                had_both_kings=state.had_both_kings[player],
                depth=min(12, self.EXACT_FORCED_WIN_MAX_DEPTH),
                external_cards_used=0, minimum_enemy_hand=min(other_sizes) - 2, memo=memo,
            )
            if proof.status != ForcedWinStatus.PROVEN:
                return []
            minimum, _, maximum = self._forced_win_preference_key(proof)
            branches.append({"incoming": piece, "minimum_score": minimum, "maximum_score": maximum})
        return branches

    def _compare_guaranteed_receive_routes(self, state, player, actions):
        if state.phase != "receive" or len(actions) < 2:
            return None
        tracker = self._track[id(state)]
        tracker.pop("guaranteed_receive_comparison", None)
        routes = list(self._proven_receive_routes(state, player, actions, tracker))
        if not routes:
            return None
        pass_action = ("pass", None, None)
        branches = self._proven_wait_branches(state, player, tracker) if pass_action in actions else []
        if branches:
            routes.append(GuaranteedRoute(
                (pass_action,), min(branch["minimum_score"] for branch in branches),
                max(branch["maximum_score"] for branch in branches),
            ))
        if len(routes) < 2:
            return None

        estimates, samples = self._paired_guaranteed_receive_scores(state, player, tracker, routes)
        best_floor = max(route.minimum for route in routes)
        eligible = [route for route in routes if route.minimum == best_floor]
        # A proven improvement can decide a tie even when sampling is
        # unavailable or failed to visit the higher paying branch.
        receive_ceiling = max(route.maximum for route in routes if len(route.actions) == 2)
        wait_target = max((branch["minimum_score"] for branch in branches), default=0)
        def fallback_key(route):
            legacy_wait = len(route.actions) == 1 and wait_target > receive_ceiling
            large_royal = (len(state.hands[player]) == 8 and len(route.actions) == 2
                           and route.actions[1][2] in ("8", "9"))
            return legacy_wait, route.maximum, large_royal, len(route.actions) == 2

        comparable = all(route in estimates for route in eligible)
        def comparison_key(route):
            estimate = round(estimates[route], 1) if comparable else 0
            return (estimate, *fallback_key(route))

        selected = max(eligible, key=comparison_key)
        if len(eligible) == 1:
            basis = "minimum_score"
        elif comparable and any(round(estimates[route], 1) < round(estimates[selected], 1) for route in eligible):
            basis = "expected_score"
        else:
            basis = "proven_score_tiebreak"
        tracker["guaranteed_receive_comparison"] = {
            "basis": basis, "samples": samples, "selected": selected.actions,
            "estimate_source": "shared_sampled_deals_and_local_policy",
            "candidates": [{"actions": route.actions, "minimum_score": route.minimum,
                            "maximum_score": route.maximum, "expected_score": estimates.get(route)}
                           for route in routes],
        }
        tracker.pop("pending_large_hand_receive_proof", None)
        tracker["pending_inferred_endgame_attack"] = None
        if selected.actions[0] == pass_action:
            receive = max((route for route in routes if len(route.actions) == 2), key=lambda route: route.minimum)
            tracker["last_royal_wait_comparison"] = {
                "source": "public_information", "selected": "pass",
                "receive_action": receive.actions[0], "receive_minimum_score": receive.minimum,
                "wait_minimum_score": selected.minimum, "wait_target_score": wait_target,
                "branches": branches,
            }
            self._set_score_fallback_detail(
                f"pass_full_receive_cover_public_route_safe_{int(selected.minimum)}_target_{int(wait_target)}"
            )
        else:
            receive, attack = selected.actions
            after = self._timed_search_apply(state, player, receive)
            seen = dict(tracker["public_seen_counts"])
            seen[receive[1]] += 1
            key = self._guaranteed_receive_followup_key(after, player, seen)
            tracker["pending_guaranteed_receive_attack"] = (key, attack, len(state.hands[player]) == 8)
            large_proof = tracker.get("large_hand_receive_proofs", {}).get(receive[1])
            if large_proof:
                tracker["pending_large_hand_receive_proof"] = {"key": key, "action": attack}
            detail = "receive_win_after" if len(after.hands[player]) == 1 else "receive_tsume_after"
            self._set_score_fallback_detail(detail)
        self._set_decision_reason("score_fallback")
        return selected.actions[0]

    @staticmethod
    def _guaranteed_receive_followup_key(state, player, seen):
        return (tuple(sorted(state.hands[player])), tuple(sorted(seen.items())),
                state.king_block_used, tuple(len(state.hands[s]) for s in ALL_SEATS))

    def _select_rule_based_action(self, state, player, actions):
        tracker = self._track.get(id(state), {})
        tracker.pop("guaranteed_receive_comparison", None)
        pending = tracker.pop("pending_guaranteed_receive_attack", None)
        if pending:
            key, attack, large_hand = pending
            tracker.pop("pending_large_hand_receive_proof", None)
            if ((not large_hand or self.LARGE_HAND_RECEIVE_PUBLIC_PROOF_ENABLED)
                    and state.phase == "attack" and state.attacker == player and state.current_attack is None
                    and attack in actions
                    and key == self._guaranteed_receive_followup_key(state, player, tracker["public_seen_counts"])):
                proof = self._forced_win_result_after_attack_action(state, player, attack)
                if proof.status == ForcedWinStatus.PROVEN:
                    self.last_attack_candidate_scores = []
                    self.last_attack_intent_comparison = None
                    self._commit_timed_search_action(state, player, attack)
                    self._set_decision_reason("win_now" if len(state.hands[player]) == 1 else "tsume")
                    self._set_score_fallback_detail(f"high_score_{int(proof.minimum_score)}")
                    return attack
        return super()._select_rule_based_action(state, player, actions)

    def _finalize_attack_candidate_snapshot(self, actions, chosen):
        super()._finalize_attack_candidate_snapshot(actions, chosen)
        tracker = self._track.get(getattr(self, "_live_plan_review_state_id", None), {})
        comparison = tracker.get("guaranteed_receive_comparison")
        if comparison and comparison["selected"][0] == chosen:
            self.last_attack_candidate_snapshot["guaranteed_receive_comparison"] = comparison

    def _paired_guaranteed_receive_scores(self, state, player, tracker, routes):
        fixed = {route: route.minimum for route in routes if route.minimum == route.maximum}
        variable = [route for route in routes if route not in fixed]
        if not variable:
            return fixed, 0
        deadline = perf_counter() + self.GUARANTEED_RECEIVE_SECONDS
        samples = self._timed_search_generate_sample_states(state, player, tracker, self.GUARANTEED_RECEIVE_SAMPLES)
        try:
            _, worlds = self._information_set_search_worlds(state, player, tracker, samples)
        except ValueError:
            return fixed, 0
        totals = dict.fromkeys(variable, 0.0)
        mass = 0.0
        count = 0
        for world in worlds:
            if perf_counter() >= deadline:
                break
            values = {}
            for route in variable:
                value = self._guaranteed_receive_rollout(world.state, player, tracker, route, deadline)
                if value is None or not route.minimum <= value <= route.maximum:
                    break
                values[route] = value
            if len(values) != len(variable):
                continue
            for route, value in values.items():
                totals[route] += world.probability * value
            mass += world.probability
            count += 1
        if count < 6 or mass < 0.7:
            return fixed, count
        return {**fixed, **{route: total / mass for route, total in totals.items()}}, count

    def _guaranteed_receive_rollout(self, state, player, tracker, route, deadline):
        current = state
        seen = dict(tracker["public_seen_counts"])
        team = self._timed_search_team(player)
        actions = list(route.actions)
        for _ in range(128):
            if current.finished:
                return current.team_score[team] - state.team_score[team]
            if perf_counter() >= deadline:
                return None
            actor = current.turn
            legal = current.legal_actions(actor)
            if actions:
                action = actions.pop(0)
            elif len(legal) == 1:
                action = legal[0]
            elif actor == player:
                self._track[id(current)] = {"public_seen_counts": seen}
                try:
                    proofs = []
                    for move in legal:
                        if move[0] == "pass":
                            continue
                        prove = (self._forced_win_result_after_receive_action if move[0] == "receive"
                                 else self._forced_win_result_after_attack_action)
                        proof = prove(current, player, move)
                        if proof.status == ForcedWinStatus.PROVEN:
                            proofs.append((self._forced_win_preference_key(proof), move))
                finally:
                    self._track.pop(id(current), None)
                if not proofs:
                    return None
                (minimum, _, maximum), action = max(proofs, key=lambda entry: entry[0])
                if minimum == maximum:
                    return minimum
            else:
                action = self._guaranteed_receive_simulated_move(current, actor, legal)
            kind, block, attack = action
            if kind == "receive" or (kind == "attack_after_block" and actor == player):
                seen[block] += 1
            if attack is not None:
                seen[attack] += 1
            current = self._timed_search_apply(current, actor, action)
        return None

    def _guaranteed_receive_simulated_move(self, state, actor, actions):
        """Use only the acting seat's hand, visible sizes and the current attack."""
        hand = state.hands[actor]
        def priority(action):
            kind, block, attack = action
            if kind == "pass":
                return 0.0
            if kind == "receive":
                if len(hand) == 2:
                    return 1000.0
                if self._same_team(actor, state.attacker):
                    return -1.0
                return -1.0 if block in ("8", "9") else 1.0
            if len(hand) == 1 + int(block is not None):
                return 1000.0 + self._forced_win_finish_score(block, attack)
            priority = 20 * (hand.count(attack) - 1) + 0.3 * POINTS[attack]
            if attack in ("8", "9"):
                priority -= 30
            if block is not None:
                priority -= 0.15 * POINTS[block] + 10 * (hand.count(block) - 1)
            return priority
        return max(actions, key=priority)

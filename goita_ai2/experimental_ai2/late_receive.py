"""Compare late receipts with their continuations, rather than a fixed pass tax.

Only AI2 uses this mixin. Hypothetical deals are weighted public-information
worlds, never the actual opposing hands. A sampled route is not a win proof.
"""
from __future__ import annotations

import copy
import math
import time
from collections import Counter

from goita_ai2.constants import POINTS
from goita_ai2.current_ai.information_set_policy import InformationSetPolicy
from goita_ai2.current_ai.information_set_search import (
    InformationSetSearchCancelled, InformationSetSearchDeadline,
    InformationSetSearchOutcome, InformationSetSearchWorld,
)


class LateReceiveMixin:
    LATE_RECEIVE_ROUTE_SAMPLES = 48
    LATE_RECEIVE_ROUTE_DEPTH = 10
    LATE_RECEIVE_ROUTE_SECONDS = 2.0
    LATE_RECEIVE_ROUTE_MARGIN = 200.0

    def _is_late_enemy_receive(self, state, player, tracker):
        return bool(
            tracker and state.phase == "receive" and state.attacker is not None
            and state.current_attack not in (None, "1")
            and not self._same_team(state.attacker, player)
            and int(tracker.get("enemy_attack_counts", {}).get(state.attacker, 0)) >= 2
        )

    def _score_receive_phase(self, state, player, action_type, block):
        value = super()._score_receive_phase(state, player, action_type, block)
        if (value >= 1e8 or not self._is_late_enemy_receive(
                state, player, self._track.get(id(state)))):
            return value
        # Cancel only the old late-non-shi offsets. Proven finishes, immediate
        # threats, shi tactics, public evidence and the other score terms stay.
        if action_type == "pass":
            return value + 10000.0
        if action_type == "receive" and block is not None:
            return value - (1000.0 if block in ("8", "9") else 10000.0)
        return value

    @staticmethod
    def _late_receive_followup_key(state, player):
        return (tuple(sorted(state.hands[player])), state.turn, state.phase,
                state.attacker, state.current_attack, int(state.king_block_used),
                tuple(len(state.hands[seat]) for seat in "ABCD"))

    def _late_public_royal_bridge_score(self, state, player, tracker):
        """Prove a natural receive -> unreceivable royal -> final hidden pair.

        This is a shortcut for a completed continuation, not a bonus for pass.
        Counts are conservative public counts; sampled enemy cards are unused.
        """
        hand = list(state.hands[player])
        royals = [piece for piece in hand if piece in ("8", "9")]
        if (state.finished or state.turn != player or state.phase != "receive"
                or len(hand) != 4 or len(royals) != 1 or not state.king_block_used
                or state.current_attack not in hand or state.current_attack in ("8", "9")):
            return None
        pool = self._public_unknown_piece_pool(state, player, tracker)
        if pool is None or pool.get("8", 0) or pool.get("9", 0):
            return None
        hand.remove(state.current_attack)
        royal = royals[0]
        if not self._forced_win_attack_is_legal(
                tuple(hand), royal, block=None, king_used=True,
                had_both_kings=bool(state.had_both_kings.get(player, False))):
            return None
        hand.remove(royal)
        return float(self._best_pair_finish_score_from_hand(hand))

    def _late_other_action_value(self, state, actor, action, tracker):
        """Forecast a standard player's plan using their own cards and public actions.

        In particular, an opponent must not pick the piece missing from our
        private hand merely because a materialized sample contains that hand.
        """
        kind, block, attack = action
        hand = state.hands[actor]
        if kind == "pass":
            return 25.0 if self._same_team(actor, state.attacker) else 0.0
        if kind == "receive":
            after = self._timed_search_apply(state, actor, action)
            if any(len(after.hands[actor]) == 1 and move[0] == "attack"
                   for move in after.legal_actions(actor)):
                return 10000.0 + max(POINTS.get(move[2], 0) for move in after.legal_actions(actor))
            strength = max(Counter(after.hands[actor]).values(), default=0)
            return (22.0 + max(0, strength - 1) * 8.0
                    - (35.0 if block in ("8", "9") else 0.0)
                    - (35.0 if self._same_team(actor, state.attacker) else 0.0))
        remaining = list(hand)
        if block is not None:
            remaining.remove(block)
        remaining.remove(attack)
        if not remaining:
            return 10000.0 + self._forced_win_finish_score(block, attack)
        counts = Counter(hand)
        public_attacks = (tracker or {}).get("public_hand_models", {}).get(actor, {}).get("attacks", {})
        return (POINTS.get(attack, 0) * 0.25 + max(0, counts[attack] - 1) * 12.0
                + len(set(remaining)) * 3.0
                + (self._best_pair_finish_score_from_hand(remaining) * 0.3 if len(remaining) == 2 else 0.0)
                + (30.0 if public_attacks.get(attack, 0) else 0.0)
                - POINTS.get(block, 0) * 0.15
                - (30.0 if attack in ("8", "9") else 0.0)
                + ((25.0 if counts["1"] >= 3 else -10.0) if attack == "1" else 0.0))

    def _late_immediate_finish(self, state):
        """Resolve an actor's legal one/two-action finish, including at a leaf."""
        if state.finished:
            return state
        actor = state.turn
        if len(state.hands[actor]) > 2:
            return None
        finishes = []
        for action in state.legal_actions(actor):
            if action[0] == "pass":
                continue
            after = self._timed_search_apply(state, actor, action)
            if after.finished:
                finishes.append(after)
            elif action[0] == "receive":
                for attack in after.legal_actions(actor):
                    final = self._timed_search_apply(after, actor, attack)
                    if final.finished:
                        finishes.append(final)
        if not finishes:
            return None
        team = self._timed_search_team(actor)
        return max(finishes, key=lambda final: final.team_score.get(team, 0))

    def _late_other_action_distribution(self, state, actor, tracker):
        scored = sorted(((self._late_other_action_value(state, actor, action, tracker), action)
                         for action in state.legal_actions(actor)), reverse=True)[:2]
        if not scored:
            return []
        peak = scored[0][0]
        choices = [(action, math.exp(max(-60.0, (score - peak) / 12.0))) for score, action in scored]
        choices = [(action, weight) for action, weight in choices if weight >= 0.01]
        total = sum(weight for _action, weight in choices)
        return [(action, weight / total) for action, weight in choices]

    def _late_predicted_other_bundle(
            self, root_state, worlds, root_player, root_information_set, baseline_scores,
            depth, deadline, stats, policy, tracker, public_history, cancel_event,
            tracker_cache, action_model_cache):
        groups = {}
        for world in worlds:
            key = self._information_set_decision_key(
                world.state, world.state.turn, root_information_set.key, public_history)
            groups.setdefault(key.digest, []).append(world)
        values = {}
        context = self._late_receive_route_context
        for group in groups.values():
            sample = group[0].state
            actor = sample.turn
            projected = self._information_set_project_tracker(tracker, root_player, public_history, sample)
            choices = self._late_other_action_distribution(sample, actor, projected)
            if not choices:
                values.update({world.index: self._timed_search_static_value(
                    world.state, root_player, baseline_scores) for world in group})
                continue
            observations = {}
            for action, weight in choices:
                observed = self._information_set_observed_action(root_player, actor, action)
                observations.setdefault(observed, []).append((action, weight))
            group_values = {world.index: 0.0 for world in group}
            for observed, alternatives in observations.items():
                children, parent_ids = [], {}
                for world in group:
                    for action, weight in alternatives:
                        index = context["next_world_index"]
                        context["next_world_index"] += 1
                        parent_ids[index] = (world.index, weight)
                        child = self._timed_search_apply(world.state, actor, action)
                        children.append(InformationSetSearchWorld(
                            index, child, world.probability * weight, world.confidence))
                outcome = self._information_set_search_bundle(
                    root_state, tuple(children), root_player, root_information_set,
                    baseline_scores, depth - 1, deadline, stats, policy, tracker,
                    tuple(public_history) + (observed,), cancel_event, tracker_cache, action_model_cache)
                for index, value in outcome.world_values:
                    parent, weight = parent_ids[index]
                    group_values[parent] += weight * value
            values.update(group_values)
        return InformationSetSearchOutcome(
            self._information_set_weighted_value(worlds, values), tuple(sorted(values.items())), len(policy))

    def _information_set_search_bundle(
            self, root_state, worlds, root_player, root_information_set,
            baseline_scores, depth, deadline, stats, policy, tracker,
            public_history=(), cancel_event=None, tracker_cache=None,
            action_model_cache=None):
        context = getattr(self, "_late_receive_route_context", None)
        if context is not None and worlds:
            if cancel_event is not None and cancel_event.is_set():
                raise InformationSetSearchCancelled()
            if time.perf_counter() >= deadline:
                raise InformationSetSearchDeadline("time_limit")
            terminal_values, ongoing = {}, []
            for world in worlds:
                final = self._late_immediate_finish(world.state)
                if final is None:
                    ongoing.append(world)
                else:
                    terminal_values[world.index] = self._timed_search_terminal_value(final, root_player, baseline_scores)
            if terminal_values:
                stats["nodes"] += len(terminal_values)
                if stats["nodes"] > stats["max_nodes"]:
                    raise InformationSetSearchDeadline("node_limit")
                if ongoing:
                    rest = self._information_set_search_bundle(
                        root_state, tuple(ongoing), root_player, root_information_set,
                        baseline_scores, depth, deadline, stats, policy, tracker,
                        public_history, cancel_event, tracker_cache, action_model_cache)
                    terminal_values.update(rest.values_dict())
                return InformationSetSearchOutcome(
                    self._information_set_weighted_value(worlds, terminal_values),
                    tuple(sorted(terminal_values.items())), len(policy))
            sample = worlds[0].state
            if (depth > 0 and all(not world.state.finished and world.state.turn != root_player
                                  for world in worlds)):
                cost = max(1, len(worlds))
                if stats["nodes"] + cost > stats["max_nodes"]:
                    raise InformationSetSearchDeadline("node_limit")
                stats["nodes"] += cost
                return self._late_predicted_other_bundle(
                    root_state, worlds, root_player, root_information_set, baseline_scores,
                    depth, deadline, stats, policy, tracker, public_history, cancel_event,
                    tracker_cache, action_model_cache)
            key = self._late_receive_followup_key(sample, root_player)
            if all(not world.state.finished and self._late_receive_followup_key(
                    world.state, root_player) == key for world in worlds):
                projected = self._information_set_project_tracker(
                    tracker, root_player, public_history, sample)
                cache_key = (key, tuple(sorted(projected.get("public_seen_counts", {}).items())))
                cache = context["bridge_cache"]
                if cache_key not in cache:
                    cache[cache_key] = self._late_public_royal_bridge_score(sample, root_player, projected)
                score = cache[cache_key]
                if score is not None:
                    cost = max(1, len(worlds))
                    if stats["nodes"] + cost > stats["max_nodes"]:
                        raise InformationSetSearchDeadline("node_limit")
                    stats["nodes"] += cost
                    context["bridge_hits"] += 1
                    context["bridge_scores"].add(score)
                    # Exactly the same scale as an actual terminal team win.
                    value = 100000.0 + score * 500.0
                    return InformationSetSearchOutcome(
                        value, tuple((world.index, value) for world in worlds), len(policy))
        return super()._information_set_search_bundle(
            root_state, worlds, root_player, root_information_set, baseline_scores,
            depth, deadline, stats, policy, tracker, public_history, cancel_event,
            tracker_cache, action_model_cache)

    def _late_receive_hypotheses(self, state, player, tracker, worlds):
        """Describe a bounded set of alternative attacker hands and hidden pairs."""
        hands = Counter()
        for world in worlds:
            hands[tuple(sorted(world.state.hands[state.attacker]))] += world.probability
        rows = []
        for hand, probability in hands.most_common(6):
            sample = copy.copy(worlds[0].state)
            sample.hands = dict(sample.hands)
            sample.hands[state.attacker] = list(hand)
            sample.turn, sample.phase = state.attacker, "attack"
            plans = []
            for action in sample.legal_actions(state.attacker):
                if action[0] != "attack_after_block":
                    continue
                remaining = list(hand)
                remaining.remove(action[1])
                remaining.remove(action[2])
                plans.append({"block": action[1], "attack": action[2], "remaining": sorted(remaining)})
            rows.append({"hand": list(hand), "probability": round(probability, 6), "plans": plans})
        # A lower bound in the weighted candidate set: in these worlds nobody
        # after our pass can legally receive before the attacker acts again.
        forced_return, estimated_return = 0.0, 0.0
        for world in worlds:
            cursor = self._timed_search_apply(world.state, player, ("pass", None, None))
            probability, forced = 1.0, True
            while cursor.turn != state.attacker:
                actor = cursor.turn
                if any(action[0] == "receive" for action in cursor.legal_actions(actor)):
                    forced = False
                choices = self._late_other_action_distribution(cursor, actor, tracker)
                probability *= sum(weight for action, weight in choices if action[0] == "pass")
                cursor = self._timed_search_apply(cursor, actor, ("pass", None, None))
            if forced:
                forced_return += world.probability
            estimated_return += world.probability * probability
        return rows, round(forced_return, 6), round(estimated_return, 6)

    def _late_royal_reserve_wait_available(self, state, player, tracker, actions):
        hand = tuple(state.hands[player])
        royals = [piece for piece in hand if piece in ("8", "9")]
        if len(royals) != 1 or not state.king_block_used:
            return False
        remaining = list(hand)
        remaining.remove(royals[0])
        if len(set(remaining)) != 3:
            return False  # A real repeat attack is evaluated on its own merits.
        if any(action[0] == "receive" and action[1] not in ("8", "9") for action in actions):
            return False
        pool = self._public_unknown_piece_pool(state, player, tracker)
        if pool is None or pool.get("8", 0) or pool.get("9", 0):
            return False
        incoming = [piece for piece, count in pool.items() if count > 0]
        return (all(self._forced_win_receive_options(hand, piece) for piece in incoming)
                and sum(piece in remaining for piece in incoming) >= 2)

    def _compare_late_receive_routes(self, state, player, actions, budget_seconds=None):
        tracker = self._track.get(id(state))
        if (not self.TIME_SEARCH_ENABLED or not self._is_late_enemy_receive(state, player, tracker)
                or getattr(self, "_time_search_cancel_event", None) is not None
                or len(state.hands[player]) != 4
                or not any(piece in ("8", "9") for piece in state.hands[player])
                or ("pass", None, None) not in actions
                or not any(action[0] == "receive" for action in actions)
                or any(len(state.hands[seat]) <= 2 for seat in "ABCD"
                       if not self._same_team(seat, player))):
            return None
        routes = [(("pass", None, None),)]
        for receive in actions:
            if receive[0] == "receive":
                after = self._timed_search_apply(state, player, receive)
                routes.extend((receive, attack) for attack in after.legal_actions(player))
        if len(routes) < 2:
            return None
        started = time.perf_counter()
        budget = min(self.LATE_RECEIVE_ROUTE_SECONDS, self.TIME_SEARCH_HARD_MAX_SECONDS,
                     self.TIME_SEARCH_HARD_MAX_SECONDS if budget_seconds is None else max(0.0, budget_seconds))
        if budget <= 0:
            return None
        deadline = started + budget
        stats = {"nodes": 0, "max_nodes": min(60000, int(self.TIME_SEARCH_MAX_NODES))}
        context = {"bridge_cache": {}, "bridge_hits": 0, "bridge_scores": set(), "next_world_index": 1000000}
        previous = getattr(self, "_late_receive_route_context", None)
        self._late_receive_route_context = context
        try:
            samples = self._timed_search_sample_states(state, player, tracker, self.LATE_RECEIVE_ROUTE_SAMPLES)
            information_set, worlds = self._information_set_search_worlds(state, player, tracker, samples)
            if len(worlds) < 8:
                return None
            values, world_values = {}, {}
            for route in routes:
                children = []
                for world in worlds:
                    child = world.state
                    for action in route:
                        child = self._timed_search_apply(child, player, action)
                    children.append(InformationSetSearchWorld(
                        world.index, child, world.probability, world.confidence))
                history = tuple(self._information_set_observed_action(player, player, action) for action in route)
                result = self._information_set_search_bundle(
                    state, tuple(children), player, information_set, dict(state.team_score),
                    self.LATE_RECEIVE_ROUTE_DEPTH - len(route), deadline, stats,
                    InformationSetPolicy(), tracker, history,
                    getattr(self, "_time_search_cancel_event", None), {}, {})
                values[route] = result.value
                world_values[route] = result.values_dict()
            hypotheses, forced_return, estimated_return = self._late_receive_hypotheses(state, player, tracker, worlds)
        except (InformationSetSearchDeadline, InformationSetSearchCancelled, TypeError, ValueError):
            return None  # Partial routes are never compared with completed ones.
        finally:
            self._late_receive_route_context = previous
        best_receive = max(routes[1:], key=lambda route: values[route])
        pass_value = values[routes[0]]
        receive_value = values[best_receive]
        best = routes[0] if pass_value > receive_value else best_receive
        differences = [(world.probability, world_values[routes[0]][world.index]
                        - world_values[best_receive][world.index]) for world in worlds]
        mean_difference = sum(weight * value for weight, value in differences)
        effective_samples = information_set.effective_candidate_count
        variance = sum(weight * (value - mean_difference) ** 2 for weight, value in differences)
        # This is dispersion of the paired hypothetical worlds, not a calibrated
        # confidence interval or a percentage chance of winning a real game.
        dispersion_band = 2.0 * math.sqrt(variance / max(1.0, effective_samples))
        reserve_wait = bool(
            information_set.confidence >= 0.5 and effective_samples >= 8
            and abs(pass_value - receive_value) <= dispersion_band
            and self._late_royal_reserve_wait_available(state, player, tracker, actions))
        if reserve_wait:
            best = routes[0]
        return {"source": "weighted_public_information_routes", "proven": False,
                "attacker": state.attacker,
                "chosen": best, "margin": abs(pass_value - receive_value),
                "pass_value": round(pass_value, 3), "receive_value": round(receive_value, 3),
                "selection_basis": "royal_reserve_within_sample_dispersion" if reserve_wait else "paired_route_value",
                "dispersion_band": round(dispersion_band, 3),
                "effective_samples": round(effective_samples, 3),
                "routes": [{"actions": list(route), "value": round(values[route], 3)} for route in routes],
                "samples": len(worlds), "depth": self.LATE_RECEIVE_ROUTE_DEPTH,
                "confidence": round(information_set.confidence, 6),
                "reliable": information_set.confidence >= 0.5 and effective_samples >= 8,
                "attacker_hypotheses": hypotheses,
                "forced_round_return_mass": forced_return,
                "round_return_estimate": estimated_return,
                "round_return_basis": "uncalibrated_actor_hand_and_public_action_model",
                "public_bridge_hits": context["bridge_hits"],
                "public_bridge_scores": sorted(context["bridge_scores"]),
                "nodes": stats["nodes"], "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)}

    def _record_generic_response_evidence(
            self, state, player, actions, baseline_action, selected_action, **evidence):
        deferred = getattr(self, "_late_receive_deferred_records", None)
        if isinstance(deferred, list):
            deferred.append((state, player, list(actions), baseline_action, selected_action, evidence))
            return False
        return super()._record_generic_response_evidence(
            state, player, actions, baseline_action, selected_action, **evidence)

    def _select_action_with_measurement(self, state, player, actions):
        previous = getattr(self, "_late_receive_deferred_records", None)
        eligible = (self._is_late_enemy_receive(state, player, self._track.get(id(state)))
                    and len(state.hands[player]) == 4
                    and any(piece in ("8", "9") for piece in state.hands[player]))
        self._late_receive_deferred_records = [] if eligible else previous
        try:
            chosen = self._select_late_receive_action(state, player, actions)
            if eligible:
                records = self._late_receive_deferred_records or []
                self._late_receive_deferred_records = previous
                for recorded_state, actor, legal, baseline, proposed, evidence in records:
                    # Covers both newly searched and cached response plans.
                    if proposed == chosen:
                        super()._record_generic_response_evidence(
                            recorded_state, actor, legal, baseline, chosen, **evidence)
            return chosen
        finally:
            self._late_receive_deferred_records = previous

    def _select_late_receive_action(self, state, player, actions):
        started = time.perf_counter()
        tracker = self._track.get(id(state), {})
        pending = tracker.pop("pending_late_receive_route", None)
        if (pending and pending["key"] == self._late_receive_followup_key(state, player)
                and pending["action"] in actions):
            chosen = pending["action"]
            self._commit_timed_search_action(state, player, chosen)
            self._set_decision_reason("time_search")
            self._set_score_fallback_detail(f"late_receive_followup_{chosen[2]}")
            self.last_rule_search_authority = "strong"
            comparison = tracker.get("last_late_receive_comparison", {})
            self.last_attack_candidate_scores = [
                {"action": row["actions"][1], "score": row["value"],
                 "candidate_role": "paired_receive_followup"}
                for row in comparison.get("routes", ())
                if len(row["actions"]) == 2 and row["actions"][0] == comparison["chosen"][0]
            ]
            return chosen
        before = copy.deepcopy(tracker)
        chosen = super()._select_action_with_measurement(state, player, actions)
        authority = self._rule_search_authority(self.last_decision_reason, self.last_score_fallback_detail)
        if authority != "ordinary" or self.last_rule_search_authority not in (None, "", "ordinary"):
            return chosen
        remaining_seconds = float(self.TIME_SEARCH_HARD_MAX_SECONDS) - (time.perf_counter() - started)
        with self._measure_performance("search"):
            comparison = self._compare_late_receive_routes(state, player, actions, remaining_seconds)
        if comparison is None:
            return chosen
        comparison["applied"] = ((comparison.get("reliable", False)
                                  and comparison["margin"] >= self.LATE_RECEIVE_ROUTE_MARGIN)
                                 or comparison.get("selection_basis") == "royal_reserve_within_sample_dispersion")
        tracker = self._track[id(state)]
        tracker["last_late_receive_comparison"] = comparison
        if not comparison["applied"]:
            return chosen
        comparison["baseline_candidate_scores"] = copy.deepcopy(self.last_attack_candidate_scores)
        route = comparison["chosen"]
        if route[0] != chosen:
            tracker.clear()
            tracker.update(before)
            tracker["last_late_receive_comparison"] = comparison
            self._commit_timed_search_action(state, player, route[0])
        if len(route) == 2:
            after = self._timed_search_apply(state, player, route[0])
            tracker["pending_late_receive_route"] = {
                "key": self._late_receive_followup_key(after, player), "action": route[1]}
        root_values = {}
        for row in comparison["routes"]:
            action = tuple(row["actions"][0])
            root_values[action] = max(root_values.get(action, -float("inf")), row["value"])
        self.last_attack_candidate_scores = [
            {"action": action, "score": value, "candidate_role": "paired_receive_route"}
            for action, value in root_values.items()
        ]
        self._set_decision_reason("time_search")
        self._set_score_fallback_detail(
            f"late_receive_routes_depth_{comparison['depth']}_samples_{comparison['samples']}_"
            f"{'pass' if len(route) == 1 else 'receive_attack_' + str(route[1][2])}"
            f"{'_reserve_uncertain' if comparison.get('selection_basis') == 'royal_reserve_within_sample_dispersion' else ''}")
        self.last_rule_search_authority = "strong"
        return route[0]

    def _finalize_attack_candidate_snapshot(self, actions, chosen):
        super()._finalize_attack_candidate_snapshot(actions, chosen)
        tracker = self._track.get(getattr(self, "_live_plan_review_state_id", None), {})
        comparison = tracker.get("last_late_receive_comparison")
        if comparison and self.last_score_fallback_detail.startswith("late_receive_"):
            self.last_attack_candidate_snapshot["late_receive_comparison"] = copy.deepcopy(comparison)

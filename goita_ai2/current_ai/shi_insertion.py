"""Evaluate shi insertion as a receive-timing plan rather than a fixed move.
The planner compares receiving now with waiting one cycle, then ranks every
legal follow-up attack using public information, ally progress, and safety.
"""

from __future__ import annotations

import copy
from collections import Counter
from typing import Dict, List, Optional, Tuple

from goita_ai2.constants import PIECE_TOTALS, POINTS


Action = Tuple[str, Optional[str], Optional[str]]


class ShiInsertionStrategyMixin:
    """Builds and scores receive-first plans that may insert shi to the ally."""

    @staticmethod
    def _shi_insertion_detail(analysis, timing, piece):
        detail = f"shi_insertion_{timing}_{piece}"
        if piece != "1" and timing in ("immediate", "followup") and analysis:
            uncovered = [row for row in analysis.get("next_attack_candidates", ())
                         if not row["can_receive"] and row["weight"] > 0]
            if uncovered:
                threat = max(uncovered, key=lambda row: row["weight"])["piece"]
                detail += f"_avoid_{threat}"
        return detail

    def _shi_insertion_piece_probability(
        self,
        state,
        player: str,
        seat: str,
        piece: str,
    ) -> float:
        """Return a public-information probability that ``seat`` still holds a piece."""
        tracker = self._track.get(id(state))
        if tracker is None:
            return 0.0

        posterior = tracker.get("probabilistic_hand_inference")
        if posterior is not None:
            try:
                summary = next(item for item in posterior.players if item.seat == seat)
                return max(
                    0.0,
                    min(1.0, float(summary.piece(piece).current_holding_probability)),
                )
            except (AttributeError, StopIteration, TypeError, ValueError):
                pass

        minimum, maximum = self._estimate_remaining_range(tracker, seat, piece)
        if minimum > 0:
            return 0.90
        if maximum <= 0:
            return 0.0
        current_size = max(1, len(state.hands.get(seat, ())))
        return max(0.20, min(0.75, float(maximum) / float(current_size)))

    def _shi_insertion_followup_actions(
        self,
        state,
        player: str,
        receive_piece: str,
    ) -> List[Action]:
        """Enumerate attacks that would be legal immediately after the receive."""
        try:
            after_receive = copy.deepcopy(state)
            after_receive.apply_receive(player, receive_piece)
        except (ValueError, TypeError):
            return []
        return [
            action
            for action in after_receive.legal_actions(player)
            if action[0] == "attack" and action[2] is not None
        ]

    def _shi_insertion_followup_score(
        self,
        state,
        player: str,
        receive_piece: str,
        attack_piece: str,
        *,
        downstream: str,
        downstream_hidden_count: int,
    ) -> Tuple[float, Dict[str, float]]:
        tracker = self._track.get(id(state))
        if tracker is None:
            return -1e18, {}

        remaining = Counter(state.hands[player])
        if remaining.get(receive_piece, 0) <= 0:
            return -1e18, {}
        remaining[receive_piece] -= 1
        if remaining.get(attack_piece, 0) <= 0:
            return -1e18, {}

        public_seen = tracker.get("public_seen_counts", {})
        outside = max(
            0,
            int(PIECE_TOTALS[attack_piece])
            - int(public_seen.get(attack_piece, 0))
            - int(remaining.get(attack_piece, 0)),
        )
        components: Dict[str, float] = {
            "piece_value": float(POINTS.get(attack_piece, 0)) * 0.8,
            "repeat_strength": max(0, int(remaining.get(attack_piece, 0)) - 1) * 24.0,
            "scarcity": max(0.0, 3.0 - float(outside)) * 20.0,
        }

        royal_count = int(remaining.get("8", 0)) + int(remaining.get("9", 0))
        if attack_piece == "1":
            ally = self._ally_of(player)
            ally_shi_probability = self._shi_insertion_piece_probability(
                state,
                player,
                ally,
                "1",
            )
            downstream_shi_probability = self._shi_insertion_piece_probability(
                state,
                player,
                downstream,
                "1",
            )
            components["information_value"] = float(self.SHI_INSERTION_INFORMATION_VALUE)

            if downstream_hidden_count == 1:
                # The first hidden block is frequently shi. Treat this as soft
                # evidence that less shi remains to intercept the insertion.
                components["downstream_one_hidden_shi"] = float(
                    self.SHI_INSERTION_ONE_HIDDEN_VALUE
                )
                downstream_shi_probability = max(
                    0.0,
                    downstream_shi_probability
                    - float(self.SHI_INSERTION_ONE_HIDDEN_SHI_REDUCTION),
                )
            # Marginal approximation: possessing shi does not mean the partner
            # gets a turn to receive it. The intervening opponent acts first.
            delivery_probability = (1.0 - downstream_shi_probability) * ally_shi_probability
            components["shi_pressure"] = (
                (1.0 - downstream_shi_probability) * float(self.SHI_INSERTION_SHI_ATTACK_VALUE)
            )
            components["ally_reach_probability"] = (
                delivery_probability * float(self.SHI_INSERTION_ALLY_PROGRESS_VALUE)
            )
            ally_cards = len(state.hands.get(ally, ()))
            if ally_cards <= 2:
                components["ally_near_finish"] = 110.0 * delivery_probability
            elif ally_cards <= 4:
                components["ally_near_reach"] = 70.0 * delivery_probability
            components["downstream_interception_risk"] = (
                -downstream_shi_probability
                * float(self.SHI_INSERTION_INTERCEPTION_PENALTY)
            )

        if attack_piece in ("6", "7"):
            components["big_piece_royal_bridge"] = royal_count * 34.0
        elif attack_piece in ("3", "4", "5") and outside <= 1:
            components["fourth_middle_pressure"] = 55.0

        if attack_piece in ("8", "9") and len(state.hands[player]) > 2:
            components["early_royal_attack"] = -80.0
        return sum(components.values()), components

    def _shi_insertion_wait_risk(self, state, player):
        """Estimate exposure to possible next attacks, using public beliefs.

        These weights describe plausible attacks, not a calibrated prediction
        of the opponent's policy. Never read their real hidden hand.
        """
        tracker = self._track[id(state)]
        hand = state.hands[player]
        candidates = []
        for piece in PIECE_TOTALS:
            probability = self._shi_insertion_piece_probability(
                state, player, state.attacker, piece,
            )
            if probability <= 0:
                continue
            estimate = self._estimated_current_piece(tracker, state.attacker, piece) or {}
            # A repeated attack becomes more plausible with estimated copies.
            weight = probability * max(1.0, float(estimate.get("expected", 1.0)))
            can_receive = piece in hand or (
                piece not in ("1", "2") and any(royal in hand for royal in ("8", "9"))
            )
            candidates.append({"piece": piece, "weight": weight, "can_receive": can_receive})
        total = sum(row["weight"] for row in candidates)
        for row in candidates:
            row["weight"] /= total
        exposure = sum(row["weight"] for row in candidates if not row["can_receive"])
        return exposure, candidates

    def _shi_insertion_purpose_evidence(self, state, player, downstream, ally):
        """Return evidence that the shi is meant to hand the turn to the ally.

        A shi insertion is deliberately gated by purpose. Possessing two shi
        and a royal is not enough: the next enemy must be close to a reach, or
        the ally must be a publicly supported two-card finish candidate.
        """
        tracker = self._track.get(id(state)) or {}
        joint = tracker.get("joint_hand_inference", {})
        map_current = joint.get("map_current_counts", {}) if joint.get("feasible") else {}
        enemy_map = map_current.get(downstream, {})
        ally_map = map_current.get(ally, {})
        enemy_shi = int(enemy_map.get("1", 0)) if enemy_map else None
        ally_shi = int(ally_map.get("1", 0)) if ally_map else None
        enemy_estimate = self._estimated_current_piece(tracker, downstream, "1") or {}
        ally_estimate = self._estimated_current_piece(tracker, ally, "1") or {}
        enemy_few = (
            enemy_shi is not None and enemy_shi <= 2
        ) or (
            enemy_shi is None
            and int(enemy_estimate.get("max", 99)) <= 2
            and float(enemy_estimate.get("expected", 99.0)) <= 2.0
        )
        # A feasible joint count is stronger evidence than a posterior
        # probability.  In particular, do not revive a zero-count ally with a
        # broad 0.75 fallback: that would turn a missing shi into a guessed
        # shi and defeat the purpose gate.
        if ally_shi is not None:
            ally_can_receive = ally_shi >= 1
        else:
            ally_can_receive = (
                int(ally_estimate.get("min", 0)) >= 1
                or self._shi_insertion_piece_probability(state, player, ally, "1") >= 0.75
            )
        enemy_reach = len(state.hands.get(downstream, ())) <= 2
        ally_finish = (
            len(state.hands.get(ally, ())) == 2
            and ally_can_receive
            and (
                enemy_shi == 0
                or (
                    enemy_shi is None
                    and int(enemy_estimate.get("max", 99)) <= 0
                )
            )
        )
        return {
            "enemy": downstream,
            "ally": ally,
            "enemy_shi_estimate": enemy_shi if enemy_shi is not None else enemy_estimate,
            "ally_shi_estimate": ally_shi if ally_shi is not None else ally_estimate,
            "enemy_shi_few": enemy_few,
            "ally_can_receive": ally_can_receive,
            "enemy_reach_purpose": enemy_reach and enemy_few,
            "ally_finish_purpose": ally_finish,
            "purpose_confirmed": enemy_few and ally_can_receive and (enemy_reach or ally_finish),
        }

    def _shi_insertion_plan_analysis(
        self,
        state,
        player: str,
        actions: List[Action],
    ) -> Optional[Dict[str, object]]:
        """Compare immediate and one-cycle delayed receive plans."""
        if (
            not bool(getattr(self, "SHI_INSERTION_ENABLED", True))
            or state.phase != "receive"
            or state.current_attack in (None, "1", "2")
            or state.attacker is None
            or self._same_team(state.attacker, player)
        ):
            return None

        current_attack = str(state.current_attack)
        pass_action = next((action for action in actions if action[0] == "pass"), None)
        receive_action = next(
            (
                action
                for action in actions
                if action[0] == "receive" and action[1] == current_attack
            ),
            None,
        )
        if pass_action is None or receive_action is None:
            return None

        hand = state.hands[player]
        royal_count = hand.count("8") + hand.count("9")
        if hand.count("1") < 2 or royal_count <= 0:
            return None

        # Waiting one cycle only has a clear meaning for the final responder:
        # a pass returns control to the attacker, who must block and attack.
        downstream = state.next_player(player)
        final_responder = downstream == state.attacker
        if not final_responder:
            return None
        tracker = self._track.get(id(state))
        if tracker is None:
            return None
        downstream_hidden_count = int(
            tracker.get("hidden_block_counts", {}).get(downstream, 0)
        )
        purpose = self._shi_insertion_purpose_evidence(
            state,
            player,
            downstream,
            self._ally_of(player),
        )
        if not purpose["purpose_confirmed"]:
            return None

        followups = self._shi_insertion_followup_actions(
            state,
            player,
            current_attack,
        )
        if not followups:
            return None

        # Use the same post-receive attack evaluation as ordinary play, with a
        # scale conversion to the tactical route scores. Keep the real tracker
        # untouched while updating the simulated public receive event.
        after_receive = copy.deepcopy(state)
        attack_agent = copy.deepcopy(self)
        attack_agent._track[id(after_receive)] = copy.deepcopy(tracker)
        after_receive.apply_receive(player, current_attack)
        attack_agent.on_public_action(after_receive, player, receive_action)
        scored_followups = []
        for action in followups:
            attack_piece = str(action[2])
            score, components = self._shi_insertion_followup_score(
                state,
                player,
                current_attack,
                attack_piece,
                downstream=downstream,
                downstream_hidden_count=downstream_hidden_count,
            )
            common = attack_agent._score_attack_phase(
                after_receive, player, "attack", None, attack_piece,
                has_non_king_attack_option=any(a[2] not in ("8", "9") for a in followups),
            )
            components["common_attack_evaluation"] = common * self.SHI_INSERTION_COMMON_ATTACK_WEIGHT
            score += components["common_attack_evaluation"]
            scored_followups.append({
                "action": action,
                "attack": attack_piece,
                "score": round(score, 3),
                "components": components,
            })
        scored_followups.sort(
            key=lambda item: (float(item["score"]), int(POINTS.get(str(item["attack"]), 0))),
            reverse=True,
        )
        scored_followups = scored_followups[: max(1, int(self.SHI_INSERTION_MAX_FOLLOWUPS))]

        royal_safety = (
            float(self.SHI_INSERTION_BOTH_ROYALS_VALUE)
            if royal_count >= 2
            else float(self.SHI_INSERTION_ONE_ROYAL_VALUE)
        )
        matching_receive = float(self.SHI_INSERTION_MATCHING_RECEIVE_VALUE)
        enemy_cards = len(state.hands.get(state.attacker, ()))
        immediate_danger = max(0, 4 - enemy_cards) * float(
            self.SHI_INSERTION_ENEMY_PROGRESS_PENALTY
        )
        pending_wait = tracker.get("pending_shi_insertion_wait")
        waited_once = bool(
            isinstance(pending_wait, dict)
            and str(pending_wait.get("attack")) == current_attack
            and str(pending_wait.get("attacker")) == str(state.attacker)
        )

        wait_exposure, next_attacks = self._shi_insertion_wait_risk(state, player)
        wait_penalty = wait_exposure * self.SHI_INSERTION_LOST_RECEIVE_PENALTY
        routes = []
        for followup in scored_followups:
            immediate_score = (
                float(followup["score"])
                + royal_safety
                + matching_receive
                - immediate_danger
                + (float(self.SHI_INSERTION_EXTRA_BLOCK_VALUE) if waited_once else 0.0)
            )
            routes.append({
                "timing": "immediate",
                "root_action": receive_action,
                "followup": followup["attack"],
                "score": round(immediate_score, 3),
                "components": {
                    "followup": float(followup["score"]),
                    "royal_safety": royal_safety,
                    "matching_receive": matching_receive,
                    "enemy_progress_risk": -immediate_danger,
                    "wait_completed": (
                        float(self.SHI_INSERTION_EXTRA_BLOCK_VALUE)
                        if waited_once
                        else 0.0
                    ),
                },
            })

        remaining_min, remaining_max = self._estimate_remaining_range(
            tracker,
            state.attacker,
            current_attack,
        )
        if final_responder and remaining_max > 0 and not waited_once:
            repeat_probability = 0.85 if remaining_min > 0 else 0.55
            repeated_attack_risk = max(0, remaining_max - 1) * float(
                self.SHI_INSERTION_REPEAT_ATTACK_PENALTY
            )
            extra_block_value = float(self.SHI_INSERTION_EXTRA_BLOCK_VALUE)
            if downstream_hidden_count == 1:
                extra_block_value += float(self.SHI_INSERTION_WAIT_AFTER_ONE_HIDDEN_VALUE)
            for followup in scored_followups:
                delayed_score = (
                    float(followup["score"])
                    + royal_safety
                    + matching_receive
                    + repeat_probability * extra_block_value
                    - repeated_attack_risk
                    - immediate_danger * 1.35
                    - wait_penalty
                )
                routes.append({
                    "timing": "delayed",
                    "root_action": pass_action,
                    "followup": followup["attack"],
                    "score": round(delayed_score, 3),
                    "components": {
                        "followup": float(followup["score"]),
                        "royal_safety": royal_safety,
                        "matching_receive": matching_receive,
                        "repeat_probability": repeat_probability,
                        "extra_hidden_block": repeat_probability * extra_block_value,
                        "repeat_attack_risk": -repeated_attack_risk,
                        "enemy_progress_risk": -immediate_danger * 1.35,
                        "lost_receive_opportunity": -wait_penalty,
                    },
                })

        if remaining_max <= 0 and not waited_once:
            # Passing still deserves comparison when the same attack cannot
            # repeat. Discount a waiting opportunity by uncovered next attacks.
            best_now = max(routes, key=lambda item: float(item["score"]))
            routes.append({
                "timing": "wait", "root_action": pass_action,
                "followup": best_now["followup"],
                "score": round(float(best_now["score"]) - wait_penalty - matching_receive, 3),
                "components": {"future_opportunity_upper_estimate": float(best_now["score"]),
                               "lost_receive_opportunity": -wait_penalty,
                               "current_receive_opportunity": -matching_receive},
            })

        routes.sort(key=lambda item: float(item["score"]), reverse=True)
        if not routes:
            return None
        best = routes[0]
        second_score = float(routes[1]["score"]) if len(routes) > 1 else -1e18
        return {
            "version": 1,
            "current_attack": current_attack,
            "attacker": state.attacker,
            "downstream": downstream,
            "downstream_hidden_count": downstream_hidden_count,
            "royal_count": royal_count,
            "final_responder": final_responder,
            "waited_once": waited_once,
            "next_attack_candidates": next_attacks,
            "uncovered_next_attack_weight": wait_exposure,
            "next_attack_estimate_kind": "public_holding_weighted_heuristic",
            "purpose": purpose,
            "followups": scored_followups,
            "routes": routes[: max(2, int(self.SHI_INSERTION_MAX_ROUTES))],
            "recommended": best,
            "margin": round(float(best["score"]) - second_score, 3),
        }

    def _shi_insertion_plan_action(
        self,
        state,
        player: str,
        actions: List[Action],
    ) -> Optional[Action]:
        """Return a sufficiently distinct tactical root and retain its follow-up."""
        analysis = self._shi_insertion_plan_analysis(state, player, actions)
        tracker = self._track.get(id(state))
        if tracker is not None and isinstance(tracker.get("pending_shi_insertion_wait"), dict):
            pending = tracker["pending_shi_insertion_wait"]
            if (
                str(pending.get("attack")) != str(state.current_attack)
                or str(pending.get("attacker")) != str(state.attacker)
            ):
                tracker["pending_shi_insertion_wait"] = None
        if tracker is not None:
            tracker["last_shi_insertion_analysis"] = analysis
        if analysis is None:
            return None

        recommended = analysis["recommended"]
        if float(analysis["margin"]) < float(self.SHI_INSERTION_MIN_ROUTE_MARGIN):
            return None

        action = tuple(recommended["root_action"])
        if action[0] == "receive" and tracker is not None:
            tracker["pending_shi_insertion_attack_piece"] = str(
                recommended["followup"]
            )
            tracker["pending_shi_insertion_wait"] = None
        if tracker is not None:
            tracker["pending_shi_insertion_wait"] = (
                {
                    "attack": str(analysis["current_attack"]),
                    "followup": str(recommended["followup"]),
                    "attacker": str(analysis["attacker"]),
                }
                if action[0] == "pass"
                else None
            )
        return action

    def _commit_shi_insertion_root(
        self,
        state,
        player: str,
        actions: List[Action],
        action: Action,
    ) -> None:
        """Retain the best follow-up belonging to a search-selected root."""
        tracker = self._track.get(id(state))
        if tracker is None:
            return
        analysis = self._shi_insertion_plan_analysis(state, player, actions)
        tracker["last_shi_insertion_analysis"] = analysis
        if analysis is None:
            return
        routes = [
            route
            for route in analysis["routes"]
            if tuple(route["root_action"]) == tuple(action)
        ]
        if not routes:
            return
        route = max(routes, key=lambda item: float(item["score"]))
        if action[0] == "receive":
            tracker["pending_shi_insertion_attack_piece"] = str(route["followup"])
            tracker["pending_shi_insertion_wait"] = None
        elif action[0] == "pass":
            tracker["pending_shi_insertion_wait"] = {
                "attack": str(analysis["current_attack"]),
                "followup": str(route["followup"]),
                "attacker": str(analysis["attacker"]),
            }

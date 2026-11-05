"""Compare every upside root before spending the budget on deeper searches.

The saved AI keeps its original planner. This experiment uses the same shared
information-set policies and risk gates; unfinished iterations are never scores.
"""
from __future__ import annotations

import time
from typing import Optional, Sequence

from goita_ai2.current_ai.upside_finish import Action, UpsideFinishResult
from goita_ai2.current_ai.information_set_search import (
    InformationSetSearchCancelled, InformationSetSearchDeadline,
)


class UpsideSearchMixin:
    def _select_rule_based_action(self, state, player, actions):
        tracker = self._track.get(id(state), {})
        tracker.pop("last_upside_finish_comparison", None)
        self.last_upside_finish_metrics = {}
        self.last_upside_finish_comparison = None
        chosen = super()._select_rule_based_action(state, player, actions)
        tracker = self._track.get(id(state), {})
        if self.last_decision_reason == "upside_finish":
            piece = chosen[2]
            support = []
            for other in "ABCD":
                if self._same_team(other, player):
                    continue
                model = tracker.get("public_hand_models", {}).get(other, {})
                if any(event["piece"] == piece for event in
                       model.get("royal_instead_of_same_evidence", ())):
                    support.append(f"{other}:royal_instead_of_same_{piece}")
                if any(event.get("response_context") == "enemy_near_reach" for event in
                       tracker.get("piece_pass_evidence", {}).get(other, {}).get(piece, ())):
                    support.append(f"{other}:declined_near_reach_{piece}")
            if support:
                self.last_upside_finish_metrics["public_response_evidence"] = support
                tracker["last_upside_finish_metrics"] = dict(self.last_upside_finish_metrics)
                self._set_score_fallback_detail(self.last_score_fallback_detail + f"_public_response_{piece}")
            comparison = tracker.get("last_upside_finish_comparison")
            if comparison is not None:
                self.last_upside_finish_comparison = {
                    "source": comparison["source"], "proven": False,
                    "candidates": [dict(action=action, **row)
                                   for action, row in comparison["candidates"].items()],
                }
        return chosen

    def _finalize_attack_candidate_snapshot(self, actions, chosen):
        super()._finalize_attack_candidate_snapshot(actions, chosen)
        if self.last_decision_reason == "upside_finish" and self.last_attack_candidate_snapshot is not None:
            # Keep the estimate's source and actual completed depth in reports.
            self.last_attack_candidate_snapshot["upside_finish"] = dict(self.last_upside_finish_metrics)
            self.last_attack_candidate_snapshot["upside_finish_comparison"] = self.last_upside_finish_comparison

    def _search_upside_candidates(self, state, player, tracker, candidates,
                                  worlds, information_set, baseline, deadline):
        maximum_depth = max(1, int(self.UPSIDE_FINISH_MAX_DEPTH))
        first_depth = min(5, maximum_depth)
        depths = list(range(first_depth, maximum_depth + 1, 2))
        if depths[-1] != maximum_depth:
            depths.append(maximum_depth)
        completed, nodes = {}, 0
        solved = set()
        rows = {action: {"maximum_score": score, "completed_depth": None,
                         "interrupted": False} for action, score in candidates}
        tracker["last_upside_finish_comparison"] = {
            "source": "public_information_set", "proven": False,
            "candidates": rows, "cancelled": False,
        }
        for depth in depths:
            pending = [(action, score) for action, score in candidates if action not in solved]
            if depth > first_depth:
                # A completed shared-policy estimate orders further work; it
                # never bypasses the final score/risk/unknown-probability gates.
                pending.sort(key=lambda item: (
                    completed[item[0]][1].value if item[0] in completed else float("-inf")
                ), reverse=True)
            for index, (action, score) in enumerate(pending):
                now = time.perf_counter()
                available_nodes = int(self.UPSIDE_FINISH_MAX_NODES) - nodes
                if now >= deadline or available_nodes <= 0:
                    return completed, nodes
                remaining = len(pending) - index
                stats = {"nodes": 0, "max_nodes": max(1, available_nodes // remaining)}
                # A slow root gets only its share; later roots still get a turn.
                root_deadline = min(deadline, now + (deadline - now) / remaining)
                try:
                    outcome = self._information_set_search_root_action(
                        state, worlds, player, action, information_set, baseline,
                        depth, root_deadline, stats, tracker, None,
                    )
                except InformationSetSearchCancelled:
                    tracker["last_upside_finish_comparison"]["cancelled"] = True
                    return {}, nodes + stats["nodes"]
                except InformationSetSearchDeadline:
                    rows[action]["interrupted"] = True
                else:
                    # Every result covers the same whole set of worlds.
                    completed[action] = (depth, outcome)
                    rows[action]["completed_depth"] = depth
                    values = outcome.values_dict()
                    if all(abs(values.get(world.index, 0.0)) >= 100000 for world in worlds):
                        solved.add(action)
                nodes += stats["nodes"]
        return completed, nodes

    def _upside_finish_action(
        self,
        state,
        player: str,
        actions: Sequence[Action],
        safe_action: Action,
        safe_score: float,
    ) -> Optional[UpsideFinishResult]:
        """Return a higher-upside root only when its measured risk is acceptable."""
        if (
            not bool(self.UPSIDE_FINISH_ENABLED)
            or state.phase != "attack"
            or state.turn != player
            or len(state.hands[player]) > int(self.UPSIDE_FINISH_MAX_HAND_SIZE)
            or safe_action[0] not in ("attack", "attack_after_block")
        ):
            return None

        team = self._upside_team(player)
        enemy = "BD" if team == "AC" else "AC"
        if (
            int(state.team_score.get(team, 0)) + int(safe_score)
            >= int(self.UPSIDE_FINISH_MATCH_TARGET)
        ):
            return None

        candidates = self._upside_finish_candidates(
            state,
            player,
            actions,
            safe_action,
            safe_score,
        )
        if not candidates:
            return None

        tracker = self._track.get(id(state))
        if tracker is None:
            return None

        started = time.perf_counter()
        deadline = started + min(
            10.0,
            max(0.01, float(self.UPSIDE_FINISH_MAX_SECONDS)),
        )
        samples = self._timed_search_sample_states(
            state,
            player,
            tracker,
            int(self.UPSIDE_FINISH_SAMPLE_COUNT),
        )
        if not samples or time.perf_counter() >= deadline:
            return None
        try:
            information_set, worlds = self._information_set_search_worlds(
                state,
                player,
                tracker,
                samples,
            )
        except (TypeError, ValueError):
            return None
        if (
            information_set is None
            or not worlds
            or float(information_set.confidence)
            < float(self.UPSIDE_FINISH_MIN_CONFIDENCE)
        ):
            return None

        baseline_scores = {
            "AC": int(state.team_score.get("AC", 0)),
            "BD": int(state.team_score.get("BD", 0)),
        }
        score_lead = baseline_scores[team] - baseline_scores[enemy]
        allowed_risk = self._upside_allowed_failure_risk(state, player)
        completed, total_nodes = self._search_upside_candidates(
            state, player, tracker, candidates, worlds, information_set,
            baseline_scores, deadline,
        )
        evaluated = []
        for action, maximum_score in candidates:
            if action not in completed:
                continue
            depth, outcome = completed[action]
            values = outcome.values_dict()

            high = safe_or_better = lower = loss = unknown = 0.0
            expected_team = expected_enemy = match_loss = 0.0
            for world in worlds:
                probability = float(world.probability)
                value = values.get(world.index)
                if value is None:
                    unknown += probability
                    continue
                if value >= 100000.0:
                    score = max(0.0, (float(value) - 100000.0) / 500.0)
                    expected_team += probability * score
                    if score >= maximum_score:
                        high += probability
                    if score >= safe_score:
                        safe_or_better += probability
                    else:
                        lower += probability
                elif value <= -100000.0:
                    score = max(0.0, (-float(value) - 100000.0) / 500.0)
                    loss += probability
                    expected_enemy += probability * score
                    if (
                        int(state.team_score.get(enemy, 0)) + score
                        >= int(self.UPSIDE_FINISH_MATCH_TARGET)
                    ):
                        match_loss += probability
                else:
                    unknown += probability

            confidence_penalty = max(
                0.0,
                float(self.UPSIDE_FINISH_CONFIDENCE_REFERENCE)
                - float(information_set.confidence),
            ) * float(self.UPSIDE_FINISH_CONFIDENCE_RISK_WEIGHT)
            adjusted_failure = (
                loss
                + unknown * float(self.UPSIDE_FINISH_UNKNOWN_RISK_WEIGHT)
                + confidence_penalty
            )
            expected_net = expected_team - expected_enemy
            effective_allowed_risk = allowed_risk
            if (
                score_lead
                <= int(self.UPSIDE_FINISH_STRONG_VALUE_MAX_SCORE_LEAD)
                and high >= float(self.UPSIDE_FINISH_STRONG_VALUE_MIN_HIGH)
                and safe_or_better
                >= float(self.UPSIDE_FINISH_STRONG_VALUE_MIN_SAFE)
                and expected_net
                >= safe_score
                + float(self.UPSIDE_FINISH_STRONG_VALUE_MIN_EXPECTED_GAIN)
            ):
                effective_allowed_risk = max(
                    effective_allowed_risk,
                    float(self.UPSIDE_FINISH_STRONG_VALUE_RISK),
                )
            result = UpsideFinishResult(
                action=action,
                safe_action=safe_action,
                safe_score=safe_score,
                maximum_score=maximum_score,
                high_score_probability=high,
                safe_retention_probability=safe_or_better,
                lower_win_probability=lower,
                loss_probability=loss,
                unknown_probability=unknown,
                adjusted_failure_risk=adjusted_failure,
                allowed_failure_risk=effective_allowed_risk,
                expected_team_score=expected_team,
                expected_enemy_score=expected_enemy,
                information_confidence=float(information_set.confidence),
                depth=depth,
                samples=len(worlds),
                nodes=total_nodes,
                elapsed_seconds=time.perf_counter() - started,
            )
            eligible = (
                high >= float(self.UPSIDE_FINISH_MIN_HIGH_SCORE_PROBABILITY)
                and safe_or_better
                >= float(self.UPSIDE_FINISH_MIN_SAFE_RETENTION_PROBABILITY)
                and unknown <= float(self.UPSIDE_FINISH_MAX_UNKNOWN_PROBABILITY)
                and adjusted_failure <= effective_allowed_risk
                and match_loss
                <= float(self.UPSIDE_FINISH_MAX_MATCH_LOSS_PROBABILITY)
                and expected_net
                >= safe_score + float(self.UPSIDE_FINISH_MIN_EXPECTED_GAIN)
            )
            tracker["last_upside_finish_comparison"]["candidates"][action].update({
                "metrics": result.as_dict(), "eligible": eligible,
            })
            if eligible:
                evaluated.append((expected_net, high, -adjusted_failure, result))

        if not evaluated:
            return None
        evaluated.sort(key=lambda item: item[:3], reverse=True)
        chosen = evaluated[0][3]
        self.last_upside_finish_metrics = chosen.as_dict()
        self.last_upside_finish_metrics["search"] = "all_candidates_iterative_deepening"
        self.last_upside_finish_metrics["proven"] = False
        tracker["last_upside_finish_metrics"] = dict(self.last_upside_finish_metrics)
        return chosen

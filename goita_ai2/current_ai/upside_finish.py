"""Compares a proven finish with higher-scoring endgame alternatives.
The planner keeps the guaranteed route as a fallback, evaluates risky roots on
the same inferred deals, and accepts upside only within a score-aware risk cap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from goita_ai2.constants import POINTS
from goita_ai2.current_ai.information_set_search import (
    InformationSetSearchCancelled,
    InformationSetSearchDeadline,
)


Action = Tuple[str, Optional[str], Optional[str]]


@dataclass(frozen=True)
class UpsideFinishResult:
    """One accepted risky finish and the public probabilities behind it."""

    action: Action
    safe_action: Action
    safe_score: float
    maximum_score: float
    high_score_probability: float
    safe_retention_probability: float
    lower_win_probability: float
    loss_probability: float
    unknown_probability: float
    adjusted_failure_risk: float
    allowed_failure_risk: float
    expected_team_score: float
    expected_enemy_score: float
    information_confidence: float
    depth: int
    samples: int
    nodes: int
    elapsed_seconds: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "action": self.action,
            "safe_action": self.safe_action,
            "safe_score": round(self.safe_score, 2),
            "maximum_score": round(self.maximum_score, 2),
            "high_score_probability": round(self.high_score_probability, 4),
            "safe_retention_probability": round(
                self.safe_retention_probability, 4
            ),
            "lower_win_probability": round(self.lower_win_probability, 4),
            "loss_probability": round(self.loss_probability, 4),
            "unknown_probability": round(self.unknown_probability, 4),
            "adjusted_failure_risk": round(self.adjusted_failure_risk, 4),
            "allowed_failure_risk": round(self.allowed_failure_risk, 4),
            "expected_team_score": round(self.expected_team_score, 2),
            "expected_enemy_score": round(self.expected_enemy_score, 2),
            "information_confidence": round(self.information_confidence, 4),
            "depth": self.depth,
            "samples": self.samples,
            "nodes": self.nodes,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
        }


class UpsideFinishMixin:
    """Adds bounded, probability-aware upside choices above a safe finish."""

    @staticmethod
    def _upside_team(player: str) -> str:
        return "AC" if player in ("A", "C") else "BD"

    def _upside_allowed_failure_risk(self, state, player: str) -> float:
        team = self._upside_team(player)
        enemy = "BD" if team == "AC" else "AC"
        difference = int(state.team_score.get(team, 0)) - int(
            state.team_score.get(enemy, 0)
        )
        if difference <= -80:
            return float(self.UPSIDE_FINISH_TRAILING_80_RISK)
        if difference <= -50:
            return float(self.UPSIDE_FINISH_TRAILING_50_RISK)
        if difference >= 50:
            return float(self.UPSIDE_FINISH_LEADING_RISK)
        return float(self.UPSIDE_FINISH_BASE_RISK)

    def _upside_action_maximum_score(
        self,
        state,
        player: str,
        action: Action,
    ) -> float:
        action_type, block, attack = action
        if action_type not in ("attack", "attack_after_block") or attack is None:
            return 0.0
        remaining = self._remaining_hand_after_attack_action(
            state,
            player,
            block,
            attack,
        )
        if remaining is None:
            return 0.0
        if not remaining:
            return float(self._forced_win_finish_score(block, attack))

        maximum = max(float(POINTS.get(piece, 0)) for piece in remaining)
        if len(remaining) == 2:
            first, second = remaining
            maximum = max(
                maximum,
                float(self._forced_win_finish_score(first, second)),
                float(self._forced_win_finish_score(second, first)),
            )
        elif "8" in remaining and "9" in remaining:
            maximum = max(maximum, 100.0)
        return maximum

    def _upside_finish_candidates(
        self,
        state,
        player: str,
        actions: Sequence[Action],
        safe_action: Action,
        safe_score: float,
    ) -> Tuple[Tuple[Action, float], ...]:
        candidates = []
        for action in actions:
            if action == safe_action:
                continue
            maximum = self._upside_action_maximum_score(state, player, action)
            if maximum <= safe_score:
                continue
            candidates.append((action, maximum))
        candidates.sort(
            key=lambda item: (
                item[1],
                float(POINTS.get(str(item[0][2]), 0)),
                item[0],
            ),
            reverse=True,
        )
        return tuple(candidates)

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
        evaluated = []
        total_nodes = 0
        for action, maximum_score in candidates:
            stats = {
                "nodes": 0,
                "max_nodes": max(
                    1,
                    int(self.UPSIDE_FINISH_MAX_NODES) - total_nodes,
                ),
            }
            try:
                outcome = self._information_set_search_root_action(
                    state,
                    worlds,
                    player,
                    action,
                    information_set,
                    baseline_scores,
                    int(self.UPSIDE_FINISH_MAX_DEPTH),
                    deadline,
                    stats,
                    tracker,
                    None,
                )
            except (
                InformationSetSearchCancelled,
                InformationSetSearchDeadline,
            ):
                return None
            total_nodes += int(stats["nodes"])
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
                depth=int(self.UPSIDE_FINISH_MAX_DEPTH),
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
            if eligible:
                evaluated.append((expected_net, high, -adjusted_failure, result))

        if not evaluated:
            return None
        evaluated.sort(key=lambda item: item[:3], reverse=True)
        chosen = evaluated[0][3]
        self.last_upside_finish_metrics = chosen.as_dict()
        tracker["last_upside_finish_metrics"] = chosen.as_dict()
        return chosen

"""Coordinate shi-count inference and shi-attack decisions.

The existing AI already estimates each player's shi count.  This module turns
those estimates into one coherent strategy while a shi attack is active.  It
uses public history and the joint ten-shi allocation only; actual opponent
hands are never read.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple


Action = Tuple[str, Optional[str], Optional[str]]


class ShiAttackPackageMixin:
    """Public-information analysis and high-confidence shi decisions."""

    def _shi_attack_package_analysis(
        self,
        state,
        player: str,
    ) -> Optional[Dict[str, object]]:
        tracker = self._track.get(id(state))
        if tracker is None:
            return None

        initial_shi = int(
            tracker.get("my_init_count", Counter()).get("1", 0)
        )
        participating = bool(
            tracker.get("shi_attack_mode")
            or "1" in tracker.get("my_past_attacks", set())
            or tracker.get("inherit_ally_shi_attack")
        )
        if initial_shi < 3 or not participating:
            return None

        joint = tracker.get("joint_hand_inference", {})
        if not bool(joint.get("feasible")):
            return None
        map_original = joint.get("map_original_counts", {})
        map_current = joint.get("map_current_counts", {})

        initial_distribution: Dict[str, int] = {}
        current_distribution: Dict[str, int] = {}
        used_distribution: Dict[str, int] = {}
        for seat in ("A", "B", "C", "D"):
            if seat == player:
                initial = initial_shi
                current = int(state.hands[player].count("1"))
            else:
                original_counts = map_original.get(seat, {})
                current_counts = map_current.get(seat, {})
                if "1" not in original_counts or "1" not in current_counts:
                    return None
                initial = int(original_counts.get("1", 0))
                current = int(current_counts.get("1", 0))
            initial_distribution[seat] = initial
            current_distribution[seat] = current
            used_distribution[seat] = max(0, initial - current)

        enemies = [
            seat
            for seat in ("A", "B", "C", "D")
            if seat != player and not self._same_team(seat, player)
        ]
        enemy_map_remaining = sum(current_distribution[seat] for seat in enemies)
        enemy_used = sum(used_distribution[seat] for seat in enemies)
        enemy_summary = self._opponents_piece_count_summary(
            tracker,
            player,
            "1",
        )
        expected_remaining = max(
            0.0,
            float(enemy_summary.get("expected", 0.0)),
        )
        confidence = max(
            0.0,
            min(1.0, float(enemy_summary.get("confidence", 0.0))),
        )
        attack_number = int(tracker.get("my_attack_count", 0)) + 1

        if attack_number >= 3 and enemy_map_remaining == 0:
            purpose = "finish_third_attack"
        elif enemy_map_remaining <= 1:
            purpose = "exhaust_enemy_shi"
        else:
            purpose = "continue_shi_pressure"

        analysis: Dict[str, object] = {
            "active": True,
            "purpose": purpose,
            "attack_number": attack_number,
            "own_initial_shi": initial_shi,
            "own_current_shi": int(state.hands[player].count("1")),
            "initial_distribution": initial_distribution,
            "current_distribution": current_distribution,
            "used_distribution": used_distribution,
            "enemy_seats": enemies,
            "enemy_map_remaining": enemy_map_remaining,
            "enemy_expected_remaining": round(expected_remaining, 3),
            "enemy_inference_confidence": round(confidence, 3),
            "enemy_used_shi": enemy_used,
            "source": "joint_public_inference",
        }
        tracker["last_shi_attack_package_analysis"] = analysis
        self.last_shi_attack_package_analysis = analysis
        return analysis

    @staticmethod
    def _shi_attack_distribution_label(distribution: Dict[str, int]) -> str:
        return "_".join(
            f"{seat}{int(distribution.get(seat, 0))}"
            for seat in ("A", "B", "C", "D")
        )

    @staticmethod
    def _hand_after_attack_action(hand: List[str], action: Action) -> Optional[List[str]]:
        action_type, block, attack = action
        if action_type not in ("attack", "attack_after_block") or attack is None:
            return None
        remaining = list(hand)
        if block is not None:
            if block not in remaining:
                return None
            remaining.remove(block)
        if attack not in remaining:
            return None
        remaining.remove(attack)
        return remaining

    def _shi_attack_package_action(
        self,
        state,
        player: str,
        actions: List[Action],
        *,
        has_non_king_attack_option: bool,
    ) -> Optional[Tuple[Action, Dict[str, object]]]:
        """Choose the last shi on attack three when the enemy team is exhausted."""
        if not bool(getattr(self, "SHI_ATTACK_PACKAGE_ENABLED", False)):
            return None
        if state.phase != "attack" or state.turn != player:
            return None

        analysis = self._shi_attack_package_analysis(state, player)
        if analysis is None:
            return None
        if (
            int(analysis["attack_number"]) != 3
            or int(analysis["own_current_shi"]) < 1
            or int(analysis["enemy_map_remaining"]) != 0
            or float(analysis["enemy_expected_remaining"])
            > float(self.SHI_ATTACK_PACKAGE_EXHAUSTED_MAX_EXPECTED)
            or float(analysis["enemy_inference_confidence"])
            < float(self.SHI_ATTACK_PACKAGE_MIN_CONFIDENCE)
        ):
            return None

        candidates: List[Tuple[float, Action]] = []
        scored_actions: List[Dict[str, object]] = []
        for action in actions:
            action_type, block, attack = action
            if action_type not in ("attack", "attack_after_block") or attack is None:
                continue
            score = self._score_attack_phase(
                state,
                player,
                action_type,
                block,
                attack,
                has_non_king_attack_option=has_non_king_attack_option,
            )
            if action_type == "attack_after_block":
                score += self._score_receive_phase(
                    state,
                    player,
                    "receive",
                    block,
                )
            scored_actions.append({"action": action, "score": float(score)})
            remaining = self._hand_after_attack_action(
                state.hands[player],
                action,
            )
            if attack == "1" and remaining is not None and len(remaining) == 2:
                candidates.append((float(score), action))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        chosen = candidates[0][1]
        analysis["selected_action"] = list(chosen)
        analysis["purpose"] = "finish_third_attack"
        self.last_attack_candidate_scores = scored_actions
        return chosen, analysis

    def _shi_attack_package_detail(self, analysis: Dict[str, object]) -> str:
        initial = self._shi_attack_distribution_label(
            analysis.get("initial_distribution", {})
        )
        expected_percent = int(
            round(float(analysis.get("enemy_expected_remaining", 0.0)) * 100)
        )
        return (
            "shi_package_final_attack_"
            f"enemy_map_{int(analysis.get('enemy_map_remaining', 0))}_"
            f"expected_{expected_percent}_distribution_{initial}"
        )

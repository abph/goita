"""Predicts plausible actions inside an information-set search.

Enemy choices are estimated from public rank evidence and the hand that enemy
would know. Ally choices additionally follow public signals and attack plans.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


Action = Tuple[str, Optional[str], Optional[str]]


@dataclass(frozen=True)
class InformationSetActionEvaluation:
    """A role-aware action score with inspectable public reasons."""

    actor: str
    role: str
    action: Action
    base_score: float
    adjustments: Tuple[Tuple[str, float], ...]
    total_score: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "actor": self.actor,
            "role": self.role,
            "action": self.action,
            "base_score": round(self.base_score, 3),
            "adjustments": {
                reason: round(value, 3)
                for reason, value in self.adjustments
            },
            "total_score": round(self.total_score, 3),
        }


class InformationSetActionModelMixin:
    """Scores self, ally, and enemy moves without consulting the live deal."""

    _INFORMATION_SET_STRONG_RANKS = frozenset(("SS", "S", "A", "B", "C"))
    _INFORMATION_SET_WEAK_RANKS = frozenset(("D", "E", "F", "X"))

    def _information_set_action_role(
        self,
        root_player: str,
        actor: str,
    ) -> str:
        if actor == root_player:
            return "self"
        if self._same_team(root_player, actor):
            return "ally"
        return "enemy"

    @staticmethod
    def _information_set_add_adjustment(
        adjustments: Dict[str, float],
        reason: str,
        value: float,
    ) -> None:
        if abs(value) < 1e-9:
            return
        adjustments[reason] = adjustments.get(reason, 0.0) + float(value)

    def _information_set_attack_plan_adjustments(
        self,
        sampled_state,
        actor: str,
        action: Action,
        tracker: Optional[dict],
        adjustments: Dict[str, float],
    ) -> None:
        action_type, _block, attack = action
        if action_type not in ("attack", "attack_after_block") or attack is None:
            return

        hand = sampled_state.hands[actor]
        count = hand.count(attack)
        if count >= 2:
            self._information_set_add_adjustment(
                adjustments,
                "hand_repeat_attack",
                16.0 * float(count - 1),
            )

        plan_info = self._special_attack_sequence_plan(Counter(hand))
        sequence = plan_info.get("sequence", []) if isinstance(plan_info, dict) else []
        if sequence:
            if attack == sequence[0]:
                self._information_set_add_adjustment(
                    adjustments,
                    "hand_attack_plan_next",
                    46.0,
                )
            elif attack in sequence:
                self._information_set_add_adjustment(
                    adjustments,
                    "hand_attack_plan_later",
                    10.0,
                )

        model = (tracker or {}).get("public_hand_models", {}).get(actor, {})
        public_attacks = model.get("attacks", Counter())
        if int(public_attacks.get(attack, 0)) > 0:
            self._information_set_add_adjustment(
                adjustments,
                "public_attack_continuation",
                30.0,
            )

        is_first_attack = int(model.get("attack_count", 0)) == 0
        if is_first_attack and count == 1 and attack in ("2", "3", "4", "5"):
            if hand.count("1") >= 3:
                self._information_set_add_adjustment(
                    adjustments,
                    "avoid_single_piece_opening_over_shi",
                    -34.0,
                )
        if is_first_attack and attack == "1" and hand.count("1") >= 3:
            self._information_set_add_adjustment(
                adjustments,
                "natural_multi_shi_opening",
                34.0,
            )

    def _information_set_enemy_adjustments(
        self,
        sampled_state,
        actor: str,
        action: Action,
        tracker: Optional[dict],
        adjustments: Dict[str, float],
    ) -> None:
        action_type, block, _attack = action
        hand = sampled_state.hands[actor]
        rank_estimate = self._public_hand_rank_estimate(tracker, actor)
        rank = str(rank_estimate.get("rank", "D"))
        confidence = max(0.0, min(1.0, float(rank_estimate.get("confidence", 0.0))))
        evidence_scale = 0.35 + confidence * 0.65
        current_attack = sampled_state.current_attack

        if action_type == "receive" and block is not None:
            same_piece = block == current_attack
            if same_piece:
                rank_value = 26.0 if rank in self._INFORMATION_SET_STRONG_RANKS else -20.0
                self._information_set_add_adjustment(
                    adjustments,
                    "enemy_rank_same_piece_receive",
                    rank_value * evidence_scale,
                )
                if hand.count(block) >= 2:
                    self._information_set_add_adjustment(
                        adjustments,
                        "enemy_multiple_same_piece_receive",
                        20.0,
                    )
                if block == "1" and hand.count("1") >= 2:
                    self._information_set_add_adjustment(
                        adjustments,
                        "enemy_two_shi_receive",
                        34.0,
                    )
            elif block in ("8", "9") and len(hand) > 2:
                self._information_set_add_adjustment(
                    adjustments,
                    "enemy_preserve_royal_receive",
                    -22.0,
                )

        elif action_type == "pass":
            same_piece_receives = [
                legal
                for legal in sampled_state.legal_actions(actor)
                if legal[0] == "receive" and legal[1] == current_attack
            ]
            if same_piece_receives:
                rank_value = -32.0 if rank in self._INFORMATION_SET_STRONG_RANKS else 38.0
                self._information_set_add_adjustment(
                    adjustments,
                    "enemy_rank_same_piece_pass",
                    rank_value * evidence_scale,
                )
                if current_attack == "1" and hand.count("1") >= 2:
                    self._information_set_add_adjustment(
                        adjustments,
                        "enemy_two_shi_should_receive",
                        -54.0,
                    )
                elif current_attack == "2" and rank in self._INFORMATION_SET_WEAK_RANKS:
                    self._information_set_add_adjustment(
                        adjustments,
                        "enemy_cautious_kyosha_pass",
                        18.0,
                    )

        self._information_set_attack_plan_adjustments(
            sampled_state,
            actor,
            action,
            tracker,
            adjustments,
        )

    def _information_set_ally_adjustments(
        self,
        sampled_state,
        actor: str,
        action: Action,
        tracker: Optional[dict],
        adjustments: Dict[str, float],
    ) -> None:
        action_type, _block, attack = action
        tr = tracker or {}
        if action_type in ("attack", "attack_after_block") and attack is not None:
            ally_signal = str(tr.get("ally_shi_signal", "unknown"))
            if attack == "1" and tr.get("shi_attack_mode"):
                self._information_set_add_adjustment(
                    adjustments,
                    "ally_shi_mode",
                    86.0,
                )
            if attack == "1" and ally_signal == "returned_shi":
                self._information_set_add_adjustment(
                    adjustments,
                    "ally_returned_shi_signal",
                    66.0,
                )
            elif attack == "1" and ally_signal == "weak":
                self._information_set_add_adjustment(
                    adjustments,
                    "ally_rejected_shi_signal",
                    -62.0,
                )
            elif attack == "1" and tr.get("ally_shi_sashikomi_candidate"):
                self._information_set_add_adjustment(
                    adjustments,
                    "ally_shi_sashikomi",
                    48.0,
                )

            pending = tr.get("ally_pending_response_piece")
            if pending is not None:
                if attack == pending:
                    self._information_set_add_adjustment(
                        adjustments,
                        "ally_pending_kakarigotae",
                        82.0,
                    )
                else:
                    self._information_set_add_adjustment(
                        adjustments,
                        "ally_skips_pending_response",
                        -22.0,
                    )

            if attack in tr.get("my_past_attacks", set()):
                self._information_set_add_adjustment(
                    adjustments,
                    "ally_answers_my_attack",
                    58.0,
                )
            if attack in tr.get("ally_past_attacks", set()):
                self._information_set_add_adjustment(
                    adjustments,
                    "ally_continues_attack_piece",
                    32.0,
                )
            if attack == tr.get("pending_ally_force_king_attack_piece"):
                self._information_set_add_adjustment(
                    adjustments,
                    "ally_force_king_plan",
                    64.0,
                )

        self._information_set_attack_plan_adjustments(
            sampled_state,
            actor,
            action,
            tracker,
            adjustments,
        )

    def _information_set_action_evaluation(
        self,
        root_state,
        sampled_state,
        root_player: str,
        actor: str,
        action: Action,
        tracker: Optional[dict] = None,
    ) -> InformationSetActionEvaluation:
        """Predict one actor's preference using only its information-set world."""
        role = self._information_set_action_role(root_player, actor)
        base_score = self._timed_search_action_priority(sampled_state, actor, action)
        adjustments: Dict[str, float] = {}
        tr = tracker if tracker is not None else self._track.get(id(root_state))

        if role == "enemy":
            self._information_set_enemy_adjustments(
                sampled_state,
                actor,
                action,
                tr,
                adjustments,
            )
        elif role == "ally":
            self._information_set_ally_adjustments(
                sampled_state,
                actor,
                action,
                tr,
                adjustments,
            )

        total_score = base_score + sum(adjustments.values())
        return InformationSetActionEvaluation(
            actor=actor,
            role=role,
            action=action,
            base_score=base_score,
            adjustments=tuple(sorted(adjustments.items())),
            total_score=total_score,
        )

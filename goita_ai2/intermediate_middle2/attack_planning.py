"""Plans every remaining attack before the AI chooses a hidden piece.
The plan assumes each future attack returns after a full round, then is rebuilt
whenever public play changes the hand, royal usage, or piece-count inference.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from goita_ai2.constants import POINTS


Action = Tuple[str, Optional[str], Optional[str]]
PlanStep = Tuple[str, str]


class AttackPlanningMixin:
    """Searches provisional block-and-attack routes through the whole hand."""

    def _planned_receive_width(self, hand: List[str]) -> float:
        """Approximate how many different attacks the remaining hand can receive."""
        counts = Counter(hand)
        normal_types = sum(
            1
            for piece in ("1", "2", "3", "4", "5", "6", "7")
            if counts.get(piece, 0) > 0
        )
        royal_count = counts.get("8", 0) + counts.get("9", 0)
        return float(normal_types + royal_count * 2)

    def _planned_finish_score(self, block: str, attack: str) -> float:
        if block == attack:
            return float(POINTS.get(attack, 0)) * 2.0
        if {block, attack} == {"8", "9"}:
            return 100.0
        return float(POINTS.get(attack, 0))

    def _planned_future_attack_value(
        self,
        state,
        player: str,
        piece: str,
        attack_no: int,
        previous_attack: Optional[str],
        current_hand: Tuple[str, ...],
    ) -> float:
        tr = self._track.get(id(state))
        value = float(POINTS.get(piece, 0)) * 0.5

        if piece in ("8", "9"):
            return value - self.GENERAL_ATTACK_PLAN_EARLY_ROYAL_PENALTY

        if piece == "1":
            value -= self.GENERAL_ATTACK_PLAN_SHI_PENALTY
            if tr is not None and tr.get("shi_attack_mode"):
                value += self.GENERAL_ATTACK_PLAN_SHI_MODE_BONUS
        elif piece == "2":
            value += self.GENERAL_ATTACK_PLAN_KYOSHA_BONUS

        value += max(
            -self.GENERAL_ATTACK_PLAN_INFERENCE_CAP,
            min(
                self.GENERAL_ATTACK_PLAN_INFERENCE_CAP,
                self._piece_count_attack_adjustment(state, player, piece),
            ),
        )
        value += min(
            self.GENERAL_ATTACK_PLAN_PUBLIC_SAFETY_CAP,
            self._public_attack_safety_bonus(state, player, piece),
        )

        if tr is not None:
            if piece in tr.get("my_past_attacks", set()):
                value += self.GENERAL_ATTACK_PLAN_PAST_ATTACK_BONUS
            if previous_attack == piece:
                value += self.GENERAL_ATTACK_PLAN_CONTINUATION_BONUS
            if self._is_kakarigotae_piece(piece) and (
                piece == tr.get("ally_first_attack")
                or piece in tr.get("ally_past_attacks", set())
            ):
                value += self.GENERAL_ATTACK_PLAN_KAKARI_BONUS

        profile = self._attack_shape_profile_for_piece(Counter(current_hand), piece)
        if profile is not None:
            value += float(profile["value"]) * self.GENERAL_ATTACK_PLAN_SHAPE_WEIGHT

        if attack_no == 3 and self._is_fourth_middle_attack(state, player, piece):
            value += self.GENERAL_ATTACK_PLAN_THIRD_FOURTH_BONUS

        return value

    def _planned_future_block_value(
        self,
        state,
        player: str,
        block: str,
        current_hand: Tuple[str, ...],
    ) -> float:
        penalty = {
            "9": 120.0,
            "8": 120.0,
            "7": 4.0,
            "6": 4.0,
            "5": 7.0,
            "4": 7.0,
            "3": 3.0,
            "2": 18.0,
            "1": 1.0,
        }.get(block, 0.0)
        value = -(penalty * self.GENERAL_ATTACK_PLAN_FUTURE_BLOCK_WEIGHT)
        value += self._piece_count_hidden_block_adjustment(state, player, block)

        if block == "1" and current_hand.count("1") <= 1:
            value -= self.GENERAL_ATTACK_PLAN_KEEP_LAST_SHI_PENALTY
        return value

    def _search_future_attack_plan(
        self,
        state,
        player: str,
        remaining_hand: List[str],
        previous_attack: Optional[str],
        next_attack_no: int,
    ) -> Dict[str, object]:
        start_hand = tuple(sorted(str(piece) for piece in remaining_hand))

        @lru_cache(maxsize=None)
        def search(
            hand: Tuple[str, ...],
            previous: Optional[str],
            attack_no: int,
        ) -> Tuple[float, Tuple[PlanStep, ...], float]:
            if not hand:
                return 0.0, tuple(), 0.0
            if len(hand) % 2 != 0:
                return -1e9, tuple(), -1.0

            if len(hand) == 2:
                candidates: List[Tuple[Tuple[float, float, int, int], float, PlanStep]] = []
                for attack in sorted(set(hand)):
                    cards = list(hand)
                    cards.remove(attack)
                    block = cards[0]
                    finish_score = self._planned_finish_score(block, attack)
                    route_score = finish_score * self.GENERAL_ATTACK_PLAN_FINISH_WEIGHT
                    key = (
                        route_score,
                        finish_score,
                        POINTS.get(attack, 0),
                        -POINTS.get(block, 0),
                    )
                    candidates.append((key, finish_score, (block, attack)))
                candidates.sort(key=lambda item: item[0], reverse=True)
                best_key, finish_score, step = candidates[0]
                return best_key[0], (step,), finish_score

            best_key: Optional[Tuple[float, float, int, int, str, str]] = None
            best_score = -1e9
            best_steps: Tuple[PlanStep, ...] = tuple()
            best_finish = -1.0

            for attack in sorted(set(hand)):
                after_attack = list(hand)
                after_attack.remove(attack)
                for block in sorted(set(after_attack)):
                    after_step = list(after_attack)
                    after_step.remove(block)
                    future_score, future_steps, finish_score = search(
                        tuple(sorted(after_step)),
                        attack,
                        attack_no + 1,
                    )
                    if future_score <= -1e8:
                        continue

                    step_score = self._planned_future_attack_value(
                        state,
                        player,
                        attack,
                        attack_no,
                        previous,
                        hand,
                    )
                    step_score += self._planned_future_block_value(
                        state,
                        player,
                        block,
                        hand,
                    )
                    total = step_score + future_score
                    key = (
                        total,
                        finish_score,
                        POINTS.get(attack, 0),
                        -POINTS.get(block, 0),
                        attack,
                        block,
                    )
                    if best_key is None or key > best_key:
                        best_key = key
                        best_score = total
                        best_steps = ((block, attack),) + future_steps
                        best_finish = finish_score

            return best_score, best_steps, best_finish

        score, steps, finish_score = search(
            start_hand,
            previous_attack,
            next_attack_no,
        )
        return {
            "score": score,
            "steps": list(steps),
            "attacks": [attack for _block, attack in steps],
            "finish_score": finish_score,
        }

    def _future_attack_plan_for_action(
        self,
        state,
        player: str,
        action: Action,
    ) -> Optional[Dict[str, object]]:
        action_type, block, attack = action
        if action_type not in ("attack", "attack_after_block") or attack is None:
            return None

        remaining = list(state.hands[player])
        if block is not None:
            if block not in remaining:
                return None
            remaining.remove(block)
        if attack not in remaining:
            return None
        remaining.remove(attack)

        if not remaining:
            return {
                "score": 0.0,
                "steps": [],
                "attacks": [],
                "finish_score": 0.0,
                "remaining_hand": [],
            }
        if len(remaining) % 2 != 0:
            return None

        tr = self._track.get(id(state))
        attack_count = 0
        if tr is not None:
            attack_count = max(
                int(tr.get("my_attack_count", 0)),
                len(tr.get("my_attack_history", [])),
            )
        plan = self._search_future_attack_plan(
            state,
            player,
            remaining,
            attack,
            attack_count + 2,
        )
        plan["remaining_hand"] = list(remaining)
        return plan

    def _future_attack_plan_adjustment(
        self,
        state,
        player: str,
        action_type: str,
        block: Optional[str],
        attack: Optional[str],
    ) -> float:
        plan = self._future_attack_plan_for_action(
            state,
            player,
            (action_type, block, attack),
        )
        if plan is None or len(plan.get("remaining_hand", [])) < 4:
            return 0.0
        value = float(plan["score"]) * self.GENERAL_ATTACK_PLAN_WEIGHT
        return max(
            -self.GENERAL_ATTACK_PLAN_BONUS_CAP,
            min(self.GENERAL_ATTACK_PLAN_BONUS_CAP, value),
        )

    def _eight_card_shallow_plan_action(
        self,
        state,
        player: str,
        actions: List[Action],
        *,
        has_non_king_attack_option: bool,
    ) -> Optional[Tuple[Action, Dict[str, object]]]:
        """Choose an opening by comparing the whole provisional four-attack route.

        This is intentionally shallower than the exact endgame solver. It assumes
        each planned attack returns after one round, but it scores the opening,
        future attack shapes, hidden pieces, receive width, and final pair
        together instead of choosing the first hidden piece in isolation.
        """
        tr = self._track.get(id(state))
        if (
            tr is None
            or state.phase != "attack"
            or state.current_attack is not None
            or state.attacker is not None
            or len(state.hands[player]) != 8
            or int(tr.get("my_attack_count", 0)) != 0
        ):
            return None

        candidates: List[
            Tuple[Tuple[float, float, float, float, int, int], Action, Dict[str, object]]
        ] = []
        for action in actions:
            action_type, block, attack = action
            if action_type != "attack_after_block" or block is None or attack is None:
                continue

            future = self._future_attack_plan_for_action(state, player, action)
            if (
                future is None
                or len(future.get("remaining_hand", [])) != 6
                or len(future.get("steps", [])) != 3
            ):
                continue

            opening_score = self._score_attack_phase(
                state,
                player,
                action_type,
                block,
                attack,
                has_non_king_attack_option=has_non_king_attack_option,
            )
            # The normal attack score already includes a capped future-plan
            # adjustment. Replace it with the uncapped eight-card evaluation.
            opening_score -= self._future_attack_plan_adjustment(
                state,
                player,
                action_type,
                block,
                attack,
            )

            remaining = list(future["remaining_hand"])
            receive_width = self._planned_receive_width(remaining)
            future_score = float(future["score"])
            finish_score = float(future["finish_score"])
            total_score = (
                opening_score
                + future_score * self.EIGHT_CARD_SHALLOW_FUTURE_WEIGHT
                + receive_width * self.EIGHT_CARD_SHALLOW_RECEIVE_WIDTH_WEIGHT
            )

            steps = [(block, attack)] + list(future["steps"])
            plan = {
                "source_hand": sorted(str(piece) for piece in state.hands[player]),
                "steps": steps,
                "attacks": [step_attack for _step_block, step_attack in steps],
                "final_pair": steps[-1],
                "finish_score": finish_score,
                "receive_width_after_opening": receive_width,
                "projected_score": total_score,
                "inference_revision": int(tr.get("piece_inference_revision", 0)),
            }
            key = (
                total_score,
                finish_score,
                receive_width,
                future_score,
                POINTS.get(attack, 0),
                -POINTS.get(block, 0),
            )
            candidates.append((key, action, plan))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        _key, action, plan = candidates[0]
        tr["shallow_eight_card_plan"] = plan
        return action, plan

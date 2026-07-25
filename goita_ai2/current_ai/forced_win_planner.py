"""Represents every guaranteed finish with one shared plan object.
Initial hand patterns and public-position proofs are normalized here so the
decision layer can compare guaranteed score, upside, and safety only once.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from goita_ai2.constants import POINTS
from goita_ai2.current_ai.endgame import ForcedWinStatus

Action = Tuple[str, Optional[str], Optional[str]]
PlanStep = Tuple[Optional[str], str]


class ForcedWinTiming(str, Enum):
    """Whether the win existed in the initial hand or arose after a trigger."""

    INITIAL = "initial"
    CONDITIONAL = "conditional"


class ForcedWinScoreMode(str, Enum):
    """Whether every branch scores alike or some branches offer more points."""

    FIXED = "fixed"
    HIGH_SCORE_BRANCH = "high_score_branch"


class ForcedWinWinner(str, Enum):
    """The teammate expected to finish the round."""

    SELF = "self"
    ALLY = "ally"


class ForcedWinRoute(str, Enum):
    """Whether the attack route is linear or reacts to public play."""

    LINEAR = "linear"
    BRANCHING = "branching"


class ForcedWinProofSource(str, Enum):
    """The evidence used to treat the route as guaranteed."""

    INITIAL_PATTERN = "initial_pattern"
    PUBLIC_EXACT = "public_exact"
    PUBLIC_ABSOLUTE_SAFE = "public_absolute_safe"
    PUBLIC_INFERENCE = "public_inference"


@dataclass(frozen=True)
class ForcedWinPlan:
    """One authoritative guaranteed-win decision and its scoring contract."""

    action: Action
    timing: ForcedWinTiming
    score_mode: ForcedWinScoreMode
    winner: ForcedWinWinner
    route: ForcedWinRoute
    proof_source: ForcedWinProofSource
    minimum_score: float
    expected_score: float
    maximum_score: float
    immediate: bool
    trigger: str
    planned_steps: Tuple[PlanStep, ...] = tuple()

    @property
    def category(self) -> str:
        """Return one of the four primary guaranteed-win categories."""
        prefix = "initial" if self.timing == ForcedWinTiming.INITIAL else "conditional"
        suffix = (
            "fixed"
            if self.score_mode == ForcedWinScoreMode.FIXED
            else "high_score_branch"
        )
        return f"{prefix}_{suffix}"

    def as_dict(self) -> Dict[str, object]:
        """Return a tracker-safe representation used by logs and tests."""
        return {
            "category": self.category,
            "action": self.action,
            "timing": self.timing.value,
            "score_mode": self.score_mode.value,
            "winner": self.winner.value,
            "route": self.route.value,
            "proof_source": self.proof_source.value,
            "minimum_score": self.minimum_score,
            "expected_score": self.expected_score,
            "maximum_score": self.maximum_score,
            "immediate": self.immediate,
            "trigger": self.trigger,
            "planned_steps": list(self.planned_steps),
        }


class ForcedWinPlannerMixin:
    """Normalizes initial patterns and exact endgame proofs into one plan."""

    @staticmethod
    def _forced_win_finish_score(block: Optional[str], attack: str) -> float:
        if block == attack:
            return float(POINTS.get(attack, 0)) * 2.0
        if block is not None and {block, attack} == {"8", "9"}:
            return 100.0
        return float(POINTS.get(attack, 0))

    @staticmethod
    def _is_initial_forced_win_position(state, player: str, tr: Optional[dict]) -> bool:
        return (
            state.phase == "attack"
            and state.turn == player
            and state.attacker is None
            and state.current_attack is None
            and len(state.hands[player]) == 8
            and (tr is None or int(tr.get("my_attack_count", 0)) == 0)
        )

    def _initial_upside_forced_win_plan(
        self,
        state,
        player: str,
        actions: List[Action],
    ) -> Optional[ForcedWinPlan]:
        """Recognize the one-kyosha, big-pair, double-royal opening."""
        hand = list(state.hands[player])
        counts = Counter(hand)
        big_pair = next(
            (piece for piece in ("7", "6") if counts.get(piece, 0) == 2),
            None,
        )
        if not (
            len(hand) == 8
            and counts.get("1", 0) == 3
            and counts.get("2", 0) == 1
            and counts.get("8", 0) == 1
            and counts.get("9", 0) == 1
            and big_pair is not None
        ):
            return None

        action = ("attack_after_block", "1", "2")
        if action not in actions:
            return None

        return ForcedWinPlan(
            action=action,
            timing=ForcedWinTiming.INITIAL,
            score_mode=ForcedWinScoreMode.HIGH_SCORE_BRANCH,
            winner=ForcedWinWinner.SELF,
            route=ForcedWinRoute.BRANCHING,
            proof_source=ForcedWinProofSource.INITIAL_PATTERN,
            minimum_score=50.0,
            expected_score=50.0,
            maximum_score=100.0,
            immediate=False,
            trigger="initial_hand_one_kyosha_big_pair_double_royal",
            planned_steps=(("1", "2"),),
        )

    def _initial_fixed_forced_win_plan(
        self,
        state,
        player: str,
        actions: List[Action],
    ) -> Optional[ForcedWinPlan]:
        route = self._plan_perfect_game(list(state.hands[player]))
        if not route:
            return None

        first_block, first_attack = route[0]
        action = ("attack_after_block", first_block, first_attack)
        if action not in actions:
            return None

        final_block, final_attack = route[-1]
        finish_score = self._forced_win_finish_score(final_block, final_attack)
        return ForcedWinPlan(
            action=action,
            timing=ForcedWinTiming.INITIAL,
            score_mode=ForcedWinScoreMode.FIXED,
            winner=ForcedWinWinner.SELF,
            route=ForcedWinRoute.LINEAR,
            proof_source=ForcedWinProofSource.INITIAL_PATTERN,
            minimum_score=finish_score,
            expected_score=finish_score,
            maximum_score=finish_score,
            immediate=False,
            trigger="initial_hand_fixed_route",
            planned_steps=tuple(route),
        )

    def _exact_forced_win_plans(
        self,
        state,
        player: str,
        actions: List[Action],
        *,
        has_non_king_attack_option: bool,
    ) -> List[Tuple[ForcedWinPlan, float]]:
        tr = self._track.get(id(state))
        plans: List[Tuple[ForcedWinPlan, float]] = []
        for action in actions:
            action_type, block, attack = action
            if action_type not in ("attack", "attack_after_block") or attack is None:
                continue

            result = self._forced_win_result_after_attack_action(
                state,
                player,
                action,
            )
            proof_source = ForcedWinProofSource.PUBLIC_EXACT
            minimum_score = (
                float(result.minimum_score)
                if (
                    result.status == ForcedWinStatus.PROVEN
                    and result.minimum_score is not None
                )
                else None
            )
            expected_score = (
                minimum_score
                if result.expected_score is None
                else float(result.expected_score)
            )
            maximum_score = (
                expected_score
                if result.maximum_score is None
                else float(result.maximum_score)
            )

            if minimum_score is None and tr is not None:
                remaining = self._remaining_hand_after_attack_action(
                    state,
                    player,
                    block,
                    attack,
                )
                if (
                    remaining is not None
                    and len(remaining) == 2
                    and self._is_absolute_safe_for_tsume(
                        state,
                        player,
                        attack,
                        tr,
                    )
                ):
                    minimum_score = self._pair_finish_score(remaining)
                    expected_score = minimum_score
                    maximum_score = minimum_score
                    proof_source = ForcedWinProofSource.PUBLIC_ABSOLUTE_SAFE

            if minimum_score is None:
                continue

            score_mode = (
                ForcedWinScoreMode.HIGH_SCORE_BRANCH
                if maximum_score > minimum_score or expected_score > minimum_score
                else ForcedWinScoreMode.FIXED
            )
            route = (
                ForcedWinRoute.BRANCHING
                if score_mode == ForcedWinScoreMode.HIGH_SCORE_BRANCH
                else ForcedWinRoute.LINEAR
            )
            immediate = self._finish_score_after_action(state, player, action) is not None
            heuristic = self._score_attack_phase(
                state,
                player,
                action_type,
                block,
                attack,
                has_non_king_attack_option=has_non_king_attack_option,
            )
            plans.append((
                ForcedWinPlan(
                    action=action,
                    timing=ForcedWinTiming.CONDITIONAL,
                    score_mode=score_mode,
                    winner=ForcedWinWinner.SELF,
                    route=route,
                    proof_source=proof_source,
                    minimum_score=minimum_score,
                    expected_score=expected_score,
                    maximum_score=maximum_score,
                    immediate=immediate,
                    trigger="current_public_position",
                ),
                heuristic,
            ))
        return plans

    @staticmethod
    def _forced_win_plan_preference_key(
        plan: ForcedWinPlan,
        heuristic: float,
    ) -> Tuple[float, float, float, int, int, float]:
        return (
            plan.minimum_score,
            plan.expected_score,
            plan.maximum_score,
            1 if plan.immediate else 0,
            1 if plan.action[2] not in ("8", "9") else 0,
            heuristic,
        )

    def _forced_win_plan_action(
        self,
        state,
        player: str,
        actions: List[Action],
        *,
        has_non_king_attack_option: bool,
    ) -> Optional[ForcedWinPlan]:
        """Return the highest-value guaranteed action before normal strategy."""
        tr = self._track.get(id(state))
        candidates: List[Tuple[ForcedWinPlan, float]] = []

        if self._is_initial_forced_win_position(state, player, tr):
            upside_plan = self._initial_upside_forced_win_plan(
                state,
                player,
                actions,
            )
            if upside_plan is not None:
                candidates.append((upside_plan, 0.0))

            fixed_plan = self._initial_fixed_forced_win_plan(
                state,
                player,
                actions,
            )
            if fixed_plan is not None:
                candidates.append((fixed_plan, 0.0))

        candidates.extend(self._exact_forced_win_plans(
            state,
            player,
            actions,
            has_non_king_attack_option=has_non_king_attack_option,
        ))

        if not candidates:
            if tr is not None:
                tr["active_forced_win_plan"] = None
            return None

        plan, _heuristic = max(
            candidates,
            key=lambda item: self._forced_win_plan_preference_key(
                item[0],
                item[1],
            ),
        )
        if tr is not None:
            plan_data = plan.as_dict()
            tr["active_forced_win_plan"] = plan_data
            tr["last_forced_win_score_plan"] = {
                "minimum_score": plan.minimum_score,
                "expected_score": plan.expected_score,
                "maximum_score": plan.maximum_score,
                "attack": plan.action[2],
                "category": plan.category,
                "proof_source": plan.proof_source.value,
            }
        return plan

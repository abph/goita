"""Evaluates generated attack graphs with public-information guarantees.
It combines existing forced-win proofs, minimum scores, receive width, and
explicit danger reports while avoiding probabilities and hidden-hand access.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from enum import IntEnum
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from goita_ai2.constants import PIECE_TOTALS
from goita_ai2.current_ai.branched_attack_plan import (
    Action,
    AttackPlanBranch,
    AttackPlanEvaluation,
    AttackPlanNode,
    BranchedAttackPlan,
    PlanActorScope,
    PublicPlanEventKind,
)
from goita_ai2.current_ai.branched_attack_inference import (
    BranchSupportLevel,
    BranchedAttackBranchSupport,
)
from goita_ai2.current_ai.endgame import ForcedWinResult, ForcedWinStatus


class AttackRouteDangerSeverity(IntEnum):
    """Ordinal severity used only for deterministic route comparison."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass(frozen=True)
class AttackRouteDanger:
    """One concrete weakness found in an action or response branch."""

    code: str
    severity: AttackRouteDangerSeverity
    node_id: str
    branch_label: str
    reason: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity.name.lower(),
            "severity_value": int(self.severity),
            "node_id": self.node_id,
            "branch_label": self.branch_label,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AttackNodeEvaluation:
    """Public proof and defensive shape after one planned attack."""

    node_id: str
    attack_number: int
    action: Optional[Action]
    proof_status: ForcedWinStatus
    minimum_score: float
    maximum_score: float
    receive_width: float
    remaining_hand_size: int
    dangers: Tuple[AttackRouteDanger, ...] = tuple()

    def as_dict(self) -> Dict[str, object]:
        return {
            "node_id": self.node_id,
            "attack_number": self.attack_number,
            "action": self.action,
            "proof_status": self.proof_status.value,
            "minimum_score": self.minimum_score,
            "maximum_score": self.maximum_score,
            "receive_width": self.receive_width,
            "remaining_hand_size": self.remaining_hand_size,
            "dangers": [danger.as_dict() for danger in self.dangers],
        }


@dataclass(frozen=True)
class AttackBranchInferenceEvaluation:
    """Current-inference support and projected result for one branch."""

    support: BranchedAttackBranchSupport
    target_node_id: str
    outcome_kind: str
    route_continues: bool
    minimum_score: float
    maximum_score: float
    receive_width: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "support": self.support.as_dict(),
            "target_node_id": self.target_node_id,
            "outcome_kind": self.outcome_kind,
            "route_continues": self.route_continues,
            "minimum_score": self.minimum_score,
            "maximum_score": self.maximum_score,
            "receive_width": self.receive_width,
        }


@dataclass(frozen=True)
class AttackPlanInferenceSummary:
    """Plan-level consequences from ordinal and probabilistic support."""

    supported_branches: int
    likely_branches: int
    impossible_branches: int
    likely_minimum_score: float
    likely_maximum_score: float
    likely_receive_width: float
    inference_wrong_receive_width: float
    low_confidence_branches: int
    route_end_risks: int
    uncertainty_penalty: float
    probability_coverage: float = 0.0
    expected_minimum_score: float = 0.0
    expected_score: float = 0.0
    expected_maximum_score: float = 0.0
    expected_receive_width: float = 0.0
    probability_failure_risk: float = 0.0

    def as_dict(self) -> Dict[str, object]:
        return {
            "supported_branches": self.supported_branches,
            "likely_branches": self.likely_branches,
            "impossible_branches": self.impossible_branches,
            "likely_minimum_score": self.likely_minimum_score,
            "likely_maximum_score": self.likely_maximum_score,
            "likely_receive_width": self.likely_receive_width,
            "inference_wrong_receive_width": self.inference_wrong_receive_width,
            "low_confidence_branches": self.low_confidence_branches,
            "route_end_risks": self.route_end_risks,
            "uncertainty_penalty": self.uncertainty_penalty,
            "probability_coverage": self.probability_coverage,
            "expected_minimum_score": self.expected_minimum_score,
            "expected_score": self.expected_score,
            "expected_maximum_score": self.expected_maximum_score,
            "expected_receive_width": self.expected_receive_width,
            "probability_failure_risk": self.probability_failure_risk,
        }


@dataclass(frozen=True)
class AttackPlanEvaluationReport:
    """Full stage-three explanation for one generated plan."""

    plan_id: str
    root_proof_status: ForcedWinStatus
    proof_scope: str
    node_evaluations: Tuple[AttackNodeEvaluation, ...]
    branch_inference: Tuple[AttackBranchInferenceEvaluation, ...]
    inference_summary: AttackPlanInferenceSummary
    dangerous_branches: Tuple[AttackRouteDanger, ...]
    evaluation: AttackPlanEvaluation

    def as_dict(self) -> Dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "root_proof_status": self.root_proof_status.value,
            "proof_scope": self.proof_scope,
            "node_evaluations": [item.as_dict() for item in self.node_evaluations],
            "branch_inference": [item.as_dict() for item in self.branch_inference],
            "inference_summary": self.inference_summary.as_dict(),
            "dangerous_branches": [danger.as_dict() for danger in self.dangerous_branches],
            "evaluation": self.evaluation.as_dict(),
        }


@dataclass(frozen=True)
class EvaluatedBranchedAttackPlan:
    """A generated plan paired with its stage-three report."""

    plan: BranchedAttackPlan
    report: AttackPlanEvaluationReport

    def as_dict(self) -> Dict[str, object]:
        return {
            "plan": self.plan.as_dict(),
            "report": self.report.as_dict(),
        }


class BranchedAttackEvaluatorMixin:
    """Scores stage-two plans without yet choosing a production action."""

    @staticmethod
    def _branched_receive_width(hand: Sequence[str]) -> float:
        """Count attack types that the remaining hand can receive."""
        counts = Counter(str(piece) for piece in hand)
        has_royal = counts.get("8", 0) + counts.get("9", 0) > 0
        width = 0
        for attack in (str(piece) for piece in range(1, 10)):
            if counts.get(attack, 0) > 0:
                width += 1
            elif has_royal and attack not in ("1", "2"):
                width += 1
        return float(width)

    def _branched_initial_forced_result(
        self,
        state,
        player: str,
        root_action: Action,
    ) -> Optional[ForcedWinResult]:
        """Map both initial guaranteed-win pattern categories to one result."""
        tr = self._track.get(id(state))
        if not self._is_initial_forced_win_position(state, player, tr):
            return None
        candidates = []
        upside = self._initial_upside_forced_win_plan(state, player, [root_action])
        if upside is not None:
            candidates.append(upside)
        fixed_score = self._branched_initial_fixed_score_after_action(
            state,
            player,
            root_action,
        )
        if fixed_score is not None:
            return ForcedWinResult(
                status=ForcedWinStatus.PROVEN,
                minimum_score=fixed_score,
                expected_score=fixed_score,
                maximum_score=fixed_score,
            )
        if not candidates:
            return None
        best = max(
            candidates,
            key=lambda plan: (
                plan.minimum_score,
                plan.maximum_score,
                plan.expected_score,
            ),
        )
        return ForcedWinResult(
            status=ForcedWinStatus.PROVEN,
            minimum_score=best.minimum_score,
            expected_score=best.minimum_score,
            maximum_score=best.maximum_score,
        )

    def _branched_initial_fixed_score_after_action(
        self,
        state,
        player: str,
        root_action: Action,
    ) -> Optional[float]:
        """Prove a fixed initial route while keeping the requested first action."""
        hand = list(state.hands[player])
        if len(hand) != 8 or root_action[0] != "attack_after_block":
            return None
        safe_pieces = self._absolute_safe_pieces_from_initial_hand(hand)
        root_attack = root_action[2]
        if root_attack is None or root_attack not in safe_pieces:
            return None
        remaining = self._branched_remove_action(hand, root_action)
        if remaining is None:
            return None

        def search(current: Tuple[str, ...], attack_number: int) -> Optional[float]:
            if len(current) == 2:
                return max(0.0, float(self._pair_finish_score(list(current))))
            best: Optional[float] = None
            for attack in sorted(set(current)):
                if attack_number < 4 and attack not in safe_pieces:
                    continue
                after_attack = list(current)
                after_attack.remove(attack)
                for block in sorted(set(after_attack)):
                    after_step = list(after_attack)
                    after_step.remove(block)
                    score = search(tuple(sorted(after_step)), attack_number + 1)
                    if score is not None and (best is None or score > best):
                        best = score
            return best

        return search(remaining, 2)

    def _branched_root_forced_result(
        self,
        state,
        player: str,
        root_action: Action,
    ) -> ForcedWinResult:
        initial = self._branched_initial_forced_result(state, player, root_action)
        if initial is not None:
            return initial
        # The main decision pipeline has already checked exact guaranteed wins.
        # In the bounded branch planner, avoid starting another deep proof from
        # a large hand because that single call cannot be interrupted safely.
        if getattr(self, "_branched_attack_evaluation_deadline", None) is not None:
            remaining = self._branched_remove_action(
                state.hands[player],
                root_action,
            )
            if remaining is not None and len(remaining) > 4:
                return self._forced_win_unknown()
        return self._forced_win_result_after_attack_action(state, player, root_action)

    def _branched_node_forced_result(
        self,
        state,
        player: str,
        node: AttackPlanNode,
        pool: Optional[Counter],
        minimum_enemy_hand: int,
        memo: Dict[tuple, ForcedWinResult],
    ) -> ForcedWinResult:
        if node.action is None or node.action[2] is None:
            return self._forced_win_unknown()
        action_type, block, attack = node.action
        remaining = tuple(sorted(node.reserved_pieces))
        if not remaining:
            return self._forced_win_proven(
                self._forced_win_finish_score(block, str(attack))
            )
        if pool is None:
            return self._forced_win_unknown()
        if len(remaining) > int(getattr(self, "EXACT_FORCED_WIN_MAX_HAND", 6)):
            return self._forced_win_unknown()
        if (
            getattr(self, "_branched_attack_evaluation_deadline", None) is not None
            and len(remaining) > 4
        ):
            return self._forced_win_unknown()
        if action_type not in ("attack", "attack_after_block"):
            return self._forced_win_counterexample()
        return self._forced_win_resolve_attack(
            remaining,
            pool.copy(),
            attack=str(attack),
            king_used=int(getattr(state, "king_block_used", 0)) > 0,
            had_both_kings=bool(state.had_both_kings.get(player, False)),
            depth=int(getattr(self, "EXACT_FORCED_WIN_MAX_DEPTH", 18)),
            external_cards_used=0,
            minimum_enemy_hand=minimum_enemy_hand,
            memo=memo,
        )

    @staticmethod
    def _branched_result_scores(result: ForcedWinResult) -> Tuple[float, float]:
        if result.status != ForcedWinStatus.PROVEN or result.minimum_score is None:
            return 0.0, 0.0
        minimum = float(result.minimum_score)
        maximum = (
            minimum
            if result.maximum_score is None
            else float(result.maximum_score)
        )
        return minimum, maximum

    def _branched_fourth_piece_count(
        self,
        state,
        hand: Sequence[str],
    ) -> int:
        tr = self._track.get(id(state))
        public_seen = tr.get("public_seen_counts", {}) if tr is not None else {}
        counts = Counter(hand)
        return sum(
            1
            for piece in ("2", "3", "4", "5")
            if counts.get(piece, 0) > 0
            and counts.get(piece, 0) + int(public_seen.get(piece, 0)) >= PIECE_TOTALS[piece]
        )

    def _branched_action_dangers(
        self,
        state,
        node: AttackPlanNode,
        result: ForcedWinResult,
    ) -> Tuple[AttackRouteDanger, ...]:
        if node.action is None:
            return tuple()
        _action_type, block, attack = node.action
        remaining = tuple(node.reserved_pieces)
        hand_before = list(remaining)
        if block is not None:
            hand_before.append(block)
        if attack is not None:
            hand_before.append(attack)
        dangers: List[AttackRouteDanger] = []

        def add(code: str, severity: AttackRouteDangerSeverity, reason: str) -> None:
            dangers.append(AttackRouteDanger(
                code=code,
                severity=severity,
                node_id=node.node_id,
                branch_label="",
                reason=reason,
            ))

        if block in ("8", "9") and remaining:
            add(
                "royal_hidden_before_finish",
                AttackRouteDangerSeverity.CRITICAL,
                "A royal is hidden before the finishing pair, removing receive width.",
            )
        if attack in ("8", "9") and node.attack_number < 3 and len(remaining) > 2:
            add(
                "early_royal_attack",
                AttackRouteDangerSeverity.HIGH,
                "A royal is attacked before the third attack without an immediate finish.",
            )
        if block == "1" and remaining.count("1") == 0 and len(remaining) > 2:
            add(
                "last_shi_hidden",
                AttackRouteDangerSeverity.HIGH,
                "The last shi is hidden while later enemy shi attacks remain possible.",
            )
        if (
            attack in ("2", "3", "4", "5")
            and node.attack_number < 3
            and hand_before.count(str(attack)) == 1
        ):
            add(
                "single_middle_attacked_early",
                AttackRouteDangerSeverity.MEDIUM,
                "A lone middle piece is shown before the third attack.",
            )
        if (
            attack in ("2", "3", "4", "5")
            and node.attack_number < 3
            and hand_before.count(str(attack)) == 1
            and self._branched_fourth_piece_count(state, hand_before) > 0
        ):
            tr = self._track.get(id(state))
            seen = tr.get("public_seen_counts", {}) if tr is not None else {}
            if int(seen.get(str(attack), 0)) + 1 >= PIECE_TOTALS[str(attack)]:
                add(
                    "fourth_middle_spent_before_third",
                    AttackRouteDangerSeverity.HIGH,
                    "A public fourth middle piece is spent before the third attack.",
                )

        receive_width = self._branched_receive_width(remaining)
        if len(remaining) > 2 and receive_width == 0:
            add(
                "no_receive_width",
                AttackRouteDangerSeverity.CRITICAL,
                "The route leaves no known receive type before the finish.",
            )
        elif len(remaining) > 2 and receive_width <= 1:
            add(
                "narrow_receive_width",
                AttackRouteDangerSeverity.HIGH,
                "The route leaves only one known receive type.",
            )
        if len(remaining) > 2 and "1" not in remaining:
            tr = self._track.get(id(state))
            pool = tr.get("unknown_piece_pool", {}) if tr is not None else {}
            if int(pool.get("1", 0)) > 0:
                add(
                    "enemy_shi_exposure",
                    AttackRouteDangerSeverity.HIGH,
                    "No shi remains to receive a possible enemy shi attack.",
                )
        if result.status == ForcedWinStatus.COUNTEREXAMPLE:
            add(
                "forced_win_counterexample",
                AttackRouteDangerSeverity.HIGH,
                "The public forced-win solver found at least one reply that stops this route.",
            )
        elif result.status == ForcedWinStatus.UNKNOWN:
            add(
                "forced_win_unresolved",
                AttackRouteDangerSeverity.LOW,
                "The public forced-win solver could not prove this continuation.",
            )
        return tuple(dangers)

    def _branched_branch_dangers(
        self,
        plan: BranchedAttackPlan,
    ) -> Tuple[AttackRouteDanger, ...]:
        dangers: List[AttackRouteDanger] = []
        for node in plan.nodes:
            for branch in node.branches:
                target = plan.node(branch.target_node_id)
                if target.checkpoint and not any(
                    nested.condition.kind != PublicPlanEventKind.ALWAYS
                    for nested in target.branches
                ):
                    dangers.append(AttackRouteDanger(
                        code="response_checkpoint_without_receive_route",
                        severity=AttackRouteDangerSeverity.CRITICAL,
                        node_id=node.node_id,
                        branch_label=branch.label,
                        reason="The response branch waits at a checkpoint with no continuation.",
                    ))
                    continue
                if not target.terminal:
                    continue
                purpose = target.purpose.lower()
                if branch.condition.kind == PublicPlanEventKind.ALWAYS:
                    severity = AttackRouteDangerSeverity.LOW
                    code = "unexpected_response_replan"
                    reason = "An unclassified public response discards the current route."
                elif "no legal continuation" in purpose or "invalid" in purpose:
                    severity = AttackRouteDangerSeverity.CRITICAL
                    code = "public_branch_without_continuation"
                    reason = "A classified public response has no generated continuation."
                else:
                    severity = AttackRouteDangerSeverity.HIGH
                    code = "public_branch_ends_plan"
                    reason = "A classified public response ends the generated route."
                dangers.append(AttackRouteDanger(
                    code=code,
                    severity=severity,
                    node_id=node.node_id,
                    branch_label=branch.label,
                    reason=reason,
                ))
        unique = {
            (danger.code, danger.node_id, danger.branch_label): danger
            for danger in dangers
        }
        return tuple(unique.values())

    @staticmethod
    def _branched_should_assess_branch(
        node: AttackPlanNode,
        branch: AttackPlanBranch,
    ) -> bool:
        if node.action is not None:
            return True
        return branch.condition.actor_scope == PlanActorScope.SELF

    def _branched_reachable_action_evaluations(
        self,
        plan: BranchedAttackPlan,
        node_id: str,
        evaluations: Dict[str, AttackNodeEvaluation],
        visited: Optional[set] = None,
    ) -> Tuple[AttackNodeEvaluation, ...]:
        visited = set() if visited is None else set(visited)
        if node_id in visited:
            return tuple()
        visited.add(node_id)
        node = plan.node(node_id)
        current = evaluations.get(node_id)
        if current is not None:
            return (current,)
        if node.terminal:
            return tuple()
        reachable: Dict[str, AttackNodeEvaluation] = {}
        for nested in node.branches:
            for item in self._branched_reachable_action_evaluations(
                plan,
                nested.target_node_id,
                evaluations,
                visited,
            ):
                reachable[item.node_id] = item
        return tuple(reachable.values())

    def _branched_branch_inference_evaluations(
        self,
        state,
        player: str,
        plan: BranchedAttackPlan,
        node_evaluations: Sequence[AttackNodeEvaluation],
    ) -> Tuple[AttackBranchInferenceEvaluation, ...]:
        evaluation_map = {
            item.node_id: item
            for item in node_evaluations
        }
        results: List[AttackBranchInferenceEvaluation] = []
        deadline = getattr(self, "_branched_attack_evaluation_deadline", None)
        for node in plan.nodes:
            if deadline is not None and time.perf_counter() >= float(deadline):
                break
            for branch in node.branches:
                if deadline is not None and time.perf_counter() >= float(deadline):
                    break
                if not self._branched_should_assess_branch(node, branch):
                    continue
                support = self._branched_branch_support(
                    state,
                    player,
                    node,
                    branch,
                )
                target = plan.node(branch.target_node_id)
                reachable = (
                    tuple()
                    if support.level <= BranchSupportLevel.LOW
                    else self._branched_reachable_action_evaluations(
                        plan,
                        target.node_id,
                        evaluation_map,
                    )
                )
                minimum_score = min(
                    (item.minimum_score for item in reachable),
                    default=0.0,
                )
                maximum_score = max(
                    (item.maximum_score for item in reachable),
                    default=0.0,
                )
                widths = [
                    item.receive_width
                    for item in reachable
                    if item.remaining_hand_size > 2
                ]
                if not widths:
                    widths = [item.receive_width for item in reachable]
                receive_width = min(widths, default=0.0)
                if target.terminal:
                    outcome_kind = "route_ends"
                    route_continues = False
                elif target.checkpoint:
                    outcome_kind = "wait_or_replan"
                    route_continues = bool(reachable) or (
                        support.level <= BranchSupportLevel.LOW
                    )
                else:
                    outcome_kind = "planned_continuation"
                    route_continues = True
                results.append(AttackBranchInferenceEvaluation(
                    support=support,
                    target_node_id=target.node_id,
                    outcome_kind=outcome_kind,
                    route_continues=route_continues,
                    minimum_score=minimum_score,
                    maximum_score=maximum_score,
                    receive_width=receive_width,
                ))
        return tuple(results)

    @staticmethod
    def _branched_inference_adjusted_branch_dangers(
        dangers: Sequence[AttackRouteDanger],
        branch_inference: Sequence[AttackBranchInferenceEvaluation],
    ) -> Tuple[AttackRouteDanger, ...]:
        support_by_branch = {
            (item.support.node_id, item.support.branch_label): item.support
            for item in branch_inference
        }
        adjusted = []
        for danger in dangers:
            support = support_by_branch.get((danger.node_id, danger.branch_label))
            if support is None:
                adjusted.append(danger)
                continue
            if support.level == BranchSupportLevel.IMPOSSIBLE:
                continue
            reduction = (
                2
                if support.level == BranchSupportLevel.LOW
                else 1
                if support.level == BranchSupportLevel.POSSIBLE
                else 0
            )
            severity = AttackRouteDangerSeverity(max(
                int(AttackRouteDangerSeverity.LOW),
                int(danger.severity) - reduction,
            ))
            if severity != danger.severity:
                danger = replace(
                    danger,
                    severity=severity,
                    reason=(
                        f"{danger.reason} Current hand inference rates this "
                        f"branch as {support.level.label}."
                    ),
                )
            adjusted.append(danger)
        return tuple(adjusted)

    @staticmethod
    def _branched_inference_certainty(
        branch_inference: Sequence[AttackBranchInferenceEvaluation],
    ) -> float:
        by_node: Dict[str, List[BranchedAttackBranchSupport]] = {}
        for item in branch_inference:
            if item.support.event_kind == PublicPlanEventKind.ALWAYS:
                continue
            by_node.setdefault(item.support.node_id, []).append(item.support)
        if not by_node:
            return 0.0
        node_values = []
        for supports in by_node.values():
            strongest = max(
                supports,
                key=lambda support: (
                    support.probability_confidence,
                    support.event_probability or 0.0,
                    support.level,
                ),
            )
            if strongest.event_probability is not None:
                node_values.append(strongest.probability_confidence)
            else:
                confidence_factor = 0.5 + min(1.0, strongest.confidence) * 0.5
                node_values.append(
                    strongest.level.comparison_weight * confidence_factor
                )
        return max(0.0, min(1.0, sum(node_values) / len(node_values)))

    @staticmethod
    def _branched_probability_consequences(
        branch_inference: Sequence[AttackBranchInferenceEvaluation],
    ) -> Dict[str, float]:
        """Normalize overlapping public-event marginals within each plan node."""
        by_node: Dict[str, List[AttackBranchInferenceEvaluation]] = {}
        for item in branch_inference:
            if item.support.event_kind == PublicPlanEventKind.ALWAYS:
                continue
            if item.support.event_probability is None:
                continue
            by_node.setdefault(item.support.node_id, []).append(item)
        if not by_node:
            return {
                "coverage": 0.0,
                "minimum": 0.0,
                "score": 0.0,
                "maximum": 0.0,
                "receive_width": 0.0,
                "failure_risk": 0.0,
            }

        node_values = []
        for items in by_node.values():
            total = sum(max(0.0, item.support.event_probability or 0.0) for item in items)
            if total <= 0.0:
                continue
            minimum = maximum = width = continuation = 0.0
            for item in items:
                weight = max(0.0, item.support.event_probability or 0.0) / total
                minimum += weight * item.minimum_score
                maximum += weight * item.maximum_score
                width += weight * item.receive_width
                continuation += weight * (1.0 if item.route_continues else 0.0)
            node_values.append({
                "coverage": min(1.0, total),
                "minimum": minimum,
                "score": (minimum + maximum) * 0.5,
                "maximum": maximum,
                "receive_width": width,
                "failure_risk": max(0.0, 1.0 - continuation),
            })
        if not node_values:
            return {
                "coverage": 0.0,
                "minimum": 0.0,
                "score": 0.0,
                "maximum": 0.0,
                "receive_width": 0.0,
                "failure_risk": 0.0,
            }
        count = float(len(node_values))
        return {
            key: sum(item[key] for item in node_values) / count
            for key in (
                "coverage",
                "minimum",
                "score",
                "maximum",
                "receive_width",
                "failure_risk",
            )
        }

    @staticmethod
    def _branched_plan_inference_summary(
        branch_inference: Sequence[AttackBranchInferenceEvaluation],
    ) -> AttackPlanInferenceSummary:
        """Aggregate branch estimates without treating them as proof."""
        considered = [
            item
            for item in branch_inference
            if item.support.event_kind != PublicPlanEventKind.ALWAYS
        ]
        supported = [
            item
            for item in considered
            if item.support.level != BranchSupportLevel.IMPOSSIBLE
        ]
        likely = [
            item
            for item in considered
            if item.support.level >= BranchSupportLevel.LIKELY
        ]
        likely_continuations = [item for item in likely if item.route_continues]
        likely_widths = [
            item.receive_width
            for item in likely_continuations
            if item.receive_width > 0
        ]
        wrong_case_continuations = [
            item
            for item in supported
            if item.route_continues
        ]
        wrong_case_widths = [
            item.receive_width
            for item in wrong_case_continuations
            if item.receive_width > 0
        ]
        low_confidence = sum(
            1
            for item in supported
            if item.support.level <= BranchSupportLevel.POSSIBLE
            or item.support.confidence < 0.35
        )
        route_end_risks = sum(
            1
            for item in supported
            if item.support.level >= BranchSupportLevel.POSSIBLE
            and not item.route_continues
        )
        unsupported_continuation = bool(supported) and any(
            not item.route_continues
            for item in supported
        )
        inference_wrong_receive_width = (
            0.0
            if unsupported_continuation
            else min(wrong_case_widths, default=0.0)
        )
        uncertainty_penalty = min(
            0.35,
            low_confidence * 0.04
            + route_end_risks * 0.08
            + (0.06 if supported and inference_wrong_receive_width <= 0 else 0.0),
        )
        probability = BranchedAttackEvaluatorMixin._branched_probability_consequences(
            branch_inference
        )
        return AttackPlanInferenceSummary(
            supported_branches=len(supported),
            likely_branches=len(likely),
            impossible_branches=sum(
                1
                for item in considered
                if item.support.level == BranchSupportLevel.IMPOSSIBLE
            ),
            likely_minimum_score=min(
                (item.minimum_score for item in likely_continuations),
                default=0.0,
            ),
            likely_maximum_score=max(
                (item.maximum_score for item in likely_continuations),
                default=0.0,
            ),
            likely_receive_width=min(likely_widths, default=0.0),
            inference_wrong_receive_width=inference_wrong_receive_width,
            low_confidence_branches=low_confidence,
            route_end_risks=route_end_risks,
            uncertainty_penalty=round(uncertainty_penalty, 4),
            probability_coverage=round(probability["coverage"], 6),
            expected_minimum_score=round(probability["minimum"], 3),
            expected_score=round(probability["score"], 3),
            expected_maximum_score=round(probability["maximum"], 3),
            expected_receive_width=round(probability["receive_width"], 3),
            probability_failure_risk=round(probability["failure_risk"], 6),
        )

    @staticmethod
    def _branched_failure_risk(dangers: Iterable[AttackRouteDanger]) -> float:
        """Convert explicit severities to a bounded heuristic, not a probability."""
        weights = {
            AttackRouteDangerSeverity.LOW: 0.04,
            AttackRouteDangerSeverity.MEDIUM: 0.14,
            AttackRouteDangerSeverity.HIGH: 0.42,
            AttackRouteDangerSeverity.CRITICAL: 0.78,
        }
        values = sorted(
            (weights[danger.severity] for danger in dangers),
            reverse=True,
        )
        if not values:
            return 0.0
        return min(1.0, values[0] + sum(values[1:]) * 0.04)

    def _evaluate_branched_attack_plan(
        self,
        state,
        player: str,
        plan: BranchedAttackPlan,
    ) -> EvaluatedBranchedAttackPlan:
        """Evaluate one graph and replace its stage-two seed metrics."""
        self._ensure_trackers(state)
        tr = self._track.get(id(state))
        pool = self._public_unknown_piece_pool(state, player, tr) if tr is not None else None
        enemies = [
            seat
            for seat in ("A", "B", "C", "D")
            if seat != player and not self._same_team(seat, player)
        ]
        minimum_enemy_hand = min((len(state.hands[seat]) for seat in enemies), default=0)
        root = plan.node(plan.root_node_id)
        if root.action is None:
            root_result = self._forced_win_unknown()
        else:
            root_result = self._branched_root_forced_result(state, player, root.action)

        memo: Dict[tuple, ForcedWinResult] = {}
        node_evaluations: List[AttackNodeEvaluation] = []
        action_dangers: List[AttackRouteDanger] = []
        deadline = getattr(self, "_branched_attack_evaluation_deadline", None)
        for node in plan.nodes:
            if node.action is None:
                continue
            if (
                node_evaluations
                and deadline is not None
                and time.perf_counter() >= float(deadline)
            ):
                break
            result = (
                root_result
                if node.node_id == plan.root_node_id
                else self._branched_node_forced_result(
                    state,
                    player,
                    node,
                    pool,
                    minimum_enemy_hand,
                    memo,
                )
            )
            minimum_score, maximum_score = self._branched_result_scores(result)
            dangers = self._branched_action_dangers(state, node, result)
            action_dangers.extend(dangers)
            node_evaluations.append(AttackNodeEvaluation(
                node_id=node.node_id,
                attack_number=node.attack_number,
                action=node.action,
                proof_status=result.status,
                minimum_score=minimum_score,
                maximum_score=maximum_score,
                receive_width=self._branched_receive_width(node.reserved_pieces),
                remaining_hand_size=len(node.reserved_pieces),
                dangers=dangers,
            ))

        branch_inference = self._branched_branch_inference_evaluations(
            state,
            player,
            plan,
            node_evaluations,
        )
        inference_summary = self._branched_plan_inference_summary(branch_inference)
        branch_dangers = self._branched_inference_adjusted_branch_dangers(
            self._branched_branch_dangers(plan),
            branch_inference,
        )
        all_dangers = tuple(action_dangers) + branch_dangers
        root_minimum, root_maximum = self._branched_result_scores(root_result)
        proven_maxima = [
            item.maximum_score
            for item in node_evaluations
            if item.proof_status == ForcedWinStatus.PROVEN
        ]
        maximum_score = max(
            [float(plan.evaluation.maximum_score), root_maximum] + proven_maxima
        )
        root_node_evaluation = next(
            (
                item
                for item in node_evaluations
                if item.node_id == plan.root_node_id
            ),
            None,
        )
        width_candidates = (
            [root_node_evaluation.receive_width]
            if root_node_evaluation is not None
            else []
        )
        width_candidates.extend(
            item.receive_width
            for item in branch_inference
            if item.route_continues
            and item.receive_width > 0
            and item.support.level >= BranchSupportLevel.POSSIBLE
        )
        minimum_receive_width = min(width_candidates, default=0.0)
        remaining_hands = [
            node.reserved_pieces
            for node in plan.nodes
            if node.action is not None and len(node.reserved_pieces) > 2
        ]
        preserved_royals = min(
            (hand.count("8") + hand.count("9") for hand in remaining_hands),
            default=0,
        )
        preserved_fourth = min(
            (self._branched_fourth_piece_count(state, hand) for hand in remaining_hands),
            default=0,
        )
        covered_responses = sum(
            1
            for item in branch_inference
            if item.support.event_kind != PublicPlanEventKind.ALWAYS
            and item.support.level != BranchSupportLevel.IMPOSSIBLE
        )
        conclusive = root_result.status != ForcedWinStatus.UNKNOWN
        inference_certainty = self._branched_inference_certainty(branch_inference)
        public_certainty = (
            1.0
            if conclusive
            else max(
                0.0,
                inference_certainty * (1.0 - inference_summary.uncertainty_penalty),
            )
        )
        failure_risk = self._branched_failure_risk(all_dangers)
        if root_result.status != ForcedWinStatus.PROVEN:
            failure_risk = min(
                1.0,
                max(failure_risk, inference_summary.probability_failure_risk)
                + inference_summary.uncertainty_penalty,
            )
        evaluation = AttackPlanEvaluation(
            guaranteed_win=root_result.status == ForcedWinStatus.PROVEN,
            minimum_score=root_minimum,
            maximum_score=max(root_minimum, maximum_score),
            covered_public_responses=covered_responses,
            public_certainty=public_certainty,
            receive_width=minimum_receive_width,
            preserved_royals=preserved_royals,
            preserved_fourth_pieces=preserved_fourth,
            failure_risk=failure_risk,
            route_length=max(
                (item.attack_number for item in node_evaluations),
                default=0,
            ),
            expected_score=max(
                root_minimum,
                min(
                    max(root_minimum, maximum_score),
                    inference_summary.expected_score,
                ),
            ),
            probability_coverage=inference_summary.probability_coverage,
            probability_failure_risk=(
                0.0
                if root_result.status == ForcedWinStatus.PROVEN
                else inference_summary.probability_failure_risk
            ),
        )
        evaluated_plan = replace(plan, evaluation=evaluation)
        report = AttackPlanEvaluationReport(
            plan_id=plan.plan_id,
            root_proof_status=root_result.status,
            proof_scope="root action with public-event replanning",
            node_evaluations=tuple(node_evaluations),
            branch_inference=branch_inference,
            inference_summary=inference_summary,
            dangerous_branches=branch_dangers,
            evaluation=evaluation,
        )
        return EvaluatedBranchedAttackPlan(evaluated_plan, report)

    def _evaluate_branched_attack_plans(
        self,
        state,
        player: str,
        plans: Sequence[BranchedAttackPlan],
    ) -> Tuple[EvaluatedBranchedAttackPlan, ...]:
        """Evaluate every generated root candidate without selecting one."""
        return tuple(
            self._evaluate_branched_attack_plan(state, player, plan)
            for plan in plans
        )


__all__ = [
    "AttackBranchInferenceEvaluation",
    "AttackNodeEvaluation",
    "AttackPlanInferenceSummary",
    "AttackPlanEvaluationReport",
    "AttackRouteDanger",
    "AttackRouteDangerSeverity",
    "BranchedAttackEvaluatorMixin",
    "EvaluatedBranchedAttackPlan",
]

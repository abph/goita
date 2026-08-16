"""Generates attack-one candidates and their public response branches.
Each route plans attacks two and three while keeping hidden opponent hands out
of the generator; later stages will score and persist the generated plans.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from goita_ai2.constants import PIECE_TOTALS, POINTS
from goita_ai2.current_ai.branched_attack_plan import (
    Action,
    AttackPlanBranch,
    AttackPlanEvaluation,
    AttackPlanNode,
    BranchedAttackPlan,
    PlanActorScope,
    PublicBranchCondition,
    PublicPlanEventKind,
)


@dataclass(frozen=True)
class AttackContinuationCandidate:
    """One legal-looking future attack and its remaining own hand."""

    action: Action
    remaining_hand: Tuple[str, ...]
    projected_score: float
    finish_score: float


@dataclass(frozen=True)
class PublicResponseBranch:
    """A possible public response to one planned attack."""

    condition: PublicBranchCondition
    label: str
    requires_replan_before_attack: bool


class _PlanGraphBuilder:
    """Builds immutable nodes while assigning stable local identifiers."""

    def __init__(self) -> None:
        self.nodes: Dict[str, AttackPlanNode] = {}
        self._counter = 0

    def node_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def add(self, node: AttackPlanNode) -> str:
        if node.node_id in self.nodes:
            raise ValueError(f"duplicate generated node: {node.node_id}")
        self.nodes[node.node_id] = node
        return node.node_id

    def terminal(self, attack_number: int, purpose: str) -> str:
        node_id = self.node_id("terminal")
        return self.add(AttackPlanNode(
            node_id=node_id,
            action=None,
            attack_number=attack_number,
            purpose=purpose,
            terminal=True,
        ))


class BranchedAttackGeneratorMixin:
    """Creates stage-two attack trees without selecting a live action."""

    BRANCHED_ATTACK_MAX_ATTACKS = 3
    BRANCHED_ATTACK_CONTINUATION_BEAM = 3

    @staticmethod
    def _branched_root_attack_candidates(actions: Sequence[Action]) -> Tuple[Action, ...]:
        """Return every current legal attack candidate in deterministic order."""
        return tuple(sorted(
            (
                action
                for action in actions
                if action[0] in ("attack", "attack_after_block")
                and action[2] is not None
            ),
            key=lambda action: (
                action[2] or "",
                action[1] or "",
                action[0],
            ),
        ))

    @staticmethod
    def _branched_remove_action(
        hand: Sequence[str],
        action: Action,
    ) -> Optional[Tuple[str, ...]]:
        action_type, block, attack = action
        if action_type not in ("attack", "attack_after_block") or attack is None:
            return None
        remaining = list(str(piece) for piece in hand)
        if block is not None:
            if block not in remaining:
                return None
            remaining.remove(block)
        if attack not in remaining:
            return None
        remaining.remove(attack)
        return tuple(sorted(remaining))

    def _branched_scope_may_hold_piece(
        self,
        state,
        player: str,
        tr: Optional[dict],
        scope: PlanActorScope,
        piece: str,
    ) -> bool:
        """Use only public inference maxima to decide if a branch is possible."""
        if tr is None:
            return True
        seats = []
        for seat in ("A", "B", "C", "D"):
            if seat == player:
                continue
            is_ally = self._same_team(seat, player)
            if scope == PlanActorScope.ALLY and is_ally:
                seats.append(seat)
            elif scope == PlanActorScope.ENEMY and not is_ally:
                seats.append(seat)
        if not seats:
            return False

        estimates = tr.get("estimated_current_hands", {})
        saw_estimate = False
        for seat in seats:
            estimate = estimates.get(seat, {}).get(piece)
            if isinstance(estimate, dict):
                saw_estimate = True
                if int(estimate.get("max", 0)) > 0:
                    return True
        if saw_estimate:
            return False

        return int(tr.get("unknown_piece_pool", {}).get(piece, 0)) > 0

    def _branched_public_response_branches(
        self,
        state,
        player: str,
        attack: str,
        attack_number: int,
    ) -> Tuple[PublicResponseBranch, ...]:
        """Generate same-piece, royal, full-lap, and fallback responses."""
        tr = self._track.get(id(state))
        branches: List[PublicResponseBranch] = []
        for scope in (PlanActorScope.ENEMY, PlanActorScope.ALLY):
            if self._branched_scope_may_hold_piece(state, player, tr, scope, attack):
                branches.append(PublicResponseBranch(
                    condition=PublicBranchCondition(
                        PublicPlanEventKind.SAME_PIECE_RECEIVE,
                        actor_scope=scope,
                        piece=attack,
                        attack_number=attack_number,
                    ),
                    label=f"{scope.value}_same_piece_receive_{attack}",
                    requires_replan_before_attack=True,
                ))

        if attack not in ("1", "2"):
            for scope in (PlanActorScope.ENEMY, PlanActorScope.ALLY):
                if any(
                    self._branched_scope_may_hold_piece(
                        state,
                        player,
                        tr,
                        scope,
                        royal,
                    )
                    for royal in ("8", "9")
                ):
                    branches.append(PublicResponseBranch(
                        condition=PublicBranchCondition(
                            PublicPlanEventKind.ROYAL_RECEIVE,
                            actor_scope=scope,
                            attack_number=attack_number,
                        ),
                        label=f"{scope.value}_royal_receive",
                        requires_replan_before_attack=True,
                    ))

        for scope in (PlanActorScope.ENEMY, PlanActorScope.ALLY):
            branches.append(PublicResponseBranch(
                condition=PublicBranchCondition(
                    PublicPlanEventKind.PASS,
                    actor_scope=scope,
                    attack_number=attack_number,
                ),
                label=f"{scope.value}_pass",
                requires_replan_before_attack=False,
            ))

        branches.append(PublicResponseBranch(
            condition=PublicBranchCondition(
                PublicPlanEventKind.LAP_COMPLETED,
                attack_number=attack_number,
            ),
            label="attack_returned_after_full_lap",
            requires_replan_before_attack=False,
        ))
        branches.append(PublicResponseBranch(
            condition=PublicBranchCondition(PublicPlanEventKind.ALWAYS),
            label="unexpected_public_response",
            requires_replan_before_attack=True,
        ))
        return tuple(branches)

    def _branched_future_royal_attack_allowed(
        self,
        state,
        player: str,
        hand_size: int,
        *,
        requires_block: bool,
    ) -> bool:
        return (
            bool(state.had_both_kings.get(player, False))
            or int(getattr(state, "king_block_used", 0)) > 0
            or hand_size == (2 if requires_block else 1)
        )

    def _branched_continuation_candidates(
        self,
        state,
        player: str,
        hand: Sequence[str],
        *,
        previous_attack: Optional[str],
        attack_number: int,
        requires_block: bool,
        preferred_attack: Optional[str] = None,
        future_preferred_attacks: Sequence[str] = tuple(),
        permanently_protected_pieces: Sequence[str] = tuple(),
        limit: Optional[int] = None,
    ) -> Tuple[AttackContinuationCandidate, ...]:
        """Generate and rank future attack/block pairs from the owner's hand."""
        current_hand = tuple(sorted(str(piece) for piece in hand))
        if not current_hand:
            return tuple()
        royal_allowed = self._branched_future_royal_attack_allowed(
            state,
            player,
            len(current_hand),
            requires_block=requires_block,
        )
        candidates: List[AttackContinuationCandidate] = []

        if requires_block:
            raw_actions: Iterable[Action] = (
                ("attack_after_block", block, attack)
                for block in sorted(set(current_hand))
                for attack in sorted(set(current_hand))
                if block != attack or current_hand.count(block) >= 2
            )
        else:
            raw_actions = (
                ("attack", None, attack)
                for attack in sorted(set(current_hand))
            )

        for action in raw_actions:
            action_type, block, attack = action
            if attack is None:
                continue
            if attack in ("8", "9") and not royal_allowed:
                continue
            remaining = self._branched_remove_action(current_hand, action)
            if remaining is None:
                continue
            required_after = Counter(str(piece) for piece in future_preferred_attacks)
            permanent = Counter(str(piece) for piece in permanently_protected_pieces)
            if any(
                remaining.count(piece) < max(required_after[piece], permanent[piece])
                for piece in set(required_after) | set(permanent)
            ):
                continue

            score = self._planned_future_attack_value(
                state,
                player,
                attack,
                attack_number,
                previous_attack,
                current_hand,
            )
            if block is not None:
                score += self._planned_future_block_value(
                    state,
                    player,
                    block,
                    current_hand,
                )

            finish_score = 0.0
            if len(remaining) == 2:
                finish_score = max(0.0, float(self._pair_finish_score(list(remaining))))
            elif remaining and len(remaining) % 2 == 0:
                future = self._search_future_attack_plan(
                    state,
                    player,
                    list(remaining),
                    attack,
                    attack_number + 1,
                )
                score += float(future.get("score", 0.0))
                finish_score = max(0.0, float(future.get("finish_score", 0.0)))

            candidates.append(AttackContinuationCandidate(
                action=action,
                remaining_hand=remaining,
                projected_score=score,
                finish_score=finish_score,
            ))

        candidates.sort(
            key=lambda candidate: (
                1 if preferred_attack is not None and candidate.action[2] == preferred_attack else 0,
                candidate.projected_score,
                candidate.finish_score,
                POINTS.get(candidate.action[2] or "1", 0),
                -POINTS.get(candidate.action[1] or "1", 0),
                candidate.action,
            ),
            reverse=True,
        )
        beam = self.BRANCHED_ATTACK_CONTINUATION_BEAM if limit is None else limit
        return tuple(candidates[:max(0, int(beam))])

    def _branched_wait_checkpoint(
        self,
        builder: _PlanGraphBuilder,
        state,
        player: str,
        hand: Tuple[str, ...],
        previous_attack: str,
        next_attack_number: int,
        preferred_attacks: Tuple[str, ...] = tuple(),
        permanently_protected_pieces: Tuple[str, ...] = tuple(),
    ) -> str:
        """Wait for the owner to receive a returned attack, then continue."""
        branches: List[AttackPlanBranch] = []
        for block in sorted(set(hand)):
            after_receive = list(hand)
            after_receive.remove(block)
            continuations = self._branched_continuation_candidates(
                state,
                player,
                after_receive,
                previous_attack=previous_attack,
                attack_number=next_attack_number,
                requires_block=False,
                preferred_attack=self._branched_preferred_attack(
                    preferred_attacks,
                    next_attack_number,
                ),
                future_preferred_attacks=preferred_attacks[next_attack_number:],
                permanently_protected_pieces=permanently_protected_pieces,
                limit=1,
            )
            if not continuations:
                continue
            target = self._branched_build_attack_node(
                builder,
                state,
                player,
                tuple(sorted(after_receive)),
                continuations[0].action,
                next_attack_number,
                preferred_attacks,
                permanently_protected_pieces,
            )
            kind = (
                PublicPlanEventKind.ROYAL_RECEIVE
                if block in ("8", "9")
                else PublicPlanEventKind.SAME_PIECE_RECEIVE
            )
            branches.append(AttackPlanBranch(
                condition=PublicBranchCondition(
                    kind,
                    actor_scope=PlanActorScope.SELF,
                    piece=block,
                ),
                target_node_id=target,
                label=f"self_receive_{block}_then_attack",
            ))

        fallback = builder.terminal(
            max(0, next_attack_number - 1),
            "rebuild after an unplanned returned attack",
        )
        branches.append(AttackPlanBranch(
            PublicBranchCondition(PublicPlanEventKind.ALWAYS),
            fallback,
            "returned attack was outside the generated receive branches",
        ))
        checkpoint_id = builder.node_id("wait_for_return")
        return builder.add(AttackPlanNode(
            node_id=checkpoint_id,
            action=None,
            attack_number=max(0, next_attack_number - 1),
            branches=tuple(branches),
            purpose="wait until the plan owner receives a public returned attack",
            reserved_pieces=hand,
            checkpoint=True,
        ))

    def _branched_build_attack_node(
        self,
        builder: _PlanGraphBuilder,
        state,
        player: str,
        hand_before_action: Tuple[str, ...],
        action: Action,
        attack_number: int,
        preferred_attacks: Tuple[str, ...] = tuple(),
        permanently_protected_pieces: Tuple[str, ...] = tuple(),
    ) -> str:
        remaining = self._branched_remove_action(hand_before_action, action)
        if remaining is None:
            return builder.terminal(attack_number, "generated action became invalid")

        node_id = builder.node_id(f"attack_{attack_number}")
        attack = str(action[2])
        if (
            attack_number >= self.BRANCHED_ATTACK_MAX_ATTACKS
            or len(remaining) <= 2
        ):
            return builder.add(AttackPlanNode(
                node_id=node_id,
                action=action,
                attack_number=attack_number,
                purpose=f"planned attack {attack_number}",
                reserved_pieces=remaining,
            ))

        response_specs = self._branched_public_response_branches(
            state,
            player,
            attack,
            attack_number,
        )
        wait_checkpoint = self._branched_wait_checkpoint(
            builder,
            state,
            player,
            remaining,
            attack,
            attack_number + 1,
            preferred_attacks,
            permanently_protected_pieces,
        )
        continuations = self._branched_continuation_candidates(
            state,
            player,
            remaining,
            previous_attack=attack,
            attack_number=attack_number + 1,
            requires_block=True,
            preferred_attack=self._branched_preferred_attack(
                preferred_attacks,
                attack_number + 1,
            ),
            future_preferred_attacks=preferred_attacks[attack_number + 1:],
            permanently_protected_pieces=permanently_protected_pieces,
            limit=1,
        )
        lap_target = (
            self._branched_build_attack_node(
                builder,
                state,
                player,
                remaining,
                continuations[0].action,
                attack_number + 1,
                preferred_attacks,
                permanently_protected_pieces,
            )
            if continuations
            else builder.terminal(
                attack_number,
                "no legal continuation after a full lap",
            )
        )
        fallback_target = builder.terminal(
            attack_number,
            "discard route after an unexpected public response",
        )
        response_checkpoint_id = builder.node_id("wait_for_response")

        def target_for(response: PublicResponseBranch) -> str:
            if response.condition.kind == PublicPlanEventKind.ALWAYS:
                return fallback_target
            if response.condition.kind == PublicPlanEventKind.LAP_COMPLETED:
                return lap_target
            if response.condition.kind == PublicPlanEventKind.PASS:
                return response_checkpoint_id
            return wait_checkpoint

        checkpoint_branches = tuple(
            AttackPlanBranch(
                response.condition,
                target_for(response),
                response.label,
            )
            for response in response_specs
        )
        builder.add(AttackPlanNode(
            node_id=response_checkpoint_id,
            action=None,
            attack_number=attack_number,
            branches=checkpoint_branches,
            purpose="keep observing passes until a receive or full lap is public",
            reserved_pieces=remaining,
            checkpoint=True,
        ))
        branches = list(checkpoint_branches)

        return builder.add(AttackPlanNode(
            node_id=node_id,
            action=action,
            attack_number=attack_number,
            branches=tuple(branches),
            purpose=f"planned attack {attack_number} and observe its public response",
            reserved_pieces=remaining,
        ))

    @staticmethod
    def _branched_preferred_attack(
        preferred_attacks: Sequence[str],
        attack_number: int,
    ) -> Optional[str]:
        """Read a one-indexed preferred attack from a template sequence."""
        index = int(attack_number) - 1
        if index < 0 or index >= len(preferred_attacks):
            return None
        piece = str(preferred_attacks[index])
        return piece if piece in PIECE_TOTALS else None

    def _branched_plan_evaluation_seed(
        self,
        state,
        player: str,
        root_action: Action,
        nodes: Sequence[AttackPlanNode],
    ) -> AttackPlanEvaluation:
        """Attach public structural metrics; proof and route scoring come next."""
        remaining = self._branched_remove_action(state.hands[player], root_action) or tuple()
        tr = self._track.get(id(state))
        public_seen = tr.get("public_seen_counts", {}) if tr is not None else {}
        fourth_pieces = sum(
            1
            for piece in ("2", "3", "4", "5")
            if remaining.count(piece) > 0
            and remaining.count(piece) + int(public_seen.get(piece, 0)) >= PIECE_TOTALS[piece]
        )
        response_count = sum(
            1
            for node in nodes
            for branch in node.branches
            if branch.condition.kind != PublicPlanEventKind.ALWAYS
        )
        maximum_score = (
            max(0.0, float(self._best_pair_finish_score_from_hand(list(remaining))))
            if len(remaining) >= 2
            else 0.0
        )
        return AttackPlanEvaluation(
            guaranteed_win=False,
            minimum_score=0.0,
            maximum_score=maximum_score,
            covered_public_responses=response_count,
            public_certainty=1.0,
            receive_width=self._planned_receive_width(list(remaining)),
            preserved_royals=remaining.count("8") + remaining.count("9"),
            preserved_fourth_pieces=fourth_pieces,
            failure_risk=1.0,
            route_length=max(
                (node.attack_number for node in nodes if node.action is not None),
                default=0,
            ),
        )

    def _generate_branched_attack_plans(
        self,
        state,
        player: str,
        actions: Sequence[Action],
    ) -> Tuple[BranchedAttackPlan, ...]:
        """Bundle root candidates and attacks two/three into immutable graphs."""
        self._ensure_trackers(state)
        tr = self._track.get(id(state))
        revision = int(tr.get("piece_inference_revision", 0)) if tr is not None else 0
        attack_number = 1
        if tr is not None:
            attack_number = int(tr.get("my_attack_count", 0)) + 1

        plans: List[BranchedAttackPlan] = []
        for index, root_action in enumerate(
            self._branched_root_attack_candidates(actions),
            start=1,
        ):
            plans.append(self._build_branched_attack_plan(
                state,
                player,
                root_action,
                attack_number=attack_number,
                revision=revision,
                candidate_label=f"c{index}",
            ))
        return tuple(plans)

    def _build_branched_attack_plan(
        self,
        state,
        player: str,
        root_action: Action,
        *,
        attack_number: int,
        revision: int,
        candidate_label: str,
        preferred_attacks: Sequence[str] = tuple(),
        permanently_protected_pieces: Sequence[str] = tuple(),
        source: str = "public_branch_generation",
        assumptions: Sequence[str] = tuple(),
    ) -> BranchedAttackPlan:
        """Build one graph, optionally steering continuation toward a template."""
        sequence = tuple(str(piece) for piece in preferred_attacks)
        builder = _PlanGraphBuilder()
        root_id = self._branched_build_attack_node(
            builder,
            state,
            player,
            tuple(sorted(str(piece) for piece in state.hands[player])),
            root_action,
            attack_number,
            sequence,
            tuple(str(piece) for piece in permanently_protected_pieces),
        )
        nodes = tuple(builder.nodes.values())
        base_assumptions = (
            "branches use public count bounds rather than real opponent hands",
            "future actions must be revalidated when their public event occurs",
        )
        return BranchedAttackPlan(
            plan_id=(
                f"{source}-r{revision}-n{attack_number}-{candidate_label}-"
                f"{root_action[0]}-{root_action[1] or 'x'}-{root_action[2]}"
            ),
            root_node_id=root_id,
            nodes=nodes,
            evaluation=self._branched_plan_evaluation_seed(
                state,
                player,
                root_action,
                nodes,
            ),
            source=source,
            public_revision=revision,
            assumptions=base_assumptions + tuple(str(item) for item in assumptions),
        )


__all__ = [
    "AttackContinuationCandidate",
    "BranchedAttackGeneratorMixin",
    "PublicResponseBranch",
]

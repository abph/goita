"""Defines the shared foundation for branched attack plans.
The module records public-event branches, future actions, and route quality
without reading an opponent's hidden hand or changing live AI behavior yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, Iterable, Optional, Tuple


Action = Tuple[str, Optional[str], Optional[str]]
PIECES = frozenset(str(piece) for piece in range(1, 10))
ACTION_TYPES = frozenset(("pass", "receive", "attack", "attack_after_block"))


class AttackDecisionPriority(IntEnum):
    """Target priority shared by existing and future attack planners."""

    NORMAL_SCORING = 100
    SHALLOW_WHOLE_HAND = 200
    FIXED_ATTACK_SEQUENCE = 300
    BRANCHED_ATTACK_PLAN = 400
    PUBLIC_TACTIC = 500
    GUARANTEED_SCORE = 600
    IMMEDIATE_WIN = 700


@dataclass(frozen=True)
class AttackDecisionLayer:
    """One audited layer in the current attack-decision pipeline."""

    name: str
    priority: AttackDecisionPriority
    current_hook: str
    protected_from_time_search: bool
    description: str


# This is an architectural audit, not a second implementation of decision.py.
# The target branched planner sits below proven/public tactics and above the
# existing fixed sequence and shallow route heuristics.
ATTACK_DECISION_AUDIT: Tuple[AttackDecisionLayer, ...] = (
    AttackDecisionLayer(
        name="immediate_win",
        priority=AttackDecisionPriority.IMMEDIATE_WIN,
        current_hook="_high_score_tsume_action / _win_now_bonus",
        protected_from_time_search=True,
        description="Finish now using the highest publicly proven score.",
    ),
    AttackDecisionLayer(
        name="guaranteed_score",
        priority=AttackDecisionPriority.GUARANTEED_SCORE,
        current_hook="_forced_win_plan_action / exact endgame helpers",
        protected_from_time_search=True,
        description="Choose a guaranteed route by its minimum finishing score.",
    ),
    AttackDecisionLayer(
        name="public_tactic",
        priority=AttackDecisionPriority.PUBLIC_TACTIC,
        current_hook="conditional tsume / kakari / shi signal / inferred endgame",
        protected_from_time_search=True,
        description="Honor a tactic proved or requested by public play.",
    ),
    AttackDecisionLayer(
        name="branched_attack_plan",
        priority=AttackDecisionPriority.BRANCHED_ATTACK_PLAN,
        current_hook="future integration point before fixed sequences",
        protected_from_time_search=False,
        description="Follow a route that changes after observable responses.",
    ),
    AttackDecisionLayer(
        name="fixed_attack_sequence",
        priority=AttackDecisionPriority.FIXED_ATTACK_SEQUENCE,
        current_hook="_special_attack_sequence_action",
        protected_from_time_search=True,
        description="Follow a known hand-shape attack order.",
    ),
    AttackDecisionLayer(
        name="shallow_whole_hand",
        priority=AttackDecisionPriority.SHALLOW_WHOLE_HAND,
        current_hook="_eight_card_shallow_plan_action",
        protected_from_time_search=False,
        description="Compare one provisional route through the whole hand.",
    ),
    AttackDecisionLayer(
        name="normal_scoring",
        priority=AttackDecisionPriority.NORMAL_SCORING,
        current_hook="_score_attack_phase",
        protected_from_time_search=False,
        description="Use local attack, block, inference, and safety scores.",
    ),
)


class PlanActorScope(str, Enum):
    """Relationship between the observed player and the plan owner."""

    ANY = "any"
    SELF = "self"
    ALLY = "ally"
    ENEMY = "enemy"


class PublicPlanEventKind(str, Enum):
    """Observable events that are allowed to drive a plan branch."""

    PASS = "pass"
    RECEIVE = "receive"
    SAME_PIECE_RECEIVE = "same_piece_receive"
    ROYAL_RECEIVE = "royal_receive"
    ATTACK = "attack"
    REACH = "reach"
    LAP_COMPLETED = "lap_completed"
    ROUND_FINISHED = "round_finished"
    ALWAYS = "always"


@dataclass(frozen=True)
class PublicPlanEvent:
    """A branch input made only from publicly observable game state."""

    kind: PublicPlanEventKind
    actor_scope: PlanActorScope = PlanActorScope.ANY
    piece: Optional[str] = None
    current_attack: Optional[str] = None
    attack_number: Optional[int] = None
    reached: bool = False
    round_finished: bool = False

    def __post_init__(self) -> None:
        if self.piece is not None and self.piece not in PIECES:
            raise ValueError(f"invalid public piece: {self.piece}")
        if self.current_attack is not None and self.current_attack not in PIECES:
            raise ValueError(f"invalid current attack: {self.current_attack}")
        if self.attack_number is not None and self.attack_number < 1:
            raise ValueError("attack_number must be positive")


@dataclass(frozen=True)
class PublicBranchCondition:
    """A serializable predicate over one public event."""

    kind: PublicPlanEventKind
    actor_scope: PlanActorScope = PlanActorScope.ANY
    piece: Optional[str] = None
    attack_number: Optional[int] = None

    def __post_init__(self) -> None:
        if self.piece is not None and self.piece not in PIECES:
            raise ValueError(f"invalid condition piece: {self.piece}")
        if self.attack_number is not None and self.attack_number < 1:
            raise ValueError("attack_number must be positive")
        if self.kind == PublicPlanEventKind.ALWAYS and (
            self.piece is not None
            or self.attack_number is not None
            or self.actor_scope != PlanActorScope.ANY
        ):
            raise ValueError("an always condition cannot contain filters")

    def matches(self, event: PublicPlanEvent) -> bool:
        """Return whether the public event selects this branch."""
        if self.kind == PublicPlanEventKind.ALWAYS:
            return True
        if self.actor_scope not in (PlanActorScope.ANY, event.actor_scope):
            return False
        if self.attack_number is not None and self.attack_number != event.attack_number:
            return False
        if self.piece is not None and self.piece != event.piece:
            return False

        if self.kind == PublicPlanEventKind.SAME_PIECE_RECEIVE:
            return (
                event.kind in (
                    PublicPlanEventKind.RECEIVE,
                    PublicPlanEventKind.SAME_PIECE_RECEIVE,
                    PublicPlanEventKind.ROYAL_RECEIVE,
                )
                and event.piece is not None
                and event.piece == event.current_attack
            )
        if self.kind == PublicPlanEventKind.ROYAL_RECEIVE:
            return (
                event.kind in (
                    PublicPlanEventKind.RECEIVE,
                    PublicPlanEventKind.SAME_PIECE_RECEIVE,
                    PublicPlanEventKind.ROYAL_RECEIVE,
                )
                and event.piece in ("8", "9")
            )
        if self.kind == PublicPlanEventKind.REACH:
            return event.kind == PublicPlanEventKind.REACH or event.reached
        if self.kind == PublicPlanEventKind.ROUND_FINISHED:
            return event.kind == PublicPlanEventKind.ROUND_FINISHED or event.round_finished
        return self.kind == event.kind

    def as_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind.value,
            "actor_scope": self.actor_scope.value,
            "piece": self.piece,
            "attack_number": self.attack_number,
        }


@dataclass(frozen=True)
class AttackPlanEvaluation:
    """Lexicographic route quality with proof kept above probability."""

    guaranteed_win: bool = False
    minimum_score: float = 0.0
    maximum_score: float = 0.0
    covered_public_responses: int = 0
    public_certainty: float = 0.0
    receive_width: float = 0.0
    preserved_royals: int = 0
    preserved_fourth_pieces: int = 0
    failure_risk: float = 1.0
    route_length: int = 0
    expected_score: float = 0.0
    probability_coverage: float = 0.0
    probability_failure_risk: float = 0.0

    def __post_init__(self) -> None:
        if self.minimum_score < 0.0 or self.maximum_score < self.minimum_score:
            raise ValueError("scores must satisfy 0 <= minimum_score <= maximum_score")
        if self.covered_public_responses < 0:
            raise ValueError("covered_public_responses cannot be negative")
        if not 0.0 <= self.public_certainty <= 1.0:
            raise ValueError("public_certainty must be between 0 and 1")
        if self.receive_width < 0.0:
            raise ValueError("receive_width cannot be negative")
        if self.preserved_royals < 0 or self.preserved_fourth_pieces < 0:
            raise ValueError("preserved piece counts cannot be negative")
        if not 0.0 <= self.failure_risk <= 1.0:
            raise ValueError("failure_risk must be between 0 and 1")
        if self.route_length < 0:
            raise ValueError("route_length cannot be negative")
        if self.expected_score < 0.0:
            raise ValueError("expected_score cannot be negative")
        if not 0.0 <= self.probability_coverage <= 1.0:
            raise ValueError("probability_coverage must be between 0 and 1")
        if not 0.0 <= self.probability_failure_risk <= 1.0:
            raise ValueError("probability_failure_risk must be between 0 and 1")

    def preference_key(self) -> Tuple[object, ...]:
        """Rank guaranteed points first, then public upside and safety."""
        return (
            1 if self.guaranteed_win else 0,
            self.minimum_score,
            self.expected_score,
            self.maximum_score,
            self.covered_public_responses,
            self.public_certainty,
            -self.failure_risk,
            self.receive_width,
            self.preserved_royals,
            self.preserved_fourth_pieces,
            -self.route_length,
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "guaranteed_win": self.guaranteed_win,
            "minimum_score": self.minimum_score,
            "maximum_score": self.maximum_score,
            "covered_public_responses": self.covered_public_responses,
            "public_certainty": self.public_certainty,
            "receive_width": self.receive_width,
            "preserved_royals": self.preserved_royals,
            "preserved_fourth_pieces": self.preserved_fourth_pieces,
            "failure_risk": self.failure_risk,
            "route_length": self.route_length,
            "expected_score": self.expected_score,
            "probability_coverage": self.probability_coverage,
            "probability_failure_risk": self.probability_failure_risk,
        }


@dataclass(frozen=True)
class AttackPlanBranch:
    """One public condition and the node selected when it becomes true."""

    condition: PublicBranchCondition
    target_node_id: str
    label: str = ""

    def __post_init__(self) -> None:
        if not self.target_node_id:
            raise ValueError("target_node_id is required")

    def as_dict(self) -> Dict[str, object]:
        return {
            "condition": self.condition.as_dict(),
            "target_node_id": self.target_node_id,
            "label": self.label,
        }


@dataclass(frozen=True)
class AttackPlanNode:
    """One intended action followed by ordered public-response branches."""

    node_id: str
    action: Optional[Action]
    attack_number: int
    branches: Tuple[AttackPlanBranch, ...] = tuple()
    purpose: str = ""
    reserved_pieces: Tuple[str, ...] = tuple()
    terminal: bool = False
    checkpoint: bool = False

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id is required")
        if self.attack_number < 0:
            raise ValueError("attack_number cannot be negative")
        if self.terminal and self.checkpoint:
            raise ValueError("a node cannot be both terminal and a checkpoint")
        if self.terminal:
            if self.action is not None or self.branches:
                raise ValueError("a terminal node cannot contain an action or branches")
        elif self.checkpoint:
            if self.action is not None or not self.branches:
                raise ValueError("a checkpoint requires branches and no action")
        elif self.action is None:
            raise ValueError("a non-terminal node requires an action")
        else:
            action_type, block, attack = self.action
            if action_type not in ACTION_TYPES:
                raise ValueError(f"invalid action type: {action_type}")
            if block is not None and block not in PIECES:
                raise ValueError(f"invalid block piece: {block}")
            if attack is not None and attack not in PIECES:
                raise ValueError(f"invalid attack piece: {attack}")

        for piece in self.reserved_pieces:
            if piece not in PIECES:
                raise ValueError(f"invalid reserved piece: {piece}")
        fallback_indexes = [
            index
            for index, branch in enumerate(self.branches)
            if branch.condition.kind == PublicPlanEventKind.ALWAYS
        ]
        if len(fallback_indexes) > 1:
            raise ValueError("a node can have only one fallback branch")
        if fallback_indexes and fallback_indexes[0] != len(self.branches) - 1:
            raise ValueError("the fallback branch must be last")

    def next_node_id(self, event: PublicPlanEvent) -> Optional[str]:
        for branch in self.branches:
            if branch.condition.matches(event):
                return branch.target_node_id
        return None

    def as_dict(self) -> Dict[str, object]:
        return {
            "node_id": self.node_id,
            "action": self.action,
            "attack_number": self.attack_number,
            "branches": [branch.as_dict() for branch in self.branches],
            "purpose": self.purpose,
            "reserved_pieces": list(self.reserved_pieces),
            "terminal": self.terminal,
            "checkpoint": self.checkpoint,
        }


@dataclass(frozen=True)
class BranchedAttackPlan:
    """A validated graph of actions selected by future public events."""

    plan_id: str
    root_node_id: str
    nodes: Tuple[AttackPlanNode, ...]
    evaluation: AttackPlanEvaluation
    source: str
    public_revision: int = 0
    assumptions: Tuple[str, ...] = tuple()
    _node_map: Dict[str, AttackPlanNode] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.plan_id:
            raise ValueError("plan_id is required")
        if self.public_revision < 0:
            raise ValueError("public_revision cannot be negative")
        node_map = {node.node_id: node for node in self.nodes}
        if len(node_map) != len(self.nodes):
            raise ValueError("node ids must be unique")
        if self.root_node_id not in node_map:
            raise ValueError("root_node_id must identify a node")
        for node in self.nodes:
            for branch in node.branches:
                if branch.target_node_id not in node_map:
                    raise ValueError(
                        f"unknown branch target {branch.target_node_id!r} "
                        f"from node {node.node_id!r}"
                    )
        object.__setattr__(self, "_node_map", node_map)

    @property
    def priority(self) -> AttackDecisionPriority:
        return AttackDecisionPriority.BRANCHED_ATTACK_PLAN

    def node(self, node_id: str) -> AttackPlanNode:
        return self._node_map[node_id]

    def advance(self, node_id: str, event: PublicPlanEvent) -> Optional[AttackPlanNode]:
        target = self.node(node_id).next_node_id(event)
        return None if target is None else self.node(target)

    def as_dict(self) -> Dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "root_node_id": self.root_node_id,
            "nodes": [node.as_dict() for node in self.nodes],
            "evaluation": self.evaluation.as_dict(),
            "source": self.source,
            "public_revision": self.public_revision,
            "assumptions": list(self.assumptions),
            "priority": int(self.priority),
        }


def choose_preferred_attack_plan(
    plans: Iterable[BranchedAttackPlan],
) -> Optional[BranchedAttackPlan]:
    """Choose by the shared stage-one lexicographic evaluation contract."""
    candidates = list(plans)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda plan: (plan.evaluation.preference_key(), plan.plan_id),
    )


__all__ = [
    "ATTACK_DECISION_AUDIT",
    "Action",
    "AttackDecisionLayer",
    "AttackDecisionPriority",
    "AttackPlanBranch",
    "AttackPlanEvaluation",
    "AttackPlanNode",
    "BranchedAttackPlan",
    "PlanActorScope",
    "PublicBranchCondition",
    "PublicPlanEvent",
    "PublicPlanEventKind",
    "choose_preferred_attack_plan",
]

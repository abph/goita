"""Classifies public attack-plan branches with the current hand estimates.

It combines hard min/max bounds with a sampled joint posterior to estimate the
chance of receives, royal receives, passes, and full laps. Hard public facts
remain authoritative, and the module never reads another player's hidden hand.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Dict, Iterable, Optional, Sequence, Tuple

from goita_ai2.current_ai.branched_attack_plan import (
    AttackPlanBranch,
    AttackPlanNode,
    PlanActorScope,
    PublicPlanEventKind,
)


class BranchSupportLevel(IntEnum):
    """Ordinal support supplied by the current public hand inference."""

    IMPOSSIBLE = 0
    LOW = 1
    POSSIBLE = 2
    LIKELY = 3
    CERTAIN = 4

    @property
    def label(self) -> str:
        return self.name.lower()

    @property
    def comparison_weight(self) -> float:
        return {
            BranchSupportLevel.IMPOSSIBLE: 0.0,
            BranchSupportLevel.LOW: 0.2,
            BranchSupportLevel.POSSIBLE: 0.5,
            BranchSupportLevel.LIKELY: 0.8,
            BranchSupportLevel.CERTAIN: 1.0,
        }[self]


@dataclass(frozen=True)
class BranchedAttackBranchSupport:
    """Auditable evidence for one public response branch."""

    node_id: str
    branch_label: str
    event_kind: PublicPlanEventKind
    actor_scope: PlanActorScope
    pieces: Tuple[str, ...]
    seats: Tuple[str, ...]
    level: BranchSupportLevel
    minimum_count: int
    maximum_count: int
    expected_count: float
    map_count: int
    confidence: float
    holding_probability: Optional[float] = None
    event_probability: Optional[float] = None
    probability_confidence: float = 0.0
    probability_source: str = "ordinal_only"
    probability_sample_count: int = 0
    evidence: Tuple[str, ...] = tuple()

    def as_dict(self) -> Dict[str, object]:
        return {
            "node_id": self.node_id,
            "branch_label": self.branch_label,
            "event_kind": self.event_kind.value,
            "actor_scope": self.actor_scope.value,
            "pieces": list(self.pieces),
            "seats": list(self.seats),
            "level": self.level.label,
            "level_value": int(self.level),
            "comparison_weight": self.level.comparison_weight,
            "minimum_count": self.minimum_count,
            "maximum_count": self.maximum_count,
            "expected_count": self.expected_count,
            "map_count": self.map_count,
            "confidence": self.confidence,
            "holding_probability": (
                None
                if self.holding_probability is None
                else round(self.holding_probability, 6)
            ),
            "event_probability": (
                None
                if self.event_probability is None
                else round(self.event_probability, 6)
            ),
            "probability_confidence": round(self.probability_confidence, 6),
            "probability_source": self.probability_source,
            "probability_sample_count": self.probability_sample_count,
            "evidence": list(self.evidence),
        }


class BranchedAttackInferenceCache:
    """Thread-safe bounded LRU for immutable branch-support results."""

    def __init__(self, max_entries: int = 512) -> None:
        self.max_entries = max(1, int(max_entries))
        self._entries: "OrderedDict[tuple, BranchedAttackBranchSupport]" = OrderedDict()
        self._lock = threading.RLock()
        self._counters = {"hits": 0, "misses": 0, "stores": 0, "evictions": 0}

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def get(self, key: tuple) -> Optional[BranchedAttackBranchSupport]:
        with self._lock:
            value = self._entries.get(key)
            if value is None:
                self._counters["misses"] += 1
                return None
            self._entries.move_to_end(key)
            self._counters["hits"] += 1
            return value

    def put(self, key: tuple, value: BranchedAttackBranchSupport) -> None:
        with self._lock:
            self._entries.pop(key, None)
            self._entries[key] = value
            self._counters["stores"] += 1
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self._counters["evictions"] += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            hits = int(self._counters["hits"])
            misses = int(self._counters["misses"])
            lookups = hits + misses
            return {
                **self._counters,
                "lookups": lookups,
                "hit_rate": round(hits / lookups if lookups else 0.0, 5),
                "size": len(self._entries),
                "max_entries": self.max_entries,
            }


class BranchedAttackInferenceMixin:
    """Converts existing public hand estimates into branch support levels."""

    def _initialize_branched_attack_inference(self) -> None:
        self._branched_attack_inference_cache = BranchedAttackInferenceCache(
            max_entries=int(self.BRANCHED_ATTACK_INFERENCE_CACHE_MAX_ENTRIES),
        )

    def _branched_inference_cache(self) -> BranchedAttackInferenceCache:
        cache = getattr(self, "_branched_attack_inference_cache", None)
        if cache is None:
            self._initialize_branched_attack_inference()
            cache = self._branched_attack_inference_cache
        return cache

    @staticmethod
    def _branched_receive_propensities(
        hand: Sequence[str],
        attack: str,
        attack_number: int,
        estimated_rank: str,
    ) -> Tuple[float, float, float]:
        """Estimate same-piece, royal, and pass behavior in one sampled deal."""
        direct_count = tuple(hand).count(attack)
        royal_count = tuple(hand).count("8") + tuple(hand).count("9")
        strong_rank = estimated_rank in ("SS", "S", "A", "B", "C")
        if direct_count <= 0:
            direct_receive = 0.0
        elif attack_number >= 3:
            direct_receive = 0.98
        elif attack_number == 2:
            direct_receive = 0.94
        elif attack == "1" and direct_count >= 2:
            direct_receive = 0.96
        elif attack == "2":
            direct_receive = 0.82 if strong_rank else 0.48
        else:
            direct_receive = 0.86 if strong_rank else 0.62

        if attack in ("1", "2") or royal_count <= 0:
            royal_receive = 0.0
        elif len(hand) <= 2 or attack_number >= 3:
            royal_receive = 0.98
        elif attack_number == 2:
            royal_receive = 0.62
        else:
            royal_receive = 0.20
        if direct_receive > 0.0:
            royal_receive *= 0.18

        any_receive = 1.0 - (1.0 - direct_receive) * (1.0 - royal_receive)
        return direct_receive, royal_receive, max(0.0, 1.0 - any_receive)

    def _branched_probabilistic_event_probability(
        self,
        state,
        player: str,
        node: AttackPlanNode,
        branch: AttackPlanBranch,
        support: BranchedAttackBranchSupport,
    ) -> BranchedAttackBranchSupport:
        """Attach a posterior event probability while preserving hard bounds."""
        if support.level == BranchSupportLevel.IMPOSSIBLE:
            return replace(
                support,
                holding_probability=0.0,
                event_probability=0.0,
                probability_confidence=1.0,
                probability_source="public_hard_bound",
            )
        if support.level == BranchSupportLevel.CERTAIN and branch.condition.kind in (
            PublicPlanEventKind.PASS,
            PublicPlanEventKind.LAP_COMPLETED,
        ):
            return replace(
                support,
                holding_probability=0.0,
                event_probability=1.0,
                probability_confidence=1.0,
                probability_source="public_hard_bound",
            )
        if branch.condition.kind == PublicPlanEventKind.ALWAYS:
            return replace(
                support,
                holding_probability=None,
                event_probability=None,
                probability_source="fallback_branch",
            )

        try:
            probability_seconds = float(getattr(
                self,
                "BRANCHED_ATTACK_PROBABILISTIC_MAX_SECONDS",
                0.04,
            ))
            planner_deadline = getattr(
                self,
                "_branched_attack_evaluation_deadline",
                None,
            )
            if planner_deadline is not None:
                remaining = float(planner_deadline) - time.perf_counter()
                if remaining <= 0.002:
                    raise ValueError("branched probability budget exhausted")
                probability_seconds = min(
                    probability_seconds,
                    max(0.001, remaining * 0.7),
                )
            posterior = self._posterior_probabilistic_hand_inference(
                state,
                player,
                sample_count=int(getattr(
                    self,
                    "BRANCHED_ATTACK_PROBABILISTIC_SAMPLE_COUNT",
                    64,
                )),
                activate=True,
                max_seconds=probability_seconds,
            )
        except (AttributeError, ValueError):
            return replace(
                support,
                holding_probability=(
                    1.0
                    if support.level == BranchSupportLevel.CERTAIN
                    else support.level.comparison_weight
                ),
                event_probability=support.level.comparison_weight,
                probability_confidence=max(0.0, min(1.0, support.confidence * 0.5)),
                probability_source="ordinal_fallback",
            )

        condition = branch.condition
        scope = (
            PlanActorScope.ANY
            if condition.kind == PublicPlanEventKind.LAP_COMPLETED
            else condition.actor_scope
        )
        seats = self._branched_inference_scope_seats(player, scope)
        attack = str(
            condition.piece
            or (node.action[2] if node.action is not None else "")
            or ""
        )
        attack_number = int(condition.attack_number or node.attack_number or 1)
        tracker = self._track.get(id(state)) or {}
        public_models = tracker.get("public_hand_models", {})
        event_probability = 0.0
        holding_probability = 0.0
        for candidate in posterior.candidates:
            sampled_hands = dict(candidate.prediction.opponent_hands)
            direct_values = []
            royal_values = []
            pass_values = []
            scope_hands = []
            for seat in seats:
                hand = (
                    tuple(node.reserved_pieces)
                    if seat == player
                    else tuple(sampled_hands.get(seat, ()))
                )
                rank = str(public_models.get(seat, {}).get("estimated_rank", "D"))
                direct, royal, passed = self._branched_receive_propensities(
                    hand,
                    attack,
                    attack_number,
                    rank,
                )
                direct_values.append(direct)
                royal_values.append(royal)
                pass_values.append(passed)
                scope_hands.append(hand)

            if condition.kind == PublicPlanEventKind.ROYAL_RECEIVE:
                candidate_holding = any(
                    tuple(hand).count("8") + tuple(hand).count("9") > 0
                    for hand in scope_hands
                )
            else:
                candidate_holding = any(
                    tuple(hand).count(attack) > 0
                    for hand in scope_hands
                )
            holding_probability += candidate.probability * float(candidate_holding)

            if condition.kind == PublicPlanEventKind.SAME_PIECE_RECEIVE:
                candidate_event = 1.0
                for value in direct_values:
                    candidate_event *= 1.0 - value
                candidate_event = 1.0 - candidate_event
            elif condition.kind == PublicPlanEventKind.ROYAL_RECEIVE:
                candidate_event = 1.0
                for value in royal_values:
                    candidate_event *= 1.0 - value
                candidate_event = 1.0 - candidate_event
            elif condition.kind == PublicPlanEventKind.RECEIVE:
                candidate_event = 1.0
                for direct, royal in zip(direct_values, royal_values):
                    candidate_event *= (1.0 - direct) * (1.0 - royal)
                candidate_event = 1.0 - candidate_event
            elif condition.kind == PublicPlanEventKind.LAP_COMPLETED:
                candidate_event = 1.0
                for value in pass_values:
                    candidate_event *= value
            elif condition.kind == PublicPlanEventKind.PASS:
                candidate_event = (
                    sum(pass_values) / len(pass_values)
                    if pass_values
                    else 0.0
                )
            else:
                candidate_event = support.level.comparison_weight
            event_probability += candidate.probability * candidate_event

        event_probability = max(0.0, min(1.0, event_probability))
        holding_probability = max(0.0, min(1.0, holding_probability))
        probability_confidence = max(
            0.0,
            min(
                1.0,
                posterior.confidence * posterior.retained_probability_mass,
            ),
        )
        evidence = tuple(support.evidence) + (
            f"posterior_event_probability:{event_probability:.3f}",
        )
        return replace(
            support,
            holding_probability=holding_probability,
            event_probability=event_probability,
            probability_confidence=probability_confidence,
            probability_source="joint_action_weighted_posterior",
            probability_sample_count=posterior.accepted_samples,
            evidence=evidence,
        )

    @staticmethod
    def _branched_inference_scope_seats(
        player: str,
        scope: PlanActorScope,
    ) -> Tuple[str, ...]:
        if scope == PlanActorScope.SELF:
            return (player,)
        seats = []
        for seat in ("A", "B", "C", "D"):
            if seat == player:
                continue
            same_team = (seat in ("A", "C")) == (player in ("A", "C"))
            if scope == PlanActorScope.ALLY and same_team:
                seats.append(seat)
            elif scope == PlanActorScope.ENEMY and not same_team:
                seats.append(seat)
            elif scope == PlanActorScope.ANY:
                seats.append(seat)
        return tuple(seats)

    @staticmethod
    def _branched_support_level_from_counts(
        *,
        minimum_count: int,
        maximum_count: int,
        expected_count: float,
        map_count: int,
        confidence: float,
    ) -> BranchSupportLevel:
        if maximum_count <= 0:
            return BranchSupportLevel.IMPOSSIBLE
        if minimum_count > 0:
            return BranchSupportLevel.CERTAIN
        if (
            map_count > 0
            and expected_count >= 0.6
            and confidence >= 0.3
        ) or expected_count >= 0.9:
            return BranchSupportLevel.LIKELY
        if expected_count <= 0.15 and confidence >= 0.55:
            return BranchSupportLevel.LOW
        return BranchSupportLevel.POSSIBLE

    @staticmethod
    def _branched_lower_support(
        level: BranchSupportLevel,
        amount: int = 1,
    ) -> BranchSupportLevel:
        return BranchSupportLevel(max(int(BranchSupportLevel.LOW), int(level) - amount))

    @staticmethod
    def _branched_raise_support(
        level: BranchSupportLevel,
        amount: int = 1,
    ) -> BranchSupportLevel:
        return BranchSupportLevel(min(int(BranchSupportLevel.CERTAIN), int(level) + amount))

    def _branched_estimated_piece_support(
        self,
        state,
        player: str,
        node: AttackPlanNode,
        scope: PlanActorScope,
        pieces: Sequence[str],
    ) -> BranchedAttackBranchSupport:
        tr = self._track.get(id(state))
        seats = self._branched_inference_scope_seats(player, scope)
        normalized_pieces = tuple(sorted(set(str(piece) for piece in pieces)))
        evidence = []

        if scope == PlanActorScope.SELF:
            reserved = tuple(str(piece) for piece in node.reserved_pieces)
            counts = [reserved.count(piece) for piece in normalized_pieces]
            total = sum(counts)
            level = (
                BranchSupportLevel.CERTAIN
                if total > 0
                else BranchSupportLevel.IMPOSSIBLE
            )
            return BranchedAttackBranchSupport(
                node_id=node.node_id,
                branch_label="",
                event_kind=PublicPlanEventKind.RECEIVE,
                actor_scope=scope,
                pieces=normalized_pieces,
                seats=seats,
                level=level,
                minimum_count=total,
                maximum_count=total,
                expected_count=float(total),
                map_count=total,
                confidence=1.0,
                evidence=("projected_owner_reserved_hand",),
            )

        estimates = tr.get("estimated_current_hands", {}) if tr is not None else {}
        unknown_pool = tr.get("unknown_piece_pool", {}) if tr is not None else {}
        minimum_count = 0
        maximum_count = 0
        expected_count = 0.0
        map_count = 0
        confidences = []
        saw_estimate = False
        for seat in seats:
            for piece in normalized_pieces:
                estimate = estimates.get(seat, {}).get(piece)
                if not isinstance(estimate, dict):
                    continue
                saw_estimate = True
                minimum_count += int(estimate.get("min", 0))
                maximum_count += int(estimate.get("max", 0))
                expected_count += float(estimate.get("expected", 0.0))
                map_count += int(estimate.get("map_count", 0))
                confidences.append(float(estimate.get("confidence", 0.0)))
                source = str(estimate.get("source", "public_pool"))
                if source not in evidence:
                    evidence.append(f"estimate:{source}")

        if not saw_estimate:
            maximum_count = sum(
                int(unknown_pool.get(piece, 0))
                for piece in normalized_pieces
            )
            evidence.append("fallback:unknown_piece_pool")
        confidence = (
            sum(confidences) / len(confidences)
            if confidences
            else 0.0
        )
        level = self._branched_support_level_from_counts(
            minimum_count=minimum_count,
            maximum_count=maximum_count,
            expected_count=expected_count,
            map_count=map_count,
            confidence=confidence,
        )

        if tr is not None and level not in (
            BranchSupportLevel.IMPOSSIBLE,
            BranchSupportLevel.CERTAIN,
        ):
            models = tr.get("public_hand_models", {})
            strategy_support = False
            for seat in seats:
                model = models.get(seat, {})
                first_attack = str(model.get("first_attack") or "")
                strategy_active = bool(model.get("inferred_attack_strategy_active"))
                strategy_broken = bool(model.get("strategy_broken"))
                if (
                    strategy_active
                    and not strategy_broken
                    and first_attack in normalized_pieces
                    and map_count > 0
                ):
                    strategy_support = True
                    evidence.append(f"{seat}:active_first_attack_strategy")
            if strategy_support:
                level = self._branched_raise_support(level)

            strong_passes = 0
            pass_evidence = tr.get("piece_pass_evidence", {})
            for seat in seats:
                for piece in normalized_pieces:
                    events = pass_evidence.get(seat, {}).get(piece, ())
                    for event in events:
                        if (
                            str(event.get("relation")) == "enemy"
                            and int(event.get("attack_no", 1)) >= 2
                        ):
                            strong_passes += 1
            if strong_passes:
                evidence.append(f"strong_enemy_passes:{strong_passes}")
            if strong_passes >= 2:
                level = self._branched_lower_support(level)

        return BranchedAttackBranchSupport(
            node_id=node.node_id,
            branch_label="",
            event_kind=PublicPlanEventKind.RECEIVE,
            actor_scope=scope,
            pieces=normalized_pieces,
            seats=seats,
            level=level,
            minimum_count=minimum_count,
            maximum_count=maximum_count,
            expected_count=round(expected_count, 3),
            map_count=map_count,
            confidence=round(confidence, 3),
            evidence=tuple(evidence),
        )

    def _branched_pass_support(
        self,
        state,
        player: str,
        node: AttackPlanNode,
        scope: PlanActorScope,
        attack: str,
        *,
        full_lap: bool,
    ) -> BranchedAttackBranchSupport:
        direct = self._branched_estimated_piece_support(
            state,
            player,
            node,
            scope,
            (attack,),
        )
        royal = None
        if attack not in ("1", "2"):
            royal = self._branched_estimated_piece_support(
                state,
                player,
                node,
                scope,
                ("8", "9"),
            )
        receive_level = max(
            direct.level,
            royal.level if royal is not None else BranchSupportLevel.IMPOSSIBLE,
        )
        if receive_level == BranchSupportLevel.IMPOSSIBLE:
            level = BranchSupportLevel.CERTAIN
        elif receive_level in (BranchSupportLevel.CERTAIN, BranchSupportLevel.LIKELY):
            level = BranchSupportLevel.LOW
        elif receive_level == BranchSupportLevel.POSSIBLE:
            level = BranchSupportLevel.POSSIBLE
        else:
            level = BranchSupportLevel.LIKELY

        tr = self._track.get(id(state))
        seats = direct.seats
        prior_passes = 0
        if tr is not None:
            pass_evidence = tr.get("piece_pass_evidence", {})
            prior_passes = sum(
                len(pass_evidence.get(seat, {}).get(attack, ()))
                for seat in seats
            )
        if prior_passes and level not in (
            BranchSupportLevel.CERTAIN,
            BranchSupportLevel.IMPOSSIBLE,
        ):
            level = self._branched_raise_support(level)

        evidence = list(direct.evidence)
        evidence.append(f"receive_support:{receive_level.label}")
        if prior_passes:
            evidence.append(f"prior_matching_passes:{prior_passes}")
        if full_lap:
            evidence.append("all_responders_must_pass")
        return BranchedAttackBranchSupport(
            node_id=node.node_id,
            branch_label="",
            event_kind=(
                PublicPlanEventKind.LAP_COMPLETED
                if full_lap
                else PublicPlanEventKind.PASS
            ),
            actor_scope=scope,
            pieces=(attack,),
            seats=seats,
            level=level,
            minimum_count=direct.minimum_count,
            maximum_count=direct.maximum_count,
            expected_count=direct.expected_count,
            map_count=direct.map_count,
            confidence=direct.confidence,
            evidence=tuple(evidence),
        )

    def _branched_branch_support_uncached(
        self,
        state,
        player: str,
        node: AttackPlanNode,
        branch: AttackPlanBranch,
    ) -> BranchedAttackBranchSupport:
        condition = branch.condition
        attack = str(node.action[2]) if node.action is not None and node.action[2] else ""
        if condition.kind == PublicPlanEventKind.SAME_PIECE_RECEIVE:
            support = self._branched_estimated_piece_support(
                state,
                player,
                node,
                condition.actor_scope,
                (str(condition.piece or attack),),
            )
        elif condition.kind == PublicPlanEventKind.ROYAL_RECEIVE:
            support = self._branched_estimated_piece_support(
                state,
                player,
                node,
                condition.actor_scope,
                ("8", "9"),
            )
        elif condition.kind == PublicPlanEventKind.PASS:
            support = self._branched_pass_support(
                state,
                player,
                node,
                condition.actor_scope,
                attack,
                full_lap=False,
            )
        elif condition.kind == PublicPlanEventKind.LAP_COMPLETED:
            support = self._branched_pass_support(
                state,
                player,
                node,
                PlanActorScope.ANY,
                attack,
                full_lap=True,
            )
        elif condition.kind == PublicPlanEventKind.ALWAYS:
            support = BranchedAttackBranchSupport(
                node_id=node.node_id,
                branch_label=branch.label,
                event_kind=condition.kind,
                actor_scope=condition.actor_scope,
                pieces=tuple(),
                seats=tuple(),
                level=BranchSupportLevel.LOW,
                minimum_count=0,
                maximum_count=0,
                expected_count=0.0,
                map_count=0,
                confidence=0.0,
                evidence=("fallback_for_unclassified_public_event",),
            )
        else:
            pieces = (condition.piece,) if condition.piece is not None else tuple()
            support = BranchedAttackBranchSupport(
                node_id=node.node_id,
                branch_label=branch.label,
                event_kind=condition.kind,
                actor_scope=condition.actor_scope,
                pieces=pieces,
                seats=self._branched_inference_scope_seats(
                    player,
                    condition.actor_scope,
                ),
                level=BranchSupportLevel.POSSIBLE,
                minimum_count=0,
                maximum_count=0,
                expected_count=0.0,
                map_count=0,
                confidence=0.0,
                evidence=("public_event_without_piece_holding_test",),
            )
        support = self._branched_probabilistic_event_probability(
            state,
            player,
            node,
            branch,
            support,
        )
        return BranchedAttackBranchSupport(
            node_id=node.node_id,
            branch_label=branch.label,
            event_kind=condition.kind,
            actor_scope=condition.actor_scope,
            pieces=support.pieces,
            seats=support.seats,
            level=support.level,
            minimum_count=support.minimum_count,
            maximum_count=support.maximum_count,
            expected_count=support.expected_count,
            map_count=support.map_count,
            confidence=support.confidence,
            holding_probability=support.holding_probability,
            event_probability=support.event_probability,
            probability_confidence=support.probability_confidence,
            probability_source=support.probability_source,
            probability_sample_count=support.probability_sample_count,
            evidence=support.evidence,
        )

    def _branched_branch_support_cache_key(
        self,
        state,
        player: str,
        node: AttackPlanNode,
        branch: AttackPlanBranch,
    ) -> tuple:
        tr = self._track.get(id(state)) or {}
        condition = branch.condition
        scope = (
            PlanActorScope.ANY
            if condition.kind == PublicPlanEventKind.LAP_COMPLETED
            else condition.actor_scope
        )
        seats = self._branched_inference_scope_seats(player, scope)
        attack = str(node.action[2]) if node.action is not None and node.action[2] else ""
        if condition.kind == PublicPlanEventKind.ROYAL_RECEIVE:
            pieces = ("8", "9")
        elif condition.kind in (
            PublicPlanEventKind.RECEIVE,
            PublicPlanEventKind.SAME_PIECE_RECEIVE,
            PublicPlanEventKind.PASS,
            PublicPlanEventKind.LAP_COMPLETED,
        ):
            pieces = (str(condition.piece or attack),)
            if condition.kind in (
                PublicPlanEventKind.PASS,
                PublicPlanEventKind.LAP_COMPLETED,
            ) and attack not in ("1", "2"):
                pieces += ("8", "9")
        else:
            pieces = tuple()
        pieces = tuple(sorted(set(pieces)))
        player_revisions = tr.get("piece_inference_player_revisions", {})
        pass_evidence = tr.get("piece_pass_evidence", {})
        public_models = tr.get("public_hand_models", {})
        estimates = tr.get("estimated_current_hands", {})
        return (
            "branched_inference_v2",
            player,
            node.node_id,
            node.action,
            tuple(node.reserved_pieces),
            branch.label,
            condition.kind.value,
            scope.value,
            condition.piece,
            condition.attack_number,
            tuple((seat, int(player_revisions.get(seat, 0))) for seat in seats),
            tuple(
                (
                    seat,
                    piece,
                    int(estimates.get(seat, {}).get(piece, {}).get("min", 0)),
                    int(estimates.get(seat, {}).get(piece, {}).get("max", 0)),
                    float(estimates.get(seat, {}).get(piece, {}).get("expected", 0.0)),
                    int(estimates.get(seat, {}).get(piece, {}).get("map_count", 0)),
                    float(estimates.get(seat, {}).get(piece, {}).get("confidence", 0.0)),
                )
                for seat in seats
                for piece in pieces
            ),
            tuple((piece, int(tr.get("unknown_piece_pool", {}).get(piece, 0))) for piece in pieces),
            tuple(
                (
                    seat,
                    piece,
                    len(pass_evidence.get(seat, {}).get(piece, ())),
                )
                for seat in seats
                for piece in pieces
            ),
            tuple(
                (
                    seat,
                    str(public_models.get(seat, {}).get("first_attack") or ""),
                    bool(public_models.get(seat, {}).get("inferred_attack_strategy_active")),
                    bool(public_models.get(seat, {}).get("strategy_broken")),
                )
                for seat in seats
            ),
            int(tr.get("piece_inference_revision", 0)),
            int(getattr(self, "BRANCHED_ATTACK_PROBABILISTIC_SAMPLE_COUNT", 64)),
            float(getattr(self, "BRANCHED_ATTACK_PROBABILISTIC_MAX_SECONDS", 0.04)),
        )

    def _branched_branch_support(
        self,
        state,
        player: str,
        node: AttackPlanNode,
        branch: AttackPlanBranch,
    ) -> BranchedAttackBranchSupport:
        key = self._branched_branch_support_cache_key(
            state,
            player,
            node,
            branch,
        )
        cache = self._branched_inference_cache()
        cached = cache.get(key)
        if cached is not None:
            return cached
        support = self._branched_branch_support_uncached(
            state,
            player,
            node,
            branch,
        )
        cache.put(key, support)
        return support

    def branched_attack_inference_cache_snapshot(self) -> Dict[str, object]:
        return self._branched_inference_cache().snapshot()

    def clear_branched_attack_inference_cache(self) -> None:
        self._branched_inference_cache().clear()

    def _branched_all_branch_support(
        self,
        state,
        player: str,
        nodes: Iterable[AttackPlanNode],
    ) -> Tuple[BranchedAttackBranchSupport, ...]:
        return tuple(
            self._branched_branch_support(state, player, node, branch)
            for node in nodes
            for branch in node.branches
        )


__all__ = [
    "BranchSupportLevel",
    "BranchedAttackInferenceCache",
    "BranchedAttackBranchSupport",
    "BranchedAttackInferenceMixin",
]

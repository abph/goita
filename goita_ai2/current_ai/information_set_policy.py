"""Chooses one future move for worlds an acting player cannot distinguish.

The policy groups candidate deals by the actor's own hand and public history,
then records one reusable action for self and ally information sets.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from goita_ai2.constants import ALL_SEATS
from goita_ai2.current_ai.information_set import (
    InformationSet,
    InformationSetCandidate,
    PublicInformationSetKey,
)


Action = Tuple[str, Optional[str], Optional[str]]
PublicAction = Tuple[str, str, Optional[str], Optional[str]]


def _freeze_history(value):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze_history(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze_history(item) for item in value), key=repr))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_history(item) for item in value)
    return repr(value)


@dataclass(frozen=True)
class InformationSetDecisionKey:
    """A future decision identity from the acting player's perspective."""

    digest: str
    actor: str
    phase: str
    turn: str
    hand_size: int
    history_length: int


@dataclass(frozen=True)
class InformationSetWorld:
    state: object
    probability: float
    confidence: float


@dataclass(frozen=True)
class SharedInformationSetDecision:
    """One action shared by every world indistinguishable to its actor."""

    key: InformationSetDecisionKey
    actor: str
    action: Action
    candidate_count: int
    probability_mass: float
    confidence: float
    action_scores: Tuple[Tuple[Action, float], ...]
    role: str = "self"
    reason_scores: Tuple[Tuple[str, float], ...] = ()
    reused: bool = False


class InformationSetPolicy:
    """Stores one immutable action choice per future information set."""

    def __init__(self) -> None:
        self._decisions: Dict[str, SharedInformationSetDecision] = {}

    def get(
        self,
        key: InformationSetDecisionKey,
    ) -> Optional[SharedInformationSetDecision]:
        return self._decisions.get(key.digest)

    def record(self, decision: SharedInformationSetDecision) -> None:
        existing = self._decisions.get(decision.key.digest)
        if existing is not None and existing.action != decision.action:
            raise ValueError("information-set policy cannot change an existing action")
        self._decisions[decision.key.digest] = decision

    def __len__(self) -> int:
        return len(self._decisions)

    def as_dict(self) -> Dict[str, object]:
        return {
            digest: {
                "actor": decision.actor,
                "action": decision.action,
                "candidate_count": decision.candidate_count,
                "probability_mass": round(decision.probability_mass, 6),
                "confidence": round(decision.confidence, 6),
                "role": decision.role,
                "reason_scores": {
                    reason: round(score, 6)
                    for reason, score in decision.reason_scores
                },
                "reused": decision.reused,
            }
            for digest, decision in sorted(self._decisions.items())
        }


class InformationSetPolicyMixin:
    """Builds shared future decisions for self and ally candidate worlds."""

    @staticmethod
    def _information_set_observed_action(
        observer: str,
        actor: str,
        action: Action,
    ) -> PublicAction:
        action_type, block, attack = action
        visible_block = block
        if action_type == "attack_after_block" and observer != actor:
            visible_block = None
        return actor, action_type, visible_block, attack

    def _information_set_decision_key(
        self,
        state,
        actor: str,
        root_key: PublicInformationSetKey,
        public_history: Sequence[PublicAction] = (),
    ) -> InformationSetDecisionKey:
        payload = (
            "information-decision-v1",
            root_key.digest,
            actor,
            tuple(sorted(state.hands[actor])),
            tuple(sorted(state.face_down_hidden[actor])),
            bool(state.had_both_kings.get(actor, False)),
            state.dealer,
            state.turn,
            state.phase,
            state.current_attack,
            state.attacker,
            state.last_block_player,
            state.last_block if state.last_block_player == actor else None,
            int(state.king_block_used),
            bool(state.finished),
            state.winner,
            tuple(sorted(state.team_score.items())),
            tuple((seat, len(state.hands[seat])) for seat in ALL_SEATS),
            tuple(_freeze_history(item) for item in public_history),
        )
        return InformationSetDecisionKey(
            digest=hashlib.sha256(repr(payload).encode("utf-8")).hexdigest(),
            actor=actor,
            phase=str(state.phase),
            turn=str(state.turn),
            hand_size=len(state.hands[actor]),
            history_length=len(public_history),
        )

    @staticmethod
    def _information_set_materialize_candidate(
        state,
        player: str,
        candidate: InformationSetCandidate,
    ):
        sampled = copy.copy(state)
        sampled.hands = {seat: list(state.hands[seat]) for seat in ALL_SEATS}
        sampled.face_down_hidden = {
            seat: list(state.face_down_hidden[seat])
            for seat in ALL_SEATS
        }
        sampled.had_both_kings = dict(state.had_both_kings)
        sampled.team_score = dict(state.team_score)
        for seat, hand in candidate.prediction.opponent_hands:
            sampled.hands[seat] = list(hand)
        for seat, hidden in candidate.prediction.opponent_hidden:
            sampled.face_down_hidden[seat] = list(hidden)
        for seat, had_both in candidate.prediction.opponent_had_both_kings:
            sampled.had_both_kings[seat] = bool(had_both)
        sampled.hands[player] = list(state.hands[player])
        sampled.face_down_hidden[player] = list(state.face_down_hidden[player])
        sampled.last_block = candidate.prediction.last_block
        return sampled

    def _information_set_worlds(
        self,
        state,
        player: str,
        information_set: InformationSet,
    ) -> Tuple[InformationSetWorld, ...]:
        return tuple(
            InformationSetWorld(
                state=self._information_set_materialize_candidate(
                    state,
                    player,
                    candidate,
                ),
                probability=candidate.probability,
                confidence=candidate.confidence,
            )
            for candidate in information_set.candidates
        )

    def _information_set_shared_future_decisions(
        self,
        state,
        root_player: str,
        actor: str,
        information_set: InformationSet,
        *,
        public_history: Sequence[PublicAction] = (),
        policy: Optional[InformationSetPolicy] = None,
    ) -> Tuple[SharedInformationSetDecision, ...]:
        active_policy = policy if policy is not None else InformationSetPolicy()
        tracker = self._track.get(id(state))
        role = self._information_set_action_role(root_player, actor)
        groups: Dict[str, List[InformationSetWorld]] = {}
        keys: Dict[str, InformationSetDecisionKey] = {}
        for world in self._information_set_worlds(state, root_player, information_set):
            if world.state.finished or world.state.turn != actor:
                continue
            decision_key = self._information_set_decision_key(
                world.state,
                actor,
                information_set.key,
                public_history,
            )
            keys[decision_key.digest] = decision_key
            groups.setdefault(decision_key.digest, []).append(world)

        decisions = []
        for digest, worlds in sorted(groups.items()):
            decision_key = keys[digest]
            legal_sets = [set(world.state.legal_actions(actor)) for world in worlds]
            common_actions = set.intersection(*legal_sets) if legal_sets else set()
            if not common_actions:
                continue

            mass = sum(world.probability for world in worlds)
            action_scores = []
            reason_totals: Dict[str, float] = {}
            for action in sorted(common_actions):
                weighted_score = 0.0
                action_reasons: Dict[str, float] = {}
                for world in worlds:
                    evaluation = self._information_set_action_evaluation(
                        state,
                        world.state,
                        root_player,
                        actor,
                        action,
                        tracker,
                    )
                    weighted_score += world.probability * evaluation.total_score
                    action_reasons["base_priority"] = (
                        action_reasons.get("base_priority", 0.0)
                        + world.probability * evaluation.base_score
                    )
                    for reason, value in evaluation.adjustments:
                        action_reasons[reason] = (
                            action_reasons.get(reason, 0.0)
                            + world.probability * value
                        )
                score = weighted_score / max(mass, 1e-12)
                action_scores.append((action, score))
                for reason, value in action_reasons.items():
                    reason_totals[f"{action}:{reason}"] = value / max(mass, 1e-12)
            action_scores.sort(key=lambda item: (item[1], item[0]), reverse=True)

            existing = active_policy.get(decision_key)
            reused = existing is not None and existing.action in common_actions
            chosen = existing.action if reused else action_scores[0][0]
            confidence = sum(
                world.probability * world.confidence
                for world in worlds
            ) / max(mass, 1e-12)
            decision = SharedInformationSetDecision(
                key=decision_key,
                actor=actor,
                action=chosen,
                candidate_count=len(worlds),
                probability_mass=mass,
                confidence=confidence,
                action_scores=tuple(action_scores),
                role=role,
                reason_scores=tuple(sorted(reason_totals.items())),
                reused=reused,
            )
            if not reused:
                active_policy.record(decision)
            decisions.append(decision)
        return tuple(decisions)

    def _information_set_self_future_decisions(
        self,
        state,
        player: str,
        information_set: InformationSet,
        **kwargs,
    ) -> Tuple[SharedInformationSetDecision, ...]:
        return self._information_set_shared_future_decisions(
            state,
            player,
            player,
            information_set,
            **kwargs,
        )

    def _information_set_ally_future_decisions(
        self,
        state,
        player: str,
        information_set: InformationSet,
        **kwargs,
    ) -> Tuple[SharedInformationSetDecision, ...]:
        return self._information_set_shared_future_decisions(
            state,
            player,
            self._ally_of(player),
            information_set,
            **kwargs,
        )

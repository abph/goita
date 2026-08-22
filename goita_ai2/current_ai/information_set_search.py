"""Searches several hidden deals as one public-information game.

Indistinguishable worlds share one action policy at every future decision.
Posterior probability and inference confidence combine their resulting values.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from goita_ai2.current_ai.information_set import InformationSet
from goita_ai2.current_ai.information_set_policy import (
    InformationSetPolicy,
    SharedInformationSetDecision,
)


Action = Tuple[str, Optional[str], Optional[str]]
PublicAction = Tuple[str, str, Optional[str], Optional[str]]


class InformationSetSearchDeadline(Exception):
    """Stops an unfinished information-set iteration at its time limit."""


class InformationSetSearchCancelled(Exception):
    """Stops speculative information-set work after public state changes."""


@dataclass(frozen=True)
class InformationSetSearchWorld:
    index: int
    state: object
    probability: float
    confidence: float


@dataclass(frozen=True)
class InformationSetSearchOutcome:
    value: float
    world_values: Tuple[Tuple[int, float], ...]
    policy_decisions: int

    def values_dict(self) -> Dict[int, float]:
        return dict(self.world_values)


class InformationSetSearchMixin:
    """Runs a bounded policy search over weighted candidate deals."""

    def _information_set_search_worlds(
        self,
        state,
        player: str,
        tracker: dict,
        samples: Sequence[object],
    ) -> Tuple[InformationSet, Tuple[InformationSetSearchWorld, ...]]:
        information_set = self._build_information_set(
            state,
            player,
            tracker,
            samples,
        )
        worlds = tuple(
            InformationSetSearchWorld(
                index=index,
                state=self._information_set_materialize_candidate(
                    state,
                    player,
                    candidate,
                ),
                probability=float(candidate.probability),
                confidence=float(candidate.confidence),
            )
            for index, candidate in enumerate(information_set.candidates)
        )
        return information_set, worlds

    @staticmethod
    def _information_set_weighted_quantile(
        weighted_values: Sequence[Tuple[float, float]],
        quantile: float,
    ) -> float:
        if not weighted_values:
            return 0.0
        ordered = sorted(weighted_values, key=lambda item: item[1])
        mass = sum(max(0.0, weight) for weight, _value in ordered)
        if mass <= 0.0:
            return ordered[0][1]
        target = mass * max(0.0, min(1.0, quantile))
        cumulative = 0.0
        for weight, value in ordered:
            cumulative += max(0.0, weight)
            if cumulative >= target:
                return value
        return ordered[-1][1]

    def _information_set_weighted_value(
        self,
        worlds: Sequence[InformationSetSearchWorld],
        values: Dict[int, float],
    ) -> float:
        available = [world for world in worlds if world.index in values]
        mass = sum(world.probability for world in available)
        if mass <= 0.0:
            return 0.0
        mean = sum(
            world.probability * values[world.index]
            for world in available
        ) / mass
        confidence = sum(
            world.probability * world.confidence
            for world in available
        ) / mass
        lower_quartile = self._information_set_weighted_quantile(
            [
                (world.probability, values[world.index])
                for world in available
            ],
            0.20,
        )
        caution_weight = 0.12 + (1.0 - max(0.0, min(1.0, confidence))) * 0.18
        return mean * (1.0 - caution_weight) + lower_quartile * caution_weight

    def _information_set_group_action_model(
        self,
        root_state,
        worlds: Sequence[InformationSetSearchWorld],
        root_player: str,
        actor: str,
        action: Action,
        tracker: Optional[dict],
    ) -> Tuple[float, Dict[str, float]]:
        mass = sum(world.probability for world in worlds)
        weighted_score = 0.0
        reasons: Dict[str, float] = {}
        for world in worlds:
            evaluation = self._information_set_action_evaluation(
                root_state,
                world.state,
                root_player,
                actor,
                action,
                tracker,
            )
            weighted_score += world.probability * evaluation.total_score
            reasons["base_priority"] = (
                reasons.get("base_priority", 0.0)
                + world.probability * evaluation.base_score
            )
            for reason, value in evaluation.adjustments:
                reasons[reason] = reasons.get(reason, 0.0) + world.probability * value
        divisor = max(mass, 1e-12)
        return (
            weighted_score / divisor,
            {reason: value / divisor for reason, value in reasons.items()},
        )

    def _information_set_search_bundle(
        self,
        root_state,
        worlds: Sequence[InformationSetSearchWorld],
        root_player: str,
        root_information_set: InformationSet,
        baseline_scores: Dict[str, int],
        depth: int,
        deadline: float,
        stats: Dict[str, int],
        policy: InformationSetPolicy,
        tracker: Optional[dict],
        public_history: Sequence[PublicAction] = (),
        cancel_event=None,
    ) -> InformationSetSearchOutcome:
        if cancel_event is not None and cancel_event.is_set():
            raise InformationSetSearchCancelled()
        maximum_nodes = int(stats.get("max_nodes", self.TIME_SEARCH_MAX_NODES))
        node_cost = max(1, len(worlds))
        if time.perf_counter() >= deadline or stats["nodes"] + node_cost > maximum_nodes:
            raise InformationSetSearchDeadline()
        stats["nodes"] += node_cost

        terminal_values: Dict[int, float] = {}
        active_worlds = []
        for world in worlds:
            if world.state.finished or depth <= 0:
                terminal_values[world.index] = self._timed_search_static_value(
                    world.state,
                    root_player,
                    baseline_scores,
                )
            else:
                active_worlds.append(world)
        if not active_worlds:
            return InformationSetSearchOutcome(
                value=self._information_set_weighted_value(worlds, terminal_values),
                world_values=tuple(sorted(terminal_values.items())),
                policy_decisions=len(policy),
            )

        groups: Dict[str, list[InformationSetSearchWorld]] = {}
        decision_keys = {}
        for world in active_worlds:
            actor = str(world.state.turn)
            key = self._information_set_decision_key(
                world.state,
                actor,
                root_information_set.key,
                public_history,
            )
            groups.setdefault(key.digest, []).append(world)
            decision_keys[key.digest] = key

        combined_values = dict(terminal_values)
        for digest, group in sorted(groups.items()):
            if cancel_event is not None and cancel_event.is_set():
                raise InformationSetSearchCancelled()
            actor = str(group[0].state.turn)
            key = decision_keys[digest]
            legal_sets = [set(world.state.legal_actions(actor)) for world in group]
            common_actions = set.intersection(*legal_sets) if legal_sets else set()
            if not common_actions:
                for world in group:
                    combined_values[world.index] = self._timed_search_static_value(
                        world.state,
                        root_player,
                        baseline_scores,
                    )
                continue

            existing = policy.get(key)
            reused = existing is not None and existing.action in common_actions
            action_models = {}
            action_reasons = {}
            for action in common_actions:
                model_score, reasons = self._information_set_group_action_model(
                    root_state,
                    group,
                    root_player,
                    actor,
                    action,
                    tracker,
                )
                action_models[action] = model_score
                action_reasons[action] = reasons

            enemy_third_attack_node = (
                tracker is not None
                and root_state.phase == "receive"
                and root_state.attacker is not None
                and actor == root_state.attacker
                and not self._same_team(actor, root_player)
                and group[0].state.phase == "attack"
                and int(
                    tracker.get("enemy_attack_counts", {}).get(actor, 0)
                ) == 2
            )
            if reused and not enemy_third_attack_node:
                candidates = [existing.action]
            else:
                ordered_candidates = sorted(
                    common_actions,
                    key=lambda action: (action_models[action], action),
                    reverse=True,
                )
                candidates = (
                    ordered_candidates
                    if enemy_third_attack_node
                    else ordered_candidates[: max(1, int(self.TIME_SEARCH_BRANCH_BEAM))]
                )

            role = self._information_set_action_role(root_player, actor)
            maximizing = role in ("self", "ally")
            prior_weight = float(
                getattr(self, "TIME_SEARCH_INFORMATION_SET_ACTION_PRIOR_WEIGHT", 0.18)
            )
            prior_cap = float(
                getattr(self, "TIME_SEARCH_INFORMATION_SET_ACTION_PRIOR_CAP", 200.0)
            )
            evaluated = []
            for action in candidates:
                child_worlds = tuple(
                    InformationSetSearchWorld(
                        index=world.index,
                        state=self._timed_search_apply(world.state, actor, action),
                        probability=world.probability,
                        confidence=world.confidence,
                    )
                    for world in group
                )
                observed = self._information_set_observed_action(
                    root_player,
                    actor,
                    action,
                )
                child = self._information_set_search_bundle(
                    root_state,
                    child_worlds,
                    root_player,
                    root_information_set,
                    baseline_scores,
                    depth - 1,
                    deadline,
                    stats,
                    policy,
                    tracker,
                    tuple(public_history) + (observed,),
                    cancel_event,
                )
                bounded_prior = max(-prior_cap, min(prior_cap, action_models[action]))
                selection_value = child.value + (
                    prior_weight * bounded_prior
                    if maximizing
                    else -prior_weight * bounded_prior
                )
                evaluated.append((selection_value, action, child))

            if not evaluated:
                for world in group:
                    combined_values[world.index] = self._timed_search_static_value(
                        world.state,
                        root_player,
                        baseline_scores,
                    )
                continue
            evaluated.sort(key=lambda item: (item[0], item[1]), reverse=maximizing)
            _selection_value, chosen_action, chosen_outcome = evaluated[0]
            combined_values.update(chosen_outcome.values_dict())

            if not reused:
                mass = sum(world.probability for world in group)
                confidence = sum(
                    world.probability * world.confidence
                    for world in group
                ) / max(mass, 1e-12)
                reason_scores = []
                for action in candidates:
                    for reason, value in action_reasons[action].items():
                        reason_scores.append((f"{action}:{reason}", value))
                policy.record(
                    SharedInformationSetDecision(
                        key=key,
                        actor=actor,
                        action=chosen_action,
                        candidate_count=len(group),
                        probability_mass=mass,
                        confidence=confidence,
                        action_scores=tuple(
                            (action, selection_value)
                            for selection_value, action, _child in evaluated
                        ),
                        role=role,
                        reason_scores=tuple(sorted(reason_scores)),
                    )
                )

        return InformationSetSearchOutcome(
            value=self._information_set_weighted_value(worlds, combined_values),
            world_values=tuple(sorted(combined_values.items())),
            policy_decisions=len(policy),
        )

    def _information_set_search_root_action(
        self,
        root_state,
        worlds: Sequence[InformationSetSearchWorld],
        root_player: str,
        root_action: Action,
        root_information_set: InformationSet,
        baseline_scores: Dict[str, int],
        depth: int,
        deadline: float,
        stats: Dict[str, int],
        tracker: Optional[dict],
        cancel_event=None,
    ) -> InformationSetSearchOutcome:
        child_worlds = tuple(
            InformationSetSearchWorld(
                index=world.index,
                state=self._timed_search_apply(world.state, root_player, root_action),
                probability=world.probability,
                confidence=world.confidence,
            )
            for world in worlds
        )
        policy = InformationSetPolicy()
        observed = self._information_set_observed_action(
            root_player,
            root_player,
            root_action,
        )
        return self._information_set_search_bundle(
            root_state,
            child_worlds,
            root_player,
            root_information_set,
            baseline_scores,
            depth - 1,
            deadline,
            stats,
            policy,
            tracker,
            (observed,),
            cancel_event,
        )

"""Keeps one evaluated attack plan alive across public turns.
It advances only from observable actions, preserves waiting plans through
unrelated play, and records why a route was completed or must be rebuilt.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

from goita_ai2.current_ai.branched_attack_plan import (
    Action,
    BranchedAttackPlan,
    PlanActorScope,
    PublicPlanEvent,
    PublicPlanEventKind,
    choose_preferred_attack_plan,
)


class AttackPlanLifecycleStatus(str, Enum):
    """Current execution state of a persisted attack route."""

    READY = "ready"
    OBSERVING = "observing"
    WAITING = "waiting"
    REPLAN_REQUIRED = "replan_required"
    COMPLETED = "completed"
    INVALIDATED = "invalidated"


@dataclass(frozen=True)
class AttackPlanTransition:
    """One auditable lifecycle change caused by a public event."""

    from_node_id: str
    to_node_id: Optional[str]
    event_kind: str
    actor: Optional[str]
    reason: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "from_node_id": self.from_node_id,
            "to_node_id": self.to_node_id,
            "event_kind": self.event_kind,
            "actor": self.actor,
            "reason": self.reason,
        }


@dataclass
class ActiveAttackPlanState:
    """Mutable cursor over an otherwise immutable branched plan."""

    plan: BranchedAttackPlan
    owner: str
    current_node_id: str
    status: AttackPlanLifecycleStatus
    installed_revision: int
    observed_revision: int
    action_committed: bool = False
    active_public_attack: Optional[str] = None
    reason: str = "installed"
    transitions: List[AttackPlanTransition] = field(default_factory=list)

    @property
    def current_node(self):
        return self.plan.node(self.current_node_id)

    def as_dict(self) -> Dict[str, object]:
        node = self.current_node
        return {
            "plan_id": self.plan.plan_id,
            "source": self.plan.source,
            "owner": self.owner,
            "current_node_id": self.current_node_id,
            "attack_number": node.attack_number,
            "planned_action": node.action,
            "status": self.status.value,
            "installed_revision": self.installed_revision,
            "observed_revision": self.observed_revision,
            "action_committed": self.action_committed,
            "active_public_attack": self.active_public_attack,
            "reason": self.reason,
            "transitions": [item.as_dict() for item in self.transitions[-12:]],
        }


class BranchedAttackLifecycleMixin:
    """Persists, advances, invalidates, and rebuilds attack plans."""

    def _initialize_branched_attack_lifecycle(self) -> None:
        self._active_branched_attack_plans: Dict[int, ActiveAttackPlanState] = {}

    def _branched_lifecycle_store(self) -> Dict[int, ActiveAttackPlanState]:
        store = getattr(self, "_active_branched_attack_plans", None)
        if store is None:
            self._initialize_branched_attack_lifecycle()
            store = self._active_branched_attack_plans
        return store

    def _branched_plan_tracker_summary(
        self,
        state,
        active: Optional[ActiveAttackPlanState],
    ) -> None:
        tr = self._track.get(id(state))
        if tr is not None:
            tr["active_branched_attack_plan"] = (
                None if active is None else active.as_dict()
            )

    def _active_branched_attack_plan(
        self,
        state,
    ) -> Optional[ActiveAttackPlanState]:
        return self._branched_lifecycle_store().get(id(state))

    def _install_branched_attack_plan(
        self,
        state,
        player: str,
        plan,
    ) -> ActiveAttackPlanState:
        """Install a generated or evaluated plan at its root action."""
        if hasattr(plan, "plan"):
            plan = plan.plan
        if not isinstance(plan, BranchedAttackPlan):
            raise TypeError("plan must be a BranchedAttackPlan")
        if self.me is not None and player != self.me:
            raise ValueError("an agent can persist only its own attack plan")

        self._ensure_trackers(state)
        root = plan.node(plan.root_node_id)
        if root.action is None or root.terminal or root.checkpoint:
            raise ValueError("an installed attack plan must start with an action")
        legal = tuple(state.legal_actions(player))
        if root.action not in legal:
            raise ValueError("the root action is no longer legal")

        tr = self._track.get(id(state))
        revision = int(tr.get("piece_inference_revision", 0)) if tr else 0
        old = self._active_branched_attack_plan(state)
        if old is not None:
            self._archive_branched_attack_plan(
                state,
                old,
                AttackPlanLifecycleStatus.INVALIDATED,
                "replaced_by_new_plan",
            )

        active = ActiveAttackPlanState(
            plan=plan,
            owner=player,
            current_node_id=plan.root_node_id,
            status=AttackPlanLifecycleStatus.READY,
            installed_revision=revision,
            observed_revision=revision,
        )
        self._branched_lifecycle_store()[id(state)] = active
        self._branched_plan_tracker_summary(state, active)
        return active

    def _refresh_branched_attack_plan(
        self,
        state,
        active: ActiveAttackPlanState,
        plan: BranchedAttackPlan,
        *,
        reason: str,
    ) -> ActiveAttackPlanState:
        """Replace a ready route after current public inference revalidates it."""
        root = plan.node(plan.root_node_id)
        if root.action is None or root.terminal or root.checkpoint:
            raise ValueError("a refreshed attack plan must start with an action")
        if root.action not in tuple(state.legal_actions(active.owner)):
            raise ValueError("the refreshed root action is no longer legal")
        tr = self._track.get(id(state))
        revision = int(tr.get("piece_inference_revision", 0)) if tr else 0
        previous_node_id = active.current_node_id
        # Revalidation updates the route but keeps its public lifecycle identity.
        plan = replace(plan, plan_id=active.plan.plan_id)
        active.plan = plan
        active.current_node_id = plan.root_node_id
        active.status = AttackPlanLifecycleStatus.READY
        active.installed_revision = revision
        active.observed_revision = revision
        active.action_committed = False
        active.active_public_attack = None
        active.reason = reason
        active.transitions.append(AttackPlanTransition(
            from_node_id=previous_node_id,
            to_node_id=plan.root_node_id,
            event_kind="inference_revalidation",
            actor=active.owner,
            reason=reason,
        ))
        self._branched_plan_tracker_summary(state, active)
        return active

    def _archive_branched_attack_plan(
        self,
        state,
        active: ActiveAttackPlanState,
        status: AttackPlanLifecycleStatus,
        reason: str,
    ) -> None:
        active.status = status
        active.reason = reason
        tr = self._track.get(id(state))
        if tr is not None:
            history = tr.setdefault("branched_attack_plan_history", [])
            history.append(active.as_dict())
            if len(history) > 12:
                del history[:-12]

    def _invalidate_branched_attack_plan(
        self,
        state,
        reason: str,
        *,
        request_replan: bool = True,
    ) -> Optional[ActiveAttackPlanState]:
        active = self._active_branched_attack_plan(state)
        if active is None:
            return None
        status = (
            AttackPlanLifecycleStatus.REPLAN_REQUIRED
            if request_replan
            else AttackPlanLifecycleStatus.INVALIDATED
        )
        active.status = status
        active.reason = reason
        self._branched_plan_tracker_summary(state, active)
        return active

    def _branched_actor_scope(self, owner: str, actor: str) -> PlanActorScope:
        if actor == owner:
            return PlanActorScope.SELF
        if self._same_team(owner, actor):
            return PlanActorScope.ALLY
        return PlanActorScope.ENEMY

    def _branched_public_events(
        self,
        state,
        active: ActiveAttackPlanState,
        actor: str,
        action: Action,
    ) -> Tuple[PublicPlanEvent, ...]:
        """Translate a post-action state into public plan events."""
        action_type, block, attack = action
        scope = self._branched_actor_scope(active.owner, actor)
        attack_number = active.current_node.attack_number

        if action_type == "pass":
            if (
                state.phase == "attack"
                and state.turn == state.attacker
                and state.attacker is not None
            ):
                return (PublicPlanEvent(
                    PublicPlanEventKind.LAP_COMPLETED,
                    actor_scope=scope,
                    current_attack=active.active_public_attack,
                    attack_number=attack_number,
                ),)
            return (PublicPlanEvent(
                PublicPlanEventKind.PASS,
                actor_scope=scope,
                current_attack=active.active_public_attack,
                attack_number=attack_number,
            ),)

        if action_type == "receive":
            current_attack = active.active_public_attack
            kind = PublicPlanEventKind.RECEIVE
            if block in ("8", "9"):
                kind = PublicPlanEventKind.ROYAL_RECEIVE
            elif block is not None and block == current_attack:
                kind = PublicPlanEventKind.SAME_PIECE_RECEIVE
            return (PublicPlanEvent(
                kind,
                actor_scope=scope,
                piece=block,
                current_attack=current_attack,
                attack_number=attack_number,
            ),)

        if action_type in ("attack", "attack_after_block"):
            return (PublicPlanEvent(
                PublicPlanEventKind.ATTACK,
                actor_scope=scope,
                piece=attack,
                current_attack=attack,
                attack_number=attack_number,
                reached=len(state.hands.get(actor, ())) == 2,
            ),)
        return tuple()

    @staticmethod
    def _branched_terminal_requires_replan(purpose: str) -> bool:
        lowered = purpose.lower()
        return any(token in lowered for token in (
            "discard",
            "unexpected",
            "invalid",
            "no legal",
            "rebuild",
        ))

    def _branched_enter_node(
        self,
        state,
        active: ActiveAttackPlanState,
        target_node_id: str,
        event: PublicPlanEvent,
        actor: str,
    ) -> None:
        previous = active.current_node_id
        active.current_node_id = target_node_id
        active.action_committed = False
        target = active.current_node
        active.transitions.append(AttackPlanTransition(
            from_node_id=previous,
            to_node_id=target_node_id,
            event_kind=event.kind.value,
            actor=actor,
            reason="public_branch_matched",
        ))
        if target.terminal:
            if self._branched_terminal_requires_replan(target.purpose):
                active.status = AttackPlanLifecycleStatus.REPLAN_REQUIRED
                active.reason = target.purpose
            else:
                active.status = AttackPlanLifecycleStatus.COMPLETED
                active.reason = target.purpose or "planned_segment_completed"
        elif target.checkpoint:
            active.status = AttackPlanLifecycleStatus.WAITING
            active.reason = target.purpose
        else:
            active.status = AttackPlanLifecycleStatus.READY
            active.reason = "next_planned_action_ready"

    def _advance_branched_attack_plan_for_public_action(
        self,
        state,
        actor: str,
        action: Action,
        tr: Optional[dict] = None,
    ) -> Optional[ActiveAttackPlanState]:
        """Advance the active plan after tracking has consumed one action."""
        active = self._active_branched_attack_plan(state)
        if active is None:
            return None
        if tr is None:
            tr = self._track.get(id(state))
        if tr is not None:
            active.observed_revision = int(tr.get("piece_inference_revision", 0))

        if state.finished:
            status = (
                AttackPlanLifecycleStatus.COMPLETED
                if state.winner == active.owner
                else AttackPlanLifecycleStatus.INVALIDATED
            )
            self._archive_branched_attack_plan(
                state,
                active,
                status,
                "round_finished",
            )
            self._branched_lifecycle_store().pop(id(state), None)
            self._branched_plan_tracker_summary(state, None)
            return active

        if active.status in (
            AttackPlanLifecycleStatus.COMPLETED,
            AttackPlanLifecycleStatus.INVALIDATED,
            AttackPlanLifecycleStatus.REPLAN_REQUIRED,
        ):
            return active

        node = active.current_node
        if active.status == AttackPlanLifecycleStatus.READY:
            if actor != active.owner:
                return self._invalidate_branched_attack_plan(
                    state,
                    "another_player_acted_while_owner_action_was_expected",
                )
            if action != node.action:
                return self._invalidate_branched_attack_plan(
                    state,
                    "owner_selected_a_different_action",
                )
            if tuple(sorted(state.hands[active.owner])) != tuple(node.reserved_pieces):
                return self._invalidate_branched_attack_plan(
                    state,
                    "owner_hand_no_longer_matches_reserved_route",
                )
            active.action_committed = True
            active.active_public_attack = action[2]
            active.transitions.append(AttackPlanTransition(
                from_node_id=node.node_id,
                to_node_id=node.node_id,
                event_kind="planned_action",
                actor=actor,
                reason="planned_action_committed",
            ))
            if not node.branches:
                active.status = AttackPlanLifecycleStatus.COMPLETED
                active.reason = "planned_segment_completed"
            else:
                active.status = AttackPlanLifecycleStatus.OBSERVING
                active.reason = "waiting_for_public_response"
            self._branched_plan_tracker_summary(state, active)
            return active

        events = self._branched_public_events(state, active, actor, action)
        if action[0] in ("attack", "attack_after_block") and action[2] is not None:
            active.active_public_attack = action[2]

        # A receive-response checkpoint intentionally survives unrelated play
        # until this plan owner is publicly shown receiving another attack.
        if active.status == AttackPlanLifecycleStatus.WAITING and (
            "owner receives" in node.purpose
        ):
            relevant = (
                action[0] == "receive"
                and actor == active.owner
            )
            if not relevant:
                self._branched_plan_tracker_summary(state, active)
                return active

        for event in events:
            target_id = node.next_node_id(event)
            if target_id is None:
                continue
            self._branched_enter_node(state, active, target_id, event, actor)
            self._branched_plan_tracker_summary(state, active)
            return active

        return self._invalidate_branched_attack_plan(
            state,
            "public_response_did_not_match_the_active_plan",
        )

    def _branched_planned_action(
        self,
        state,
        player: str,
        actions: Sequence[Action],
    ) -> Optional[Action]:
        """Return the next persisted action, invalidating it if no longer legal."""
        active = self._active_branched_attack_plan(state)
        if active is None or active.owner != player:
            return None
        if active.status != AttackPlanLifecycleStatus.READY:
            return None
        tr = self._track.get(id(state))
        expected_attack_number = int(tr.get("my_attack_count", 0)) + 1 if tr else 1
        if active.current_node.attack_number != expected_attack_number:
            self._invalidate_branched_attack_plan(
                state,
                "tracked_attack_number_no_longer_matches_the_plan",
            )
            return None
        action = active.current_node.action
        if action not in actions:
            self._invalidate_branched_attack_plan(
                state,
                "planned_action_is_no_longer_legal",
            )
            return None
        return action

    def _rebuild_branched_attack_plan(
        self,
        state,
        player: str,
        actions: Sequence[Action],
    ) -> Optional[ActiveAttackPlanState]:
        """Regenerate and evaluate a plan after its public assumptions break."""
        generated = self._generate_branched_attack_plans(state, player, actions)
        evaluated = self._evaluate_branched_attack_plans(state, player, generated)
        preferred = choose_preferred_attack_plan(item.plan for item in evaluated)
        if preferred is None:
            self._invalidate_branched_attack_plan(
                state,
                "no_replacement_attack_plan",
                request_replan=False,
            )
            return None
        return self._install_branched_attack_plan(state, player, preferred)


__all__ = [
    "ActiveAttackPlanState",
    "AttackPlanLifecycleStatus",
    "AttackPlanTransition",
    "BranchedAttackLifecycleMixin",
]

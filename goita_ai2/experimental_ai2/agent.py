"""強化中AI2: neural-first play with the rule engine as a safety guard."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from goita_ai2.current_ai.agent import RuleBasedAgent as CurrentRuleBasedAgent
from goita_ai2.neural_policy import NeuralPolicyModel, live_state_payload


Action = Tuple[str, Optional[str], Optional[str]]
MODEL_PATH = Path(__file__).resolve().parent / "data" / "neural_policy.json"


class RuleBasedAgent(CurrentRuleBasedAgent):
    """Use the learned policy first, while preserving proven rule decisions."""

    _shared_neural_model: Optional[NeuralPolicyModel] = None
    _shared_neural_error: Optional[str] = None

    def __init__(self, name: str = "強化中AI2"):
        super().__init__(name=name)
        self.PRESERVE_SHI_FOR_THIRD_ATTACK_ENABLED = True
        self.ALLY_GUARANTEED_WIN_NO_SELF_FINISH_ENABLED = True
        self.NEURAL_PRIMARY_ENABLED = True
        self.NEURAL_PRIMARY_MIN_MARGIN = 0.0
        self.last_neural_shadow: Dict[str, Any] = {}
        self._neural_public_history_by_state_id: Dict[int, List[dict]] = {}

    @classmethod
    def _neural_model(cls) -> Optional[NeuralPolicyModel]:
        if cls._shared_neural_model is not None:
            return cls._shared_neural_model
        if cls._shared_neural_error is not None:
            return None
        try:
            cls._shared_neural_model = NeuralPolicyModel.load(MODEL_PATH)
        except Exception as error:  # A missing model must not stop a game.
            cls._shared_neural_error = f"{type(error).__name__}: {error}"
        return cls._shared_neural_model

    def on_public_action(self, state, player: str, action: Action) -> None:
        super().on_public_action(state, player, action)
        action_type, block, attack = action
        visible_block = block
        if action_type == "attack_after_block" and player != self.me:
            visible_block = None
        self._neural_public_history_by_state_id.setdefault(id(state), []).append({
            "player": str(player),
            "action": [action_type, visible_block, attack],
        })

    def select_action(self, state, player: str, actions: List[Action]) -> Action:
        # The current AI still evaluates the position so proven wins and other
        # exact guards remain available. Keep a clean tracker snapshot because
        # its chosen move may be replaced by the learned human policy.
        self._ensure_trackers(state)
        tracker = self._track.get(id(state))
        tracker_before = copy.deepcopy(tracker) if tracker is not None else None
        rule_action = super().select_action(state, player, actions)
        rule_reason = str(self.last_decision_reason or "")
        rule_detail = str(self.last_score_fallback_detail or "")
        rule_authority = str(self.last_rule_search_authority or "ordinary")

        self._record_neural_shadow(state, player, actions, rule_action)
        snapshot = self.last_neural_shadow
        recommended = tuple(snapshot.get("recommended_action", ()))
        neural_available = bool(snapshot.get("available")) and recommended in actions
        margin = float(snapshot.get("margin") or 0.0)
        safety_locked = rule_authority == "proven"
        apply_neural = bool(
            self.NEURAL_PRIMARY_ENABLED
            and neural_available
            and recommended != rule_action
            and not safety_locked
            and margin >= float(self.NEURAL_PRIMARY_MIN_MARGIN)
        )

        selected = recommended if apply_neural else rule_action
        snapshot.update({
            "mode": "primary",
            "applied": apply_neural,
            "selected_action": list(selected),
            "safety_locked": safety_locked,
            "rule_reason": rule_reason,
            "rule_detail": rule_detail,
            "rule_authority": rule_authority,
        })
        if not apply_neural:
            return rule_action

        if tracker_before is not None:
            live_tracker = self._track.get(id(state))
            if live_tracker is not None:
                live_tracker.clear()
                live_tracker.update(tracker_before)
        self._commit_neural_primary_action(state, player, selected)
        self.last_attack_candidate_scores = [
            dict(item) for item in snapshot.get("top_candidates", [])
            if isinstance(item, dict)
        ]
        self.last_attack_intent_comparison = None
        self.last_attack_candidate_snapshot = {}
        self._finalize_attack_candidate_snapshot(actions, selected)
        self._set_decision_reason("neural_primary")
        self._set_score_fallback_detail(
            f"neural_primary_margin_{margin:.3f}"
        )
        return selected

    def _commit_neural_primary_action(
        self,
        state,
        player: str,
        action: Action,
    ) -> None:
        """Commit only state needed before on_public_action sees the move."""
        tracker = self._track.get(id(state))
        if tracker is None:
            return
        for key in (
            "pending_weak_hand_shi_signal",
            "pending_ally_force_king_attack_piece",
            "pending_inferred_endgame_attack",
            "pending_low_reentry_attack_piece",
            "pending_kyosha_receive_attack_piece",
            "pending_kyosha_receive_source",
            "pending_conditional_response_attack_piece",
            "pending_shi_insertion_attack_piece",
        ):
            tracker[key] = False if key == "pending_weak_hand_shi_signal" else None

        action_type, block, _attack = action
        if action_type in ("attack", "attack_after_block"):
            self._commit_timed_search_action(state, player, action)
        elif (
            action_type == "receive"
            and block == "1"
            and state.attacker == tracker.get("ally")
            and state.current_attack == "1"
        ):
            tracker["my_shi_approval_pending"] = True

    def _record_neural_shadow(
        self,
        state,
        player: str,
        actions: List[Action],
        chosen: Action,
    ) -> None:
        started = time.perf_counter()
        model = self._neural_model()
        if model is None:
            self.last_neural_shadow = {
                "available": False,
                "error": self._shared_neural_error or "model unavailable",
            }
            return
        try:
            self._ensure_trackers(state)
            initial_hand = self._get_my_initial_hand(state)
            history = self._neural_public_history_by_state_id.get(id(state), [])
            payload = live_state_payload(
                state,
                player,
                initial_hand=initial_hand,
                history=history,
            )
            ranked = model.rank_actions(payload, actions)
            recommended, top_score = ranked[0]
            second_score = ranked[1][1] if len(ranked) > 1 else top_score
            self.last_neural_shadow = {
                "available": True,
                "mode": "primary_candidate",
                "rule_action": list(chosen),
                "recommended_action": list(recommended),
                "match": tuple(recommended) == tuple(chosen),
                "margin": round(float(top_score - second_score), 6),
                "top_candidates": [
                    {"action": list(action), "score": round(float(score), 6)}
                    for action, score in ranked[:3]
                ],
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        except Exception as error:
            self.last_neural_shadow = {
                "available": False,
                "error": f"{type(error).__name__}: {error}",
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            }


__all__ = ["RuleBasedAgent"]

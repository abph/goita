"""中級者（中3）: 2026年10月5日時点の強化中AI2を固定保存した安定版。"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from goita_ai2.intermediate_middle3.base_agent import RuleBasedAgent as FrozenRuleBasedAgent
from goita_ai2.intermediate_middle3.neural_policy import NeuralPolicyModel, live_state_payload


Action = Tuple[str, Optional[str], Optional[str]]
MODEL_PATH = Path(__file__).resolve().parent / "data" / "neural_policy.json"


class RuleBasedAgent(FrozenRuleBasedAgent):
    """Keep the current AI decision unless its own scores show a close tie."""

    _shared_neural_model: Optional[NeuralPolicyModel] = None
    _shared_neural_error: Optional[str] = None

    def __init__(self, name: str = "中級者（中3）"):
        super().__init__(name=name)
        self.PRESERVE_SHI_FOR_THIRD_ATTACK_ENABLED = True
        self.ALLY_SHI_SPARE_THIRD_BIG_ATTACK_ENABLED = True
        self.SHI_ATTACK_PACKAGE_ENABLED = True
        self.ALLY_GUARANTEED_WIN_NO_SELF_FINISH_ENABLED = True
        self.PRESERVE_PUBLIC_UNSTOPPABLE_FINISH_ENABLED = True
        # Retained for diagnostics and compatibility with older snapshots.
        self.NEURAL_PRIMARY_ENABLED = False
        self.NEURAL_TIEBREAK_ENABLED = True
        self.NEURAL_TIEBREAK_MIN_NEURAL_MARGIN = 0.5
        self.NEURAL_TIEBREAK_MIN_CURRENT_GAP = 5.0
        self.NEURAL_TIEBREAK_MAX_CURRENT_GAP = 20.0
        self.NEURAL_TIEBREAK_CURRENT_RELATIVE_GAP = 0.05
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
        protect_shi_continuation = rule_detail.startswith(
            "attack_enemy_team_shi_remaining_"
        )
        protect_attack_reserve = self._rule_preserves_continuation_reserve(
            state,
            player,
            rule_action,
        )
        protect_unstoppable_finish = rule_detail.startswith(
            "pass_preserve_public_unstoppable_finish_piece_"
        )
        if protect_shi_continuation or protect_attack_reserve:
            rule_authority = "strong"
        forced_receive = self._enemy_immediate_finish_receive_action(
            state,
            player,
            actions,
        )
        prevent_enemy_immediate_finish = forced_receive is not None
        if prevent_enemy_immediate_finish:
            if forced_receive != rule_action:
                rule_action = forced_receive
                if tracker_before is not None:
                    live_tracker = self._track.get(id(state))
                    if live_tracker is not None:
                        live_tracker.clear()
                        live_tracker.update(tracker_before)
                self._commit_neural_tiebreak_action(
                    state,
                    player,
                    rule_action,
                )
            rule_reason = "score_fallback"
            rule_detail = "receive_prevent_enemy_immediate_finish"
            rule_authority = "proven"
            self._set_decision_reason(rule_reason)
            self._set_score_fallback_detail(rule_detail)
            self.last_rule_search_authority = rule_authority

        current_comparison = self._current_ai_tiebreak_candidates(
            actions,
            rule_action,
            rule_authority,
        )
        self._record_neural_shadow(
            state,
            player,
            actions,
            rule_action,
            eligible_actions=current_comparison["eligible_actions"],
        )
        snapshot = self.last_neural_shadow
        recommended = tuple(snapshot.get("recommended_action", ()))
        neural_available = bool(snapshot.get("available")) and recommended in actions
        margin = float(snapshot.get("margin") or 0.0)
        current_comparison["selected_gap"] = snapshot.get(
            "current_score_gap"
        )
        safety_locked = rule_authority == "proven"
        block_only_disagreement = self._neural_block_only_disagreement(
            rule_action,
            recommended,
        )
        required_margin = float(self.NEURAL_TIEBREAK_MIN_NEURAL_MARGIN)
        current_choice_status = str(current_comparison["status"])
        confidence_deferred = bool(
            neural_available
            and recommended != rule_action
            and margin < required_margin
        )
        apply_neural = bool(
            self.NEURAL_TIEBREAK_ENABLED
            and neural_available
            and recommended != rule_action
            and not safety_locked
            and current_choice_status == "close"
            and recommended in current_comparison["eligible_actions"]
            and margin >= required_margin
        )

        selected = recommended if apply_neural else rule_action
        snapshot.update({
            "mode": "tiebreak",
            "applied": apply_neural,
            "selected_action": list(selected),
            "safety_locked": safety_locked,
            "confidence_deferred": confidence_deferred,
            "required_margin": required_margin,
            "block_only_disagreement": block_only_disagreement,
            "current_choice_status": current_choice_status,
            "current_score_gap": current_comparison.get("selected_gap"),
            "current_gap_limit": current_comparison.get("gap_limit"),
            "current_candidate_count": len(
                current_comparison["eligible_actions"]
            ),
            "prevent_enemy_immediate_finish": (
                prevent_enemy_immediate_finish
            ),
            "protect_shi_continuation": protect_shi_continuation,
            "protect_attack_reserve": protect_attack_reserve,
            "protect_unstoppable_finish": protect_unstoppable_finish,
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
        self._commit_neural_tiebreak_action(state, player, selected)
        self.last_attack_intent_comparison = None
        self.last_attack_candidate_snapshot = {}
        self._finalize_attack_candidate_snapshot(actions, selected)
        self._set_decision_reason("neural_tiebreak")
        self._set_score_fallback_detail(
            f"neural_tiebreak_current_gap_"
            f"{float(current_comparison['selected_gap'] or 0.0):.3f}_"
            f"neural_margin_{margin:.3f}"
        )
        return selected

    def _rule_preserves_continuation_reserve(
        self,
        state,
        player: str,
        rule_action: Action,
    ) -> bool:
        """Whether the rule keeps the last copy for a repeated attack plan."""
        action_type, block, attack = rule_action
        if (
            action_type != "attack_after_block"
            or attack is None
            or block == attack
        ):
            return False
        initial_hand = self._get_my_initial_hand(state)
        tracker = self._track.get(id(state), {})
        return bool(
            initial_hand.count(attack) >= 3
            and state.hands[player].count(attack) == 2
            and tracker.get("my_last_attack") == attack
        )

    def _enemy_immediate_finish_receive_action(
        self,
        state,
        player: str,
        actions: List[Action],
    ) -> Optional[Action]:
        """Return a receive when passing gives an enemy an immediate finish."""
        attacker = state.attacker
        if not (
            state.phase == "receive"
            and attacker is not None
            and not self._same_team(attacker, player)
            and len(state.hands.get(attacker, ())) == 2
            and state.next_player(player) == attacker
            and any(action[0] == "pass" for action in actions)
        ):
            return None
        receives = [action for action in actions if action[0] == "receive"]
        if not receives:
            return None
        scores = {
            tuple(item.get("action", ())): float(item.get("score"))
            for item in self.last_attack_candidate_scores
            if isinstance(item, dict)
            and isinstance(item.get("action"), (list, tuple))
            and len(item.get("action")) == 3
            and item.get("score") is not None
        }
        return max(receives, key=lambda action: scores.get(action, -math.inf))

    def _current_ai_tiebreak_candidates(
        self,
        actions: List[Action],
        rule_action: Action,
        rule_authority: str,
    ) -> Dict[str, Any]:
        """Return only actions that the current AI itself scored as very close."""
        result: Dict[str, Any] = {
            "status": "rule_locked",
            "eligible_actions": [rule_action],
            "selected_gap": None,
            "gap_limit": None,
        }
        if rule_authority != "ordinary" or len(actions) <= 1:
            return result

        scores: Dict[Action, float] = {}
        for item in self.last_attack_candidate_scores:
            if not isinstance(item, dict):
                continue
            raw_action = item.get("action")
            raw_score = item.get("score")
            if not isinstance(raw_action, (list, tuple)) or len(raw_action) != 3:
                continue
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score):
                continue
            action = tuple(raw_action)
            if action in actions:
                scores[action] = score

        if rule_action not in scores or len(scores) < 2:
            result["status"] = "score_unavailable"
            return result
        rule_score = scores[rule_action]
        best_score = max(scores.values())
        if rule_score < best_score - 1e-6:
            result["status"] = "rule_selected_outside_score"
            return result

        relative_gap = abs(rule_score) * float(
            self.NEURAL_TIEBREAK_CURRENT_RELATIVE_GAP
        )
        gap_limit = min(
            float(self.NEURAL_TIEBREAK_MAX_CURRENT_GAP),
            max(
                float(self.NEURAL_TIEBREAK_MIN_CURRENT_GAP),
                relative_gap,
            ),
        )
        close_actions = []
        for action in actions:
            if action not in scores:
                continue
            gap = rule_score - scores[action]
            if gap < -1e-6 or gap > gap_limit + 1e-6:
                continue
            close_actions.append(action)

        result["gap_limit"] = round(gap_limit, 6)
        if len(close_actions) < 2:
            result["status"] = "clear"
            return result
        result["status"] = "close"
        result["eligible_actions"] = close_actions
        return result

    @staticmethod
    def _neural_block_only_disagreement(
        rule_action: Action,
        neural_action: Action,
    ) -> bool:
        """Whether both policies chose the same attack but different blocks."""
        return bool(
            len(rule_action) == 3
            and len(neural_action) == 3
            and rule_action[0] == neural_action[0] == "attack_after_block"
            and rule_action[2] == neural_action[2]
            and rule_action[1] != neural_action[1]
        )

    def _commit_neural_tiebreak_action(
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
        *,
        eligible_actions: List[Action],
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
            eligible = set(eligible_actions)
            eligible_ranked = [item for item in ranked if item[0] in eligible]
            considered = eligible_ranked if len(eligible_ranked) > 1 else ranked
            recommended, top_score = considered[0]
            second_score = considered[1][1] if len(considered) > 1 else top_score
            selected_gap = None
            current_scores: Dict[Action, float] = {}
            for item in self.last_attack_candidate_scores:
                if not isinstance(item, dict):
                    continue
                raw_action = item.get("action")
                if not isinstance(raw_action, (list, tuple)) or len(raw_action) != 3:
                    continue
                try:
                    score = float(item.get("score"))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(score):
                    current_scores[tuple(raw_action)] = score
            if chosen in current_scores and recommended in current_scores:
                selected_gap = round(
                    current_scores[chosen] - current_scores[recommended],
                    6,
                )
            self.last_neural_shadow = {
                "available": True,
                "mode": "tiebreak_candidate",
                "rule_action": list(chosen),
                "recommended_action": list(recommended),
                "global_recommended_action": list(ranked[0][0]),
                "match": tuple(recommended) == tuple(chosen),
                "margin": round(float(top_score - second_score), 6),
                "current_score_gap": selected_gap,
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

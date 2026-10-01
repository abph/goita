"""強化中AI plus a neural recommendation that cannot change live actions."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from goita_ai2.current_ai.agent import RuleBasedAgent as CurrentRuleBasedAgent
from goita_ai2.neural_policy import NeuralPolicyModel, live_state_payload


Action = Tuple[str, Optional[str], Optional[str]]
MODEL_PATH = Path(__file__).resolve().parent / "data" / "neural_policy.json"


class RuleBasedAgent(CurrentRuleBasedAgent):
    """Keep the current AI decision and record the neural policy beside it."""

    _shared_neural_model: Optional[NeuralPolicyModel] = None
    _shared_neural_error: Optional[str] = None

    def __init__(self, name: str = "強化中AI2"):
        super().__init__(name=name)
        self.PRESERVE_SHI_FOR_THIRD_ATTACK_ENABLED = True
        self.ALLY_GUARANTEED_WIN_NO_SELF_FINISH_ENABLED = True
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
        chosen = super().select_action(state, player, actions)
        self._record_neural_shadow(state, player, actions, chosen)
        return chosen

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
                "mode": "shadow",
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

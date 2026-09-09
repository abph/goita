"""攻め駒の目的を共通形式で記録し、未達時に再評価します。

攻め順 (``special_attack_plan``) は「どの駒を先に使うか」を表します。
このモジュールの ``attack_intent`` は、それとは別に「その攻めで何を
起こしたいのか」を表します。両者を分けることで、固定順が一度決まった
目的を無条件に上書きすることを防ぎます。
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple


Action = Tuple[str, Optional[str], Optional[str]]


class AttackIntentMixin:
    """攻めの目的のライフサイクルと再比較を担当します。"""

    _ATTACK_INTENT_CONTINUABLE_KINDS = frozenset({"force_enemy_royal"})

    def _set_attack_intent_plan(
        self,
        state,
        player: str,
        *,
        kind: str,
        attack_piece: Optional[str],
        source: str,
        target_team: str = "enemy",
        success_condition: Optional[Dict[str, object]] = None,
        expires: str = "next_attack_cycle",
        evidence: Optional[Dict[str, object]] = None,
    ) -> Optional[dict]:
        """次に出す攻め駒の目的を予約します。

        これは受けを選んだ直後など、まだ攻め駒を実際には出していない
        場面で呼びます。目的が実際の攻めとして公開された時点で
        ``_activate_attack_intent_from_action`` が active に移します。
        """
        tracker = self._track.get(id(state))
        if tracker is None or attack_piece is None:
            return None

        intent = {
            "kind": str(kind),
            "attack_piece": str(attack_piece),
            "source": str(source),
            "target_team": str(target_team),
            "success_condition": copy.deepcopy(success_condition or {}),
            "status": "created",
            "expires": str(expires),
            "evidence": copy.deepcopy(evidence or {}),
        }
        existing = tracker.get("planned_attack_intent")
        if isinstance(existing, dict):
            if (
                str(existing.get("kind")) == intent["kind"]
                and str(existing.get("attack_piece")) == intent["attack_piece"]
            ):
                existing.update(intent)
                return existing
        tracker["planned_attack_intent"] = intent
        return intent

    def _archive_attack_intent(
        self,
        tracker: dict,
        *,
        status: str,
        reason: str,
    ) -> None:
        active = tracker.get("attack_intent")
        if not isinstance(active, dict):
            return
        archived = copy.deepcopy(active)
        archived["status"] = str(status)
        archived["resolution_reason"] = str(reason)
        tracker.setdefault("attack_intent_history", []).append(archived)
        tracker["attack_intent"] = None

    def _activate_attack_intent_from_action(
        self,
        state,
        player: str,
        action: Action,
    ) -> Optional[dict]:
        """実際に選ばれた攻めを目的に結び付けます。"""
        if player != self.me or action[0] not in ("attack", "attack_after_block"):
            return None
        attack = action[2]
        if attack is None:
            return None
        tracker = self._track.get(id(state))
        if tracker is None:
            return None

        active = tracker.get("attack_intent")
        if isinstance(active, dict):
            if str(active.get("attack_piece")) == str(attack) and active.get(
                "status"
            ) in ("created", "unresolved"):
                same_public_action = (
                    active.get("status") == "created"
                    and not active.get("responses")
                    and active.get("last_attack_action") == list(action)
                )
                active["status"] = "created"
                if not same_public_action:
                    active["continuation_count"] = int(
                        active.get("continuation_count", 0)
                    ) + 1
                active["responses"] = []
                active["last_attack_piece"] = str(attack)
                active["last_attack_action"] = list(action)
                tracker["planned_attack_intent"] = None
                return active
            self._archive_attack_intent(
                tracker,
                status="invalidated",
                reason="different_attack_selected",
            )

        planned = tracker.get("planned_attack_intent")
        if not isinstance(planned, dict):
            # 旧フラグからの移行期間に、予約だけが作られていない
            # ルートでも目的を失わないようにする。
            pending_piece = tracker.get("pending_ally_force_king_attack_piece")
            if pending_piece is not None and str(pending_piece) == str(attack):
                planned = {
                    "kind": "force_enemy_royal",
                    "attack_piece": str(attack),
                    "source": "legacy_pending_ally_force_king",
                    "target_team": "enemy",
                    "success_condition": {"receive_piece": ["8", "9"]},
                    "status": "created",
                    "expires": "next_attack_cycle",
                    "evidence": {},
                }
        if not isinstance(planned, dict):
            return None
        if str(planned.get("attack_piece")) != str(attack):
            tracker["planned_attack_intent"] = None
            return None

        active = copy.deepcopy(planned)
        active["status"] = "created"
        active["created_attack_count"] = int(tracker.get("my_attack_count", 0))
        active["continuation_count"] = int(active.get("continuation_count", 0))
        active["responses"] = []
        active["last_attack_piece"] = str(attack)
        active["last_attack_action"] = list(action)
        tracker["attack_intent"] = active
        tracker["planned_attack_intent"] = None
        return active

    def _intent_targets_player(self, intent: dict, player: str) -> bool:
        target = str(intent.get("target_team", "enemy"))
        if target == "ally":
            return self._same_team(player, self.me) and player != self.me
        if target == "self":
            return player == self.me
        if target == "any":
            return True
        return player != self.me and not self._same_team(player, self.me)

    def _intent_receive_succeeds(self, intent: dict, player: str, block: object) -> bool:
        if not self._intent_targets_player(intent, player):
            return False
        condition = intent.get("success_condition")
        if not isinstance(condition, dict):
            return True
        pieces = condition.get("receive_piece")
        if pieces is None:
            return True
        return str(block) in {str(piece) for piece in pieces}

    def _update_attack_intent_from_public_action(
        self,
        state,
        player: str,
        action: Action,
    ) -> None:
        """公開された応答で目的を achieved/blocked/unresolved に更新します。"""
        tracker = self._track.get(id(state))
        if tracker is None:
            return

        action_type, block, attack = action
        if player == self.me and action_type in ("attack", "attack_after_block"):
            # リプレイや検索分岐では select_action を経ずに公開通知が来る
            # ことがあるため、ここでも予約を active 化する。
            self._activate_attack_intent_from_action(state, player, action)
            return

        intent = tracker.get("attack_intent")
        if not isinstance(intent, dict):
            return
        if intent.get("status") in ("achieved", "blocked", "invalidated", "expired"):
            return

        responses = intent.setdefault("responses", [])
        responses.append(
            {
                "player": str(player),
                "action": str(action_type),
                "block": str(block) if block is not None else None,
            }
        )

        if action_type == "receive":
            if self._intent_receive_succeeds(intent, player, block):
                intent["status"] = "achieved"
                intent["resolution_reason"] = "target_received"
            else:
                intent["status"] = "blocked"
                intent["resolution_reason"] = "non_target_receive"
            tracker.setdefault("attack_intent_history", []).append(copy.deepcopy(intent))
            tracker["attack_intent"] = None
            return

        if action_type == "pass":
            # 最後の受け手がパスした後は、state が攻め元の手番に戻る。
            # その時点で「目的が未達」という再検討可能な状態にする。
            if (
                state.phase == "attack"
                and state.attacker == self.me
                and state.turn == self.me
                and str(state.current_attack) == str(intent.get("attack_piece"))
            ):
                intent["status"] = "unresolved"
                intent["unresolved_reason"] = "all_responses_passed"
                intent["response_pass_count"] = sum(
                    1 for response in responses if response.get("action") == "pass"
                )

    def _score_attack_intent_action(
        self,
        state,
        player: str,
        action: Action,
        *,
        has_non_king_attack_option: bool,
    ) -> float:
        action_type, block, attack = action
        score = float(
            self._score_attack_phase(
                state,
                player,
                action_type,
                block,
                attack,
                has_non_king_attack_option=has_non_king_attack_option,
            )
        )
        if action_type == "attack_after_block":
            score += float(self._score_receive_phase(state, player, "receive", block))
        return score

    def _attack_intent_continuation_action(
        self,
        state,
        player: str,
        actions: List[Action],
        *,
        has_non_king_attack_option: bool,
    ) -> Optional[Action]:
        """未達の目的と固定攻め順を、同じ候補評価で比較します。"""
        if not bool(getattr(self, "ATTACK_INTENT_ENABLED", True)):
            return None
        if state.phase != "attack" or state.turn != player:
            return None
        tracker = self._track.get(id(state))
        if tracker is None:
            return None
        intent = tracker.get("attack_intent")
        if not isinstance(intent, dict) or intent.get("status") != "unresolved":
            return None
        if str(intent.get("kind")) not in self._ATTACK_INTENT_CONTINUABLE_KINDS:
            return None

        target_piece = str(intent.get("attack_piece"))
        same_piece = [
            action
            for action in actions
            if action[0] in ("attack", "attack_after_block")
            and action[2] is not None
            and str(action[2]) == target_piece
        ]
        if not same_piece:
            intent["status"] = "blocked"
            intent["resolution_reason"] = "purpose_piece_unavailable"
            return None

        scored_intent = [
            (
                self._score_attack_intent_action(
                    state,
                    player,
                    action,
                    has_non_king_attack_option=has_non_king_attack_option,
                ),
                action,
            )
            for action in same_piece
        ]
        scored_intent.sort(key=lambda item: item[0], reverse=True)
        intent_score, intent_action = scored_intent[0]

        baseline_action = self._special_attack_sequence_action(
            state,
            player,
            actions,
            has_non_king_attack_option=has_non_king_attack_option,
        )
        if baseline_action is None or baseline_action[2] == target_piece:
            baseline_candidates = [
                action
                for action in actions
                if action[0] in ("attack", "attack_after_block")
                and action[2] is not None
                and action[2] != target_piece
            ]
            if baseline_candidates:
                baseline_action = max(
                    baseline_candidates,
                    key=lambda action: self._score_attack_intent_action(
                        state,
                        player,
                        action,
                        has_non_king_attack_option=has_non_king_attack_option,
                    ),
                )
            else:
                baseline_action = None

        baseline_score = None
        if baseline_action is not None:
            baseline_score = self._score_attack_intent_action(
                state,
                player,
                baseline_action,
                has_non_king_attack_option=has_non_king_attack_option,
            )

        continuation_bonus = float(
            getattr(self, "ATTACK_INTENT_CONTINUATION_BONUS", 0.0)
        )
        effective_intent_score = float(intent_score) + continuation_bonus
        choose_intent = (
            baseline_score is None
            or effective_intent_score
            >= float(baseline_score)
            - float(getattr(self, "ATTACK_INTENT_MIN_MARGIN", 0.0))
        )
        comparison = {
            "kind": str(intent.get("kind")),
            "status": "unresolved",
            "intent_action": list(intent_action),
            "intent_score": round(float(intent_score), 3),
            "continuation_bonus": round(continuation_bonus, 3),
            "effective_intent_score": round(effective_intent_score, 3),
            "baseline_action": list(baseline_action) if baseline_action else None,
            "baseline_score": round(float(baseline_score), 3)
            if baseline_score is not None
            else None,
            "selected": "intent" if choose_intent else "baseline",
        }
        tracker["last_attack_intent_comparison"] = comparison
        self.last_attack_intent_comparison = copy.deepcopy(comparison)
        self.last_attack_candidate_scores = [
            {
                "action": list(intent_action),
                "score": float(intent_score),
                "intent_kind": str(intent.get("kind")),
                "intent_status": "unresolved",
                "candidate_role": "intent_continuation",
            }
        ]
        if baseline_action is not None:
            self.last_attack_candidate_scores.append(
                {
                    "action": list(baseline_action),
                    "score": float(baseline_score),
                    "intent_kind": str(intent.get("kind")),
                    "intent_status": "unresolved",
                    "candidate_role": "fixed_plan_baseline",
                }
            )

        if not choose_intent:
            return None
        tracker["my_attack_count"] = int(tracker.get("my_attack_count", 0)) + 1
        self._set_decision_reason("attack_intent")
        self._set_score_fallback_detail(
            f"attack_intent_{intent.get('kind')}_unresolved_continue_{target_piece}"
        )
        self._activate_attack_intent_from_action(state, player, intent_action)
        return intent_action

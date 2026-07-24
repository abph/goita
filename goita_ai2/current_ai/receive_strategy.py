"""場に出た攻めに対して、受けるかパスするかを評価します。
敵の初手、香・し・大駒への対応や、王玉を使うべき場面を手駒ランクから判断します。
味方の攻めを受ける条件と、受けた後に強く攻め返せるかどうかも考慮します。
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional, Tuple

from goita_ai2.constants import POINTS
from goita_ai2.current_ai.endgame import ForcedWinStatus

Action = Tuple[str, Optional[str], Optional[str]]


class ReceiveStrategyMixin:
    """Chooses when to receive, pass, or spend a king piece."""

    def _enemy_shi_response_adjustment(
        self,
        state,
        player: str,
        action_type: str,
        block: Optional[str],
    ) -> float:
        if not getattr(self, "USE_ENEMY_SHI_RESPONSE", True):
            return 0.0

        tr = self._track.get(id(state))
        if (
            tr is None
            or state.phase != "receive"
            or state.current_attack != "1"
            or state.attacker is None
            or self._same_team(state.attacker, player)
        ):
            return 0.0

        axes = self._initial_hand_axes_for_state(state, player)
        rank = self._strategy_rank_from_axes(axes)
        current_shis = state.hands[player].count("1")
        initial_shis = tr["my_init_count"].get("1", 0)

        if action_type == "pass":
            bonus = self.ENEMY_SHI_PASS_BONUS
            if rank in ("C", "D", "F"):
                bonus += 160.0
            elif rank == "B":
                bonus += 60.0
            elif rank in ("S", "A"):
                bonus -= 100.0
            if current_shis >= 4:
                bonus -= 250.0
            return bonus

        if action_type == "receive" and block == "1":
            value = -self.ENEMY_SHI_RECEIVE_PENALTY
            if current_shis >= 4:
                value += 620.0
            elif initial_shis >= 3 and current_shis >= 2:
                value += 180.0
            if rank in ("C", "D", "F"):
                value -= 100.0
            return value

        return 0.0

    def _enemy_first_receive_strength_bonus(
        self,
        state,
        player: str,
        action_type: str,
        block: Optional[str],
    ) -> float:
        axes = self._initial_hand_axes_for_state(state, player)
        use_relative = getattr(self, "USE_RELATIVE_HAND_RANK", True)
        rank = str(axes.get("relative_rank", axes["rank"]) if use_relative else axes["rank"])
        total = int(axes["total_score"])
        attack = int(axes["attack_score"])
        receive = int(axes["receive_score"])

        if action_type == "receive":
            block_bonus = 140.0 if block not in ("8", "9") else -120.0
            if rank == "SS":
                return 5000.0 + block_bonus
            if rank == "S":
                return 1700.0 + block_bonus
            if rank == "A":
                return 1250.0 + attack * 4.0 + receive * 2.0 + block_bonus
            if rank == "B":
                return 750.0 + attack * 3.0 + receive * 4.0 + block_bonus
            if rank == "C":
                return 300.0 + attack * 2.0 + receive * 5.0 + block_bonus
            return 120.0 + attack * 1.5 + receive * 4.0 + block_bonus

        if action_type == "pass":
            if rank == "SS":
                return -2500.0
            if rank == "S":
                return 350.0
            if rank == "A":
                return 650.0 - total * 4.0
            if rank == "B":
                return 900.0 - total * 2.0
            if rank == "C":
                return 1200.0 - max(receive, -20) * 3.0
            return 1450.0

        return 0.0

    def _ally_shi_exhaust_receive_bonus(self, state, player: str, block: Optional[str]) -> float:
        if (
            block is None
            or block not in ("3", "4", "5", "6", "7")
            or state.phase != "receive"
            or state.attacker is None
            or not self._same_team(state.attacker, player)
            or state.current_attack != block
        ):
            return 0.0
        if not self._next_enemy_likely_shi_exhausted(state, player):
            return 0.0
        return self.SHI_EXHAUST_RECEIVE_BONUS

    def _piece_count_receive_adjustment(self, state, player: str, action_type: str, block: Optional[str]) -> float:
        tr = self._track.get(id(state))
        target_piece = block if block is not None else state.current_attack
        if tr is None or target_piece is None or state.attacker is None:
            return 0.0
        if not (state.phase == "receive" and not self._same_team(state.attacker, player)):
            return 0.0

        remaining_min, remaining_max = self._estimate_remaining_range(tr, state.attacker, target_piece)
        if remaining_max <= 0:
            return 0.0

        axes = self._initial_hand_axes_for_state(state, player)
        rank = self._strategy_rank_from_axes(axes)
        weakish = rank in ("D", "F")
        if action_type == "pass":
            return self.INFER_REPEAT_RECEIVE_PASS_BONUS if weakish and remaining_max >= 1 else 0.0
        if action_type == "receive":
            penalty = self.INFER_REPEAT_RECEIVE_PENALTY if weakish and remaining_max >= 1 else 0.0
            if remaining_min > 0:
                penalty += 60.0
            return -penalty
        return 0.0

    def _enemy_first_same_piece_rank_policy_action(
        self,
        state,
        player: str,
        actions: List[Action],
    ) -> Optional[Action]:
        tr = self._track.get(id(state))
        if (
            tr is None
            or state.phase != "receive"
            or state.current_attack is None
            or state.attacker is None
            or self._same_team(state.attacker, player)
        ):
            return None

        attacker_count = tr.get("enemy_attack_counts", {}).get(state.attacker, 1)
        if attacker_count != 1:
            return None

        current_attack = str(state.current_attack)
        same_piece_receive = next(
            (
                act
                for act in actions
                if act[0] == "receive" and act[1] == current_attack
            ),
            None,
        )
        if same_piece_receive is None:
            return None

        if current_attack == "2":
            pass_action = next((act for act in actions if act[0] == "pass"), None)
            if pass_action is None:
                return None

            attacker = state.attacker
            next_after_attacker = state.next_player(attacker)
            axes = self._initial_hand_axes_for_state(state, player)
            absolute_rank = str(axes.get("absolute_rank", axes.get("rank", "D")))

            if player == next_after_attacker and absolute_rank in ("SS", "S", "A", "B"):
                self._set_decision_reason("score_fallback")
                self._set_score_fallback_detail(f"enemy_first_kyosha_next_abs{absolute_rank}_receive")
                return same_piece_receive

            self._set_decision_reason("score_fallback")
            if player == next_after_attacker:
                self._set_score_fallback_detail(f"enemy_first_kyosha_next_abs{absolute_rank}_pass")
            else:
                self._set_score_fallback_detail("enemy_first_kyosha_later_pass")
            return pass_action

        if current_attack == "1":
            axes = self._initial_hand_axes_for_state(state, player)
            absolute_rank = str(axes.get("absolute_rank", axes.get("rank", "D")))
            if absolute_rank in ("SS", "S", "A", "B", "C"):
                self._set_decision_reason("score_fallback")
                self._set_score_fallback_detail(f"enemy_first_shi_abs{absolute_rank}_same_piece_receive")
                return same_piece_receive

            pass_action = next((act for act in actions if act[0] == "pass"), None)
            if pass_action is not None and absolute_rank in ("D", "E", "F", "X"):
                if state.hands[player].count("1") >= 2:
                    self._set_decision_reason("score_fallback")
                    self._set_score_fallback_detail(f"enemy_first_shi_abs{absolute_rank}_two_shi_receive")
                    return same_piece_receive
                self._set_decision_reason("score_fallback")
                self._set_score_fallback_detail(f"enemy_first_shi_abs{absolute_rank}_one_shi_pass")
                return pass_action

        attacker = state.attacker
        next_after_attacker = state.next_player(attacker)
        ally = self._ally_of(player)
        ally_passed_to_me = (
            ally == next_after_attacker
            and bool(tr.get("ally_passed_enemy_first_attack"))
            and tr.get("ally_passed_enemy_first_attack_attacker") == attacker
            and tr.get("ally_passed_enemy_first_attack_piece") == current_attack
        )
        if ally_passed_to_me:
            dealer = getattr(state, "dealer", None)
            self._set_decision_reason("score_fallback")
            if dealer is not None and attacker == dealer:
                self._set_score_fallback_detail("enemy_dealer_first_ally_passed_same_piece_receive")
            else:
                self._set_score_fallback_detail("enemy_first_ally_passed_same_piece_receive")
            return same_piece_receive

        if player != next_after_attacker:
            return None

        axes = self._initial_hand_axes_for_state(state, player)
        absolute_rank = str(axes.get("absolute_rank", axes.get("rank", "D")))
        dealer = getattr(state, "dealer", None)
        weak_shi_signal_receive = (
            attacker == dealer
            and current_attack in ("3", "4", "5")
            and absolute_rank in ("D", "E", "F", "X")
            and state.hands[player].count("1") >= 3
            and self._effective_receive_type(Counter(state.hands[player])) <= 3
        )
        if weak_shi_signal_receive:
            tr["pending_weak_hand_shi_signal"] = True
            self._set_decision_reason("score_fallback")
            self._set_score_fallback_detail(
                f"enemy_dealer_first_next_abs{absolute_rank}_weak_shi_signal_receive"
            )
            return same_piece_receive

        if absolute_rank in ("SS", "S", "A", "B", "C"):
            self._set_decision_reason("score_fallback")
            if dealer is not None and attacker == dealer:
                self._set_score_fallback_detail(f"enemy_dealer_first_next_abs{absolute_rank}_same_piece_receive")
            else:
                self._set_score_fallback_detail(f"enemy_first_next_abs{absolute_rank}_same_piece_receive")
            return same_piece_receive

        pass_action = next((act for act in actions if act[0] == "pass"), None)
        if pass_action is not None and absolute_rank in ("D", "E", "F", "X"):
            if current_attack == "1":
                if state.hands[player].count("1") >= 2:
                    self._set_decision_reason("score_fallback")
                    if dealer is not None and attacker == dealer:
                        self._set_score_fallback_detail(f"enemy_dealer_first_next_abs{absolute_rank}_two_shi_receive")
                    else:
                        self._set_score_fallback_detail(f"enemy_first_next_abs{absolute_rank}_two_shi_receive")
                    return same_piece_receive
                self._set_decision_reason("score_fallback")
                if dealer is not None and attacker == dealer:
                    self._set_score_fallback_detail(f"enemy_dealer_first_next_abs{absolute_rank}_one_shi_pass")
                else:
                    self._set_score_fallback_detail(f"enemy_first_next_abs{absolute_rank}_one_shi_pass")
                return pass_action
            self._set_decision_reason("score_fallback")
            if dealer is not None and attacker == dealer:
                self._set_score_fallback_detail(f"enemy_dealer_first_next_abs{absolute_rank}_same_piece_pass")
            else:
                self._set_score_fallback_detail(f"enemy_first_next_abs{absolute_rank}_same_piece_pass")
            return pass_action

        return None

    def _guaranteed_finish_receive_action(
        self,
        state,
        player: str,
        actions: List[Action],
    ) -> Optional[Action]:
        if state.phase != "receive":
            return None

        finish_after_receive_actions: List[Tuple[float, str, Action]] = []
        for act in actions:
            if act[0] != "receive":
                continue
            result = self._forced_win_result_after_receive_action(state, player, act)
            if result.status != ForcedWinStatus.PROVEN or result.minimum_score is None:
                continue
            detail = "receive_win_after"
            if self._win_after_receive_bonus(state, player, act) <= 0:
                detail = "receive_tsume_after"
            finish_after_receive_actions.append((result.minimum_score, detail, act))

        if not finish_after_receive_actions:
            return None

        finish_after_receive_actions.sort(key=lambda x: x[0], reverse=True)
        _score, detail, chosen = finish_after_receive_actions[0]
        self._set_decision_reason("score_fallback")
        self._set_score_fallback_detail(detail)
        return chosen

    def _no_shi_royal_endgame_commit_action(
        self,
        state,
        player: str,
        actions: List[Action],
    ) -> Optional[Action]:
        """Use a royal for the third attack route when passing leaves no shi defence."""
        tr = self._track.get(id(state))
        if (
            tr is None
            or state.phase != "receive"
            or state.attacker is None
            or self._same_team(state.attacker, player)
            or int(tr.get("my_attack_count", 0)) != 2
            or len(state.hands[player]) != 4
            or "1" in state.hands[player]
        ):
            return None

        royal_receives = [
            action
            for action in actions
            if action[0] == "receive" and action[1] in ("8", "9")
        ]
        if not royal_receives:
            return None

        royal_receives.sort(
            key=lambda action: self._score_receive_phase(
                state,
                player,
                action[0],
                action[1],
            ),
            reverse=True,
        )
        self._set_decision_reason("score_fallback")
        self._set_score_fallback_detail("receive_no_shi_royal_endgame_commit")
        return royal_receives[0]

    def _full_receive_cover_royal_wait_pass_action(
        self,
        state,
        player: str,
        actions: List[Action],
    ) -> Optional[Action]:
        """Delay a proven low-score finish when し・香・玉・王 cover every next attack."""
        tr = self._track.get(id(state))
        if (
            tr is None
            or state.phase != "receive"
            or state.current_attack not in ("3", "4", "5", "6", "7")
            or state.attacker is None
            or self._same_team(state.attacker, player)
            or int(tr.get("my_attack_count", 0)) != 2
            or not bool(state.had_both_kings.get(player, False))
        ):
            return None

        hand = state.hands[player]
        if Counter(hand) != Counter(("1", "2", "8", "9")):
            return None

        attacker = state.attacker
        if (
            int(tr.get("enemy_attack_counts", {}).get(attacker, 0)) != 2
            or len(state.hands[attacker]) != 4
        ):
            return None

        # An enemy with two cards could receive before our next turn and finish
        # immediately, so the waiting route is no longer guaranteed.
        for enemy in ("A", "B", "C", "D"):
            if enemy == player or self._same_team(enemy, player):
                continue
            if enemy != attacker and len(state.hands[enemy]) <= 2:
                return None

        pass_action = next((action for action in actions if action[0] == "pass"), None)
        if pass_action is None:
            return None

        self._set_decision_reason("score_fallback")
        self._set_score_fallback_detail("pass_full_receive_cover_royal_wait_high_score")
        return pass_action

    def _early_big_piece_same_receive_action(
        self,
        state,
        player: str,
        actions: List[Action],
    ) -> Optional[Action]:
        tr = self._track.get(id(state))
        if (
            tr is None
            or state.phase != "receive"
            or state.current_attack not in ("6", "7")
            or state.attacker is None
            or self._same_team(state.attacker, player)
        ):
            return None

        same_piece_receive = next(
            (
                act
                for act in actions
                if act[0] == "receive" and act[1] == state.current_attack
            ),
            None,
        )
        if same_piece_receive is None:
            return None

        public_seen = sum(int(v) for v in tr.get("public_seen_counts", {}).values())
        attacker_count = tr.get("enemy_attack_counts", {}).get(state.attacker, 1)
        if public_seen > 8 and attacker_count != 1:
            return None

        self._set_decision_reason("score_fallback")
        self._set_score_fallback_detail("early_big_piece_same_receive")
        return same_piece_receive

    def _early_enemy_first_king_receive_penalty(
        self,
        state,
        player: str,
        action_type: str,
        block: Optional[str],
    ) -> float:
        if action_type != "receive" or block not in ("8", "9"):
            return 0.0

        tr = self._track.get(id(state))
        if (
            tr is None
            or state.phase != "receive"
            or state.current_attack is None
            or state.attacker is None
            or self._same_team(state.attacker, player)
        ):
            return 0.0

        attacker_count = tr.get("enemy_attack_counts", {}).get(state.attacker, 1)
        if attacker_count != 1:
            return 0.0

        return self.FIRST_ENEMY_KING_RECEIVE_PENALTY

    def _enemy_second_attack_royal_reserve_pass_action(
        self,
        state,
        player: str,
        actions: List[Action],
    ) -> Optional[Action]:
        tr = self._track.get(id(state))
        if (
            tr is None
            or state.phase != "receive"
            or state.current_attack not in ("6", "7")
            or state.attacker is None
            or self._same_team(state.attacker, player)
            or state.next_player(player) != state.attacker
            or len(state.hands[player]) != 4
            or len(state.hands[state.attacker]) != 4
        ):
            return None

        attacker = state.attacker
        if int(tr.get("enemy_attack_counts", {}).get(attacker, 1)) != 2:
            return None

        hand = state.hands[player]
        royal = next((piece for piece in ("8", "9") if piece in hand), None)
        if (
            royal is None
            or hand.count("8") + hand.count("9") != 1
            or hand.count("1") < 1
            or hand.count("2") < 1
            or not any(hand.count(piece) >= 1 for piece in ("3", "4", "5"))
        ):
            return None

        attack_history = list(tr.get("my_attack_history", []))
        if len(attack_history) < 2 or attack_history[0] != "2" or attack_history[1] not in ("3", "4", "5"):
            return None

        attacker_blocks = tr.get("public_hand_models", {}).get(attacker, {}).get("blocks", Counter())
        if int(attacker_blocks.get("8", 0)) + int(attacker_blocks.get("9", 0)) < 1:
            return None
        known_royals = (
            int(tr.get("public_seen_counts", {}).get("8", 0))
            + int(tr.get("public_seen_counts", {}).get("9", 0))
            + hand.count("8")
            + hand.count("9")
        )
        if known_royals < 2:
            return None

        royal_receive = next(
            (action for action in actions if action[0] == "receive" and action[1] == royal),
            None,
        )
        pass_action = next((action for action in actions if action[0] == "pass"), None)
        if royal_receive is None or pass_action is None:
            return None

        self._set_decision_reason("score_fallback")
        self._set_score_fallback_detail("pass_royal_reserve_wait_ally_kakari")
        return pass_action

    def _strong_ally_receive_followup_pieces(self, hand: List[str]) -> List[str]:
        counts = Counter(hand)
        pieces: List[str] = []
        if counts.get("2", 0) >= 3:
            pieces.append("2")
        for piece in ("5", "4", "3"):
            if counts.get(piece, 0) >= 3:
                pieces.append(piece)
        for piece in ("7", "6"):
            if counts.get(piece, 0) >= 2:
                pieces.append(piece)
        pieces.sort(key=lambda piece: (POINTS.get(piece, 0), piece), reverse=True)
        return pieces

    def _strong_ally_receive_followup_piece_after_receive(
        self,
        state,
        player: str,
        action_type: str,
        block: Optional[str],
    ) -> Optional[str]:
        if action_type != "receive" or block is None:
            return None
        if (
            state.phase != "receive"
            or state.current_attack is None
            or state.attacker is None
            or not self._same_team(state.attacker, player)
            or block != state.current_attack
            or block == "1"
        ):
            return None

        hand_after = list(state.hands[player])
        if block not in hand_after:
            return None
        hand_after.remove(block)

        pieces = self._strong_ally_receive_followup_pieces(hand_after)
        return pieces[0] if pieces else None

    def _ally_strong_followup_receive_bonus(
        self,
        state,
        player: str,
        action_type: str,
        block: Optional[str],
    ) -> float:
        if self._strong_ally_receive_followup_piece_after_receive(state, player, action_type, block) is not None:
            return self.ALLY_STRONG_FOLLOWUP_RECEIVE_BONUS
        return 0.0

    def _kakari_saturation_receive_bonus(self, state, player: str, block: Optional[str]) -> float:
        if not self._is_kakarigotae_piece(block):
            return 0.0
        if state.attacker is None or not self._same_team(state.attacker, player):
            return 0.0
        if state.current_attack != block:
            return 0.0

        tr = self._track.get(id(state))
        if tr is None:
            return 0.0
        if not (block == tr.get("ally_first_attack") or block in tr.get("ally_past_attacks", set())):
            return 0.0
        if state.hands[player].count(block) < 2:
            return 0.0

        after_return = list(state.hands[player])
        after_return.remove(block)
        after_return.remove(block)
        followups = self._kakari_saturation_followup_pieces(after_return)
        if not followups:
            return 0.0

        bonus = self.KAKARI_SATURATION_RECEIVE_BONUS
        bonus += max(float(POINTS.get(piece, 0)) for piece in followups) / 2.0

        ally = state.attacker
        remaining_min, remaining_max = self._estimate_remaining_range(tr, ally, block)
        if remaining_min > 0 or remaining_max > 0:
            bonus += self.KAKARI_SATURATION_ALLY_REMAINING_BONUS
        return bonus

    def _attack_forces_enemy_king_receive(
        self,
        state,
        player: str,
        attack: Optional[str],
        hand_after_receive: Optional[List[str]] = None,
    ) -> bool:
        if attack is None or attack not in ("3", "4", "5", "6", "7"):
            return False

        tr = self._track.get(id(state))
        if tr is None:
            return False

        hand = hand_after_receive if hand_after_receive is not None else list(state.hands[player])
        if attack not in hand:
            return False

        total = self._piece_total(attack)
        seen = int(tr.get("public_seen_counts", {}).get(attack, 0))
        mine = hand.count(attack)
        if seen + mine >= total:
            return True

        saw_estimate = False
        for other in ("A", "B", "C", "D"):
            if other == player or self._same_team(other, player):
                continue
            if self._piece_count_estimate(tr, other, attack) is not None:
                saw_estimate = True
            _remaining_min, remaining_max = self._estimate_remaining_range(tr, other, attack)
            if remaining_max > 0:
                return False
        return saw_estimate

    def _ally_force_king_attack_piece_after_receive(
        self,
        state,
        player: str,
        action_type: str,
        block: Optional[str],
    ) -> Optional[str]:
        if action_type != "receive" or block is None:
            return None
        if (
            state.phase != "receive"
            or state.current_attack is None
            or state.attacker is None
            or not self._same_team(state.attacker, player)
            or block != state.current_attack
        ):
            return None

        hand_after = list(state.hands[player])
        if block not in hand_after:
            return None
        hand_after.remove(block)

        fourth_middle_followups: set[str] = set()
        tr = self._track.get(id(state))
        if tr is not None and int(tr.get("my_attack_count", 0)) == 2:
            fourth_middle_followups = {
                piece
                for piece in ("3", "4", "5")
                if piece in hand_after
                and (
                    int(tr.get("public_seen_counts", {}).get(piece, 0))
                    + hand_after.count(piece)
                    >= self._piece_total(piece)
                )
            }

        strong_followups = set(self._strong_ally_receive_followup_pieces(hand_after))
        candidates = [
            p
            for p in set(hand_after)
            if p in strong_followups | fourth_middle_followups
            and self._attack_forces_enemy_king_receive(state, player, p, hand_after)
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda p: (POINTS.get(p, 0), p), reverse=True)
        return candidates[0]

    def _ally_force_king_receive_bonus(
        self,
        state,
        player: str,
        action_type: str,
        block: Optional[str],
    ) -> float:
        if self._ally_force_king_attack_piece_after_receive(state, player, action_type, block) is not None:
            return self.ALLY_FORCE_KING_RECEIVE_BONUS
        return 0.0

    def _ally_kyosha_continuation_pass_action(
        self,
        state,
        player: str,
        actions: List[Action],
    ) -> Optional[Action]:
        tr = self._track.get(id(state))
        ally = state.attacker
        if (
            tr is None
            or state.phase != "receive"
            or state.current_attack != "2"
            or ally is None
            or not self._same_team(ally, player)
            or state.hands[player].count("2") != 1
            or int(tr.get("my_init_count", Counter()).get("2", 0)) != 1
        ):
            return None

        ally_model = tr.get("public_hand_models", {}).get(ally, {})
        ally_attack_count = int(ally_model.get("attack_count", 0))
        if (
            ally_model.get("first_attack") != "2"
            or ally_attack_count < 1
            or bool(ally_model.get("strategy_broken"))
        ):
            return None

        first_responder = state.next_player(ally)
        if (
            state.next_player(first_responder) != player
            or self._same_team(first_responder, player)
        ):
            return None

        _remaining_min, remaining_max = self._estimate_remaining_range(
            tr,
            ally,
            "2",
        )
        if ally_attack_count == 1 and remaining_max < 1:
            return None

        hand_after_receive = list(state.hands[player])
        hand_after_receive.remove("2")
        if any(
            hand_after_receive.count(piece) >= 3
            for piece in ("3", "4", "5")
        ):
            return None

        enemies = [
            seat
            for seat in ("A", "B", "C", "D")
            if not self._same_team(seat, player)
        ]
        if any(len(state.hands[enemy]) <= 2 for enemy in enemies):
            return None

        pass_action = next(
            (action for action in actions if action[0] == "pass"),
            None,
        )
        same_piece_receive = next(
            (
                action
                for action in actions
                if action[0] == "receive" and action[1] == "2"
            ),
            None,
        )
        if pass_action is None or same_piece_receive is None:
            return None

        self._set_decision_reason("score_fallback")
        self._set_score_fallback_detail("pass_ally_kyosha_continuation")
        return pass_action

    def _king_gyoku_opening_keep_receive_width_action(
        self,
        state,
        player: str,
        actions: List[Action],
    ) -> Optional[Action]:
        tr = self._track.get(id(state))
        if (
            tr is None
            or not tr.get("kg_plan_active")
            or int(tr.get("my_attack_count", 0)) != 0
            or state.phase != "attack"
            or state.attacker is not None
            or state.current_attack is not None
        ):
            return None

        counts = tr["my_init_count"]
        if counts.get("1", 0) < 3:
            return None
        if self._has_strong_repeat_attack(counts):
            return None

        for act in actions:
            if act[0] == "attack_after_block" and act[1] == "1" and act[2] == "1":
                tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                self._set_decision_reason("score_fallback")
                self._set_score_fallback_detail("king_gyoku_keep_receive_width")
                return act
        return None

    def _score_receive_phase(self, state, player: str, action_type: str, block: Optional[str]) -> float:
        if action_type == "pass":
            base = 0.0
        else:
            if action_type != "receive" or block is None:
                return -1e18
            bonus = self._win_after_receive_bonus(state, player, (action_type, block, None))
            if bonus > 0:
                return 1e9

            forced_result = self._forced_win_result_after_receive_action(
                state,
                player,
                (action_type, block, None),
            )
            if (
                forced_result.status == ForcedWinStatus.PROVEN
                and forced_result.minimum_score is not None
            ):
                return 1e8 + forced_result.minimum_score

            if state.attacker is not None and self._same_team(state.attacker, player):
                ally_strength = self._public_hand_strength(self._track.get(id(state)), state.attacker)
                saturation_bonus = self._kakari_saturation_receive_bonus(state, player, block)
                force_king_bonus = self._ally_force_king_receive_bonus(state, player, action_type, block)
                strong_followup_bonus = self._ally_strong_followup_receive_bonus(state, player, action_type, block)
                shi_exhaust_bonus = self._ally_shi_exhaust_receive_bonus(state, player, block)
                return (
                    -100.0
                    - min(220.0, ally_strength * 8.0)
                    + saturation_bonus
                    + force_king_bonus
                    + strong_followup_bonus
                    + shi_exhaust_bonus
                )
            base = 1.0 if block in ("8", "9") else 5.0

        tr = self._track.get(id(state))
        if tr is None:
            return base

        enemy_attack_turn = (
            state.phase == "receive"
            and state.current_attack is not None
            and state.attacker is not None
            and (not self._same_team(state.attacker, player))
        )

        if enemy_attack_turn:
            attacker = state.attacker
            attacker_count = tr.get("enemy_attack_counts", {}).get(attacker, 1)
            attacker_strength = self._public_hand_strength(tr, attacker)
            preserve_win_value = self._preserve_current_attack_for_win_value(state, player)
            base += self._piece_count_receive_adjustment(state, player, action_type, block)
            base -= self._early_enemy_first_king_receive_penalty(state, player, action_type, block)
            if preserve_win_value > 0:
                if action_type == "pass":
                    base += preserve_win_value
                elif action_type == "receive":
                    base -= self.PRESERVE_WIN_ATTACK_RECEIVE_PENALTY
            if action_type == "receive":
                base += min(260.0, max(0.0, attacker_strength) * 9.0)
            elif action_type == "pass":
                base -= min(160.0, max(0.0, attacker_strength) * 5.0)

            if attacker_count == 1:
                policy = getattr(self, "ENEMY_FIRST_ATTACK_POLICY", "strict_pass")
                if policy == "strict_pass":
                    if action_type == "pass":
                        base += 10000.0
                    else:
                        return -1e18
                elif policy == "relaxed":
                    if action_type == "pass":
                        base += 1000.0
                    elif action_type == "receive":
                        base += 900.0 if block in ("8", "9") else 950.0
                elif policy == "receive_preferred":
                    if action_type == "receive":
                        base += 1200.0 if block not in ("8", "9") else 1000.0
                    else:
                        base += 700.0
                elif policy == "neutral":
                    if action_type == "pass":
                        base += 100.0
                    elif action_type == "receive":
                        base += 100.0
                elif policy == "hand_power":
                    hand_power = self._calculate_hand_power(state, player, tr)
                    if hand_power >= 95.0:
                        if action_type == "receive":
                            base += 1300.0 if block not in ("8", "9") else 1050.0
                        else:
                            base += 600.0
                    elif hand_power >= 70.0:
                        if action_type == "receive":
                            base += 850.0 if block not in ("8", "9") else 700.0
                        else:
                            base += 900.0
                    else:
                        if action_type == "pass":
                            base += 1200.0
                        elif action_type == "receive":
                            base += 450.0 if block not in ("8", "9") else 250.0
                elif policy == "hand_power_loose":
                    hand_power = self._calculate_hand_power(state, player, tr)
                    if hand_power >= 80.0:
                        if action_type == "receive":
                            base += 1300.0 if block not in ("8", "9") else 1050.0
                        else:
                            base += 600.0
                    elif hand_power >= 55.0:
                        if action_type == "receive":
                            base += 850.0 if block not in ("8", "9") else 700.0
                        else:
                            base += 900.0
                    else:
                        if action_type == "pass":
                            base += 1200.0
                        elif action_type == "receive":
                            base += 450.0 if block not in ("8", "9") else 250.0
                elif policy == "hand_power_aggressive":
                    hand_power = self._calculate_hand_power(state, player, tr)
                    if hand_power >= 55.0:
                        if action_type == "receive":
                            base += 1300.0 if block not in ("8", "9") else 1050.0
                        else:
                            base += 600.0
                    elif hand_power >= 35.0:
                        if action_type == "receive":
                            base += 850.0 if block not in ("8", "9") else 650.0
                        else:
                            base += 900.0
                    else:
                        if action_type == "pass":
                            base += 1250.0
                        elif action_type == "receive":
                            base += 350.0 if block not in ("8", "9") else 200.0
                elif policy == "hand_strength":
                    base += self._enemy_first_receive_strength_bonus(state, player, action_type, block)
                    if action_type == "pass" and self._weak_shi_attack_plan_value(state, player) > 0:
                        base += self.SHI_ATTACK_PREPARE_PASS_BONUS
                    base += self._enemy_shi_response_adjustment(state, player, action_type, block)
                else:
                    if action_type == "pass":
                        base += 10000.0
                    else:
                        return -1e18
            else:
                if state.current_attack == "1":
                    if action_type == "receive":
                        base += 250.0 + self._enemy_shi_response_adjustment(state, player, action_type, block)
                    else:
                        base += 180.0 + self._enemy_shi_response_adjustment(state, player, action_type, block)
                else:
                    if action_type == "receive":
                        if block in ("8", "9"):
                            base += 1000.0
                        else:
                            base += 10000.0
                    else:
                        base -= 10000.0

        return base

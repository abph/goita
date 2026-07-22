"""終盤で勝ち切る手順と、その得点を評価します。
現在の盤面から詰みを探し、確定上がりの中でも得点が高い手順を優先します。
自分が低得点で上がるより、味方に上がらせた方がよい場面の判断も扱います。
"""

from __future__ import annotations

import copy
from collections import Counter
from typing import List, Optional, Tuple

from goita_ai2.constants import POINTS

Action = Tuple[str, Optional[str], Optional[str]]


class EndgameMixin:
    """Finds forced wins and prefers higher-scoring finish routes."""

    def _endgame_pair_score(self, state, player: str, pair: List[str]) -> float:
        if len(pair) != 2:
            return 0.0
        p1, p2 = pair[0], pair[1]
        tr = self._track.get(id(state))

        if p1 == p2:
            score = float(POINTS.get(p1, 0)) * 2.0
            if tr is not None and self._opponents_piece_pressure(tr, player, p1) > 0:
                score -= self.ENDGAME_PAIR_UNCERTAIN_PENALTY
            return max(0.0, score)

        if set(pair) == {"8", "9"}:
            return 100.0

        kings = [p for p in pair if p in ("8", "9")]
        non_kings = [p for p in pair if p not in ("8", "9")]
        if kings and non_kings:
            finisher = non_kings[0]
            score = float(POINTS.get(finisher, 0))
            if finisher not in ("1", "2"):
                score += self.ENDGAME_PAIR_KING_RECEIVE_BONUS
            else:
                score -= self.ENDGAME_PAIR_UNCERTAIN_PENALTY
            if tr is not None and self._opponents_piece_pressure(tr, player, finisher) > 0:
                score -= self.ENDGAME_PAIR_UNCERTAIN_PENALTY
            return max(0.0, score)

        score = max(float(POINTS.get(p1, 0)), float(POINTS.get(p2, 0)))
        if tr is not None:
            pressure = max(
                self._opponents_piece_pressure(tr, player, p1),
                self._opponents_piece_pressure(tr, player, p2),
            )
            if pressure > 0:
                score -= self.ENDGAME_PAIR_UNCERTAIN_PENALTY
        return max(0.0, score)

    def _remaining_hand_after_attack_action(
        self,
        state,
        player: str,
        block: Optional[str],
        attack: Optional[str],
    ) -> Optional[List[str]]:
        if attack is None:
            return None
        hand = list(state.hands[player])
        if block is not None:
            if block not in hand:
                return None
            hand.remove(block)
        if attack not in hand:
            return None
        hand.remove(attack)
        return hand

    def _conditional_shi_royal_finish_score(
        self,
        state,
        player: str,
        action_type: str,
        block: Optional[str],
        attack: Optional[str],
    ) -> Optional[float]:
        if action_type not in ("attack", "attack_after_block") or attack is None:
            return None

        tr = self._track.get(id(state))
        if tr is None or not self._enemy_likely_to_repeat_shi(tr, player):
            return None
        if int(getattr(state, "king_block_used", 0)) <= 0:
            return None

        after_hand = self._remaining_hand_after_attack_action(state, player, block, attack)
        if after_hand is None or len(after_hand) != 4 or "1" not in after_hand:
            return None

        best_score: Optional[float] = None
        public_seen = tr.get("public_seen_counts", {})
        for royal in ("8", "9"):
            if royal not in after_hand:
                continue
            other_royal = "9" if royal == "8" else "8"
            if int(public_seen.get(other_royal, 0)) <= 0:
                continue

            final_pair = list(after_hand)
            final_pair.remove("1")
            final_pair.remove(royal)
            if len(final_pair) != 2:
                continue
            score = self._pair_finish_score(final_pair)
            if score >= 0 and (best_score is None or score > best_score):
                best_score = score
        return best_score

    def _conditional_shi_royal_finish_adjustment(
        self,
        state,
        player: str,
        action_type: str,
        block: Optional[str],
        attack: Optional[str],
    ) -> float:
        score = self._conditional_shi_royal_finish_score(
            state,
            player,
            action_type,
            block,
            attack,
        )
        if score is None:
            return 0.0
        return (
            self.CONDITIONAL_SHI_ROYAL_ROUTE_BASE_BONUS
            + score * self.CONDITIONAL_SHI_ROYAL_ROUTE_SCORE_WEIGHT
        )

    def _shi_sashikomi_wait_bonus_for_pair(self, state, player: str, pair: List[str]) -> float:
        if len(pair) != 2 or pair.count("1") != 1:
            return 0.0

        tr = self._track.get(id(state))
        if tr is None:
            return 0.0

        if not tr.get("ally_shi_passed_by_enemy"):
            return 0.0
        if "1" not in tr.get("ally_past_attacks", set()):
            return 0.0

        finisher = next((p for p in pair if p != "1"), None)
        if finisher is None:
            return 0.0

        bonus = self.SHI_SASHIKOMI_WAIT_BONUS
        if finisher in ("8", "9"):
            bonus += 60.0
        elif finisher in ("6", "7"):
            bonus += 40.0
        elif finisher in ("2", "3", "4", "5"):
            bonus += 20.0
        return bonus

    def _shi_sashikomi_wait_bonus(
        self,
        state,
        player: str,
        block: Optional[str],
        attack: Optional[str],
    ) -> float:
        hand = self._remaining_hand_after_attack_action(state, player, block, attack)
        if hand is None or len(hand) != 2:
            return 0.0
        return self._shi_sashikomi_wait_bonus_for_pair(state, player, hand)

    def _shi_sashikomi_attack_bonus(self, state, player: str) -> float:
        tr = self._track.get(id(state))
        if tr is None:
            return 0.0
        if not tr.get("ally_shi_sashikomi_candidate"):
            return 0.0
        if state.hands[player].count("1") <= 0:
            return 0.0
        if int(tr.get("enemy_passed_my_shi_count", 0)) <= 0:
            return 0.0

        ally_remaining = max(0, 8 - int(tr.get("ally_consumed_count", 0)))
        if ally_remaining > 2:
            return 0.0

        return self.SHI_SASHIKOMI_ATTACK_BONUS

    def _endgame_remaining_pair_adjustment(
        self,
        state,
        player: str,
        block: Optional[str],
        attack: Optional[str],
    ) -> float:
        hand = self._remaining_hand_after_attack_action(state, player, block, attack)
        if hand is None:
            return 0.0
        if len(hand) != 2:
            return 0.0
        score = self._endgame_pair_score(state, player, hand) * self.ENDGAME_PAIR_SCORE_WEIGHT
        shi_count = hand.count("1")
        if shi_count == 1:
            score += self.ENDGAME_MIXED_SHI_PAIR_BONUS
            score += self._shi_sashikomi_wait_bonus_for_pair(state, player, hand)
        elif shi_count == 2:
            score -= self.ENDGAME_SHI_PAIR_PENALTY
        return score

    def _is_absolute_safe_for_tsume(self, state, player: str, attack: str, tr: dict) -> bool:
        if attack is None:
            return False
        if attack in ("8", "9"):
            return True
        total_p = 4 if attack in ("2", "3", "4", "5") else 2 if attack in ("6", "7") else 10 if attack == "1" else 1
        seen_and_mine = tr.get("public_seen_counts", {}).get(attack, 0) + state.hands[player].count(attack)
        if seen_and_mine < total_p:
            return False
        if attack in ("1", "2"):
            return True
        visible_kings = tr.get("public_seen_counts", {}).get("8", 0) + tr.get("public_seen_counts", {}).get("9", 0) + state.hands[player].count("8") + state.hands[player].count("9")
        return visible_kings == 2

    def _is_tsume_from_even(self, hand: List[str], state, player: str, tr: dict) -> bool:
        if len(hand) <= 2:
            return True

        safe_pieces = set(p for p in hand if self._is_absolute_safe_for_tsume(state, player, p, tr))
        if not safe_pieces:
            return False

        for atk in safe_pieces:
            temp1 = list(hand)
            temp1.remove(atk)
            for fuse in set(temp1):
                temp2 = list(temp1)
                temp2.remove(fuse)
                if self._is_tsume_from_even(temp2, state, player, tr):
                    return True
        return False

    def _max_tsume_score(self, hand: List[str], state, player: str, tr: dict) -> float:
        """確定上がりルートを探索し、そのルートでの最大打点を逆算して返す"""
        if len(hand) <= 2:
            if len(hand) == 2:
                p1, p2 = hand[0], hand[1]
                if p1 == p2:
                    return float(POINTS.get(p1, 0)) * 2.0
                if set([p1, p2]) == {"8", "9"}:
                    return 100.0
                return max(float(POINTS.get(p1, 0)), float(POINTS.get(p2, 0)))
            if len(hand) == 1:
                return float(POINTS.get(hand[0], 0))

        safe_pieces = set(p for p in hand if self._is_absolute_safe_for_tsume(state, player, p, tr))
        if not safe_pieces:
            return -1.0

        best_score = -1.0
        for atk in safe_pieces:
            temp1 = list(hand)
            temp1.remove(atk)
            for fuse in set(temp1):
                temp2 = list(temp1)
                temp2.remove(fuse)
                score = self._max_tsume_score(temp2, state, player, tr)
                if score > best_score:
                    best_score = score
        return best_score

    def _pair_finish_score(self, pair: List[str]) -> float:
        if len(pair) != 2:
            return -1.0
        p1, p2 = pair[0], pair[1]
        if p1 == p2:
            return float(POINTS.get(p1, 0)) * 2.0
        if set([p1, p2]) == {"8", "9"}:
            return 100.0
        return max(float(POINTS.get(p1, 0)), float(POINTS.get(p2, 0)))

    def _best_pair_finish_score_from_hand(self, hand: List[str]) -> float:
        if len(hand) < 2:
            return -1.0
        best = -1.0
        for i in range(len(hand)):
            for j in range(i + 1, len(hand)):
                score = self._pair_finish_score([hand[i], hand[j]])
                if score > best:
                    best = score
        return best

    def _best_safe_pair_pressure_finish_score(self, hand: List[str], state, player: str, tr: dict) -> float:
        best = -1.0
        counts = Counter(hand)
        for piece, count in counts.items():
            if count < 2 or piece in ("1", "2", "8", "9"):
                continue
            if not self._is_absolute_safe_for_tsume(state, player, piece, tr):
                continue
            rest = list(hand)
            rest.remove(piece)
            rest.remove(piece)
            score = self._best_pair_finish_score_from_hand(rest)
            if score > best:
                best = score
        return best

    def _kyosha_probe_expected_score_after_attack_action(
        self,
        state,
        player: str,
        action: Action,
        tr: dict,
    ) -> Optional[float]:
        action_type, block, attack = action
        if action_type != "attack" or block is not None or attack != "2":
            return None
        if state.hands[player].count("2") != 1:
            return None

        known_kyosha_before = int(tr.get("public_seen_counts", {}).get("2", 0)) + state.hands[player].count("2")
        if known_kyosha_before < 3:
            return None

        after_hand = list(state.hands[player])
        after_hand.remove("2")
        if "1" not in after_hand or not any(p in ("8", "9") for p in after_hand):
            return None

        pass_score = self._best_safe_pair_pressure_finish_score(after_hand, state, player, tr)
        if pass_score < 0:
            return None

        receive_scores: List[float] = []
        after_shi_receive = list(after_hand)
        after_shi_receive.remove("1")
        receive_scores.append(self._best_safe_pair_pressure_finish_score(after_shi_receive, state, player, tr))
        for king in ("9", "8"):
            if king in after_hand:
                after_king_receive = list(after_hand)
                after_king_receive.remove(king)
                receive_scores.append(self._best_safe_pair_pressure_finish_score(after_king_receive, state, player, tr))

        receive_scores = [score for score in receive_scores if score >= 0]
        if not receive_scores:
            return None

        receive_score = min(receive_scores)
        return max(receive_score, (pass_score + receive_score) / 2.0)

    def _apply_action_on_copy(self, state, player: str, action: Action):
        s = copy.deepcopy(state)
        t, block, attack = action
        if t == "pass":
            s.apply_pass(player)
        elif t == "receive":
            s.apply_receive(player, block)
        elif t == "attack":
            s.apply_attack(player, attack)
        elif t == "attack_after_block":
            s.apply_attack_after_block(player, block, attack)
        return s

    def _win_now_bonus(self, state, player: str, action: Action) -> float:
        try:
            s = self._apply_action_on_copy(state, player, action)
        except Exception:
            return 0.0
        return self.WIN_NOW_BONUS if getattr(s, "finished", False) and getattr(s, "winner", None) == player else 0.0

    def _win_after_receive_bonus(self, state, player: str, action: Action) -> float:
        t, block, _ = action
        if t != "receive" or block is None:
            return 0.0
        try:
            s = self._apply_action_on_copy(state, player, action)
            next_actions = s.legal_actions(player)
        except Exception:
            return 0.0

        for (nt, nb, na) in next_actions:
            if nt not in ("attack", "attack_after_block"):
                continue
            try:
                s2 = self._apply_action_on_copy(s, player, (nt, nb, na))
                if getattr(s2, "finished", False) and getattr(s2, "winner", None) == player:
                    return self.WIN_AFTER_RECEIVE_BONUS
            except Exception:
                continue
        return 0.0

    def _guaranteed_finish_score_after_attack_action(
        self,
        state,
        player: str,
        action: Action,
    ) -> Optional[float]:
        action_type, block, attack = action
        if action_type not in ("attack", "attack_after_block") or attack is None:
            return None

        finish_score = self._finish_score_after_action(state, player, action)
        if finish_score is not None:
            return finish_score

        tr = self._track.get(id(state))
        if tr is None:
            return None

        kyosha_probe_score = self._kyosha_probe_expected_score_after_attack_action(state, player, action, tr)
        if kyosha_probe_score is not None:
            return kyosha_probe_score

        if not self._is_absolute_safe_for_tsume(state, player, attack, tr):
            return None

        temp_hand = list(state.hands[player])
        if block is not None and block in temp_hand:
            temp_hand.remove(block)
        if attack in temp_hand:
            temp_hand.remove(attack)
        if not temp_hand:
            return finish_score

        score = self._max_tsume_score(temp_hand, state, player, tr)
        return score if score >= 0 else None

    def _high_score_tsume_action(
        self,
        state,
        player: str,
        actions: List[Action],
        *,
        has_non_king_attack_option: bool,
    ) -> Optional[Tuple[Action, float, bool]]:
        candidates: List[Tuple[float, int, float, Action]] = []
        for action in actions:
            action_type, block, attack = action
            if action_type not in ("attack", "attack_after_block") or attack is None:
                continue

            route_score = self._guaranteed_finish_score_after_attack_action(state, player, action)
            if route_score is None:
                continue

            immediate = 1 if self._finish_score_after_action(state, player, action) is not None else 0
            heuristic = self._score_attack_phase(
                state,
                player,
                action_type,
                block,
                attack,
                has_non_king_attack_option=has_non_king_attack_option,
            )
            candidates.append((route_score, immediate, heuristic, action))

        if not candidates:
            return None

        candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        route_score, immediate, _heuristic, action = candidates[0]
        return action, route_score, bool(immediate)

    def _finish_score_after_action(self, state, player: str, action: Action) -> Optional[float]:
        team = "AC" if player in ("A", "C") else "BD"
        before_score = float(state.team_score.get(team, 0))
        try:
            s = self._apply_action_on_copy(state, player, action)
        except Exception:
            return None
        if not (getattr(s, "finished", False) and getattr(s, "winner", None) == player):
            return None
        return float(s.team_score.get(team, 0)) - before_score

    def _best_finish_score_after_receive(self, state, player: str, action: Action) -> Optional[float]:
        t, block, _ = action
        if t != "receive" or block is None:
            return None
        try:
            after_receive = self._apply_action_on_copy(state, player, action)
        except Exception:
            return None

        best_score: Optional[float] = None
        for next_action in after_receive.legal_actions(player):
            if next_action[0] not in ("attack", "attack_after_block"):
                continue
            score = self._finish_score_after_action(after_receive, player, next_action)
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_score = score
        return best_score

    def _ally_current_attack_is_publicly_unstoppable(self, state, player: str) -> bool:
        tr = self._track.get(id(state))
        attack = state.current_attack
        ally = state.attacker
        if (
            tr is None
            or state.phase != "receive"
            or attack is None
            or ally is None
            or ally == player
            or not self._same_team(ally, player)
        ):
            return False

        # 味方が残り2枚なら、この攻めが一周した時点で伏せて上がれる。
        if len(state.hands.get(ally, [])) != 2:
            return False

        total = self._piece_total(attack)
        seen_and_mine = int(tr.get("public_seen_counts", {}).get(attack, 0)) + state.hands[player].count(attack)
        if seen_and_mine < total:
            return False

        if attack in ("1", "2"):
            return True
        if attack in ("3", "4", "5", "6", "7"):
            known_kings = (
                int(tr.get("public_seen_counts", {}).get("8", 0))
                + int(tr.get("public_seen_counts", {}).get("9", 0))
                + state.hands[player].count("8")
                + state.hands[player].count("9")
            )
            return known_kings >= 2
        return False

    def _give_way_to_ally_guaranteed_win_action(
        self,
        state,
        player: str,
        actions: List[Action],
    ) -> Optional[Action]:
        pass_action = next((act for act in actions if act[0] == "pass"), None)
        if pass_action is None:
            return None
        if not self._ally_current_attack_is_publicly_unstoppable(state, player):
            return None

        best_self_score: Optional[float] = None
        for act in actions:
            if act[0] != "receive":
                continue
            score = self._best_finish_score_after_receive(state, player, act)
            if score is None:
                continue
            if best_self_score is None or score > best_self_score:
                best_self_score = score

        if best_self_score is None:
            return None
        if best_self_score <= self.ALLY_GUARANTEED_WIN_GIVE_WAY_MAX_SCORE:
            self._set_decision_reason("score_fallback")
            self._set_score_fallback_detail(f"pass_ally_guaranteed_win_self_score_{int(best_self_score)}")
            return pass_action
        return None

    def _preserve_current_attack_for_win_value(self, state, player: str) -> float:
        tr = self._track.get(id(state))
        attack = state.current_attack
        if tr is None or state.phase != "receive" or state.attacker is None:
            return 0.0
        if attack not in ("3", "4", "5", "6", "7"):
            return 0.0
        if self._same_team(state.attacker, player):
            return 0.0

        hand = list(state.hands[player])
        if hand.count(attack) != 1:
            return 0.0
        if not any(p in ("8", "9") for p in hand):
            return 0.0
        if len(hand) > 4:
            return 0.0

        def remaining_has_king_finish_route(remaining: List[str]) -> bool:
            if not any(p in ("8", "9") for p in remaining):
                return False
            if len(remaining) <= 2:
                return True
            return self._max_tsume_score(remaining, state, player, tr) >= 0

        if len(hand) == 2:
            remaining = list(hand)
            remaining.remove(attack)
            if remaining_has_king_finish_route(remaining):
                return self.PRESERVE_WIN_ATTACK_PASS_BONUS

        for fuse in set(hand):
            if fuse == attack:
                continue
            if fuse in ("8", "9"):
                continue
            future = list(hand)
            future.remove(fuse)
            future.remove(attack)
            if remaining_has_king_finish_route(future):
                return self.PRESERVE_WIN_ATTACK_PASS_BONUS

        return 0.0

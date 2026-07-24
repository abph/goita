"""終盤で勝ち切る手順と、その得点を評価します。
現在の盤面から詰みを探し、確定上がりの中でも得点が高い手順を優先します。
自分が低得点で上がるより、味方に上がらせた方がよい場面の判断も扱います。
"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from goita_ai2.constants import PIECE_TOTALS, POINTS

Action = Tuple[str, Optional[str], Optional[str]]


class ForcedWinStatus(str, Enum):
    """Result of a public-information forced-win proof."""

    PROVEN = "proven_win"
    COUNTEREXAMPLE = "counterexample"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ForcedWinResult:
    """Three-valued result with guaranteed and branch-weighted scores."""

    status: ForcedWinStatus
    minimum_score: Optional[float] = None
    expected_score: Optional[float] = None
    maximum_score: Optional[float] = None


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
            if any(piece in ("8", "9") for piece in hand):
                tr = self._track.get(id(state))
                shi_pressure = self._opponents_piece_pressure(tr, player, "1")
                score += self.ROYAL_WAIT_SHI_BASE_BONUS
                score += (
                    min(self.ROYAL_WAIT_SHI_PRESSURE_CAP, shi_pressure)
                    * self.ROYAL_WAIT_SHI_PRESSURE_WEIGHT
                )
            else:
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

    @staticmethod
    def _forced_win_proven(
        score: float,
        *,
        expected_score: Optional[float] = None,
        maximum_score: Optional[float] = None,
    ) -> ForcedWinResult:
        minimum = float(score)
        expected = minimum if expected_score is None else float(expected_score)
        maximum = expected if maximum_score is None else float(maximum_score)
        return ForcedWinResult(
            ForcedWinStatus.PROVEN,
            minimum,
            expected,
            maximum,
        )

    @staticmethod
    def _forced_win_counterexample() -> ForcedWinResult:
        return ForcedWinResult(ForcedWinStatus.COUNTEREXAMPLE)

    @staticmethod
    def _forced_win_unknown() -> ForcedWinResult:
        return ForcedWinResult(ForcedWinStatus.UNKNOWN)

    @staticmethod
    def _forced_win_preference_key(result: ForcedWinResult) -> Tuple[float, float, float]:
        minimum = float(result.minimum_score or 0.0)
        expected = (
            minimum
            if result.expected_score is None
            else float(result.expected_score)
        )
        maximum = (
            expected
            if result.maximum_score is None
            else float(result.maximum_score)
        )
        return minimum, expected, maximum

    def _public_unknown_piece_pool(
        self,
        state,
        player: str,
        tr: dict,
    ) -> Optional[Counter]:
        public_seen = tr.get("public_seen_counts")
        if not isinstance(public_seen, dict):
            return None

        pool = Counter()
        my_hand = Counter(state.hands[player])
        for piece, total in PIECE_TOTALS.items():
            remaining = (
                int(total)
                - int(public_seen.get(piece, 0))
                - int(my_hand.get(piece, 0))
            )
            if remaining < 0:
                return None
            pool[piece] = remaining
        return pool

    @staticmethod
    def _forced_win_pool_key(pool: Counter) -> Tuple[int, ...]:
        return tuple(int(pool.get(str(i), 0)) for i in range(1, 10))

    @staticmethod
    def _forced_win_attack_is_legal(
        hand: Tuple[str, ...],
        attack: str,
        *,
        block: Optional[str],
        king_used: bool,
        had_both_kings: bool,
    ) -> bool:
        if attack not in hand:
            return False
        if block is not None:
            if block not in hand:
                return False
            if block == attack and hand.count(block) < 2:
                return False
        if attack not in ("8", "9"):
            return True
        last_finish = len(hand) == (2 if block is not None else 1)
        return had_both_kings or king_used or last_finish

    @staticmethod
    def _forced_win_finish_score(block: Optional[str], attack: str) -> float:
        base = float(POINTS.get(attack, 0))
        if block is None:
            return base
        if {block, attack} == {"8", "9"}:
            return 100.0
        if block == attack:
            return base * 2.0
        return base

    @staticmethod
    def _forced_win_receive_options(
        hand: Tuple[str, ...],
        attack: str,
    ) -> Tuple[str, ...]:
        options = []
        for piece in sorted(set(hand)):
            if piece == attack or (piece in ("8", "9") and attack not in ("1", "2")):
                options.append(piece)
        return tuple(options)

    @staticmethod
    def _forced_win_external_receivers(
        pool: Counter,
        attack: str,
    ) -> Tuple[str, ...]:
        receivers = []
        if int(pool.get(attack, 0)) > 0:
            receivers.append(attack)
        if attack not in ("1", "2"):
            for royal in ("8", "9"):
                if int(pool.get(royal, 0)) > 0 and royal not in receivers:
                    receivers.append(royal)
        return tuple(receivers)

    def _forced_win_choose_attack(
        self,
        hand: Tuple[str, ...],
        pool: Counter,
        *,
        need_block: bool,
        king_used: bool,
        had_both_kings: bool,
        depth: int,
        external_cards_used: int,
        minimum_enemy_hand: int,
        memo: Dict[tuple, ForcedWinResult],
    ) -> ForcedWinResult:
        if depth <= 0:
            return self._forced_win_unknown()

        key = (
            "choose",
            tuple(sorted(hand)),
            self._forced_win_pool_key(pool),
            bool(need_block),
            bool(king_used),
            int(depth),
            int(external_cards_used),
            int(minimum_enemy_hand),
        )
        if key in memo:
            return memo[key]

        candidates: List[ForcedWinResult] = []
        blocks: Tuple[Optional[str], ...]
        blocks = tuple(sorted(set(hand))) if need_block else (None,)
        for block in blocks:
            for attack in sorted(set(hand)):
                if not self._forced_win_attack_is_legal(
                    hand,
                    attack,
                    block=block,
                    king_used=king_used,
                    had_both_kings=had_both_kings,
                ):
                    continue

                remaining = list(hand)
                if block is not None:
                    remaining.remove(block)
                remaining.remove(attack)
                if not remaining:
                    candidates.append(self._forced_win_proven(
                        self._forced_win_finish_score(block, attack)
                    ))
                    continue

                candidates.append(self._forced_win_resolve_attack(
                    tuple(sorted(remaining)),
                    pool,
                    attack=attack,
                    king_used=king_used,
                    had_both_kings=had_both_kings,
                    depth=depth - 1,
                    external_cards_used=external_cards_used,
                    minimum_enemy_hand=minimum_enemy_hand,
                    memo=memo,
                ))

        proven = [
            result
            for result in candidates
            if result.status == ForcedWinStatus.PROVEN and result.minimum_score is not None
        ]
        if proven:
            result = max(proven, key=self._forced_win_preference_key)
        elif any(result.status == ForcedWinStatus.UNKNOWN for result in candidates):
            result = self._forced_win_unknown()
        else:
            result = self._forced_win_counterexample()
        memo[key] = result
        return result

    def _forced_win_resolve_external_return(
        self,
        hand: Tuple[str, ...],
        pool: Counter,
        *,
        returned_attack: str,
        king_used: bool,
        had_both_kings: bool,
        depth: int,
        external_cards_used: int,
        minimum_enemy_hand: int,
        memo: Dict[tuple, ForcedWinResult],
    ) -> ForcedWinResult:
        receive_results: List[ForcedWinResult] = []
        for receive_piece in self._forced_win_receive_options(hand, returned_attack):
            remaining = list(hand)
            remaining.remove(receive_piece)
            if not remaining:
                continue
            receive_results.append(self._forced_win_choose_attack(
                tuple(sorted(remaining)),
                pool,
                need_block=False,
                king_used=king_used or receive_piece in ("8", "9"),
                had_both_kings=had_both_kings,
                depth=depth - 1,
                external_cards_used=external_cards_used,
                minimum_enemy_hand=minimum_enemy_hand,
                memo=memo,
            ))

        proven = [
            result
            for result in receive_results
            if result.status == ForcedWinStatus.PROVEN and result.minimum_score is not None
        ]
        if proven:
            return max(proven, key=self._forced_win_preference_key)
        if any(result.status == ForcedWinStatus.UNKNOWN for result in receive_results):
            return self._forced_win_unknown()
        return self._forced_win_counterexample()

    def _forced_win_resolve_attack(
        self,
        hand: Tuple[str, ...],
        pool: Counter,
        *,
        attack: str,
        king_used: bool,
        had_both_kings: bool,
        depth: int,
        external_cards_used: int,
        minimum_enemy_hand: int,
        memo: Dict[tuple, ForcedWinResult],
    ) -> ForcedWinResult:
        if depth <= 0:
            return self._forced_win_unknown()

        key = (
            "resolve",
            tuple(sorted(hand)),
            self._forced_win_pool_key(pool),
            attack,
            bool(king_used),
            int(depth),
            int(external_cards_used),
            int(minimum_enemy_hand),
        )
        if key in memo:
            return memo[key]

        # Passing around the table is always a legal enemy response.
        branches: List[Tuple[float, ForcedWinResult]] = [
            (1.0, self._forced_win_choose_attack(
                hand,
                pool,
                need_block=True,
                king_used=king_used,
                had_both_kings=had_both_kings,
                depth=depth - 1,
                external_cards_used=external_cards_used,
                minimum_enemy_hand=minimum_enemy_hand,
                memo=memo,
            ))
        ]

        for receiver in self._forced_win_external_receivers(pool, attack):
            receiver_weight = max(1.0, float(pool.get(receiver, 0)))
            after_receiver = pool.copy()
            after_receiver[receiver] -= 1
            next_external_used = external_cards_used + 2
            if minimum_enemy_hand > 0 and next_external_used >= minimum_enemy_hand:
                branches.append((receiver_weight, self._forced_win_counterexample()))
                continue

            returned_pieces = [
                piece
                for piece in (str(i) for i in range(1, 10))
                if int(after_receiver.get(piece, 0)) > 0
            ]
            if not returned_pieces:
                branches.append((receiver_weight, self._forced_win_unknown()))
                continue

            for returned_attack in returned_pieces:
                branch_weight = receiver_weight * max(
                    1.0,
                    float(after_receiver.get(returned_attack, 0)),
                )
                after_return = after_receiver.copy()
                after_return[returned_attack] -= 1
                branches.append((
                    branch_weight,
                    self._forced_win_resolve_external_return(
                        hand,
                        after_return,
                        returned_attack=returned_attack,
                        king_used=king_used,
                        had_both_kings=had_both_kings,
                        depth=depth - 1,
                        external_cards_used=next_external_used,
                        minimum_enemy_hand=minimum_enemy_hand,
                        memo=memo,
                    ),
                ))

        if any(
            result.status == ForcedWinStatus.COUNTEREXAMPLE
            for _weight, result in branches
        ):
            result = self._forced_win_counterexample()
        elif any(
            result.status == ForcedWinStatus.UNKNOWN
            for _weight, result in branches
        ):
            result = self._forced_win_unknown()
        else:
            proven_branches = [
                (weight, branch)
                for weight, branch in branches
                if branch.minimum_score is not None
            ]
            minimum_score = min(
                float(branch.minimum_score)
                for _weight, branch in proven_branches
            )
            total_weight = sum(weight for weight, _branch in proven_branches)
            expected_score = sum(
                weight
                * float(
                    branch.expected_score
                    if branch.expected_score is not None
                    else branch.minimum_score
                )
                for weight, branch in proven_branches
            ) / max(1.0, total_weight)
            maximum_score = max(
                float(
                    branch.maximum_score
                    if branch.maximum_score is not None
                    else branch.minimum_score
                )
                for _weight, branch in proven_branches
            )
            result = self._forced_win_proven(
                minimum_score,
                expected_score=expected_score,
                maximum_score=maximum_score,
            )
        memo[key] = result
        return result

    def _forced_win_result_after_attack_action(
        self,
        state,
        player: str,
        action: Action,
    ) -> ForcedWinResult:
        action_type, block, attack = action
        if action_type not in ("attack", "attack_after_block") or attack is None:
            return self._forced_win_counterexample()

        finish_score = self._finish_score_after_action(state, player, action)
        if finish_score is not None:
            return self._forced_win_proven(finish_score)

        tr = self._track.get(id(state))
        if tr is None:
            return self._forced_win_unknown()
        pool = self._public_unknown_piece_pool(state, player, tr)
        if pool is None:
            return self._forced_win_unknown()

        enemies = [
            seat
            for seat in ("A", "B", "C", "D")
            if not self._same_team(seat, player)
        ]
        minimum_enemy_hand = min((len(state.hands[seat]) for seat in enemies), default=0)
        if minimum_enemy_hand <= 2:
            return self._forced_win_unknown()

        remaining = self._remaining_hand_after_attack_action(state, player, block, attack)
        if remaining is None:
            return self._forced_win_counterexample()
        max_hand = int(getattr(self, "EXACT_FORCED_WIN_MAX_HAND", 6))
        allow_seven_card_receive_followup = (
            action_type == "attack"
            and block is None
            and len(state.hands[player]) == max_hand + 1
        )
        if len(remaining) > max_hand or (
            len(state.hands[player]) > max_hand
            and not allow_seven_card_receive_followup
        ):
            return self._forced_win_unknown()

        return self._forced_win_resolve_attack(
            tuple(sorted(remaining)),
            pool,
            attack=attack,
            king_used=int(getattr(state, "king_block_used", 0)) > 0,
            had_both_kings=bool(state.had_both_kings.get(player, False)),
            depth=int(getattr(self, "EXACT_FORCED_WIN_MAX_DEPTH", 18)),
            external_cards_used=0,
            minimum_enemy_hand=minimum_enemy_hand,
            memo={},
        )

    def _forced_win_result_after_receive_action(
        self,
        state,
        player: str,
        action: Action,
    ) -> ForcedWinResult:
        action_type, block, _attack = action
        if action_type != "receive" or block is None:
            return self._forced_win_counterexample()

        max_hand = int(getattr(self, "EXACT_FORCED_WIN_MAX_HAND", 6))
        if len(state.hands[player]) > max_hand:
            return self._forced_win_unknown()

        tr = self._track.get(id(state))
        if tr is None:
            return self._forced_win_unknown()
        try:
            after_receive = self._apply_action_on_copy(state, player, action)
        except Exception:
            return self._forced_win_counterexample()

        # The copied state has consumed our receive piece. Reflect that newly
        # public piece before evaluating the available follow-up attacks.
        after_tracker = copy.deepcopy(tr)
        after_public = after_tracker.setdefault(
            "public_seen_counts",
            {str(i): 0 for i in range(1, 10)},
        )
        after_public[block] = int(after_public.get(block, 0)) + 1
        after_id = id(after_receive)
        self._track[after_id] = after_tracker
        try:
            results = [
                self._forced_win_result_after_attack_action(
                    after_receive,
                    player,
                    next_action,
                )
                for next_action in after_receive.legal_actions(player)
                if next_action[0] in ("attack", "attack_after_block")
            ]
        finally:
            self._track.pop(after_id, None)

        proven = [
            result
            for result in results
            if result.status == ForcedWinStatus.PROVEN and result.minimum_score is not None
        ]
        if proven:
            return max(proven, key=self._forced_win_preference_key)
        if any(result.status == ForcedWinStatus.UNKNOWN for result in results):
            return self._forced_win_unknown()
        return self._forced_win_counterexample()

    def _guaranteed_finish_score_after_attack_action(
        self,
        state,
        player: str,
        action: Action,
    ) -> Optional[float]:
        result = self._forced_win_result_after_attack_action(state, player, action)
        if result.status != ForcedWinStatus.PROVEN:
            return None
        return result.minimum_score

    def _estimated_piece_hold_risk(
        self,
        tr: dict,
        player: str,
        piece: str,
    ) -> Optional[float]:
        estimate = self._estimated_current_piece(tr, player, piece)
        if estimate is None:
            return None
        if int(estimate.get("min", 0)) > 0:
            return 1.0
        if int(estimate.get("max", 0)) <= 0:
            return 0.0
        return max(0.0, min(1.0, float(estimate.get("expected", 0.0))))

    def _estimated_receive_risk_for_player(
        self,
        tr: dict,
        player: str,
        attack: str,
    ) -> Optional[float]:
        attack_risk = self._estimated_piece_hold_risk(tr, player, attack)
        if attack_risk is None:
            return None
        if attack in ("1", "2", "8", "9"):
            return attack_risk

        king_risks = [
            self._estimated_piece_hold_risk(tr, player, king)
            for king in ("8", "9")
        ]
        if any(risk is None for risk in king_risks):
            return None
        return min(1.0, attack_risk + sum(float(risk) for risk in king_risks))

    def _royal_bridge_finish_action(
        self,
        state,
        player: str,
        actions: List[Action],
    ) -> Optional[Tuple[Action, float]]:
        if (
            state.phase != "attack"
            or state.turn != player
            or len(state.hands[player]) != 4
        ):
            return None

        candidates: List[Tuple[float, float, Action]] = []
        for action in actions:
            action_type, block, attack = action
            if (
                action_type != "attack_after_block"
                or block is None
                or attack not in ("8", "9")
            ):
                continue

            remaining = self._remaining_hand_after_attack_action(
                state,
                player,
                block,
                attack,
            )
            if remaining is None or len(remaining) != 2:
                continue

            finish_score = self._pair_finish_score(remaining)
            if finish_score < 0:
                continue
            heuristic = self._score_attack_phase(
                state,
                player,
                action_type,
                block,
                attack,
                has_non_king_attack_option=True,
            )
            candidates.append((finish_score, heuristic, action))

        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        finish_score, _heuristic, action = candidates[0]
        return action, finish_score

    def _reach_avoidance_conditional_tsume_action(
        self,
        state,
        player: str,
        actions: List[Action],
        *,
        has_non_king_attack_option: bool,
    ) -> Optional[Tuple[Action, float, float]]:
        tr = self._track.get(id(state))
        if (
            tr is None
            or state.phase != "attack"
            or state.turn != player
            or len(state.hands[player]) != 4
        ):
            return None

        next_enemy = state.next_player(player)
        if self._same_team(next_enemy, player) or len(state.hands[next_enemy]) != 2:
            return None
        enemy_model = tr.get("public_hand_models", {}).get(next_enemy, {})
        if int(enemy_model.get("attack_count", 0)) < 3:
            return None

        attack_actions = [
            action
            for action in actions
            if action[0] == "attack_after_block"
            and action[1] is not None
            and action[2] is not None
            and action[2] not in ("8", "9")
        ]
        if len(attack_actions) < 2:
            return None

        # A guaranteed route always has priority over this probabilistic search.
        if any(
            self._guaranteed_finish_score_after_attack_action(state, player, action) is not None
            for action in attack_actions
        ):
            return None

        candidates: List[Tuple[float, float, Action]] = []
        attack_risks: dict[str, float] = {}
        for action in attack_actions:
            action_type, block, attack = action
            remaining = self._remaining_hand_after_attack_action(state, player, block, attack)
            if remaining is None or len(remaining) != 2:
                continue
            receive_risk = self._estimated_receive_risk_for_player(tr, next_enemy, attack)
            if receive_risk is None:
                continue
            heuristic = self._score_attack_phase(
                state,
                player,
                action_type,
                block,
                attack,
                has_non_king_attack_option=has_non_king_attack_option,
            )
            heuristic += self._score_receive_phase(state, player, "receive", block)
            candidates.append((receive_risk, heuristic, action))
            attack_risks[attack] = receive_risk

        if len(attack_risks) < 2:
            return None
        lowest_risk = min(attack_risks.values())
        highest_risk = max(attack_risks.values())
        risk_gap = highest_risk - lowest_risk
        if risk_gap < self.REACH_AVOIDANCE_CONDITIONAL_TSUME_MIN_RISK_GAP:
            return None

        candidates.sort(key=lambda item: (item[0], -item[1]))
        receive_risk, _heuristic, action = candidates[0]
        return action, receive_risk, risk_gap

    def _inferred_ally_shi_sashikomi_finish_action(
        self,
        state,
        player: str,
        actions: List[Action],
    ) -> Optional[Action]:
        tr = self._track.get(id(state))
        if (
            tr is None
            or state.phase != "attack"
            or state.turn != player
            or int(tr.get("my_attack_count", 0)) != 2
        ):
            return None

        next_enemy = state.next_player(player)
        ally = state.next_player(next_enemy)
        if self._same_team(next_enemy, player) or not self._same_team(ally, player):
            return None
        if len(state.hands[ally]) != 2:
            return None

        joint = tr.get("joint_hand_inference", {})
        if not bool(joint.get("feasible")):
            return None
        map_current = joint.get("map_current_counts", {})
        enemy_counts = map_current.get(next_enemy, {})
        ally_counts = map_current.get(ally, {})
        if int(enemy_counts.get("1", 0)) != 0 or int(ally_counts.get("1", 0)) < 1:
            return None
        if sum(int(value) for value in ally_counts.values()) != len(state.hands[ally]):
            return None

        return next(
            (
                action
                for action in actions
                if action[0] in ("attack", "attack_after_block") and action[2] == "1"
            ),
            None,
        )

    @staticmethod
    def _endgame_team(player: str) -> str:
        return "AC" if player in ("A", "C") else "BD"

    def _inferred_endgame_state(self, state, player: str, tr: dict):
        joint = tr.get("joint_hand_inference", {})
        if not bool(joint.get("feasible")):
            return None

        map_current = joint.get("map_current_counts", {})
        map_hidden = joint.get("map_hidden_counts", {})
        map_original = joint.get("map_original_counts", {})
        inferred = copy.deepcopy(state)
        for other in ("A", "B", "C", "D"):
            if other == player:
                inferred.hands[other] = sorted(state.hands[other])
                continue

            current_counts = map_current.get(other)
            hidden_counts = map_hidden.get(other)
            original_counts = map_original.get(other)
            if not isinstance(current_counts, dict) or not isinstance(hidden_counts, dict):
                return None
            if not isinstance(original_counts, dict):
                return None

            current_hand = [
                piece
                for piece in (str(i) for i in range(1, 10))
                for _ in range(int(current_counts.get(piece, 0)))
            ]
            hidden_hand = [
                piece
                for piece in (str(i) for i in range(1, 10))
                for _ in range(int(hidden_counts.get(piece, 0)))
            ]
            if len(current_hand) != len(state.hands[other]):
                return None
            if len(hidden_hand) != len(state.face_down_hidden.get(other, [])):
                return None

            inferred.hands[other] = current_hand
            inferred.face_down_hidden[other] = hidden_hand
            inferred.had_both_kings[other] = (
                int(original_counts.get("8", 0)) > 0
                and int(original_counts.get("9", 0)) > 0
            )

        # The hidden block itself is not public. A receive clears it, while
        # a completed lap replaces it before the next scoring action.
        inferred.last_block = None
        inferred.last_block_player = None
        return inferred

    def _inferred_endgame_state_key(self, state) -> tuple:
        return (
            tuple(
                (seat, tuple(sorted(state.hands[seat])))
                for seat in ("A", "B", "C", "D")
            ),
            state.phase,
            state.turn,
            state.current_attack,
            state.attacker,
            state.last_block,
            state.last_block_player,
            int(state.king_block_used),
            tuple(
                (seat, bool(state.had_both_kings.get(seat, False)))
                for seat in ("A", "B", "C", "D")
            ),
            int(state.team_score.get("AC", 0)),
            int(state.team_score.get("BD", 0)),
        )

    def _solve_inferred_endgame(
        self,
        state,
        root_player: str,
        baseline_scores: dict,
        depth: int,
        memo: dict,
    ):
        if state.finished and state.winner is not None:
            winner = state.winner
            winner_team = self._endgame_team(winner)
            score = int(state.team_score[winner_team]) - int(baseline_scores[winner_team])
            if self._same_team(winner, root_player):
                utility = 100_000 + score
            else:
                utility = -100_000 - score
            return utility, winner, score, ()
        if depth <= 0:
            return None

        key = (depth, self._inferred_endgame_state_key(state))
        if key in memo:
            return memo[key]

        turn = state.turn
        legal = state.legal_actions(turn)
        if not legal:
            memo[key] = None
            return None

        outcomes = []
        for action in legal:
            try:
                next_state = self._apply_action_on_copy(state, turn, action)
            except Exception:
                memo[key] = None
                return None
            outcome = self._solve_inferred_endgame(
                next_state,
                root_player,
                baseline_scores,
                depth - 1,
                memo,
            )
            if outcome is None:
                memo[key] = None
                return None
            utility, winner, score, path = outcome
            outcomes.append((utility, winner, score, (action,) + path))

        if self._same_team(turn, root_player):
            chosen = max(outcomes, key=lambda item: item[0])
        else:
            chosen = min(outcomes, key=lambda item: item[0])
        memo[key] = chosen
        return chosen

    def _inferred_endgame_team_result_action(
        self,
        state,
        player: str,
        actions: List[Action],
    ) -> Optional[Tuple[Action, str, int]]:
        tr = self._track.get(id(state))
        if (
            tr is None
            or state.phase != "receive"
            or state.turn != player
            or state.attacker is None
            or self._same_team(state.attacker, player)
            or len(state.hands[player]) > 4
            or len(actions) < 2
        ):
            return None

        attacker_model = tr.get("public_hand_models", {}).get(state.attacker, {})
        if int(attacker_model.get("attack_count", 0)) < 2:
            return None

        inferred = self._inferred_endgame_state(state, player, tr)
        if inferred is None or any(
            len(inferred.hands[seat]) > 4 for seat in ("A", "B", "C", "D")
        ):
            return None

        baseline_scores = {
            "AC": int(inferred.team_score.get("AC", 0)),
            "BD": int(inferred.team_score.get("BD", 0)),
        }
        memo = {}
        candidates = []
        for action in actions:
            try:
                next_state = self._apply_action_on_copy(inferred, player, action)
            except Exception:
                return None
            outcome = self._solve_inferred_endgame(
                next_state,
                player,
                baseline_scores,
                48,
                memo,
            )
            if outcome is None:
                return None
            utility, winner, score, path = outcome
            candidates.append((utility, action, winner, score, (action,) + path))

        if len({candidate[0] for candidate in candidates}) < 2:
            return None

        _utility, action, winner, score, path = max(
            candidates,
            key=lambda item: item[0],
        )
        tr["pending_inferred_endgame_attack"] = None
        if action[0] == "receive" and len(path) >= 2:
            planned_attack = path[1]
            if planned_attack[0] in ("attack", "attack_after_block"):
                tr["pending_inferred_endgame_attack"] = planned_attack
        return action, winner, score

    def _high_score_tsume_action(
        self,
        state,
        player: str,
        actions: List[Action],
        *,
        has_non_king_attack_option: bool,
    ) -> Optional[Tuple[Action, float, bool]]:
        tr = self._track.get(id(state))
        candidates: List[
            Tuple[float, float, float, int, int, float, Action]
        ] = []
        for action in actions:
            action_type, block, attack = action
            if action_type not in ("attack", "attack_after_block") or attack is None:
                continue

            result = self._forced_win_result_after_attack_action(
                state,
                player,
                action,
            )
            route_score = (
                float(result.minimum_score)
                if (
                    result.status == ForcedWinStatus.PROVEN
                    and result.minimum_score is not None
                )
                else None
            )
            expected_score = (
                route_score
                if result.expected_score is None
                else float(result.expected_score)
            )
            maximum_score = (
                expected_score
                if result.maximum_score is None
                else float(result.maximum_score)
            )
            if route_score is None and tr is not None:
                remaining = self._remaining_hand_after_attack_action(
                    state,
                    player,
                    block,
                    attack,
                )
                if (
                    remaining is not None
                    and len(remaining) == 2
                    and self._is_absolute_safe_for_tsume(
                        state,
                        player,
                        attack,
                        tr,
                    )
                ):
                    route_score = self._pair_finish_score(remaining)
                    expected_score = route_score
                    maximum_score = route_score
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
            candidates.append((
                route_score,
                expected_score,
                maximum_score,
                immediate,
                1 if attack not in ("8", "9") else 0,
                heuristic,
                action,
            ))

        if not candidates:
            return None

        candidates.sort(
            key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5]),
            reverse=True,
        )
        (
            route_score,
            expected_score,
            maximum_score,
            immediate,
            _keep_royal,
            _heuristic,
            action,
        ) = candidates[0]
        if tr is not None:
            tr["last_forced_win_score_plan"] = {
                "minimum_score": route_score,
                "expected_score": expected_score,
                "maximum_score": maximum_score,
                "attack": action[2],
            }
        # The heuristic already contains the normal attack evaluation. Keeping
        # the choice here prevents lower guaranteed-score routes from rejoining.
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

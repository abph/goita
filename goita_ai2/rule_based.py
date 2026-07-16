# goita_ai2/rule_based.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from collections import Counter
import copy
import csv
from pathlib import Path

from goita_ai2.constants import POINTS

Action = Tuple[str, Optional[str], Optional[str]]
TARGET_LAST1 = ("2", "3", "4", "5", "6", "7")


class RuleBasedAgent:
    def __init__(self, name: str = "RuleBased"):
        self.name = name
        self.me: Optional[str] = None

        self._track: Dict[int, dict] = {}
        self._my_initial_hands_by_state_id: Dict[int, List[str]] = {}
        self._relative_hand_rank_table: Optional[Dict[str, Dict[str, str]]] = None

        self.WIN_NOW_BONUS = 10_000.0
        self.WIN_AFTER_RECEIVE_BONUS = 9_000.0

        self.KING_ATTACK_PENALTY = 300.0

        self.FIRST_ENEMY_RECEIVE_BONUS = 500.0
        self.FIRST_ENEMY_PASS_BONUS = 500.0
        self.FIRST_ENEMY_KING_RECEIVE_PENALTY = 12000.0

        self.LAST_ONE_BONUS = 65.0

        self.KING_GYOKU_FORCE_ORDER = True
        self.FORCE_KING_GYOKU_ON_THIRD_ATTACK = True

        self.PREFER_PUBLIC_SAFE_NONKING_ON_THIRD_ATTACK = True

        self.PUBLIC_SAFE_ATTACK_BONUS_HIGH = 60.0
        self.PUBLIC_SAFE_ATTACK_BONUS_MID = 30.0
        self.PUBLIC_SAFE_ATTACK_BONUS_LOW = 10.0

        self.KAKARI_GOTAE_BONUS = 100.0
        self.ABSOLUTE_SAFE_BONUS = 1000.0
        self.TATEWARI_BONUS = 800.0
        self.CONTINUOUS_ATTACK_BONUS = 500.0
        self.ATTACK_STRATEGY_BONUS = 120.0
        self.RECEIVE_KEEP_PENALTY = 25.0
        self.ENEMY_FIRST_ATTACK_POLICY = "hand_strength"
        self.USE_RELATIVE_HAND_RANK = True
        self.USE_PUBLIC_HAND_INFERENCE = True
        self.USE_WEAK_SHI_ATTACK_STRATEGY = True
        self.USE_ENEMY_SHI_RESPONSE = False
        self.WEAK_SHI_ATTACK_BONUS = 120.0
        self.SHI_ATTACK_MODE_BONUS = 520.0
        self.DEALER_FOUR_SHI_BLOCK_SHI_BONUS = 220.0
        self.NON_WEAK_SHI_ATTACK_PENALTY = 220.0
        self.WEAK_SHI_FALLBACK_HIGH_POINT_WEIGHT = 2.0
        self.SHI_ATTACK_PREPARE_PASS_BONUS = 180.0
        self.ENEMY_SHI_PASS_BONUS = 250.0
        self.ENEMY_SHI_RECEIVE_PENALTY = 180.0
        self.DEALER_OPENING_PLAN_ATTACK_BONUS = 220.0
        self.DEALER_OPENING_PLAN_BLOCK_PENALTY = 700.0
        self.OPPONENT_FIRST_ATTACK_STRATEGY_SAFE_PENALTY = {
            "1": 40.0,
            "2": 80.0,
            "3": 55.0,
            "4": 55.0,
            "5": 55.0,
            "6": 70.0,
            "7": 70.0,
        }
        self.INFER_REPEAT_RECEIVE_PASS_BONUS = 14.0
        self.INFER_REPEAT_RECEIVE_PENALTY = 10.0
        self.INFER_ATTACK_EXHAUSTED_BONUS = 35.0
        self.INFER_ATTACK_OVERLAP_PENALTY = 8.0
        self.INFER_KAKARI_BLOCKED_PENALTY = 22.0
        self.INFER_KAKARI_CLEAR_BONUS = 25.0
        self.INFER_BLOCK_KEEP_BONUS = 14.0
        self.INFER_ALLY_STRATEGY_KEEP_BONUS = 18.0
        self.INFER_SHI_ATTACK_ALLY_BONUS = 25.0
        self.INFER_SHI_ATTACK_ENEMY_PENALTY = 14.0
        self.INFER_FORCE_KING_PRESSURE_BONUS = 18.0
        self.KAKARI_SATURATION_RECEIVE_BONUS = 280.0
        self.KAKARI_SATURATION_ATTACK_BONUS = 150.0
        self.KAKARI_SATURATION_ALLY_REMAINING_BONUS = 45.0
        self.ALLY_FORCE_KING_RECEIVE_BONUS = 720.0
        self.ALLY_FORCE_KING_ATTACK_BONUS = 950.0
        self.ALLY_STRONG_FOLLOWUP_RECEIVE_BONUS = 620.0
        self.ENDGAME_PAIR_SCORE_WEIGHT = 1.6
        self.ENDGAME_PAIR_KING_RECEIVE_BONUS = 18.0
        self.ENDGAME_PAIR_UNCERTAIN_PENALTY = 16.0
        self.ENDGAME_MIXED_SHI_PAIR_BONUS = 180.0
        self.ENDGAME_SHI_PAIR_PENALTY = 180.0
        self.SHI_SASHIKOMI_WAIT_BONUS = 180.0
        self.SHI_SASHIKOMI_ATTACK_BONUS = 520.0
        self.SHI_EXHAUST_RECEIVE_BONUS = 760.0
        self.SHI_EXHAUST_ATTACK_BONUS = 620.0
        self.WEAK_SHI_ENDGAME_MIXED_BLOCK_BONUS = 180.0
        self.PRESERVE_WIN_ATTACK_PASS_BONUS = 26000.0
        self.PRESERVE_WIN_ATTACK_RECEIVE_PENALTY = 12000.0
        self.FUSE_KYOSHA_BLOCK_PENALTY = 90.0
        self.FUSE_KING_BLOCK_PENALTY = 80.0
        self.FUSE_KEEP_LAST_SHI_PENALTY = 35.0
        self.FUSE_ENEMY_SHI_THREAT_BLOCK_PENALTY = 110.0
        self.FUSE_THIRD_BLOCK_KING_SHI_BONUS = 60.0
        self.FUSE_ATTACK_SATURATION_BLOCK_BONUS = 30.0
        self.FUSE_KEEP_KIN_GIN_RECEIVE_BONUS = 10.0
        self.LOWER_ATTACK_SHAPE_BLOCK_BONUS = 55.0
        self.LOWER_ATTACK_SHAPE_ATTACK_PENALTY = 70.0
        self.TOP_ATTACK_SHAPE_BLOCK_PENALTY = 35.0
        self.SAME_PIECE_PAIR_SPEND_PENALTY = 75.0
        self.SINGLE_MIDDLE_AFTER_BIG_RECEIVE_FIRST_ATTACK_PENALTY = 220.0
        self.FOURTH_MIDDLE_FIRST_ATTACK_DELAY_PENALTY = 950.0
        self.SINGLE_MIDDLE_OVER_FOUR_SHI_SIGNAL_PENALTY = 260.0
        self.ALLY_GUARANTEED_WIN_GIVE_WAY_MAX_SCORE = 30.0
        self.last_decision_reason = ""
        self.last_score_fallback_detail = ""

    def _set_decision_reason(self, reason: str) -> None:
        self.last_decision_reason = reason

    def _set_score_fallback_detail(self, detail: str) -> None:
        self.last_score_fallback_detail = detail

    def bind_player(self, player: str) -> None:
        if self.me is None:
            self.me = player
        elif self.me != player:
            raise ValueError(f"{self.name}: already bound to {self.me}, cannot bind to {player}")

    def _same_team(self, p1: str, p2: str) -> bool:
        return (
            (p1 in ("A", "C") and p2 in ("A", "C")) or
            (p1 in ("B", "D") and p2 in ("B", "D"))
        )

    def _ally_of(self, me: str) -> str:
        return "C" if me == "A" else "A" if me == "C" else "D" if me == "B" else "B"

    def _get_my_initial_hand(self, state) -> List[str]:
        if self.me is None:
            return []

        sid = id(state)
        if sid not in self._my_initial_hands_by_state_id:
            self._my_initial_hands_by_state_id[sid] = list(state.hands[self.me])
        return self._my_initial_hands_by_state_id[sid]

    def _ensure_trackers(self, state) -> None:
        sid = id(state)
        if sid in self._track:
            return
        if self.me is None:
            return

        init_hand = self._get_my_initial_hand(state)
        cnt_all = Counter(init_hand)
        ally_player = self._ally_of(self.me)

        public_seen_counts = {str(i): 0 for i in range(1, 10)}

        self._track[sid] = dict(
            my_init_count=cnt_all,
            ally=ally_player,
            public_seen_counts=public_seen_counts,

            my_attack_count=0,
            kg_plan_active=(("9" in init_hand) and ("8" in init_hand)),
            kg_second=None,
            
            my_past_attacks=set(),
            ally_past_attacks=set(),
            enemy_past_attacks=set(),
            enemy_attack_counts={},
            hidden_block_counts={p: 0 for p in ("A", "B", "C", "D")},
            other_first_attack_strategy_by_player={},
            other_piece_count_estimates={},
            
            ally_responded_to_my_attacks=set(),
            ally_ignored_my_attacks=set(),
            ally_pending_response_piece=None,
            ally_passed_my_shi_count=0,
            enemy_passed_my_shi_count=0,
            ally_shi_signal="unknown",
            shi_attack_mode=False,
            shi_attack_mode_source=None,
            i_passed_ally_shi=False,
            inherit_ally_shi_attack=False,
            my_open_shi_attack_pending=False,
            ally_open_shi_attack_pending=False,
            ally_shi_passed_by_enemy=False,
            ally_shi_sashikomi_candidate=False,
            ally_consumed_count=0,
            ally_passed_enemy_dealer_first_attack=False,
            ally_passed_enemy_dealer_first_attack_piece=None,
            ally_passed_enemy_first_attack=False,
            ally_passed_enemy_first_attack_attacker=None,
            ally_passed_enemy_first_attack_piece=None,
            pending_ally_force_king_attack_piece=None,
            my_last_receive_piece=None,
            enemy_pending_shi_receive_players=set(),
            enemy_team_rejected_shi_attack=False,
            public_hand_models={
                p: dict(
                    strength=0.0,
                    attack_count=0,
                    receive_count=0,
                    pass_count=0,
                    attacks=Counter(),
                    blocks=Counter(),
                    first_attack=None,
                    inferred_attack_strategy=None,
                )
                for p in ("A", "B", "C", "D")
            },
            
            perfect_plan=None,
            perfect_plan_step=0,
        )

    def _plan_perfect_game(self, hand: List[str]) -> Optional[List[Tuple[str, str]]]:
        if len(hand) != 8:
            return None

        counts = Counter(hand)
        has_kings = ("8" in hand and "9" in hand)

        safe_pieces = set()
        if counts.get("2", 0) == 4:
            safe_pieces.add("2")
        
        if has_kings:
            for p in ("3", "4", "5"):
                if counts.get(p, 0) == 4: safe_pieces.add(p)
            for p in ("6", "7"):
                if counts.get(p, 0) == 2: safe_pieces.add(p)
            safe_pieces.add("8")
            safe_pieces.add("9")

        if not safe_pieces:
            return None

        best_score = -1
        best_plan = None

        def search(current_hand: List[str], current_plan: List[Tuple[str, str]]):
            nonlocal best_score, best_plan
            
            if best_score == 100:
                return

            if not current_hand:
                fuse, atk = current_plan[-1]
                if fuse == atk:
                    score = POINTS.get(atk, 0) * 2
                elif set([fuse, atk]) == {"8", "9"}:
                    score = 100
                else:
                    score = POINTS.get(atk, 0)

                if score > best_score:
                    best_score = score
                    best_plan = list(current_plan)
                return

            turn = len(current_plan) + 1
            if turn < 4:
                available_safe = [p for p in safe_pieces if p in current_hand]
                if not available_safe:
                    return
                for atk in set(available_safe):
                    if best_score == 100: break
                    temp_hand = list(current_hand)
                    temp_hand.remove(atk)
                    for fuse in set(temp_hand):
                        if best_score == 100: break
                        next_hand = list(temp_hand)
                        next_hand.remove(fuse)
                        current_plan.append((fuse, atk))
                        search(next_hand, current_plan)
                        current_plan.pop()
            else:
                for atk in set(current_hand):
                    if best_score == 100: break
                    temp_hand = list(current_hand)
                    temp_hand.remove(atk)
                    fuse = temp_hand[0]
                    current_plan.append((fuse, atk))
                    search([], current_plan)
                    current_plan.pop()

        search(hand, [])
        return best_plan

    def _absolute_safe_pieces_for_hand(self, hand: List[str]) -> set[str]:
        counts = Counter(hand)
        has_kings = ("8" in hand and "9" in hand)

        safe_pieces = set()
        if counts.get("2", 0) == 4:
            safe_pieces.add("2")

        if has_kings:
            for p in ("3", "4", "5"):
                if counts.get(p, 0) == 4:
                    safe_pieces.add(p)
            for p in ("6", "7"):
                if counts.get(p, 0) == 2:
                    safe_pieces.add(p)
            safe_pieces.add("8")
            safe_pieces.add("9")

        return safe_pieces

    def _absolute_safe_pieces_from_initial_hand(self, hand: List[str]) -> set[str]:
        """
        初期手牌で占有していた駒は、途中で1枚使った後も相手には受けられない。
        王玉両持ちだった場合は、片方を受けで使った後も王玉攻めの権利が残る。
        """
        counts = Counter(hand)
        has_kings = ("8" in hand and "9" in hand)

        safe_pieces = set()
        if counts.get("2", 0) == 4:
            safe_pieces.add("2")

        if has_kings:
            for p in ("3", "4", "5"):
                if counts.get(p, 0) == 4:
                    safe_pieces.add(p)
            for p in ("6", "7"):
                if counts.get(p, 0) == 2:
                    safe_pieces.add(p)
            safe_pieces.add("8")
            safe_pieces.add("9")

        return safe_pieces

    def _can_receive_piece_from_initial_attack(self, block: str, attack: str) -> bool:
        if block in ("8", "9"):
            return attack not in ("1", "2")
        return block == attack

    def _possible_enemy_initial_attacks(self, hand: List[str]) -> List[str]:
        counts = Counter(hand)
        attacks: List[str] = []

        for p in ("1", "2", "3", "4", "5", "6", "7"):
            if counts.get(p, 0) < (10 if p == "1" else 4 if p in ("2", "3", "4", "5") else 2):
                attacks.append(p)

        # 王玉攻めは、敵親が王玉両持ちのときだけ初手から可能。
        if counts.get("8", 0) == 0 and counts.get("9", 0) == 0:
            attacks.extend(["8", "9"])

        return attacks

    def _plan_perfect_game_after_first_receive(self, hand: List[str]) -> Optional[List[Tuple[Optional[str], str]]]:
        """
        非親で、敵親が出し得るすべての初攻めに対して勝ち切れるルートを探す。
        戻り値は各初攻めに対する初回の (受け駒, 攻め駒)。
        """
        if len(hand) != 8:
            return None

        counts = Counter(hand)
        initial_safe_pieces = self._absolute_safe_pieces_from_initial_hand(hand)
        def finish_score(fuse: Optional[str], atk: str) -> int:
            if fuse == atk:
                return POINTS.get(atk, 0) * 2
            if fuse is not None and set([fuse, atk]) == {"8", "9"}:
                return 100
            return POINTS.get(atk, 0)

        def find_plan_after_attack(
            current_hand: List[str],
            current_plan: List[Tuple[Optional[str], str]],
            safe_pieces: set[str],
        ) -> Optional[List[Tuple[Optional[str], str]]]:
            nonlocal best_score, best_plan

            if best_score == 100:
                return best_plan

            if not current_hand:
                fuse, atk = current_plan[-1]
                score = finish_score(fuse, atk)
                if score > best_score:
                    best_score = score
                    best_plan = list(current_plan)
                return best_plan

            remaining_attacks = (len(current_hand) + 1) // 2
            need_safe_attack = remaining_attacks > 1

            for atk in set(current_hand):
                if best_score == 100:
                    break
                if need_safe_attack and atk not in safe_pieces:
                    continue

                temp_hand = list(current_hand)
                temp_hand.remove(atk)
                for fuse in set(temp_hand):
                    if best_score == 100:
                        break
                    next_hand = list(temp_hand)
                    next_hand.remove(fuse)
                    current_plan.append((fuse, atk))
                    find_plan_after_attack(next_hand, current_plan, safe_pieces)
                    current_plan.pop()

            return best_plan

        def can_force_lap_by_passing(enemy_attack: str) -> bool:
            if enemy_attack not in ("1", "2"):
                return False
            return counts.get(enemy_attack, 0) == self._piece_total(enemy_attack) - 1

        plans: List[Tuple[Optional[str], str]] = []
        for enemy_attack in self._possible_enemy_initial_attacks(hand):
            best_score = -1
            best_plan = None
            for block in set(hand):
                if not self._can_receive_piece_from_initial_attack(block, enemy_attack):
                    continue
                temp_hand = list(hand)
                temp_hand.remove(block)
                for atk in set(temp_hand):
                    if atk not in initial_safe_pieces:
                        continue
                    next_hand = list(temp_hand)
                    next_hand.remove(atk)
                    plan: List[Tuple[Optional[str], str]] = [(block, atk)]
                    find_plan_after_attack(next_hand, plan, initial_safe_pieces)

            if best_plan is None and can_force_lap_by_passing(enemy_attack):
                safe_after_pass = set(initial_safe_pieces)
                safe_after_pass.add(enemy_attack)
                pass_route_ok = True
                follow_attacks = [
                    atk for atk in self._possible_enemy_initial_attacks(hand)
                    if atk != enemy_attack
                ]
                for follow_attack in follow_attacks:
                    best_score = -1
                    best_plan = None
                    for block in set(hand):
                        if not self._can_receive_piece_from_initial_attack(block, follow_attack):
                            continue
                        temp_hand = list(hand)
                        temp_hand.remove(block)
                        for atk in set(temp_hand):
                            if atk not in safe_after_pass:
                                continue
                            next_hand = list(temp_hand)
                            next_hand.remove(atk)
                            plan = [(block, atk)]
                            find_plan_after_attack(next_hand, plan, safe_after_pass)
                    if best_plan is None:
                        pass_route_ok = False
                        break
                if pass_route_ok:
                    best_plan = [(None, enemy_attack)]

            if best_plan is None:
                return None
            plans.append(best_plan[0])

        return plans

    def _plan_any_win_after_first_receive(self, hand: List[str]) -> Optional[List[Tuple[Optional[str], str]]]:
        if len(hand) != 8:
            return None

        initial_safe_pieces = self._absolute_safe_pieces_from_initial_hand(hand)

        def finish_score(fuse: Optional[str], atk: str) -> int:
            if fuse == atk:
                return POINTS.get(atk, 0) * 2
            if fuse is not None and set([fuse, atk]) == {"8", "9"}:
                return 100
            return POINTS.get(atk, 0)

        def find_plan_after_attack(
            current_hand: List[str],
            current_plan: List[Tuple[Optional[str], str]],
        ) -> Optional[List[Tuple[Optional[str], str]]]:
            nonlocal best_score, best_plan

            if best_score == 100:
                return best_plan

            if not current_hand:
                fuse, atk = current_plan[-1]
                score = finish_score(fuse, atk)
                if score > best_score:
                    best_score = score
                    best_plan = list(current_plan)
                return best_plan

            remaining_attacks = (len(current_hand) + 1) // 2
            need_safe_attack = remaining_attacks > 1

            for atk in set(current_hand):
                if best_score == 100:
                    break
                if need_safe_attack and atk not in initial_safe_pieces:
                    continue

                temp_hand = list(current_hand)
                temp_hand.remove(atk)
                for fuse in set(temp_hand):
                    if best_score == 100:
                        break
                    next_hand = list(temp_hand)
                    next_hand.remove(fuse)
                    current_plan.append((fuse, atk))
                    find_plan_after_attack(next_hand, current_plan)
                    current_plan.pop()

            return best_plan

        for enemy_attack in self._possible_enemy_initial_attacks(hand):
            best_score = -1
            best_plan = None
            for block in set(hand):
                if not self._can_receive_piece_from_initial_attack(block, enemy_attack):
                    continue
                temp_hand = list(hand)
                temp_hand.remove(block)
                for atk in set(temp_hand):
                    if atk not in initial_safe_pieces:
                        continue
                    next_hand = list(temp_hand)
                    next_hand.remove(atk)
                    plan: List[Tuple[Optional[str], str]] = [(block, atk)]
                    find_plan_after_attack(next_hand, plan)

            if best_plan is not None:
                return best_plan

        return None

    def _forced_attack_plan_from_hand(
        self,
        hand: List[str],
        initial_safe_pieces: set[str],
    ) -> Optional[List[Tuple[Optional[str], str]]]:
        if not hand:
            return []

        plan: List[Tuple[Optional[str], str]] = []
        current_hand = list(hand)
        while len(current_hand) > 1:
            safe_attacks = sorted(set(p for p in current_hand if p in initial_safe_pieces))
            if not safe_attacks:
                return None
            atk = safe_attacks[0]
            current_hand.remove(atk)
            plan.append((None, atk))

        plan.append((None, current_hand[0]))
        return plan

    def _forced_hidden_attack_plan_from_even_hand(
        self,
        hand: List[str],
        initial_safe_pieces: set[str],
    ) -> Optional[List[Tuple[str, str]]]:
        if len(hand) % 2 != 0:
            return None
        if not hand:
            return []

        best_score = -1
        best_plan: Optional[List[Tuple[str, str]]] = None

        def finish_score(fuse: str, atk: str) -> int:
            if fuse == atk:
                return POINTS.get(atk, 0) * 2
            if set([fuse, atk]) == {"8", "9"}:
                return 100
            return POINTS.get(atk, 0)

        def search(current_hand: List[str], current_plan: List[Tuple[str, str]]) -> None:
            nonlocal best_score, best_plan

            if best_score == 100:
                return

            if not current_hand:
                fuse, atk = current_plan[-1]
                score = finish_score(fuse, atk)
                if score > best_score:
                    best_score = score
                    best_plan = list(current_plan)
                return

            need_safe_attack = len(current_hand) > 2
            for atk in set(current_hand):
                if need_safe_attack and atk not in initial_safe_pieces:
                    continue

                temp_hand = list(current_hand)
                temp_hand.remove(atk)
                for fuse in set(temp_hand):
                    next_hand = list(temp_hand)
                    next_hand.remove(fuse)
                    current_plan.append((fuse, atk))
                    search(next_hand, current_plan)
                    current_plan.pop()

        search(list(hand), [])
        return best_plan

    def _plan_win_after_two_receives(self, hand: List[str]) -> Optional[List[Tuple[Optional[str], str]]]:
        if len(hand) != 8:
            return None

        initial_safe_pieces = self._absolute_safe_pieces_from_initial_hand(hand)

        for first_enemy_attack in self._possible_enemy_initial_attacks(hand):
            for first_block in set(hand):
                if not self._can_receive_piece_from_initial_attack(first_block, first_enemy_attack):
                    continue

                after_first_block = list(hand)
                after_first_block.remove(first_block)

                for first_attack in set(after_first_block):
                    after_first_attack = list(after_first_block)
                    after_first_attack.remove(first_attack)

                    for second_enemy_attack in self._possible_enemy_initial_attacks(after_first_attack):
                        for second_block in set(after_first_attack):
                            if not self._can_receive_piece_from_initial_attack(second_block, second_enemy_attack):
                                continue

                            after_second_block = list(after_first_attack)
                            after_second_block.remove(second_block)
                            for second_attack in set(after_second_block):
                                if second_attack not in initial_safe_pieces:
                                    continue
                                after_second_attack = list(after_second_block)
                                after_second_attack.remove(second_attack)
                                forced_plan = self._forced_hidden_attack_plan_from_even_hand(
                                    after_second_attack,
                                    initial_safe_pieces,
                                )
                                if forced_plan is not None:
                                    return [(first_block, first_attack), (second_block, second_attack)] + forced_plan

        return None

    def _calculate_hand_power(self, state, player: str, tr: dict) -> float:
        hand = state.hands[player]
        if not hand:
            return 0.0

        score = 0.0
        counts = Counter(hand)
        
        max_base_point = 0.0
        for p in counts.keys():
            pt = 50.0 if p in ("8", "9") else float(POINTS.get(p, 0))
            if pt > max_base_point:
                max_base_point = pt
        score += max_base_point

        if "9" in counts or "8" in counts:
            score += 20.0
            
        for p, count in counts.items():
            if p in ("6", "7") and count == 2:
                score += 20.0
            elif p in ("2", "3", "4", "5"):
                if count >= 3:
                    score += 20.0
                elif count == 2:
                    score += 10.0

        shi_count = counts.get("1", 0)
        score -= shi_count * 5.0

        cards_played = 8 - len(hand)
        score += cards_played * 5.0

        unique_hiragoma = sum(1 for p in ("2", "3", "4", "5", "6", "7") if counts.get(p, 0) > 0)
        has_anchor = ("9" in counts) or ("8" in counts)
        if has_anchor:
            if unique_hiragoma >= 4:
                score += 20.0
            elif unique_hiragoma == 3:
                score += 10.0

        return score

    def _classify_hand_strength(self, hand: List[str], is_dealer: bool = False) -> Tuple[str, int, List[str]]:
        """
        初期手牌の強さを、人間が確認しやすいランク・点数・理由に分解する。
        ランクは敵の初攻めに対する大まかな方針判断に使う想定。
        """
        axes = self._classify_hand_axes(hand, is_dealer=is_dealer)
        return axes["rank"], int(axes["total_score"]), list(axes["reasons"])

    def _classify_hand_axes(self, hand: List[str], is_dealer: bool = False) -> Dict[str, object]:
        """
        初期手牌を「攻めの強さ」と「受けの強さ」の2軸で評価する。
        is_dealer=True の場合だけ、初手から勝ち切れる手を SS として扱う。
        """
        counts = Counter(hand)
        attack_reasons: List[str] = []
        receive_reasons: List[str] = []
        attack_score = 0
        receive_score = 0

        has_king = counts.get("8", 0) > 0
        has_ou = counts.get("9", 0) > 0
        has_both_kings = has_king and has_ou

        if has_both_kings:
            attack_score += 24
            attack_reasons.append("王玉両持ち")
        elif has_king or has_ou:
            attack_score += 8
            attack_reasons.append("王玉片方")
        else:
            attack_reasons.append("王玉なし")

        if has_king or has_ou:
            receive_score += 10
            receive_reasons.append("王玉あり")

        if counts.get("2", 0) > 0:
            receive_score += 10
            receive_reasons.append("香あり")

        has_hisha_pair = counts.get("7", 0) == 2
        has_kaku_pair = counts.get("6", 0) == 2
        if has_hisha_pair or has_kaku_pair:
            attack_score += 20
            if has_hisha_pair and has_kaku_pair:
                attack_reasons.append("飛/角2枚")
            elif has_hisha_pair:
                attack_reasons.append("飛2枚")
            else:
                attack_reasons.append("角2枚")

        c_kyosha = counts.get("2", 0)
        if c_kyosha == 4:
            attack_score += 35
            attack_reasons.append("香4枚")
        elif c_kyosha == 3:
            attack_score += 25
            attack_reasons.append("香3枚")
        elif c_kyosha == 2:
            attack_score += 15
            attack_reasons.append("香2枚")

        for p in ("5", "4", "3"):
            c = counts.get(p, 0)
            label = {"5": "金", "4": "銀", "3": "馬"}[p]
            if c == 4:
                attack_score += 26
                attack_reasons.append(f"{label}4枚")
            elif c == 3:
                attack_score += 16
                attack_reasons.append(f"{label}3枚")
            elif c == 2:
                attack_score += 8
                attack_reasons.append(f"{label}2枚")

        shi = counts.get("1", 0)
        if shi == 0:
            attack_reasons.append("し0枚")
        elif shi == 1:
            attack_reasons.append("し1枚")
        elif shi == 2:
            attack_reasons.append("し2枚")
        elif shi == 3:
            attack_reasons.append("3し")
        else:
            attack_reasons.append("4し")

        unique_hiragoma = sum(1 for p in ("2", "3", "4", "5", "6", "7") if counts.get(p, 0) > 0)
        if unique_hiragoma >= 5:
            receive_score += 20
            receive_reasons.append("受け幅かなり広い")
        elif unique_hiragoma == 4:
            receive_score += 10
            receive_reasons.append("受け幅広い")
        elif unique_hiragoma == 3:
            receive_score -= 10
            receive_reasons.append("受け幅やや狭い")
        elif unique_hiragoma == 2:
            receive_score -= 20
            receive_reasons.append("受け幅狭い")
        elif unique_hiragoma <= 1:
            receive_score -= 30
            receive_reasons.append("受け幅かなり狭い")

        attack_score = max(-20, min(100, attack_score))
        receive_score = max(-20, min(100, receive_score))
        total_score = int(round((attack_score * 0.58) + (receive_score * 0.42)))

        dealer_perfect_plan = self._plan_perfect_game(hand) if is_dealer else None
        has_shi_and_kyosha = counts.get("1", 0) > 0 and counts.get("2", 0) > 0
        non_dealer_perfect_plan = (
            self._plan_perfect_game_after_first_receive(hand)
            if not is_dealer and has_shi_and_kyosha
            else None
        )
        one_receive_win_plan = (
            self._plan_any_win_after_first_receive(hand)
            if not is_dealer and (has_king or has_ou)
            else None
        )
        two_receive_win_plan = (
            self._plan_win_after_two_receives(hand)
            if not is_dealer and (has_king or has_ou)
            else None
        )
        distinct_piece_kinds = sum(1 for n in counts.values() if n > 0)
        effective_receive_type = self._effective_receive_type(counts)
        attack_type_profile = self._classify_attack_type(counts)
        attack_type = int(attack_type_profile["type"])
        attack_type_label = str(attack_type_profile["label"])
        attack_type_value = {2: 5, 3: 4, 4: 4, 5: 3, 6: 2, 7: 1, 8: 0, 9: 1}.get(attack_type, 0)
        receive_type_value = {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}.get(effective_receive_type, 0)
        receive_type_value += (counts.get("8", 0) + counts.get("9", 0)) * 2
        type_total = attack_type_value + receive_type_value
        two_receive_win_plan = None
        if dealer_perfect_plan is not None:
            rank = "S"
            total_score = 100
            attack_reasons.append("勝ち確定")
        elif non_dealer_perfect_plan is not None:
            rank = "S"
            total_score = 100
            attack_reasons.append("敵親全初攻め対応")
        elif one_receive_win_plan is not None:
            rank = "S"
            total_score = max(total_score, 90)
            attack_reasons.append("王玉受け後勝ち確定")
        elif type_total >= 8:
            rank = "A"
            total_score = type_total
        elif type_total == 7:
            rank = "B"
            total_score = type_total
        elif type_total == 6:
            rank = "C"
            total_score = type_total
        elif type_total == 5:
            rank = "D"
            total_score = type_total
        elif type_total == 4:
            rank = "E"
            total_score = type_total
        elif type_total == 3:
            rank = "F"
            total_score = type_total
        else:
            rank = "X"
            total_score = type_total
        """
        elif two_receive_win_plan is not None:
            if distinct_piece_kinds >= 5:
                rank = "A"
                total_score = max(total_score, 80)
                attack_reasons.append("2回受け後勝ち確定/5種類以上")
            else:
                rank = "B"
                total_score = max(total_score, 70)
                attack_reasons.append("2回受け後勝ち確定/4種類以下")
        elif total_score >= 47 and attack_score >= 55 and receive_score >= 30 and has_both_kings:
            rank = "A"
        elif total_score >= 40:
            rank = "A"
        elif total_score >= 30:
            rank = "B"
        elif total_score >= 18:
            rank = "C"
        else:
            rank = "D"
        """

        reasons = (
            [f"攻め{int(attack_score)}", f"受け{int(receive_score)}"]
            + [f"攻:{r}" for r in attack_reasons]
            + [f"受:{r}" for r in receive_reasons]
        )
        return {
            "rank": rank,
            "total_score": total_score,
            "attack_score": int(attack_score),
            "receive_score": int(receive_score),
            "attack_type": attack_type,
            "attack_type_label": attack_type_label,
            "receive_type": effective_receive_type,
            "raw_receive_type": distinct_piece_kinds,
            "attack_type_value": attack_type_value,
            "receive_type_value": receive_type_value,
            "type_total": type_total,
            "reasons": reasons,
            "attack_reasons": attack_reasons,
            "receive_reasons": receive_reasons,
        }

    def _effective_receive_type(self, counts: Counter) -> int:
        kinds = 1 if counts.get("1", 0) > 0 else 0

        for p in ("2", "3", "4", "5", "6", "7"):
            if counts.get(p, 0) == 1:
                kinds += 1

        return min(kinds, 7)

    def _hand_strength_score(self, hand: List[str]) -> int:
        _rank, score, _reasons = self._classify_hand_strength(hand)
        return score

    def _classify_attack_type(self, counts: Counter) -> Dict[str, object]:
        if counts.get("2", 0) >= 3:
            return {"type": 2, "label": "three_kyosha", "pieces": ["2"]}

        middle3 = [p for p in ("5", "4", "3") if counts.get(p, 0) >= 3]
        if middle3:
            return {"type": 3, "label": "three_middle", "pieces": middle3}

        big_pairs = [p for p in ("7", "6") if counts.get(p, 0) >= 2]
        if big_pairs:
            return {"type": 4, "label": "big_pair", "pieces": big_pairs}

        if counts.get("2", 0) == 2:
            return {"type": 5, "label": "two_kyosha", "pieces": ["2"]}

        middle2 = [p for p in ("5", "4", "3") if counts.get(p, 0) >= 2]
        if middle2:
            return {"type": 6, "label": "two_middle", "pieces": middle2}

        if counts.get("1", 0) == 3:
            return {"type": 7, "label": "three_shi", "pieces": ["1"]}

        if counts.get("1", 0) >= 4:
            return {"type": 8, "label": "four_shi", "pieces": ["1"]}

        return {"type": 9, "label": "other", "pieces": []}

    def _attack_shape_profiles(self, counts: Counter) -> List[Dict[str, object]]:
        profiles: List[Dict[str, object]] = []

        if counts.get("2", 0) >= 3:
            profiles.append({"type": 2, "label": "three_kyosha", "value": 5, "pieces": ["2"]})
        elif counts.get("2", 0) == 2:
            profiles.append({"type": 5, "label": "two_kyosha", "value": 3, "pieces": ["2"]})

        for p in ("5", "4", "3"):
            if counts.get(p, 0) >= 3:
                profiles.append({"type": 3, "label": "three_middle", "value": 4, "pieces": [p]})
            elif counts.get(p, 0) >= 2:
                profiles.append({"type": 6, "label": "two_middle", "value": 2, "pieces": [p]})

        for p in ("7", "6"):
            if counts.get(p, 0) >= 2:
                profiles.append({"type": 4, "label": "big_pair", "value": 4, "pieces": [p]})

        if counts.get("1", 0) == 3:
            profiles.append({"type": 7, "label": "three_shi", "value": 1, "pieces": ["1"]})
        elif counts.get("1", 0) >= 4:
            profiles.append({"type": 8, "label": "four_shi", "value": 0, "pieces": ["1"]})

        return profiles

    def _attack_shape_profile_for_piece(self, counts: Counter, piece: str) -> Optional[Dict[str, object]]:
        profiles = [
            profile
            for profile in self._attack_shape_profiles(counts)
            if piece in profile["pieces"]
        ]
        if not profiles:
            return None
        return max(
            profiles,
            key=lambda profile: (
                int(profile["value"]),
                POINTS.get(piece, 0),
                -int(profile["type"]),
            ),
        )

    def _multi_attack_shape_plan_adjustment(
        self,
        state,
        player: str,
        action_type: str,
        block: Optional[str],
        attack: Optional[str],
    ) -> float:
        if attack is None:
            return 0.0

        hand = list(state.hands[player])
        profiles = self._attack_shape_profiles(Counter(hand))
        if len(profiles) < 2:
            return 0.0

        selected_profile = self._attack_shape_profile_for_piece(Counter(hand), attack)
        if selected_profile is None:
            return 0.0

        best_value = max(int(p["value"]) for p in profiles)
        selected_value = int(selected_profile["value"])
        selected_pieces = {attack}
        higher_pieces = {
            piece
            for profile in profiles
            if int(profile["value"]) > selected_value
            for piece in profile["pieces"]
        }
        lower_pieces = {
            piece
            for profile in profiles
            if int(profile["value"]) < selected_value
            for piece in profile["pieces"]
        }

        tr = self._track.get(id(state))
        is_kakari = (
            tr is not None
            and attack != "1"
            and (attack == tr.get("ally_first_attack") or attack in tr.get("ally_past_attacks", set()))
        )

        value = 0.0
        if action_type == "attack_after_block" and block in lower_pieces and attack in selected_pieces:
            value += self.LOWER_ATTACK_SHAPE_BLOCK_BONUS
        if action_type == "attack_after_block" and block in selected_pieces and lower_pieces:
            value -= self.TOP_ATTACK_SHAPE_BLOCK_PENALTY
        if action_type == "attack_after_block" and block in higher_pieces and selected_value < best_value:
            value -= self.TOP_ATTACK_SHAPE_BLOCK_PENALTY
        if selected_value < best_value and not is_kakari:
            value -= self.LOWER_ATTACK_SHAPE_ATTACK_PENALTY
        return value

    def _same_piece_pair_spend_penalty(
        self,
        state,
        player: str,
        action_type: str,
        block: Optional[str],
        attack: Optional[str],
    ) -> float:
        if (
            action_type != "attack_after_block"
            or block is None
            or attack is None
            or block != attack
            or attack in ("8", "9")
        ):
            return 0.0

        hand = list(state.hands[player])
        if len(hand) <= 2:
            return 0.0
        if hand.count(attack) != 2:
            return 0.0
        return self.SAME_PIECE_PAIR_SPEND_PENALTY

    def _single_middle_after_big_receive_first_attack_penalty(
        self,
        state,
        player: str,
        action_type: str,
        attack: Optional[str],
    ) -> float:
        tr = self._track.get(id(state))
        if (
            tr is None
            or action_type not in ("attack", "attack_after_block")
            or attack not in ("3", "4", "5")
            or int(tr.get("my_attack_count", 0)) != 0
            or tr.get("my_last_receive_piece") not in ("6", "7")
            or state.hands[player].count("1") < 4
            or state.hands[player].count(attack) != 1
        ):
            return 0.0
        return self.SINGLE_MIDDLE_AFTER_BIG_RECEIVE_FIRST_ATTACK_PENALTY

    def _initial_hand_axes_for_state(self, state, player: str) -> Dict[str, object]:
        tr = self._track.get(id(state))
        hand = list(tr.get("my_init_count", Counter()).elements()) if tr is not None else list(state.hands[player])
        axes = self._classify_hand_axes(hand, is_dealer=False)
        relative = self._relative_hand_info(hand)
        if relative is not None:
            axes["absolute_rank"] = axes["rank"]
            axes["relative_rank"] = relative["relative_rank"]
            axes["relative_score"] = float(relative["relative_score"])
            axes["other_avg_rank_score"] = float(relative["other_avg_rank_score"])
        else:
            axes["absolute_rank"] = axes["rank"]
            axes["relative_rank"] = axes["rank"]
            axes["relative_score"] = 0.0
            axes["other_avg_rank_score"] = 0.0
        return axes

    def _hand_digits(self, hand: List[str]) -> str:
        return "".join(sorted(hand, key=int))

    def _relative_hand_info(self, hand: List[str]) -> Optional[Dict[str, str]]:
        table = self._load_relative_hand_rank_table()
        return table.get(self._hand_digits(hand))

    def _strategy_rank_from_axes(self, axes: Dict[str, object]) -> str:
        if getattr(self, "USE_RELATIVE_HAND_RANK", True):
            return str(axes.get("relative_rank", axes.get("rank", "D")))
        return str(axes.get("rank", "D"))

    def _has_strong_repeat_attack(self, counts: Counter) -> bool:
        profile = self._classify_attack_type(counts)
        return int(profile["type"]) in (1, 2, 3, 4, 5, 6)

    def _is_dealer_opening_attack(self, state, player: str) -> bool:
        tr = self._track.get(id(state))
        return (
            tr is not None
            and state.phase == "attack"
            and state.turn == player
            and state.attacker is None
            and state.current_attack is None
            and int(tr.get("my_attack_count", 0)) == 0
        )

    def _dealer_opening_attack_plan_pieces(
        self,
        state,
        player: str,
        attack: Optional[str] = None,
    ) -> set[str]:
        tr = self._track.get(id(state))
        if tr is None:
            return set()

        if attack is not None:
            profile = self._attack_shape_profile_for_piece(tr["my_init_count"], attack)
            if profile is None:
                return set()
            return {attack}

        profile = self._classify_attack_type(tr["my_init_count"])
        attack_type = int(profile["type"])
        if attack_type in (1, 2, 3, 4, 5, 6, 7, 8):
            return set(str(p) for p in profile["pieces"])
        return set()

    def _dealer_opening_plan_adjustment(
        self,
        state,
        player: str,
        action_type: str,
        block: Optional[str],
        attack: Optional[str],
    ) -> float:
        if action_type != "attack_after_block" or attack is None:
            return 0.0
        if not self._is_dealer_opening_attack(state, player):
            return 0.0

        plan_pieces = self._dealer_opening_attack_plan_pieces(state, player, attack)
        if not plan_pieces:
            return 0.0

        value = 0.0
        if attack in plan_pieces:
            value += self.DEALER_OPENING_PLAN_ATTACK_BONUS
        if block in plan_pieces:
            if not (block == "1" and attack == "1" and state.hands[player].count("1") >= 4):
                value -= self.DEALER_OPENING_PLAN_BLOCK_PENALTY
            else:
                value += self.DEALER_FOUR_SHI_BLOCK_SHI_BONUS
        return value

    def _weak_shi_attack_plan_value(self, state, player: str) -> float:
        if not getattr(self, "USE_WEAK_SHI_ATTACK_STRATEGY", True):
            return 0.0

        tr = self._track.get(id(state))
        if tr is None:
            return 0.0

        counts = tr["my_init_count"]
        if counts.get("1", 0) < 3:
            return 0.0

        if tr.get("ally_shi_signal") == "weak" and not tr.get("inherit_ally_shi_attack"):
            return 0.0

        mode_active = bool(tr.get("shi_attack_mode")) and state.hands[player].count("1") > 0
        if mode_active and not self._has_strong_repeat_attack(counts):
            value = self.SHI_ATTACK_MODE_BONUS
            ally_signal = str(tr.get("ally_shi_signal", "unknown"))
            if ally_signal == "passed":
                value += 120.0
            elif ally_signal == "returned_shi":
                value += 260.0
            if tr.get("inherit_ally_shi_attack"):
                value += 280.0
            return value

        axes = self._initial_hand_axes_for_state(state, player)
        rank = self._strategy_rank_from_axes(axes)
        four_shi_signal_attack = (
            counts.get("1", 0) >= 4
            and state.hands[player].count("1") >= 3
            and rank not in ("S", "A", "B")
            and not self._has_strong_repeat_attack(counts)
        )
        if rank not in ("C", "D") and not tr.get("inherit_ally_shi_attack") and not four_shi_signal_attack:
            return 0.0

        if self._has_strong_repeat_attack(counts):
            return 0.0

        if int(axes.get("receive_type", 0)) < 5 and not four_shi_signal_attack:
            return 0.0

        if "1" in tr.get("enemy_past_attacks", set()) and not tr.get("inherit_ally_shi_attack"):
            return 0.0

        already_used_shi_attack = "1" in tr.get("my_past_attacks", set())
        ally_signal = str(tr.get("ally_shi_signal", "unknown"))
        public_seen = sum(int(v) for v in tr.get("public_seen_counts", {}).values())
        is_early = public_seen <= 4

        if already_used_shi_attack and ally_signal not in ("passed", "returned_shi"):
            return 0.0
        if not already_used_shi_attack and not is_early and not tr.get("inherit_ally_shi_attack") and not four_shi_signal_attack:
            return 0.0

        ally = tr.get("ally")
        if self._public_hand_strength(tr, ally) >= 10.0:
            return 0.0

        value = self.WEAK_SHI_ATTACK_BONUS
        if four_shi_signal_attack:
            value += 120.0
        if ally_signal == "passed":
            value += 100.0
        elif ally_signal == "returned_shi":
            value += 220.0
        if tr.get("inherit_ally_shi_attack"):
            value += 280.0
        return value

    def _single_middle_over_four_shi_signal_penalty(
        self,
        state,
        player: str,
        action_type: str,
        attack: Optional[str],
    ) -> float:
        tr = self._track.get(id(state))
        if (
            tr is None
            or action_type not in ("attack", "attack_after_block")
            or attack not in ("3", "4", "5")
            or int(tr.get("my_attack_count", 0)) != 0
            or state.hands[player].count(attack) != 1
            or state.hands[player].count("1") < 3
            or tr.get("my_init_count", Counter()).get("1", 0) < 4
            or self._weak_shi_attack_plan_value(state, player) <= 0
        ):
            return 0.0
        return self.SINGLE_MIDDLE_OVER_FOUR_SHI_SIGNAL_PENALTY

    def _shi_attack_score_adjustment(self, state, player: str) -> float:
        tr = self._track.get(id(state))
        if tr is None:
            return -self.NON_WEAK_SHI_ATTACK_PENALTY

        exhaust_bonus = self._shi_exhaust_attack_bonus(state, player)
        if exhaust_bonus > 0:
            return exhaust_bonus

        sashikomi_bonus = self._shi_sashikomi_attack_bonus(state, player)
        if sashikomi_bonus > 0:
            return sashikomi_bonus

        plan_value = self._weak_shi_attack_plan_value(state, player)
        if plan_value > 0:
            return plan_value

        counts = tr["my_init_count"]
        axes = self._initial_hand_axes_for_state(state, player)
        rank = self._strategy_rank_from_axes(axes)

        penalty = self.NON_WEAK_SHI_ATTACK_PENALTY
        if rank in ("S", "A"):
            penalty += 120.0
        if self._has_strong_repeat_attack(counts):
            penalty += 90.0
        if counts.get("1", 0) <= 2:
            penalty += 80.0
        return -penalty

    def _weak_shi_fallback_high_point_attack_bonus(
        self,
        state,
        player: str,
        action_type: str,
        attack: Optional[str],
    ) -> float:
        if action_type not in ("attack", "attack_after_block") or attack is None:
            return 0.0
        if attack in ("1", "8", "9"):
            return 0.0

        tr = self._track.get(id(state))
        if tr is None:
            return 0.0
        if self._weak_shi_attack_plan_value(state, player) > 0:
            return 0.0

        initial_shi = tr["my_init_count"].get("1", 0)
        current_shi = state.hands[player].count("1")
        if initial_shi > 2 and current_shi > 2:
            return 0.0

        return max(0.0, float(POINTS.get(attack, 0) - POINTS.get("2", 20))) * self.WEAK_SHI_FALLBACK_HIGH_POINT_WEIGHT

    def _shi_attack_hidden_block_adjustment(
        self,
        state,
        player: str,
        action_type: str,
        block: Optional[str],
        attack: Optional[str],
    ) -> float:
        if action_type != "attack_after_block" or block is None:
            return 0.0
        if self._weak_shi_attack_plan_value(state, player) <= 0:
            return 0.0

        hand = list(state.hands[player])
        consumed = 0
        if block == "1":
            consumed += 1
        if attack == "1":
            consumed += 1

        remaining_len = max(0, len(hand) - 2)
        future_attack_slots = (remaining_len + 1) // 2
        remaining_shi = hand.count("1") - consumed
        shi_is_surplus = remaining_shi > future_attack_slots

        if block == "1":
            if (
                attack == "1"
                and self._is_dealer_opening_attack(state, player)
                and hand.count("1") >= 4
            ):
                return self.DEALER_FOUR_SHI_BLOCK_SHI_BONUS
            if remaining_len == 2 and remaining_shi == 1:
                return self.WEAK_SHI_ENDGAME_MIXED_BLOCK_BONUS
            return 60.0 if shi_is_surplus else -700.0
        if block in ("2", "8", "9"):
            return -260.0
        return 120.0

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

    def _load_relative_hand_rank_table(self) -> Dict[str, Dict[str, str]]:
        if self._relative_hand_rank_table is not None:
            return self._relative_hand_rank_table

        path = Path(__file__).resolve().parents[1] / "results" / "relative_hand_all_4745_500_combined.csv"
        table: Dict[str, Dict[str, str]] = {}
        if path.exists():
            with path.open(encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    table[row["hand_digits"]] = row
        self._relative_hand_rank_table = table
        return table

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

    def _last_one_remaining_bonus(self, state, player: str, attack: Optional[str]) -> float:
        if attack is None or attack not in TARGET_LAST1:
            return 0.0
        tr = self._track.get(id(state))
        if tr is None:
            return 0.0
        if attack not in state.hands[player]:
            return 0.0

        total = 4 if attack in ("2", "3", "4", "5") else 2
        seen = tr["public_seen_counts"].get(attack, 0)
        return self.LAST_ONE_BONUS if seen == total - 1 else 0.0

    def _piece_total(self, p: str) -> int:
        if p == "1":
            return 10
        if p in ("2", "3", "4", "5"):
            return 4
        if p in ("6", "7"):
            return 2
        return 1

    def _is_fourth_middle_attack(self, state, player: str, attack: Optional[str]) -> bool:
        if attack not in ("3", "4", "5"):
            return False
        tr = self._track.get(id(state))
        if tr is None:
            return False
        seen_and_mine = int(tr.get("public_seen_counts", {}).get(attack, 0)) + state.hands[player].count(attack)
        return seen_and_mine >= self._piece_total(attack)

    def _fourth_middle_first_attack_delay_penalty(
        self,
        state,
        player: str,
        action_type: str,
        attack: Optional[str],
    ) -> float:
        tr = self._track.get(id(state))
        if (
            tr is None
            or action_type not in ("attack", "attack_after_block")
            or int(tr.get("my_attack_count", 0)) != 0
            or not self._is_fourth_middle_attack(state, player, attack)
        ):
            return 0.0
        return self.FOURTH_MIDDLE_FIRST_ATTACK_DELAY_PENALTY

    def _infer_first_attack_strategy(self, attack: Optional[str]) -> Optional[Dict[str, object]]:
        if attack is None:
            return None
        if attack in ("8", "9"):
            return {
                "label": "king_pair_or_king_attack",
                "attack_types": [1],
                "pieces": ["8", "9"],
                "strength": 7.0,
            }
        if attack == "2":
            return {
                "label": "kyosha_repeat",
                "attack_types": [2, 5],
                "pieces": ["2"],
                "strength": 5.0,
            }
        if attack in ("5", "4", "3"):
            return {
                "label": "middle_repeat",
                "attack_types": [3, 6],
                "pieces": [attack],
                "strength": 3.5,
            }
        if attack in ("7", "6"):
            return {
                "label": "big_pair",
                "attack_types": [4],
                "pieces": [attack],
                "strength": 4.5,
            }
        if attack == "1":
            return {
                "label": "shi_attack",
                "attack_types": [7, 8],
                "pieces": ["1"],
                "strength": 1.5,
            }
        return None

    def _first_attack_count_range(self, attack: Optional[str]) -> Optional[Tuple[int, int]]:
        if attack in ("2", "3", "4", "5"):
            return (2, 3)
        if attack in ("6", "7"):
            return (2, 2)
        if attack == "1":
            return (3, 4)
        if attack in ("8", "9"):
            return (1, 1)
        return None

    def _ensure_piece_count_estimate(self, tr: dict, player: str, piece: str) -> Dict[str, object]:
        estimates = tr.setdefault("other_piece_count_estimates", {})
        by_piece = estimates.setdefault(player, {})
        return by_piece.setdefault(
            piece,
            {
                "min": 0,
                "max": self._piece_total(piece),
                "source": "observed",
            },
        )

    def _set_piece_count_estimate(
        self,
        tr: dict,
        player: str,
        piece: str,
        *,
        min_count: int,
        max_count: int,
        source: str,
    ) -> None:
        est = self._ensure_piece_count_estimate(tr, player, piece)
        est["min"] = max(int(est.get("min", 0)), int(min_count))
        est["max"] = min(int(est.get("max", self._piece_total(piece))), int(max_count))
        if int(est["max"]) < int(est["min"]):
            est["max"] = est["min"]
        if source != "observed" or str(est.get("source", "observed")) == "observed":
            est["source"] = source

    def _update_observed_piece_count_estimate(self, tr: dict, player: str, piece: str) -> None:
        if player == self.me:
            return
        models = tr.get("public_hand_models", {})
        model = models.get(player)
        if model is None:
            return
        observed_count = int(model.get("attacks", Counter()).get(piece, 0)) + int(
            model.get("blocks", Counter()).get(piece, 0)
        )
        if observed_count <= 0:
            return
        self._set_piece_count_estimate(
            tr,
            player,
            piece,
            min_count=observed_count,
            max_count=self._piece_total(piece),
            source="observed",
        )

    def _reconcile_piece_count_estimates(self, state, tr: dict, piece: str) -> None:
        if self.me is None:
            return
        estimates = tr.get("other_piece_count_estimates", {})
        mine = state.hands[self.me].count(piece)
        other_players = [p for p in ("A", "B", "C", "D") if p != self.me]

        for p in other_players:
            self._update_observed_piece_count_estimate(tr, p, piece)

        available_for_others = max(0, self._piece_total(piece) - mine)
        while True:
            current_min_sum = sum(
                int(estimates.get(p, {}).get(piece, {}).get("min", 0))
                for p in other_players
            )
            if current_min_sum <= available_for_others:
                break

            relaxed = False
            for p in other_players:
                est = estimates.get(p, {}).get(piece)
                if est is None or str(est.get("source", "observed")) == "observed":
                    continue
                model = tr.get("public_hand_models", {}).get(p, {})
                observed_count = int(model.get("attacks", Counter()).get(piece, 0)) + int(
                    model.get("blocks", Counter()).get(piece, 0)
                )
                if int(est.get("min", 0)) > observed_count:
                    est["min"] = max(observed_count, int(est.get("min", 0)) - 1)
                    relaxed = True
                    break
            if not relaxed:
                break

        lower_bounds = {
            p: int(estimates.get(p, {}).get(piece, {}).get("min", 0))
            for p in other_players
        }
        total = self._piece_total(piece)
        for p in other_players:
            est = estimates.get(p, {}).get(piece)
            if est is None:
                continue
            others_min = sum(v for q, v in lower_bounds.items() if q != p)
            cap = max(lower_bounds[p], total - mine - others_min)
            est["max"] = min(int(est.get("max", total)), cap)
            if int(est["max"]) < int(est["min"]):
                est["max"] = est["min"]

    def _record_first_attack_count_estimate(
        self,
        state,
        tr: dict,
        player: str,
        attack: Optional[str],
        strategy: Optional[Dict[str, object]],
    ) -> None:
        if player == self.me or attack is None:
            return
        count_range = self._first_attack_count_range(attack)
        if count_range is None:
            return
        min_count, max_count = count_range
        label = str(strategy.get("label", "first_attack")) if strategy is not None else "first_attack"
        self._set_piece_count_estimate(
            tr,
            player,
            attack,
            min_count=min_count,
            max_count=max_count,
            source=label,
        )
        if attack in ("8", "9") and strategy is not None:
            other_king = "8" if attack == "9" else "9"
            self._set_piece_count_estimate(
                tr,
                player,
                other_king,
                min_count=1,
                max_count=1,
                source=label,
            )
            self._reconcile_piece_count_estimates(state, tr, other_king)
        self._reconcile_piece_count_estimates(state, tr, attack)

    def _infer_single_remaining_unique_piece_holder(
        self,
        state,
        tr: dict,
        piece: str,
        *,
        source: str,
    ) -> None:
        if self.me is None or self._piece_total(piece) != 1:
            return
        if state.hands[self.me].count(piece) > 0:
            return
        if int(tr.get("public_seen_counts", {}).get(piece, 0)) > 0:
            return

        candidates: List[str] = []
        for other in ("A", "B", "C", "D"):
            if other == self.me:
                continue
            observed = self._observed_piece_count_for_player(tr, other, piece)
            if observed > 0:
                return
            est = self._piece_count_estimate(tr, other, piece)
            max_count = self._piece_total(piece) if est is None else int(est.get("max", self._piece_total(piece)))
            if max_count > 0:
                candidates.append(other)

        if len(candidates) != 1:
            return

        self._set_piece_count_estimate(
            tr,
            candidates[0],
            piece,
            min_count=1,
            max_count=1,
            source=source,
        )
        self._reconcile_piece_count_estimates(state, tr, piece)

    def _infer_missing_kings_from_third_visible_attack(
        self,
        state,
        tr: dict,
        player: str,
        action_type: str,
        attack: Optional[str],
    ) -> None:
        if player == self.me or action_type != "attack" or attack in ("8", "9"):
            return

        model = tr.get("public_hand_models", {}).get(player, {})
        if int(model.get("attack_count", 0)) != 3:
            return

        for king in ("8", "9"):
            if self._observed_piece_count_for_player(tr, player, king) > 0:
                continue
            self._set_piece_count_estimate(
                tr,
                player,
                king,
                min_count=0,
                max_count=0,
                source="third_attack_no_king",
            )
            self._reconcile_piece_count_estimates(state, tr, king)
            self._infer_single_remaining_unique_piece_holder(
                state,
                tr,
                king,
                source="single_remaining_after_third_attack",
            )

    def _opponent_first_attack_strategy_penalty(self, tr: dict, player: str, attack: str) -> float:
        penalty = 0.0
        guesses = tr.get("other_first_attack_strategy_by_player", {})
        for other, guess in guesses.items():
            if self._same_team(other, player):
                continue
            if attack in set(str(p) for p in guess.get("pieces", [])):
                estimates = tr.get("other_piece_count_estimates", {})
                est = estimates.get(other, {}).get(attack)
                if est is not None:
                    model = tr.get("public_hand_models", {}).get(other, {})
                    observed_count = int(model.get("attacks", Counter()).get(attack, 0)) + int(
                        model.get("blocks", Counter()).get(attack, 0)
                    )
                    if int(est.get("max", 0)) <= observed_count:
                        continue
                penalty += float(self.OPPONENT_FIRST_ATTACK_STRATEGY_SAFE_PENALTY.get(attack, 0.0))
        return penalty

    def _observed_piece_count_for_player(self, tr: dict, player: str, piece: str) -> int:
        model = tr.get("public_hand_models", {}).get(player, {})
        return int(model.get("attacks", Counter()).get(piece, 0)) + int(
            model.get("blocks", Counter()).get(piece, 0)
        )

    def _piece_count_estimate(self, tr: Optional[dict], player: Optional[str], piece: Optional[str]) -> Optional[Dict[str, object]]:
        if tr is None or player is None or piece is None:
            return None
        return tr.get("other_piece_count_estimates", {}).get(player, {}).get(piece)

    def _estimate_remaining_range(self, tr: Optional[dict], player: Optional[str], piece: Optional[str]) -> Tuple[int, int]:
        est = self._piece_count_estimate(tr, player, piece)
        if tr is None or player is None or piece is None:
            return (0, self._piece_total(piece or "1"))
        observed = self._observed_piece_count_for_player(tr, player, piece)
        if est is None:
            return (0, max(0, self._piece_total(piece) - observed))
        min_count = int(est.get("min", 0))
        max_count = int(est.get("max", self._piece_total(piece)))
        return (max(0, min_count - observed), max(0, max_count - observed))

    def _opponents_piece_pressure(self, tr: Optional[dict], player: str, piece: Optional[str]) -> float:
        if tr is None or piece is None:
            return 0.0
        pressure = 0.0
        for other in ("A", "B", "C", "D"):
            if other == player or self._same_team(other, player):
                continue
            remaining_min, remaining_max = self._estimate_remaining_range(tr, other, piece)
            if remaining_min > 0:
                pressure += 1.0 + remaining_min
            elif remaining_max > 0:
                pressure += 0.5
        return pressure

    def _opponents_piece_exhausted(self, tr: Optional[dict], player: str, piece: Optional[str]) -> bool:
        if tr is None or piece is None:
            return False
        saw_estimate = False
        for other in ("A", "B", "C", "D"):
            if other == player or self._same_team(other, player):
                continue
            est = self._piece_count_estimate(tr, other, piece)
            if est is not None:
                saw_estimate = True
            _remaining_min, remaining_max = self._estimate_remaining_range(tr, other, piece)
            if remaining_max > 0:
                return False
        return saw_estimate

    def _ally_strategy_piece_value(self, tr: Optional[dict], player: str, piece: Optional[str]) -> float:
        if tr is None or piece is None:
            return 0.0
        ally = self._ally_of(player)
        guess = tr.get("other_first_attack_strategy_by_player", {}).get(ally)
        if guess is None:
            return 0.0
        if piece not in set(str(p) for p in guess.get("pieces", [])):
            return 0.0
        remaining_min, remaining_max = self._estimate_remaining_range(tr, ally, piece)
        if remaining_min > 0:
            return 2.0
        if remaining_max > 0:
            return 1.0
        return 0.0

    def _piece_count_attack_adjustment(self, state, player: str, attack: str) -> float:
        tr = self._track.get(id(state))
        if tr is None:
            return 0.0

        value = 0.0
        if self._opponents_piece_exhausted(tr, player, attack):
            value += self.INFER_ATTACK_EXHAUSTED_BONUS

        pressure = self._opponents_piece_pressure(tr, player, attack)
        if pressure > 0:
            value -= min(160.0, self.INFER_ATTACK_OVERLAP_PENALTY * pressure)

        ally_value = self._ally_strategy_piece_value(tr, player, attack)
        if ally_value > 0:
            value += self.INFER_KAKARI_CLEAR_BONUS * ally_value

        if attack == "1":
            ally_shi = self._ally_strategy_piece_value(tr, player, "1")
            enemy_shi = self._opponents_piece_pressure(tr, player, "1")
            value += self.INFER_SHI_ATTACK_ALLY_BONUS * ally_shi
            value -= min(180.0, self.INFER_SHI_ATTACK_ENEMY_PENALTY * enemy_shi)

        if attack not in ("1", "2", "8", "9"):
            enemy_pressure = self._opponents_piece_pressure(tr, player, attack)
            visible_kings = (
                tr["public_seen_counts"].get("8", 0)
                + tr["public_seen_counts"].get("9", 0)
                + state.hands[player].count("8")
                + state.hands[player].count("9")
            )
            if enemy_pressure == 0 and visible_kings < 2:
                value += self.INFER_FORCE_KING_PRESSURE_BONUS

        return value

    def _piece_count_hidden_block_adjustment(self, state, player: str, block: Optional[str]) -> float:
        if block is None:
            return 0.0
        tr = self._track.get(id(state))
        if tr is None:
            return 0.0
        value = 0.0
        if self._opponents_piece_pressure(tr, player, block) > 0:
            value -= self.INFER_BLOCK_KEEP_BONUS
        ally_value = self._ally_strategy_piece_value(tr, player, block)
        if ally_value > 0:
            value -= self.INFER_ALLY_STRATEGY_KEEP_BONUS * ally_value
        if self._opponents_piece_exhausted(tr, player, block):
            value += self.INFER_BLOCK_KEEP_BONUS
        return value

    def _enemy_shi_attack_threat_for_fuse(self, tr: Optional[dict], player: str) -> bool:
        if tr is None:
            return False
        enemies = [p for p in ("A", "B", "C", "D") if p != player and not self._same_team(p, player)]
        models = tr.get("public_hand_models", {})
        enemy_shi_attacks = 0
        enemy_shi_attackers = 0
        for enemy in enemies:
            model = models.get(enemy, {})
            n = int(model.get("attacks", Counter()).get("1", 0))
            if n > 0:
                enemy_shi_attackers += 1
                enemy_shi_attacks += n

        if enemy_shi_attackers >= 2 or enemy_shi_attacks >= 2:
            return True

        if enemy_shi_attacks == 1 and tr.get("enemy_team_rejected_shi_attack"):
            return False

        ally = self._ally_of(player)
        ally_pass_count = int(models.get(ally, {}).get("pass_count", 0))
        return enemy_shi_attacks > 0 and ally_pass_count > 0

    def _next_enemy_likely_shi_exhausted(self, state, player: str) -> bool:
        tr = self._track.get(id(state))
        if tr is None:
            return False
        if "1" not in tr.get("my_past_attacks", set()):
            return False
        if state.hands[player].count("1") <= 0:
            return False

        next_enemy = state.next_player(player)
        if self._same_team(next_enemy, player):
            return False

        models = tr.get("public_hand_models", {})
        model = models.get(next_enemy, {})
        attacks = model.get("attacks", Counter())
        blocks = model.get("blocks", Counter())
        visible_shi_spent = int(attacks.get("1", 0)) + int(blocks.get("1", 0))
        hidden_blocks = int(tr.get("hidden_block_counts", {}).get(next_enemy, 0))

        if visible_shi_spent <= 0 or hidden_blocks < 2:
            return False

        first_attack = model.get("first_attack")
        if first_attack == "1" and int(attacks.get("1", 0)) >= 1:
            return False

        return visible_shi_spent + min(hidden_blocks, 2) >= 3

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

    def _shi_exhaust_attack_bonus(self, state, player: str) -> float:
        if not self._next_enemy_likely_shi_exhausted(state, player):
            return 0.0
        return self.SHI_EXHAUST_ATTACK_BONUS

    def _fuse_strategy_hidden_block_adjustment(
        self,
        state,
        player: str,
        action_type: str,
        block: Optional[str],
        attack: Optional[str],
    ) -> float:
        if action_type != "attack_after_block" or block is None:
            return 0.0

        tr = self._track.get(id(state))
        hand = list(state.hands[player])
        after_hand = list(hand)
        if block in after_hand:
            after_hand.remove(block)
        if attack in after_hand:
            after_hand.remove(attack)

        block_no = len(getattr(state, "face_down_hidden", {}).get(player, [])) + 1
        value = 0.0

        # 香と王玉は、点数以上に受け・終盤待ちとしての価値が高い。
        if block == "2":
            value -= self.FUSE_KYOSHA_BLOCK_PENALTY
            if tr is not None and self._opponents_piece_pressure(tr, player, "2") > 0:
                value -= 60.0
        elif block in ("8", "9"):
            value -= self.FUSE_KING_BLOCK_PENALTY

        # 金銀は実戦で受けに使う機会が多いため、大駒より少し残したい。
        if block in ("4", "5"):
            value -= self.FUSE_KEEP_KIN_GIN_RECEIVE_BONUS

        # しは基本的には伏せやすい。ただし敵方のし攻めが濃い時と、
        # 2枚目以降に最後のしを消す時は守備価値が上がる。
        if block == "1":
            remaining_shi = after_hand.count("1")
            if remaining_shi < 2 and self._enemy_shi_attack_threat_for_fuse(tr, player):
                value -= self.FUSE_ENEMY_SHI_THREAT_BLOCK_PENALTY
            has_king_after = any(p in ("8", "9") for p in after_hand)
            if block_no >= 2 and remaining_shi <= 0 and not has_king_after:
                value -= self.FUSE_KEEP_LAST_SHI_PENALTY
            if block_no >= 3 and has_king_after:
                value += self.FUSE_THIRD_BLOCK_KING_SHI_BONUS

        # 攻め駒が飽和している場合は、強い駒を1枚伏せてもよい。
        # 例: 角角飛飛、馬馬飛飛、銀銀飛飛など。
        if block in ("3", "4", "5", "6", "7") and hand.count(block) >= 2:
            before_pairs = self._strong_attack_pair_pieces(hand)
            after_pairs = self._strong_attack_pair_pieces(after_hand)
            if len(before_pairs) >= 2 and after_pairs:
                value += self.FUSE_ATTACK_SATURATION_BLOCK_BONUS
                if block in ("6", "7"):
                    value += 8.0
                elif block == "3":
                    value += 4.0

        return value

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
        if absolute_rank in ("SS", "S", "A", "B", "C"):
            self._set_decision_reason("score_fallback")
            dealer = getattr(state, "dealer", None)
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
                    dealer = getattr(state, "dealer", None)
                    if dealer is not None and attacker == dealer:
                        self._set_score_fallback_detail(f"enemy_dealer_first_next_abs{absolute_rank}_two_shi_receive")
                    else:
                        self._set_score_fallback_detail(f"enemy_first_next_abs{absolute_rank}_two_shi_receive")
                    return same_piece_receive
                self._set_decision_reason("score_fallback")
                dealer = getattr(state, "dealer", None)
                if dealer is not None and attacker == dealer:
                    self._set_score_fallback_detail(f"enemy_dealer_first_next_abs{absolute_rank}_one_shi_pass")
                else:
                    self._set_score_fallback_detail(f"enemy_first_next_abs{absolute_rank}_one_shi_pass")
                return pass_action
            self._set_decision_reason("score_fallback")
            dealer = getattr(state, "dealer", None)
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
            score = self._score_receive_phase(state, player, act[0], act[1])
            if score < 1e8:
                continue
            detail = "receive_win_after"
            if self._win_after_receive_bonus(state, player, act) <= 0:
                detail = "receive_tsume_after"
            finish_after_receive_actions.append((score, detail, act))

        if not finish_after_receive_actions:
            return None

        finish_after_receive_actions.sort(key=lambda x: x[0], reverse=True)
        _score, detail, chosen = finish_after_receive_actions[0]
        self._set_decision_reason("score_fallback")
        self._set_score_fallback_detail(detail)
        return chosen

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

    def _piece_count_kakari_adjustment(self, state, player: str, attack: str) -> float:
        tr = self._track.get(id(state))
        if tr is None or attack == "1":
            return 0.0
        if not (attack == tr.get("ally_first_attack") or attack in tr.get("ally_past_attacks", set())):
            return 0.0
        pressure = self._opponents_piece_pressure(tr, player, attack)
        if pressure > 0:
            return -min(180.0, self.INFER_KAKARI_BLOCKED_PENALTY * pressure)
        if self._opponents_piece_exhausted(tr, player, attack):
            return self.INFER_KAKARI_CLEAR_BONUS
        return 0.0

    def _strong_attack_pair_pieces(self, hand: List[str]) -> List[str]:
        counts = Counter(hand)
        pairs: List[str] = []
        for piece in ("7", "6", "5", "4", "3", "2"):
            if counts.get(piece, 0) >= 2:
                pairs.append(piece)
        return pairs

    def _kakari_saturation_followup_pieces(self, hand: List[str]) -> List[str]:
        counts = Counter(hand)
        profiles = [
            profile
            for profile in self._attack_shape_profiles(counts)
            if int(profile.get("value", 0)) >= 3
        ]
        pieces = {
            str(piece)
            for profile in profiles
            for piece in profile.get("pieces", [])
        }
        return sorted(pieces, key=lambda piece: (POINTS.get(piece, 0), piece), reverse=True)

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
        if block is None or block not in ("2", "3", "4", "5", "6", "7"):
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

    def _kakari_saturation_attack_bonus(self, state, player: str, attack: Optional[str]) -> float:
        if attack is None or attack not in ("2", "3", "4", "5", "6", "7"):
            return 0.0
        tr = self._track.get(id(state))
        if tr is None:
            return 0.0
        if self._is_fourth_middle_attack(state, player, attack):
            return 0.0
        if not (attack == tr.get("ally_first_attack") or attack in tr.get("ally_past_attacks", set())):
            return 0.0
        if attack not in state.hands[player]:
            return 0.0

        after_attack = list(state.hands[player])
        after_attack.remove(attack)
        followups = self._kakari_saturation_followup_pieces(after_attack)
        if not followups:
            return 0.0

        bonus = self.KAKARI_SATURATION_ATTACK_BONUS
        bonus += max(float(POINTS.get(piece, 0)) for piece in followups) / 3.0
        ally = self._ally_of(player)
        remaining_min, remaining_max = self._estimate_remaining_range(tr, ally, attack)
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

        strong_followups = set(self._strong_ally_receive_followup_pieces(hand_after))
        candidates = [
            p
            for p in set(hand_after)
            if p in strong_followups
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

    def _ally_force_king_attack_bonus(
        self,
        state,
        player: str,
        action_type: str,
        attack: Optional[str],
    ) -> float:
        tr = self._track.get(id(state))
        if (
            tr is not None
            and action_type == "attack"
            and attack is not None
            and attack == tr.get("pending_ally_force_king_attack_piece")
        ):
            return self.ALLY_FORCE_KING_ATTACK_BONUS
        return 0.0

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

    def _public_attack_safety_bonus(self, state, player: str, attack: str) -> float:
        tr = self._track.get(id(state))
        if tr is None:
            return 0.0

        if attack in ("8", "9"):
            return 0.0

        total = self._piece_total(attack)
        seen = tr["public_seen_counts"].get(attack, 0)
        mine = state.hands[player].count(attack)

        unseen_elsewhere = max(0, total - seen - mine)
        bonus = 0.0

        if attack in ("1", "2"):
            bonus += 15.0

        if unseen_elsewhere == 0:
            bonus += self.PUBLIC_SAFE_ATTACK_BONUS_HIGH
        elif unseen_elsewhere == 1:
            bonus += self.PUBLIC_SAFE_ATTACK_BONUS_MID
        elif unseen_elsewhere == 2:
            bonus += self.PUBLIC_SAFE_ATTACK_BONUS_LOW

        bonus -= self._opponent_first_attack_strategy_penalty(tr, player, attack)

        return bonus
        
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

    def on_public_action(self, state, player: str, action: Action) -> None:
        if self.me is None:
            return
        self._ensure_trackers(state)
        tr = self._track.get(id(state))
        if tr is None:
            return

        action_type, block, attack = action
        visible_block = block
        if action_type == "attack_after_block" and player != self.me:
            visible_block = None

        self._update_public_hand_model(state, tr, player, action_type, visible_block, attack)

        if action_type == "attack_after_block":
            hidden_counts = tr.setdefault("hidden_block_counts", {p: 0 for p in ("A", "B", "C", "D")})
            hidden_counts[player] = int(hidden_counts.get(player, 0)) + 1

        if player == tr.get("ally"):
            consumed = 0
            if action_type == "attack_after_block":
                consumed = 2
            elif action_type in ("receive", "attack"):
                consumed = 1
            if consumed:
                tr["ally_consumed_count"] = min(8, int(tr.get("ally_consumed_count", 0)) + consumed)

        if (
            action_type == "pass"
            and player == tr.get("ally")
            and state.attacker == self.me
            and state.current_attack == "1"
        ):
            tr["ally_passed_my_shi_count"] = int(tr.get("ally_passed_my_shi_count", 0)) + 1
            if tr.get("ally_shi_signal") == "unknown":
                tr["ally_shi_signal"] = "passed"

        if (
            action_type == "pass"
            and player == self.me
            and state.attacker == tr.get("ally")
            and state.current_attack == "1"
        ):
            tr["i_passed_ally_shi"] = True

        if (
            action_type == "pass"
            and player != self.me
            and not self._same_team(player, self.me)
            and state.attacker == self.me
            and state.current_attack == "1"
        ):
            tr["enemy_passed_my_shi_count"] = int(tr.get("enemy_passed_my_shi_count", 0)) + 1

        if (
            action_type == "pass"
            and player != self.me
            and not self._same_team(player, self.me)
            and state.attacker == tr.get("ally")
            and state.current_attack == "1"
        ):
            tr["ally_shi_passed_by_enemy"] = True

        if (
            action_type == "pass"
            and player == tr.get("ally")
            and state.attacker is not None
            and state.current_attack is not None
            and not self._same_team(state.attacker, self.me)
            and tr.get("enemy_attack_counts", {}).get(state.attacker, 1) == 1
            and player == state.next_player(state.attacker)
        ):
            tr["ally_passed_enemy_first_attack"] = True
            tr["ally_passed_enemy_first_attack_attacker"] = state.attacker
            tr["ally_passed_enemy_first_attack_piece"] = str(state.current_attack)

            dealer = getattr(state, "dealer", None)
            if dealer is not None and state.attacker == dealer:
                tr["ally_passed_enemy_dealer_first_attack"] = True
                tr["ally_passed_enemy_dealer_first_attack_piece"] = str(state.current_attack)

        dealer = getattr(state, "dealer", None)
        if (
            action_type == "pass"
            and dealer is not None
            and player == tr.get("ally")
            and state.attacker == dealer
            and state.current_attack is not None
            and not self._same_team(state.attacker, self.me)
            and tr.get("enemy_attack_counts", {}).get(state.attacker, 1) == 1
        ):
            tr["ally_passed_enemy_dealer_first_attack"] = True
            tr["ally_passed_enemy_dealer_first_attack_piece"] = str(state.current_attack)

        if player == self.me and action_type in ("attack", "attack_after_block"):
            tr["pending_ally_force_king_attack_piece"] = None
            tr["my_last_receive_piece"] = None

        if action_type in ("receive", "attack_after_block") and visible_block is not None:
            if visible_block in tr["public_seen_counts"]:
                tr["public_seen_counts"][visible_block] += 1
            if player == self.me and action_type == "receive":
                tr["my_last_receive_piece"] = visible_block
            if player == tr.get("ally") and action_type == "receive":
                if visible_block in tr.get("my_past_attacks", set()):
                    tr["ally_pending_response_piece"] = visible_block
                else:
                    tr["ally_pending_response_piece"] = None
            if (
                player is not None
                and player != self.me
                and not self._same_team(player, self.me)
                and action_type == "receive"
                and visible_block == "1"
            ):
                tr.setdefault("enemy_pending_shi_receive_players", set()).add(player)
            if (
                visible_block == "1"
                and state.current_attack == "1"
                and player is not None
                and not self._same_team(player, self.me)
                and tr.get("i_passed_ally_shi")
            ):
                tr["inherit_ally_shi_attack"] = True

        if action_type in ("attack", "attack_after_block") and attack is not None:
            if attack in tr["public_seen_counts"]:
                tr["public_seen_counts"][attack] += 1
                
            if player == self.me:
                tr["my_past_attacks"].add(attack)
                tr["my_last_attack"] = attack
                tr["ally_attacked_since_my_last_attack"] = False
                if attack == "1":
                    tr["my_open_shi_attack_pending"] = True
                    if tr.get("my_init_count", Counter()).get("1", 0) >= 3:
                        tr["shi_attack_mode"] = True
                        tr["shi_attack_mode_source"] = "self"
                    if tr.get("inherit_ally_shi_attack"):
                        tr["inherit_ally_shi_attack"] = False
                else:
                    tr["my_open_shi_attack_pending"] = False
                    if tr.get("shi_attack_mode") and state.hands[player].count("1") > 0:
                        tr["shi_attack_mode"] = False
                        tr["shi_attack_mode_source"] = None
            elif self._same_team(player, self.me):
                if not tr["ally_past_attacks"]:
                    tr["ally_first_attack"] = attack
                tr["ally_past_attacks"].add(attack)
                tr["ally_last_attack"] = attack
                tr["ally_attacked_since_my_last_attack"] = True
                if attack == "1":
                    tr["ally_open_shi_attack_pending"] = True
                else:
                    tr["ally_open_shi_attack_pending"] = False

                pending_response = tr.get("ally_pending_response_piece")
                if pending_response is not None and action_type == "attack":
                    if pending_response == "1" and "1" in tr["my_past_attacks"]:
                        if attack == "1":
                            tr["ally_shi_signal"] = "returned_shi"
                            tr["shi_attack_mode"] = True
                            tr["ally_shi_sashikomi_candidate"] = False
                        else:
                            if int(tr.get("enemy_passed_my_shi_count", 0)) > 0:
                                tr["ally_shi_signal"] = "sashikomi"
                                tr["ally_shi_sashikomi_candidate"] = True
                            else:
                                tr["ally_shi_signal"] = "weak"
                            tr["shi_attack_mode"] = False
                            tr["shi_attack_mode_source"] = None
                    if attack == pending_response and attack in tr["my_past_attacks"]:
                        tr["ally_responded_to_my_attacks"].add(attack)
                    elif pending_response in tr["my_past_attacks"] and pending_response not in tr["ally_responded_to_my_attacks"]:
                        tr["ally_ignored_my_attacks"].add(pending_response)
                    if pending_response == "1":
                        tr["my_open_shi_attack_pending"] = False
                    tr["ally_pending_response_piece"] = None

                if action_type == "attack_after_block":
                    if block == "1" and tr.get("my_open_shi_attack_pending"):
                        if attack == "1":
                            tr["ally_shi_signal"] = "returned_shi"
                            tr["ally_shi_sashikomi_candidate"] = False
                        elif int(tr.get("enemy_passed_my_shi_count", 0)) > 0:
                            tr["ally_shi_signal"] = "sashikomi"
                            tr["ally_shi_sashikomi_candidate"] = True
                        else:
                            tr["ally_shi_signal"] = "weak"
                        if attack == "1":
                            tr["shi_attack_mode"] = True
                        else:
                            tr["shi_attack_mode"] = False
                            tr["shi_attack_mode_source"] = None
                        tr["my_open_shi_attack_pending"] = False
                    if attack in tr["my_past_attacks"]:
                        tr["ally_responded_to_my_attacks"].add(attack)
                        
                    for past_attack in tr["my_past_attacks"]:
                        if past_attack != attack and past_attack not in tr["ally_responded_to_my_attacks"]:
                            tr["ally_ignored_my_attacks"].add(past_attack)
            else:
                if action_type in ("receive", "attack_after_block") and block == "1" and tr.get("my_open_shi_attack_pending"):
                    tr["my_open_shi_attack_pending"] = False

                pending_shi_receivers = tr.setdefault("enemy_pending_shi_receive_players", set())
                if player in pending_shi_receivers:
                    if attack != "1":
                        tr["enemy_team_rejected_shi_attack"] = True
                    pending_shi_receivers.discard(player)

                tr["enemy_past_attacks"].add(attack)
                if "enemy_attack_counts" not in tr:
                    tr["enemy_attack_counts"] = {}
                previous_count = tr["enemy_attack_counts"].get(player, 0)
                tr["enemy_attack_counts"][player] = previous_count + 1

            self._infer_missing_kings_from_third_visible_attack(
                state,
                tr,
                player,
                action_type,
                attack,
            )

    def _public_attack_evidence(self, attack: str) -> float:
        if attack in ("8", "9"):
            return 7.0
        if attack in ("6", "7"):
            return 4.0
        if attack == "2":
            return 3.5
        if attack in ("3", "4", "5"):
            return 3.0
        if attack == "1":
            return 0.5
        return 0.0

    def _update_public_hand_model(
        self,
        state,
        tr: dict,
        player: str,
        action_type: str,
        block: Optional[str],
        attack: Optional[str],
    ) -> None:
        models = tr.get("public_hand_models")
        if models is None or player not in models:
            return

        model = models[player]
        if action_type == "pass":
            model["pass_count"] += 1
            model["strength"] -= 0.25
            return

        if action_type in ("receive", "attack_after_block") and block is not None:
            model["receive_count"] += 1
            model["blocks"][block] += 1
            if player != self.me:
                self._update_observed_piece_count_estimate(tr, player, block)
                self._reconcile_piece_count_estimates(state, tr, block)
            model["strength"] += 4.0 if block in ("8", "9") else 1.5

        if action_type in ("attack", "attack_after_block") and attack is not None:
            attack_seen_before = model["attacks"][attack]
            is_first_attack = model["attack_count"] == 0
            first_attack = model.get("first_attack")
            model["attack_count"] += 1
            model["attacks"][attack] += 1
            if is_first_attack:
                strategy = self._infer_first_attack_strategy(attack)
                model["first_attack"] = attack
                model["inferred_attack_strategy"] = strategy["label"] if strategy is not None else None
                if player != self.me and strategy is not None:
                    tr.setdefault("other_first_attack_strategy_by_player", {})[player] = strategy
                    self._record_first_attack_count_estimate(state, tr, player, attack, strategy)
                    model["strength"] += float(strategy.get("strength", 0.0))
            elif player != self.me:
                self._update_observed_piece_count_estimate(tr, player, attack)
                self._reconcile_piece_count_estimates(state, tr, attack)
                if first_attack is not None and attack != first_attack:
                    model["strategy_broken"] = True
                    model["strength"] -= 2.0
                    est = self._piece_count_estimate(tr, player, first_attack)
                    if est is not None and str(est.get("source", "observed")) != "observed":
                        observed_first = self._observed_piece_count_for_player(tr, player, first_attack)
                        est["max"] = max(int(est.get("min", 0)), min(int(est.get("max", self._piece_total(first_attack))), observed_first + 1))
                        self._reconcile_piece_count_estimates(state, tr, first_attack)
            model["strength"] += self._public_attack_evidence(attack)
            if attack_seen_before >= 1:
                model["strength"] += 3.0 + attack_seen_before
            if action_type == "attack_after_block":
                model["strength"] += 1.5

        model["strength"] = max(-6.0, min(40.0, float(model["strength"])))

    def _public_hand_strength(self, tr: Optional[dict], player: Optional[str]) -> float:
        if not getattr(self, "USE_PUBLIC_HAND_INFERENCE", True):
            return 0.0
        if tr is None or player is None:
            return 0.0
        model = tr.get("public_hand_models", {}).get(player)
        if model is None:
            return 0.0
        return float(model.get("strength", 0.0))

    def _occupancy_priority_bonus(self, state, attack: str) -> float:
        tr = self._track.get(id(state))
        if tr is None:
            return 0.0
        c_all = tr["my_init_count"]

        if attack in ("2", "3", "4", "5") and c_all.get(attack, 0) == 4:
            return 80.0
        if attack in ("6", "7") and c_all.get(attack, 0) == 2:
            return 70.0
        if attack in ("2", "3", "4", "5") and c_all.get(attack, 0) == 3:
            return 55.0
        if attack in ("2", "3", "4", "5") and c_all.get(attack, 0) == 2:
            return 35.0

        if tr is not None and "1" in tr.get("ally_ignored_my_attacks", set()):
            if attack in ("6", "7"):
                return 32.0
            if attack in ("4", "5"):
                return 29.0
            if attack in ("2", "3"):
                return 26.0

        if attack == "1" and c_all.get("1", 0) == 4:
            return 25.0
        if attack == "1" and c_all.get("1", 0) == 3:
            return 10.0
        return 0.0

    def _attack_strategy_bonus(self, state, player: str, attack: str) -> float:
        tr = self._track.get(id(state))
        if tr is None:
            return 0.0

        counts = tr["my_init_count"]
        profile = self._classify_attack_type(counts)
        attack_type = int(profile["type"])
        attack_pieces = set(str(p) for p in profile["pieces"])
        receive_type = sum(1 for n in counts.values() if n > 0)

        is_kakari = attack != "1" and (
            attack == tr.get("ally_first_attack") or attack in tr.get("ally_past_attacks", set())
        )
        if is_kakari:
            return 0.0

        bonus = 0.0
        if attack_type in (2, 3):
            if attack in attack_pieces:
                bonus += self.ATTACK_STRATEGY_BONUS
        elif attack_type == 4:
            if attack in attack_pieces:
                bonus += 100.0 if receive_type <= 4 else 45.0
        elif attack_type == 5:
            if attack == "2":
                bonus += 45.0
        elif attack_type == 6:
            if attack in attack_pieces:
                bonus += 55.0
        elif attack_type in (7, 8):
            if attack == "1":
                bonus -= 40.0

        if receive_type <= 3 and counts.get(attack, 0) >= 3 and attack not in ("8", "9"):
            bonus += 35.0

        if receive_type >= 5:
            if counts.get(attack, 0) == 1 and attack not in attack_pieces and attack not in ("6", "7", "8", "9"):
                bonus -= self.RECEIVE_KEEP_PENALTY
            if receive_type >= 7 and attack in ("6", "7"):
                bonus += 45.0

        return bonus

    def _score_attack_phase(
        self,
        state,
        player: str,
        action_type: str,
        block: Optional[str],
        attack: Optional[str],
        *,
        has_non_king_attack_option: bool,
    ) -> float:
        if attack is None:
            return -1e18

        score = 0.0
        tr = self._track.get(id(state))

        if tr is not None and attack is not None:
            is_safe = self._is_absolute_safe_for_tsume(state, player, attack, tr)
            is_agari = (len(state.hands[player]) <= 2)
            
            if is_safe or is_agari:
                temp_hand = list(state.hands[player])
                if block is not None and block in temp_hand:
                    temp_hand.remove(block)
                if attack in temp_hand:
                    temp_hand.remove(attack)
                
                if len(temp_hand) == 0:
                    agari_pt = 0.0
                    if block == attack and block is not None:
                        agari_pt = float(POINTS.get(attack, 0)) * 2.0
                    elif block is not None and set([block, attack]) == {"8", "9"}:
                        agari_pt = 100.0
                    else:
                        agari_pt = float(POINTS.get(attack, 0))
                    score += 1e8 + agari_pt
                else:
                    max_sc = self._max_tsume_score(temp_hand, state, player, tr)
                    if max_sc >= 0:
                        score += 1e8 + max_sc

        score += self._last_one_remaining_bonus(state, player, attack)
        score += self._occupancy_priority_bonus(state, attack)
        score += self._public_attack_safety_bonus(state, player, attack)
        score += self._attack_strategy_bonus(state, player, attack)
        score += self._multi_attack_shape_plan_adjustment(state, player, action_type, block, attack)
        score += self._weak_shi_fallback_high_point_attack_bonus(state, player, action_type, attack)
        score += self._piece_count_attack_adjustment(state, player, attack)
        score += self._kakari_saturation_attack_bonus(state, player, attack)
        score += self._ally_force_king_attack_bonus(state, player, action_type, attack)
        score += self._endgame_remaining_pair_adjustment(state, player, block, attack)
        score += self._dealer_opening_plan_adjustment(state, player, action_type, block, attack)

        if tr is not None and tr.get("my_attack_count", 0) == 0 and state.hands[player].count(attack) == 1:
            if attack not in ("8", "9", "1"):
                score -= 30.0
        score -= self._single_middle_after_big_receive_first_attack_penalty(
            state,
            player,
            action_type,
            attack,
        )
        score -= self._single_middle_over_four_shi_signal_penalty(
            state,
            player,
            action_type,
            attack,
        )
        score -= self._fourth_middle_first_attack_delay_penalty(
            state,
            player,
            action_type,
            attack,
        )

        if tr is not None and attack != "1" and attack == tr.get("my_last_attack"):
            if tr.get("ally_attacked_since_my_last_attack"):
                if tr.get("ally_last_attack") == attack:
                    score += self.CONTINUOUS_ATTACK_BONUS * 1.5
                else:
                    score -= self.CONTINUOUS_ATTACK_BONUS
                    score -= 300.0
            else:
                score += self.CONTINUOUS_ATTACK_BONUS

        if tr is not None and attack != "1":
            ally_first = tr.get("ally_first_attack")
            is_fourth_middle = self._is_fourth_middle_attack(state, player, attack)
            if ally_first is not None and attack == ally_first:
                is_unreasonable_block = is_fourth_middle or (action_type == "attack_after_block" and block in ("8", "9"))
                if not is_unreasonable_block:
                    if tr.get("my_attack_count", 0) == 1:
                        score += self.KAKARI_GOTAE_BONUS * 10.0
                    else:
                        score += self.KAKARI_GOTAE_BONUS
            elif attack in tr.get("ally_past_attacks", set()):
                is_unreasonable_block = is_fourth_middle or (action_type == "attack_after_block" and block in ("8", "9"))
                if not is_unreasonable_block:
                    score += self.KAKARI_GOTAE_BONUS

            score += self._piece_count_kakari_adjustment(state, player, attack)

        if tr is not None:
            visible_kings = tr["public_seen_counts"].get("8", 0) + tr["public_seen_counts"].get("9", 0) + state.hands[player].count("8") + state.hands[player].count("9")
            total_p = 4 if attack in ("2", "3", "4", "5") else 2 if attack in ("6", "7") else 10 if attack == "1" else 1
            seen_and_mine = tr["public_seen_counts"].get(attack, 0) + state.hands[player].count(attack)
            is_monopoly = (seen_and_mine == total_p)

            if is_monopoly:
                if attack == "2":
                    score += self.ABSOLUTE_SAFE_BONUS
                elif attack not in ("1", "8", "9"):
                    if visible_kings == 2:
                        score += self.ABSOLUTE_SAFE_BONUS
                    else:
                        score += self.TATEWARI_BONUS

        # ★ 修正箇所：初期手札「し」が3枚以上の場合はペナルティを0点とし、空回りを恐れずに攻められるように緩和
        if action_type in ("attack", "attack_after_block") and attack == "1":
            score += self._shi_attack_score_adjustment(state, player)

        if attack in ("9", "8") and has_non_king_attack_option:
            score -= self.KING_ATTACK_PENALTY

        score += POINTS.get(attack, 0) / 10.0

        if action_type in ("attack", "attack_after_block") and block is not None:
            penalty_table = {"9": 120, "8": 120, "7": 4, "6": 4, "5": 7, "4": 7, "3": 3, "2": 18, "1": 1}
            base_penalty = float(penalty_table.get(block, 0))
            score += self._shi_attack_hidden_block_adjustment(state, player, action_type, block, attack)
            score += self._piece_count_hidden_block_adjustment(state, player, block)
            score += self._fuse_strategy_hidden_block_adjustment(state, player, action_type, block, attack)
            score -= self._same_piece_pair_spend_penalty(state, player, action_type, block, attack)
            
            # --- 新規追加：打ち出し（伏せ札）時の「し」温存ロジック ---
            # state.current_attack is None は、親の最初の手番、または他3人パスによる新たな攻めのターン（打ち出し）を指します
            is_uchidashi = (state.current_attack is None)
            if tr is not None and block == "1" and is_uchidashi:
                init_shi = tr["my_init_count"].get("1", 0)
                # 自分が3しの場合、または味方が「し」シグナルを出していて自分が3し以上の場合
                if init_shi == 3 or (tr.get("ally_first_attack") == "1" and init_shi >= 3):
                    base_penalty = 15.0
            # ----------------------------------------------
            
            context_penalty = 0.0
            
            if tr is not None and block != "1":
                consumed = 1
                if action_type == "attack_after_block" and attack == block:
                    consumed += 1
                
                remaining_blocks = state.hands[player].count(block) - consumed

                if remaining_blocks <= 0:
                    if block in tr.get("my_past_attacks", set()):
                        context_penalty += 5.0
                    if block in tr.get("ally_past_attacks", set()):
                        context_penalty += 5.0
                    if block in tr.get("enemy_past_attacks", set()):
                        context_penalty += 5.0

            score -= (base_penalty + context_penalty)

        score += self._win_now_bonus(state, player, (action_type, block, attack))
        return score

    def _score_receive_phase(self, state, player: str, action_type: str, block: Optional[str]) -> float:
        if action_type == "pass":
            base = 0.0
        else:
            if action_type != "receive" or block is None:
                return -1e18
            bonus = self._win_after_receive_bonus(state, player, (action_type, block, None))
            if bonus > 0:
                return 1e9
                
            tr = self._track.get(id(state))
            if tr is not None:
                try:
                    temp_hand = list(state.hands[player])
                    if block in temp_hand:
                        temp_hand.remove(block)
                        if len(temp_hand) <= 1:
                            return 1e8
                        
                        safe_atks = set(p for p in temp_hand if self._is_absolute_safe_for_tsume(state, player, p, tr))
                        for atk in safe_atks:
                            next_hand = list(temp_hand)
                            next_hand.remove(atk)
                            max_sc = self._max_tsume_score(next_hand, state, player, tr)
                            if max_sc >= 0:
                                return 1e8 + max_sc
                except Exception:
                    pass

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

    def _classify_score_fallback(
        self,
        state,
        player: str,
        action: Action,
        *,
        has_non_king_attack_option: bool,
    ) -> str:
        action_type, block, attack = action
        tr = self._track.get(id(state))

        enemy_attack_turn = (
            state.phase == "receive"
            and state.current_attack is not None
            and state.attacker is not None
            and (not self._same_team(state.attacker, player))
        )

        if action_type == "pass":
            if enemy_attack_turn and tr is not None:
                if self._preserve_current_attack_for_win_value(state, player) > 0:
                    return "pass_preserve_win_attack"
                if self._piece_count_receive_adjustment(state, player, action_type, block) > 0:
                    return "pass_piece_count_inference"
                attacker_count = tr.get("enemy_attack_counts", {}).get(state.attacker, 1)
                if attacker_count == 1:
                    policy = getattr(self, "ENEMY_FIRST_ATTACK_POLICY", "strict_pass")
                    if policy == "hand_strength":
                        axes = self._initial_hand_axes_for_state(state, player)
                        return (
                            f"enemy_first_pass_hand_strength_"
                            f"abs{axes['absolute_rank']}_rel{axes['relative_rank']}_"
                            f"{axes['total_score']}_"
                            f"atk{axes['attack_score']}_rcv{axes['receive_score']}"
                        )
                    return "enemy_first_pass"
                return "enemy_later_pass"
            return "pass_base"

        if action_type == "receive":
            if block is not None and self._win_after_receive_bonus(state, player, action) > 0:
                return "receive_win_after"
            if block is not None and self._score_receive_phase(state, player, action_type, block) >= 1e8:
                return "receive_tsume_after"
            if enemy_attack_turn and tr is not None:
                if self._preserve_current_attack_for_win_value(state, player) > 0:
                    return "receive_spends_win_attack"
                if self._early_enemy_first_king_receive_penalty(state, player, action_type, block) > 0:
                    return "enemy_first_receive_king_reserved"
                if self._piece_count_receive_adjustment(state, player, action_type, block) < 0:
                    return "receive_piece_count_risk"
                attacker_count = tr.get("enemy_attack_counts", {}).get(state.attacker, 1)
                if attacker_count == 1:
                    policy = getattr(self, "ENEMY_FIRST_ATTACK_POLICY", "strict_pass")
                    if policy == "hand_strength":
                        axes = self._initial_hand_axes_for_state(state, player)
                        return (
                            f"enemy_first_{action_type}_hand_strength_"
                            f"abs{axes['absolute_rank']}_rel{axes['relative_rank']}_"
                            f"{axes['total_score']}_"
                            f"atk{axes['attack_score']}_rcv{axes['receive_score']}"
                        )
                    return f"enemy_first_receive_{policy}"
                if block in ("8", "9"):
                    return "enemy_later_receive_king"
                return "enemy_later_receive"
            if state.attacker is not None and self._same_team(state.attacker, player):
                if self._ally_shi_exhaust_receive_bonus(state, player, block) > 0:
                    return "ally_shi_exhaust_receive"
                if self._ally_force_king_receive_bonus(state, player, action_type, block) > 0:
                    return "ally_force_king_receive"
                if self._ally_strong_followup_receive_bonus(state, player, action_type, block) > 0:
                    return "ally_strong_followup_receive"
                if self._kakari_saturation_receive_bonus(state, player, block) > 0:
                    return "ally_attack_receive_saturation"
                return "ally_attack_receive"
            if block in ("8", "9"):
                return "receive_king_base"
            return "receive_base"

        if action_type not in ("attack", "attack_after_block") or attack is None:
            return "other"

        if self._win_now_bonus(state, player, action) > 0:
            return "attack_win_now_score"

        if tr is not None:
            is_safe = self._is_absolute_safe_for_tsume(state, player, attack, tr)
            is_agari = (len(state.hands[player]) <= 2)
            if is_safe or is_agari:
                temp_hand = list(state.hands[player])
                if block is not None and block in temp_hand:
                    temp_hand.remove(block)
                if attack in temp_hand:
                    temp_hand.remove(attack)
                if len(temp_hand) == 0:
                    return "attack_agari_score"
                if self._max_tsume_score(temp_hand, state, player, tr) >= 0:
                    return "attack_tsume_score"

            ally_first = tr.get("ally_first_attack")
            if self._kakari_saturation_attack_bonus(state, player, attack) > 0:
                return "attack_kakari_saturation"
            if self._ally_force_king_attack_bonus(state, player, action_type, attack) > 0:
                return "attack_force_enemy_king"
            if self._fourth_middle_first_attack_delay_penalty(state, player, action_type, attack) > 0:
                return "attack_delay_fourth_middle"
            if attack != "1" and (attack == ally_first or attack in tr.get("ally_past_attacks", set())):
                if not self._is_fourth_middle_attack(state, player, attack):
                    return "attack_kakari_score"

            if attack != "1" and attack == tr.get("my_last_attack"):
                return "attack_continuous_score"

            dealer_plan_adjustment = self._dealer_opening_plan_adjustment(state, player, action_type, block, attack)
            if dealer_plan_adjustment >= self.DEALER_OPENING_PLAN_ATTACK_BONUS:
                return "dealer_opening_primary_attack"
            if dealer_plan_adjustment <= -self.DEALER_OPENING_PLAN_BLOCK_PENALTY:
                return "block_dealer_opening_primary_attack"

            multi_attack_adjustment = self._multi_attack_shape_plan_adjustment(state, player, action_type, block, attack)
            if multi_attack_adjustment >= self.LOWER_ATTACK_SHAPE_BLOCK_BONUS:
                return "block_lower_attack_shape"
            if multi_attack_adjustment <= -self.LOWER_ATTACK_SHAPE_ATTACK_PENALTY:
                return "attack_lower_attack_shape"
            if multi_attack_adjustment <= -self.TOP_ATTACK_SHAPE_BLOCK_PENALTY:
                return "block_top_attack_shape"

            visible_kings = (
                tr["public_seen_counts"].get("8", 0)
                + tr["public_seen_counts"].get("9", 0)
                + state.hands[player].count("8")
                + state.hands[player].count("9")
            )
            total_p = 4 if attack in ("2", "3", "4", "5") else 2 if attack in ("6", "7") else 10 if attack == "1" else 1
            seen_and_mine = tr["public_seen_counts"].get(attack, 0) + state.hands[player].count(attack)
            if seen_and_mine == total_p:
                if attack == "2" or (attack not in ("1", "8", "9") and visible_kings == 2):
                    return "attack_absolute_safe"
                if attack not in ("1", "8", "9"):
                    return "attack_tatewari"

            if self._last_one_remaining_bonus(state, player, attack) > 0:
                return "attack_last_one"

            if self._same_piece_pair_spend_penalty(state, player, action_type, block, attack) > 0:
                return "block_spends_attack_pair"

            if (
                attack == "1"
                and int(tr.get("my_attack_count", 0)) == 0
                and tr.get("my_last_receive_piece") in ("6", "7")
                and state.hands[player].count("1") >= 4
                and any(state.hands[player].count(p) == 1 for p in ("3", "4", "5"))
            ):
                return "attack_four_shi_over_single_middle"

            if self._single_middle_over_four_shi_signal_penalty(state, player, action_type, attack) > 0:
                return "attack_avoid_single_middle_over_four_shi"

            if (
                attack == "2"
                and self._kyosha_probe_expected_score_after_attack_action(state, player, action, tr) is not None
            ):
                return "attack_kyosha_probe_high_score"

            if attack == "1" and self._shi_exhaust_attack_bonus(state, player) > 0:
                return "attack_shi_exhaust_enemy"

            if self._weak_shi_fallback_high_point_attack_bonus(state, player, action_type, attack) > 0:
                return "attack_high_point_after_weak_shi"

            if attack == "1" and self._shi_sashikomi_attack_bonus(state, player) > 0:
                return "attack_shi_sashikomi"

            if self._shi_sashikomi_wait_bonus(state, player, block, attack) > 0:
                return "attack_keep_shi_sashikomi"

            if self._occupancy_priority_bonus(state, attack) > 0:
                return "attack_occupancy"

            if self._endgame_remaining_pair_adjustment(state, player, block, attack) >= 50.0:
                return "attack_endgame_high_score_pair"

            piece_count_adjustment = self._piece_count_attack_adjustment(state, player, attack)
            if piece_count_adjustment >= self.INFER_ATTACK_EXHAUSTED_BONUS:
                return "attack_piece_count_clear"
            if piece_count_adjustment <= -self.INFER_ATTACK_OVERLAP_PENALTY:
                return "attack_avoid_enemy_piece_count"

            kakari_adjustment = self._piece_count_kakari_adjustment(state, player, attack)
            if kakari_adjustment > 0:
                return "attack_kakari_piece_count_clear"
            if kakari_adjustment < 0:
                return "attack_kakari_piece_count_blocked"

            if self._public_attack_safety_bonus(state, player, attack) >= self.PUBLIC_SAFE_ATTACK_BONUS_MID:
                return "attack_public_safe"

            strategy_bonus = self._attack_strategy_bonus(state, player, attack)
            if strategy_bonus > 0:
                profile = self._classify_attack_type(tr["my_init_count"])
                return f"attack_strategy_type_{profile['type']}_{profile['label']}"
            if strategy_bonus < 0:
                return "attack_receive_keep_penalty"

        if attack in ("8", "9") and has_non_king_attack_option:
            return "attack_king_penalty_context"

        if action_type == "attack_after_block" and block is not None:
            fuse_adjustment = self._fuse_strategy_hidden_block_adjustment(state, player, action_type, block, attack)
            if fuse_adjustment >= self.FUSE_ATTACK_SATURATION_BLOCK_BONUS:
                return "block_attack_saturation"
            if fuse_adjustment <= -self.FUSE_KYOSHA_BLOCK_PENALTY:
                return "block_fuse_keep_key_piece"
            if block == "1" and fuse_adjustment <= -self.FUSE_KEEP_LAST_SHI_PENALTY:
                return "block_keep_shi_defense"
            if self._piece_count_hidden_block_adjustment(state, player, block) < 0:
                return "block_keep_piece_count"
            if block in ("8", "9"):
                return "block_king_penalty"
            if block == "1":
                return "block_shi_context"
            return "block_piece_penalty"

        return "attack_piece_value"

    def select_action(self, state, player: str, actions: List[Action]) -> Action:
        self._set_decision_reason("")
        self._set_score_fallback_detail("")

        if self.me is None:
            self.me = player
        elif self.me != player:
            raise ValueError(f"{self.name}: bound to me={self.me}, cannot play for {player}")

        self._ensure_trackers(state)
        tr = self._track.get(id(state))

        if tr is not None and tr.get("kg_plan_active"):
            kings_in_hand = state.hands[player].count("8") + state.hands[player].count("9")
            kings_in_past = 1 if "8" in tr.get("my_past_attacks", set()) else 0
            kings_in_past += 1 if "9" in tr.get("my_past_attacks", set()) else 0
            if kings_in_hand + kings_in_past < 2:
                tr["kg_plan_active"] = False

        has_non_king_attack_option = any(
            (t in ("attack", "attack_after_block")) and (a is not None) and (a not in ("8", "9"))
            for (t, _b, a) in actions
        )

        if tr is not None and tr.get("pending_ally_force_king_attack_piece"):
            attack_actions = [
                (t, b, a)
                for (t, b, a) in actions
                if t in ("attack", "attack_after_block") and a is not None
            ]
            win_now_actions = [
                (self._win_now_bonus(state, player, act), act)
                for act in attack_actions
                if self._win_now_bonus(state, player, act) > 0
            ]
            if win_now_actions:
                win_now_actions.sort(key=lambda x: x[0], reverse=True)
                chosen = win_now_actions[0][1]
                tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                tr["pending_ally_force_king_attack_piece"] = None
                self._set_decision_reason("win_now")
                return chosen

            pending_piece = str(tr.get("pending_ally_force_king_attack_piece"))
            for act in attack_actions:
                if act[2] == pending_piece:
                    tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                    tr["pending_ally_force_king_attack_piece"] = None
                    self._set_decision_reason("score_fallback")
                    self._set_score_fallback_detail("attack_force_enemy_king")
                    return act

        kakari_actions: List[Tuple[float, Action]] = []
        if tr is not None:
            ally_first = tr.get("ally_first_attack")
            ally_past = tr.get("ally_past_attacks", set())
            for (t, b, a) in actions:
                if t in ("attack", "attack_after_block") and a is not None and a != "1":
                    is_unreasonable_block = (
                        self._is_fourth_middle_attack(state, player, a)
                        or (t == "attack_after_block" and b in ("8", "9"))
                    )
                    if not is_unreasonable_block:
                        if (ally_first is not None and a == ally_first) or (a in ally_past):
                            sc = self._score_attack_phase(state, player, t, b, a, has_non_king_attack_option=has_non_king_attack_option)
                            if t == "attack_after_block":
                                sc += self._score_receive_phase(state, player, "receive", b)
                            kakari_actions.append((sc, (t, b, a)))

        if kakari_actions:
            kakari_actions.sort(key=lambda x: x[0], reverse=True)
            chosen = kakari_actions[0][1]
            if tr is not None:
                tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                if tr.get("kg_plan_active") and tr["my_attack_count"] == 2 and chosen[2] in ("8", "9") and tr.get("kg_second") is None:
                    tr["kg_second"] = chosen[2]
                if tr.get("kg_plan_active") and tr["my_attack_count"] >= 3:
                    tr["kg_plan_active"] = False
            self._set_decision_reason("kakari")
            return chosen

        responded_actions: List[Tuple[float, Action]] = []
        if tr is not None:
            for (t, b, a) in actions:
                if t in ("attack", "attack_after_block") and a is not None:
                    if a in ("1", "2", "3", "4", "5") and a in tr.get("ally_responded_to_my_attacks", set()):
                        is_unreasonable_block = (t == "attack_after_block" and b in ("8", "9"))
                        if not is_unreasonable_block:
                            sc = self._score_attack_phase(state, player, t, b, a, has_non_king_attack_option=has_non_king_attack_option)
                            if t == "attack_after_block":
                                sc += self._score_receive_phase(state, player, "receive", b)
                            responded_actions.append((sc, (t, b, a)))

        if responded_actions:
            responded_actions.sort(key=lambda x: x[0], reverse=True)
            chosen = responded_actions[0][1]
            if tr is not None:
                tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                if tr.get("kg_plan_active") and tr["my_attack_count"] == 2 and chosen[2] in ("8", "9") and tr.get("kg_second") is None:
                    tr["kg_second"] = chosen[2]
                if tr.get("kg_plan_active") and tr["my_attack_count"] >= 3:
                    tr["kg_plan_active"] = False
            self._set_decision_reason("responded")
            return chosen

        give_way_action = self._give_way_to_ally_guaranteed_win_action(state, player, actions)
        if give_way_action is not None:
            return give_way_action

        guaranteed_finish_receive = self._guaranteed_finish_receive_action(state, player, actions)
        if guaranteed_finish_receive is not None:
            return guaranteed_finish_receive

        if state.phase == "receive" and state.current_attack in ("1", "2"):
            rank_policy_action = self._enemy_first_same_piece_rank_policy_action(state, player, actions)
            if rank_policy_action is not None:
                return rank_policy_action

        filtered_actions = []
        if tr is not None:
            ignored = tr.get("ally_ignored_my_attacks", set())
            for act in actions:
                t, b, a = act
                if t in ("attack", "attack_after_block") and a is not None:
                    if a in ignored:
                        if a == "1":
                            continue
                        elif a in ("2", "3", "4", "5") and tr["my_init_count"].get(a, 0) == 2:
                            continue
                filtered_actions.append(act)
        else:
            filtered_actions = actions
            
        if not filtered_actions:
            filtered_actions = actions
            
        actions = filtered_actions

        early_big_receive_action = self._early_big_piece_same_receive_action(state, player, actions)
        if early_big_receive_action is not None:
            return early_big_receive_action

        rank_policy_action = self._enemy_first_same_piece_rank_policy_action(state, player, actions)
        if rank_policy_action is not None:
            return rank_policy_action

        kg_keep_width_action = self._king_gyoku_opening_keep_receive_width_action(state, player, actions)
        if kg_keep_width_action is not None:
            return kg_keep_width_action

        if tr is not None and tr.get("my_attack_count", 0) == 0 and state.attacker is None and state.current_attack is None:
            if tr.get("perfect_plan") is None:
                plan = self._plan_perfect_game(state.hands[player])
                if plan:
                    tr["perfect_plan"] = plan
                    tr["perfect_plan_step"] = 0

        if tr is not None and tr.get("perfect_plan") is not None:
            step = tr["perfect_plan_step"]
            plan = tr["perfect_plan"]
            if step < len(plan):
                expected_fuse, expected_atk = plan[step]
                for act in actions:
                    if act[0] in ("attack", "attack_after_block") and act[1] == expected_fuse and act[2] == expected_atk:
                        tr["perfect_plan_step"] += 1
                        tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                        self._set_decision_reason("perfect_plan")
                        return act

        high_score_tsume = self._high_score_tsume_action(
            state,
            player,
            actions,
            has_non_king_attack_option=has_non_king_attack_option,
        )
        if high_score_tsume is not None:
            chosen, route_score, immediate = high_score_tsume
            if tr is not None and chosen[0] in ("attack", "attack_after_block"):
                tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                if tr.get("kg_plan_active") and tr["my_attack_count"] == 2 and chosen[2] in ("8", "9") and tr.get("kg_second") is None:
                    tr["kg_second"] = chosen[2]
                if tr.get("kg_plan_active") and tr["my_attack_count"] >= 3:
                    tr["kg_plan_active"] = False
            self._set_decision_reason("win_now" if immediate else "tsume")
            self._set_score_fallback_detail(f"high_score_{int(route_score)}")
            return chosen

        win_now_actions: List[Tuple[float, Action]] = []
        for (t, b, a) in actions:
            if t in ("attack", "attack_after_block"):
                bonus = self._win_now_bonus(state, player, (t, b, a))
                if bonus > 0:
                    win_now_actions.append((bonus, (t, b, a)))
        if win_now_actions:
            win_now_actions.sort(key=lambda x: x[0], reverse=True)
            chosen = win_now_actions[0][1]
            if tr is not None and chosen[0] in ("attack", "attack_after_block"):
                tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                if tr.get("kg_plan_active") and tr["my_attack_count"] == 2 and chosen[2] in ("8", "9") and tr.get("kg_second") is None:
                    tr["kg_second"] = chosen[2]
                if tr.get("kg_plan_active") and tr["my_attack_count"] >= 3:
                    tr["kg_plan_active"] = False
            self._set_decision_reason("win_now")
            return chosen

        tsume_actions: List[Tuple[float, Action]] = []
        if tr is not None:
            for (t, b, a) in actions:
                if t in ("attack", "attack_after_block") and a is not None:
                    is_safe = self._is_absolute_safe_for_tsume(state, player, a, tr)
                    is_agari = (len(state.hands[player]) <= 2)
                    if is_safe or is_agari:
                        temp_hand = list(state.hands[player])
                        if b is not None and b in temp_hand:
                            temp_hand.remove(b)
                        if a in temp_hand:
                            temp_hand.remove(a)
                        
                        if len(temp_hand) == 0:
                            sc = self._score_attack_phase(state, player, t, b, a, has_non_king_attack_option=has_non_king_attack_option)
                            if t == "attack_after_block":
                                sc += self._score_receive_phase(state, player, "receive", b)
                            tsume_actions.append((sc, (t, b, a)))
                        else:
                            max_sc = self._max_tsume_score(temp_hand, state, player, tr)
                            if max_sc >= 0:
                                sc = self._score_attack_phase(state, player, t, b, a, has_non_king_attack_option=has_non_king_attack_option)
                                if t == "attack_after_block":
                                    sc += self._score_receive_phase(state, player, "receive", b)
                                tsume_actions.append((sc, (t, b, a)))
            
        if tsume_actions:
            tsume_actions.sort(key=lambda x: x[0], reverse=True)
            chosen = tsume_actions[0][1]
            if tr is not None:
                tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                if tr.get("kg_plan_active") and tr["my_attack_count"] == 2 and chosen[2] in ("8", "9") and tr.get("kg_second") is None:
                    tr["kg_second"] = chosen[2]
                if tr.get("kg_plan_active") and tr["my_attack_count"] >= 3:
                    tr["kg_plan_active"] = False
            self._set_decision_reason("tsume")
            return chosen

        attack_actions = [(t, b, a) for (t, b, a) in actions if t in ("attack", "attack_after_block") and a is not None]

        if tr is not None and self.FORCE_KING_GYOKU_ON_THIRD_ATTACK and attack_actions:
            next_attack_no = int(tr.get("my_attack_count", 0)) + 1
            if next_attack_no == 3:
                for p in ["8", "9"]:
                    for act in attack_actions:
                        if act[2] == p:
                            tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                            if tr.get("kg_plan_active"):
                                tr["kg_plan_active"] = False
                            self._set_decision_reason("forced_king_third")
                            return act

        if tr is not None and tr.get("kg_plan_active") and self.KING_GYOKU_FORCE_ORDER:
            next_attack_no = int(tr.get("my_attack_count", 0)) + 1
            if attack_actions and next_attack_no in (2, 3):
                hand = state.hands[player]
                has9 = "9" in hand
                has8 = "8" in hand
                if has8 or has9:
                    if next_attack_no == 2:
                        for p in ["9", "8"]:
                            if p == "9" and not has9: continue
                            if p == "8" and not has8: continue
                            for act in attack_actions:
                                if act[2] == p:
                                    tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                                    if act[2] in ("8", "9") and tr.get("kg_second") is None:
                                        tr["kg_second"] = act[2]
                                    self._set_decision_reason("king_order")
                                    return act
                    if next_attack_no == 3:
                        second = tr.get("kg_second")
                        want = "8" if second == "9" else "9" if second == "8" else None
                        if want is not None:
                            for act in attack_actions:
                                if act[2] == want:
                                    tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                                    tr["kg_plan_active"] = False
                                    self._set_decision_reason("king_order")
                                    return act
                        for p in ["9", "8"]:
                            for act in attack_actions:
                                if act[2] == p:
                                    tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                                    tr["kg_plan_active"] = False
                                    self._set_decision_reason("king_order")
                                    return act

        if tr is not None and self.PREFER_PUBLIC_SAFE_NONKING_ON_THIRD_ATTACK and attack_actions:
            next_attack_no = int(tr.get("my_attack_count", 0)) + 1
            if next_attack_no == 3:
                has_king_attack = any(act[2] in ("8", "9") for act in attack_actions)
                if has_king_attack:
                    safe_non_king = []
                    for act in attack_actions:
                        a = act[2]
                        if a is None or a in ("8", "9"): continue
                        safety = self._public_attack_safety_bonus(state, player, a)
                        if safety >= self.PUBLIC_SAFE_ATTACK_BONUS_MID:
                            safe_non_king.append(act)
                    if safe_non_king:
                        best = safe_non_king[0]
                        best_score = -1e18
                        for (t, b, a) in safe_non_king:
                            sc = self._score_attack_phase(state, player, t, b, a, has_non_king_attack_option=True)
                            if t == "attack_after_block":
                                sc += self._score_receive_phase(state, player, "receive", b)
                            if sc > best_score:
                                best_score = sc
                                best = (t, b, a)
                        tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                        self._set_decision_reason("safe_nonking_third")
                        return best

        # --- 第8位：味方の「し」攻めに対するレスポンス（しシグナルへの返答） ---
        if tr is not None:
            ally = tr["ally"]
            if state.phase == "receive" and state.current_attack == "1" and state.attacker == ally:
                initial_shis = tr["my_init_count"].get("1", 0)
                current_shis = state.hands[player].count("1")
                can_show_four_shi_signal = (
                    current_shis >= 4
                    or (current_shis >= 3 and "1" in tr.get("my_past_attacks", set()))
                    or (current_shis >= 2 and tr.get("shi_attack_mode"))
                )

                # 1. 「現在の手札」に「し」が4枚以上ある場合（し受け・し攻め）
                if can_show_four_shi_signal:
                    for act in actions:
                        if act[0] == "attack_after_block" and act[1] == "1" and act[2] == "1":
                            tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                            tr["shi_attack_mode"] = True
                            tr["shi_attack_mode_source"] = "ally_signal"
                            if tr.get("kg_plan_active") and tr["my_attack_count"] == 2 and act[2] in ("8", "9") and tr.get("kg_second") is None:
                                tr["kg_second"] = act[2]
                            if tr.get("kg_plan_active") and tr["my_attack_count"] >= 3:
                                tr["kg_plan_active"] = False
                            self._set_decision_reason("shi_signal")
                            return act
                    # 「し受け・し攻め」が物理的にできない場合は、とりあえず「し」で受ける
                    for act in actions:
                        if act[0] == "receive" and act[1] == "1":
                            tr["shi_attack_mode"] = True
                            tr["shi_attack_mode_source"] = "ally_signal_receive"
                            self._set_decision_reason("shi_signal")
                            return act

                # 2. 「配牌時の手札」に「し」が3枚だった場合（パス）
                elif initial_shis == 3:
                    for act in actions:
                        if act[0] == "pass":
                            self._set_decision_reason("shi_signal")
                            return act

                # 3. 「配牌時の手札」に「し」が1〜2枚だった場合（し受け・別の強い駒で攻め）
                elif initial_shis in (1, 2):
                    cands = [act for act in actions if act[0] == "attack_after_block" and act[1] == "1" and act[2] is not None and act[2] != "1"]
                    if cands:
                        has_non_king = any((c[2] is not None) and (c[2] not in ("8", "9")) for c in cands)
                        best = cands[0]
                        best_score = -1e18
                        for (t, b, a) in cands:
                            # 第9位のスコア計算関数を流用して、最も強い駒を選ぶ
                            sc = self._score_attack_phase(state, player, t, b, a, has_non_king_attack_option=has_non_king)
                            sc += self._score_receive_phase(state, player, "receive", b)
                            if sc > best_score:
                                best_score = sc
                                best = (t, b, a)
                        
                        tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                        if tr.get("kg_plan_active") and tr["my_attack_count"] == 2 and best[2] in ("8", "9") and tr.get("kg_second") is None:
                            tr["kg_second"] = best[2]
                        if tr.get("kg_plan_active") and tr["my_attack_count"] >= 3:
                            tr["kg_plan_active"] = False
                        self._set_decision_reason("shi_signal")
                        return best
                    
                    # 別の駒で攻められない場合（残り1枚など）はとりあえず「し」で受ける
                    for act in actions:
                        if act[0] == "receive" and act[1] == "1":
                            self._set_decision_reason("shi_signal")
                            return act


        # --- 第9位：総合スコア評価（通常時の最適解計算） ---
        if (
            tr is not None
            and state.phase == "receive"
            and state.attacker is not None
            and not self._same_team(state.attacker, player)
            and self._preserve_current_attack_for_win_value(state, player) > 0
        ):
            immediate_receive_win = any(
                act[0] == "receive" and self._win_after_receive_bonus(state, player, act) > 0
                for act in actions
            )
            if not immediate_receive_win:
                for act in actions:
                    if act[0] == "pass":
                        self._set_decision_reason("score_fallback")
                        self._set_score_fallback_detail("pass_preserve_win_attack")
                        return act

        best_action = actions[0]
        best_score = -1e18

        for (t, block, attack) in actions:
            if t == "attack_after_block":
                score = self._score_receive_phase(state, player, "receive", block)
                score += self._score_attack_phase(state, player, t, block, attack, has_non_king_attack_option=has_non_king_attack_option)
            elif t == "attack":
                score = self._score_attack_phase(state, player, t, block, attack, has_non_king_attack_option=has_non_king_attack_option)
            else:
                score = self._score_receive_phase(state, player, t, block)

            if score > best_score:
                best_score = score
                best_action = (t, block, attack)

        score_fallback_detail = self._classify_score_fallback(
            state,
            player,
            best_action,
            has_non_king_attack_option=has_non_king_attack_option,
        )

        if tr is not None:
            if best_action[0] == "receive":
                tr["pending_ally_force_king_attack_piece"] = self._ally_force_king_attack_piece_after_receive(
                    state,
                    player,
                    best_action[0],
                    best_action[1],
                )
            if best_action[0] in ("attack", "attack_after_block"):
                tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                if tr.get("kg_plan_active") and tr["my_attack_count"] == 2 and best_action[2] in ("8", "9") and tr.get("kg_second") is None:
                    tr["kg_second"] = best_action[2]
                if tr.get("kg_plan_active") and tr["my_attack_count"] >= 3:
                    tr["kg_plan_active"] = False

        self._set_decision_reason("score_fallback")
        self._set_score_fallback_detail(score_fallback_detail)
        return best_action

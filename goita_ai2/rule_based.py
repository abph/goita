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
        self.NON_WEAK_SHI_ATTACK_PENALTY = 220.0
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
        self.ENDGAME_PAIR_SCORE_WEIGHT = 1.6
        self.ENDGAME_PAIR_KING_RECEIVE_BONUS = 18.0
        self.ENDGAME_PAIR_UNCERTAIN_PENALTY = 16.0
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
            other_first_attack_strategy_by_player={},
            other_piece_count_estimates={},
            
            ally_responded_to_my_attacks=set(),
            ally_ignored_my_attacks=set(),
            ally_passed_my_shi_count=0,
            ally_shi_signal="unknown",
            i_passed_ally_shi=False,
            inherit_ally_shi_attack=False,
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

    def _dealer_opening_attack_plan_pieces(self, state, player: str) -> set[str]:
        tr = self._track.get(id(state))
        if tr is None:
            return set()
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

        plan_pieces = self._dealer_opening_attack_plan_pieces(state, player)
        if not plan_pieces:
            return 0.0

        value = 0.0
        if attack in plan_pieces:
            value += self.DEALER_OPENING_PLAN_ATTACK_BONUS
        if block in plan_pieces:
            value -= self.DEALER_OPENING_PLAN_BLOCK_PENALTY
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

        axes = self._initial_hand_axes_for_state(state, player)
        rank = self._strategy_rank_from_axes(axes)
        if rank not in ("C", "D") and not tr.get("inherit_ally_shi_attack"):
            return 0.0

        if self._has_strong_repeat_attack(counts):
            return 0.0

        if int(axes.get("receive_type", 0)) < 5:
            return 0.0

        if "1" in tr.get("enemy_past_attacks", set()) and not tr.get("inherit_ally_shi_attack"):
            return 0.0

        already_used_shi_attack = "1" in tr.get("my_past_attacks", set())
        ally_signal = str(tr.get("ally_shi_signal", "unknown"))
        public_seen = sum(int(v) for v in tr.get("public_seen_counts", {}).values())
        is_early = public_seen <= 4

        if already_used_shi_attack and ally_signal not in ("passed", "returned_shi"):
            return 0.0
        if not already_used_shi_attack and not is_early and not tr.get("inherit_ally_shi_attack"):
            return 0.0

        ally = tr.get("ally")
        if self._public_hand_strength(tr, ally) >= 10.0:
            return 0.0

        value = self.WEAK_SHI_ATTACK_BONUS
        if ally_signal == "passed":
            value += 100.0
        elif ally_signal == "returned_shi":
            value += 220.0
        if tr.get("inherit_ally_shi_attack"):
            value += 280.0
        return value

    def _shi_attack_score_adjustment(self, state, player: str) -> float:
        tr = self._track.get(id(state))
        if tr is None:
            return -self.NON_WEAK_SHI_ATTACK_PENALTY

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
        follow_pairs = self._strong_attack_pair_pieces(after_return)
        if not follow_pairs:
            return 0.0

        bonus = self.KAKARI_SATURATION_RECEIVE_BONUS
        bonus += max(float(POINTS.get(piece, 0)) for piece in follow_pairs) / 2.0

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
        if not (attack == tr.get("ally_first_attack") or attack in tr.get("ally_past_attacks", set())):
            return 0.0
        if attack not in state.hands[player]:
            return 0.0

        after_attack = list(state.hands[player])
        after_attack.remove(attack)
        follow_pairs = self._strong_attack_pair_pieces(after_attack)
        if not follow_pairs:
            return 0.0

        bonus = self.KAKARI_SATURATION_ATTACK_BONUS
        bonus += max(float(POINTS.get(piece, 0)) for piece in follow_pairs) / 3.0
        ally = self._ally_of(player)
        remaining_min, remaining_max = self._estimate_remaining_range(tr, ally, attack)
        if remaining_min > 0 or remaining_max > 0:
            bonus += self.KAKARI_SATURATION_ALLY_REMAINING_BONUS
        return bonus

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

    def _endgame_remaining_pair_adjustment(
        self,
        state,
        player: str,
        block: Optional[str],
        attack: Optional[str],
    ) -> float:
        if attack is None:
            return 0.0
        hand = list(state.hands[player])
        if block is not None:
            if block not in hand:
                return 0.0
            hand.remove(block)
        if attack not in hand:
            return 0.0
        hand.remove(attack)
        if len(hand) != 2:
            return 0.0
        return self._endgame_pair_score(state, player, hand) * self.ENDGAME_PAIR_SCORE_WEIGHT

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

    def on_public_action(self, state, player: str, action: Action) -> None:
        if self.me is None:
            return
        self._ensure_trackers(state)
        tr = self._track.get(id(state))
        if tr is None:
            return

        action_type, block, attack = action
        self._update_public_hand_model(state, tr, player, action_type, block, attack)

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

        if action_type in ("receive", "attack_after_block") and block is not None:
            if block in tr["public_seen_counts"]:
                tr["public_seen_counts"][block] += 1
            if (
                block == "1"
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
                if attack == "1" and tr.get("inherit_ally_shi_attack"):
                    tr["inherit_ally_shi_attack"] = False
            elif self._same_team(player, self.me):
                if not tr["ally_past_attacks"]:
                    tr["ally_first_attack"] = attack
                tr["ally_past_attacks"].add(attack)
                tr["ally_last_attack"] = attack
                tr["ally_attacked_since_my_last_attack"] = True
                
                if action_type == "attack_after_block":
                    if block == "1" and "1" in tr["my_past_attacks"]:
                        tr["ally_shi_signal"] = "returned_shi" if attack == "1" else "weak"
                    if attack in tr["my_past_attacks"]:
                        tr["ally_responded_to_my_attacks"].add(attack)
                        
                    for past_attack in tr["my_past_attacks"]:
                        if past_attack != attack and past_attack not in tr["ally_responded_to_my_attacks"]:
                            tr["ally_ignored_my_attacks"].add(past_attack)
            else:
                tr["enemy_past_attacks"].add(attack)
                if "enemy_attack_counts" not in tr:
                    tr["enemy_attack_counts"] = {}
                previous_count = tr["enemy_attack_counts"].get(player, 0)
                tr["enemy_attack_counts"][player] = previous_count + 1

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
        score += self._piece_count_attack_adjustment(state, player, attack)
        score += self._kakari_saturation_attack_bonus(state, player, attack)
        score += self._endgame_remaining_pair_adjustment(state, player, block, attack)
        score += self._dealer_opening_plan_adjustment(state, player, action_type, block, attack)

        if tr is not None and tr.get("my_attack_count", 0) == 0 and state.hands[player].count(attack) == 1:
            if attack not in ("8", "9", "1"):
                score -= 30.0

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
            if ally_first is not None and attack == ally_first:
                is_unreasonable_block = (action_type == "attack_after_block" and block in ("8", "9"))
                if not is_unreasonable_block:
                    if tr.get("my_attack_count", 0) == 1:
                        score += self.KAKARI_GOTAE_BONUS * 10.0
                    else:
                        score += self.KAKARI_GOTAE_BONUS
            elif attack in tr.get("ally_past_attacks", set()):
                is_unreasonable_block = (action_type == "attack_after_block" and block in ("8", "9"))
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
        if action_type == "attack" and attack == "1":
            score += self._shi_attack_score_adjustment(state, player)

        if attack in ("9", "8") and has_non_king_attack_option:
            score -= self.KING_ATTACK_PENALTY

        score += POINTS.get(attack, 0) / 10.0

        if action_type in ("attack", "attack_after_block") and block is not None:
            penalty_table = {"9": 100, "8": 100, "7": 4, "6": 4, "5": 4, "4": 4, "3": 3, "2": 8, "1": 1}
            base_penalty = float(penalty_table.get(block, 0))
            score += self._shi_attack_hidden_block_adjustment(state, player, action_type, block, attack)
            score += self._piece_count_hidden_block_adjustment(state, player, block)
            
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
                return -100.0 - min(220.0, ally_strength * 8.0) + saturation_bonus
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
            base += self._piece_count_receive_adjustment(state, player, action_type, block)
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
            if enemy_attack_turn and tr is not None:
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
            if attack != "1" and (attack == ally_first or attack in tr.get("ally_past_attacks", set())):
                return "attack_kakari_score"

            if attack != "1" and attack == tr.get("my_last_attack"):
                return "attack_continuous_score"

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

        kakari_actions: List[Tuple[float, Action]] = []
        if tr is not None:
            ally_first = tr.get("ally_first_attack")
            ally_past = tr.get("ally_past_attacks", set())
            for (t, b, a) in actions:
                if t in ("attack", "attack_after_block") and a is not None and a != "1":
                    is_unreasonable_block = (t == "attack_after_block" and b in ("8", "9"))
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

                # 1. 「現在の手札」に「し」が4枚以上ある場合（し受け・し攻め）
                if current_shis >= 4:
                    for act in actions:
                        if act[0] == "attack_after_block" and act[1] == "1" and act[2] == "1":
                            tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                            if tr.get("kg_plan_active") and tr["my_attack_count"] == 2 and act[2] in ("8", "9") and tr.get("kg_second") is None:
                                tr["kg_second"] = act[2]
                            if tr.get("kg_plan_active") and tr["my_attack_count"] >= 3:
                                tr["kg_plan_active"] = False
                            self._set_decision_reason("shi_signal")
                            return act
                    # 「し受け・し攻め」が物理的にできない場合は、とりあえず「し」で受ける
                    for act in actions:
                        if act[0] == "receive" and act[1] == "1":
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

        if tr is not None:
            if best_action[0] in ("attack", "attack_after_block"):
                tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                if tr.get("kg_plan_active") and tr["my_attack_count"] == 2 and best_action[2] in ("8", "9") and tr.get("kg_second") is None:
                    tr["kg_second"] = best_action[2]
                if tr.get("kg_plan_active") and tr["my_attack_count"] >= 3:
                    tr["kg_plan_active"] = False

        self._set_decision_reason("score_fallback")
        self._set_score_fallback_detail(
            self._classify_score_fallback(
                state,
                player,
                best_action,
                has_non_king_attack_option=has_non_king_attack_option,
            )
        )
        return best_action

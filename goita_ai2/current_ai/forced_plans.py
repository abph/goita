"""手駒から勝ちが確定する攻め手順を組み立てます。
初期状態、敵の攻めを1回受けた後、2回受けた後という場面ごとに勝ち筋を調べます。
SS・S判定や、通常の評価より優先して実行する確定手順の基礎になります。
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional, Tuple

from goita_ai2.constants import POINTS


class ForcedPlansMixin:
    """Builds deterministic winning plans from the current hand shape."""

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

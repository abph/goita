# goita_ai2/rule_based.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from collections import Counter
import copy

from goita_ai2.constants import POINTS

Action = Tuple[str, Optional[str], Optional[str]]
TARGET_LAST1 = ("2", "3", "4", "5", "6", "7")


class RuleBasedAgent:
    def __init__(self, name: str = "RuleBased"):
        self.name = name
        self.me: Optional[str] = None

        self._track: Dict[int, dict] = {}
        self._my_initial_hands_by_state_id: Dict[int, List[str]] = {}

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
            
            ally_responded_to_my_attacks=set(),
            ally_ignored_my_attacks=set(),
            
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
            # 残り2枚（または1枚）なら確実に上がれるので、最大点を計算
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
            return -1.0  # 確定上がりルートではない
            
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

        if action_type in ("receive", "attack_after_block") and block is not None:
            if block in tr["public_seen_counts"]:
                tr["public_seen_counts"][block] += 1

        if action_type in ("attack", "attack_after_block") and attack is not None:
            if attack in tr["public_seen_counts"]:
                tr["public_seen_counts"][attack] += 1
                
            if player == self.me:
                tr["my_past_attacks"].add(attack)
                tr["my_last_attack"] = attack
                tr["ally_attacked_since_my_last_attack"] = False
            elif self._same_team(player, self.me):
                if not tr["ally_past_attacks"]:
                    tr["ally_first_attack"] = attack
                tr["ally_past_attacks"].add(attack)
                tr["ally_last_attack"] = attack
                tr["ally_attacked_since_my_last_attack"] = True
                
                if action_type == "attack_after_block":
                    if attack in tr["my_past_attacks"]:
                        tr["ally_responded_to_my_attacks"].add(attack)
                        
                    for past_attack in tr["my_past_attacks"]:
                        if past_attack != attack and past_attack not in tr["ally_responded_to_my_attacks"]:
                            tr["ally_ignored_my_attacks"].add(past_attack)
            else:
                tr["enemy_past_attacks"].add(attack)
                if "enemy_attack_counts" not in tr:
                    tr["enemy_attack_counts"] = {}
                tr["enemy_attack_counts"][player] = tr["enemy_attack_counts"].get(player, 0) + 1

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

        if state.attacker is None and state.current_attack is None and attack == "1":
            my_shis = tr["my_init_count"].get("1", 0) if tr is not None else 0
            has_kg = tr is not None and tr.get("kg_plan_active", False)
            if my_shis >= 4 or has_kg:
                score -= 0.0
            elif my_shis == 3:
                score -= 30.0
            else:
                score -= 100.0

        if attack in ("9", "8") and has_non_king_attack_option:
            score -= self.KING_ATTACK_PENALTY

        score += POINTS.get(attack, 0) / 10.0

        if action_type in ("attack", "attack_after_block") and block is not None:
            penalty_table = {"9": 100, "8": 100, "7": 4, "6": 4, "5": 4, "4": 4, "3": 3, "2": 8, "1": 1}
            base_penalty = float(penalty_table.get(block, 0))
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
                return -100.0
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

            if attacker_count == 1:
                if action_type == "pass":
                    base += 10000.0
                else:
                    return -1e18
            else:
                if action_type == "receive":
                    if block in ("8", "9"):
                        base += 1000.0
                    else:
                        base += 10000.0
                else:
                    base -= 10000.0

        return base

    def select_action(self, state, player: str, actions: List[Action]) -> Action:
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
                                    return act
                    if next_attack_no == 3:
                        second = tr.get("kg_second")
                        want = "8" if second == "9" else "9" if second == "8" else None
                        if want is not None:
                            for act in attack_actions:
                                if act[2] == want:
                                    tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                                    tr["kg_plan_active"] = False
                                    return act
                        for p in ["9", "8"]:
                            for act in attack_actions:
                                if act[2] == p:
                                    tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                                    tr["kg_plan_active"] = False
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
                            return act
                    # 「し受け・し攻め」が物理的にできない場合は、とりあえず「し」で受ける
                    for act in actions:
                        if act[0] == "receive" and act[1] == "1":
                            return act

                # 2. 「配牌時の手札」に「し」が3枚だった場合（パス）
                elif initial_shis == 3:
                    for act in actions:
                        if act[0] == "pass":
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
                        return best
                    
                    # 別の駒で攻められない場合（残り1枚など）はとりあえず「し」で受ける
                    for act in actions:
                        if act[0] == "receive" and act[1] == "1":
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

        return best_action
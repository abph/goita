# goita_ai2/rule_based.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from collections import Counter
import copy

from goita_ai2.constants import POINTS  # 共通化された定数を読み込む

Action = Tuple[str, Optional[str], Optional[str]]  # (action_type, block, attack)
TARGET_LAST1 = ("2", "3", "4", "5", "6", "7")  # 残り1枚狙い対象


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
        self.FIRST_ENEMY_SHI_FORCE = 800.0

        self.KING_GYOKU_FORCE_ORDER = True
        self.FORCE_KING_GYOKU_ON_THIRD_ATTACK = True

        self.PREFER_PUBLIC_SAFE_NONKING_ON_THIRD_ATTACK = True
        self.KEEP_KING_GYOKU_FOR_LAST_WHEN_TWO_LEFT = True

        # ===== "し"(=駒"1") 攻め戦略（過激派チューニング） =====
        self.SHI_PLAN_ATTACK_FORCE = 5_000.0   # 元2000.0: し攻めへの執着を最大化
        self.SHI_PLAN_RECEIVE_FORCE = 5_000.0  # 元2000.0: し受けの優先度を最大化
        self.SHI_SIGNAL_ACTIVATE = 2.0         # 元3.0: 少しの気配でシグナル発動

        # 公開情報ベースの「通りやすさ」加点
        self.PUBLIC_SAFE_ATTACK_BONUS_HIGH = 60.0
        self.PUBLIC_SAFE_ATTACK_BONUS_MID = 30.0
        self.PUBLIC_SAFE_ATTACK_BONUS_LOW = 10.0

        # ★ 連携・戦術ボーナス
        self.KAKARI_GOTAE_BONUS = 100.0
        self.ABSOLUTE_SAFE_BONUS = 1000.0  # 絶対安全駒（パス確定）
        self.TATEWARI_BONUS = 800.0        # 盾割り（王・玉あぶり出し）
        self.CONTINUOUS_ATTACK_BONUS = 500.0 # 連打ボーナス

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

        # 「しシグナル」の初期値を自分の手札から与える
        shi_signal = 0.0
        my_ones = cnt_all.get("1", 0)
        if my_ones >= 4:
            shi_signal += 2.0
        elif my_ones == 3:
            shi_signal += 1.0

        public_seen_counts = {str(i): 0 for i in range(1, 10)}

        self._track[sid] = dict(
            my_init_count=cnt_all,
            ally=ally_player,
            first_enemy_attack_seen=False,
            first_enemy_attack_skipped=False,
            public_seen_counts=public_seen_counts,

            shi_signal=shi_signal,
            shi_plan_active=False,
            shi_message_sent=False,
            shi_chain_attacker=None,
            shi_chain_passed=False,
            shi_chain_first_passer=None,

            my_attack_count=0,
            kg_plan_active=(("9" in init_hand) and ("8" in init_hand)),
            kg_second=None,
            
            my_past_attacks=set(),
            ally_past_attacks=set(),
            enemy_past_attacks=set(),
            
            # ★ 新規追加：連打とシグナル応答のトラッキング
            my_last_attack=None,
            ally_attacked_since_my_last_attack=False,
            ally_last_attack=None,
            ally_first_attack=None,
            
            perfect_plan=None,
            perfect_plan_step=0,
        )

    # ★ 「完全読み（詰めごいた）」全探索アルゴリズム
    def _plan_perfect_game(self, hand: List[str]) -> Optional[List[Tuple[str, str]]]:
        if len(hand) != 8:
            return None

        counts = Counter(hand)
        has_kings = ("8" in hand and "9" in hand)

        # 絶対安全駒のリストアップ（王玉独占による無敵化 or 香の独占）
        safe_pieces = set()
        if counts.get("2", 0) == 4:
            safe_pieces.add("2")  # パターンB, E, F: 無敵の矛
        
        if has_kings:
            for p in ("3", "4", "5"):
                if counts.get(p, 0) == 4: safe_pieces.add(p)  # パターンA, G: 盾の独占＋4枚組
            for p in ("6", "7"):
                if counts.get(p, 0) == 2: safe_pieces.add(p)  # パターンD: 飛角の双龍
            safe_pieces.add("8")
            safe_pieces.add("9")

        # 確定上がり不可の場合は探索終了
        if not safe_pieces:
            return None

        best_score = -1
        best_plan = None

        def search(current_hand: List[str], current_plan: List[Tuple[str, str]]):
            nonlocal best_score, best_plan
            
            if best_score == 100:
                return  # 最大スコア（100点）に到達した時点で探索を打ち切り（超高速化）

            # 手札を出し切った（上がり）の得点計算
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
                # 道中（1〜3ターン）は絶対に受けられない安全駒でのみ攻める
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
                # 最終ターン（4ターン目）：上がりには受けが発生しないため、残りの任意の2枚で上がれる
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

    def _strong_initial_hand(self, state) -> bool:
        tr = self._track.get(id(state))
        if tr is None:
            return False
        c_all = tr["my_init_count"]

        for x in ("2", "3", "4", "5"):
            if c_all.get(x, 0) >= 3:
                return True
        for x in ("6", "7"):
            if c_all.get(x, 0) == 2:
                return True
        return False

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
        return 1  # 8,9

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
        
    # ★ 新規追加：中盤の詰めごいた（事実上の確定勝利）を判定する専用ヘルパー
    def _is_absolute_safe_for_tsume(self, state, player: str, attack: str, tr: dict) -> bool:
        if attack is None:
            return False
        # 王・玉の攻めは絶対にブロックされない
        if attack in ("8", "9"):
            return True
        total_p = 4 if attack in ("2", "3", "4", "5") else 2 if attack in ("6", "7") else 10 if attack == "1" else 1
        seen_and_mine = tr.get("public_seen_counts", {}).get(attack, 0) + state.hands[player].count(attack)
        if seen_and_mine < total_p:
            return False
        # 香(2)・し(1)を独占している場合は王玉でも切れないため絶対安全
        if attack in ("1", "2"):
            return True
        # それ以外の駒は、王と玉の両方が場に出ている（または自分が持っている）場合のみ絶対安全
        visible_kings = tr.get("public_seen_counts", {}).get("8", 0) + tr.get("public_seen_counts", {}).get("9", 0) + state.hands[player].count("8") + state.hands[player].count("9")
        return visible_kings == 2

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

        # ===== "し"(=1) 攻め戦略：公開情報からのトラッキング =====
        if action_type in ("attack", "attack_after_block") and attack == "1":
            tr["shi_chain_attacker"] = player
            tr["shi_chain_passed"] = False
            tr["shi_chain_first_passer"] = None

            if self._same_team(player, self.me):
                tr["shi_signal"] += 2.0
                if tr["shi_signal"] >= self.SHI_SIGNAL_ACTIVATE:
                    tr["shi_plan_active"] = True

            if action_type == "attack_after_block" and block == "1" and self._same_team(player, self.me):
                tr["shi_message_sent"] = True
                tr["shi_signal"] += 1.0
                if tr["shi_signal"] >= self.SHI_SIGNAL_ACTIVATE:
                    tr["shi_plan_active"] = True

        if action_type == "pass":
            if state.current_attack == "1" and tr.get("shi_chain_attacker") is not None and tr["shi_chain_attacker"] != player:
                tr["shi_chain_passed"] = True
                if tr.get("shi_chain_first_passer") is None:
                    tr["shi_chain_first_passer"] = player

                if self._same_team(tr["shi_chain_attacker"], self.me):
                    tr["shi_signal"] += 2.0
                    if tr["shi_signal"] >= self.SHI_SIGNAL_ACTIVATE:
                        tr["shi_plan_active"] = True

        if action_type == "receive" and block is not None:
            if block in tr["public_seen_counts"]:
                tr["public_seen_counts"][block] += 1

            if block == "1" and self._same_team(player, self.me):
                tr["shi_signal"] += 1.0
                if tr["shi_signal"] >= self.SHI_SIGNAL_ACTIVATE:
                    tr["shi_plan_active"] = True

        if action_type in ("attack", "attack_after_block") and attack is not None:
            if attack in tr["public_seen_counts"]:
                tr["public_seen_counts"][attack] += 1
                
            # ★ 自分と味方の攻め履歴とシグナル応答をトラッキング
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
            else:
                tr["enemy_past_attacks"].add(attack)

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

        score += self._last_one_remaining_bonus(state, player, attack)
        score += self._occupancy_priority_bonus(state, attack)
        score += self._public_attack_safety_bonus(state, player, attack)

        # ★ 連打（Renda）ボーナスと、味方のスルーによる連打キャンセル判定
        if tr is not None and attack == tr.get("my_last_attack"):
            if tr.get("ally_attacked_since_my_last_attack"):
                if tr.get("ally_last_attack") == attack:
                    # 味方がかかりごたえをしてくれた！超連打で押し切る！
                    score += self.CONTINUOUS_ATTACK_BONUS * 1.5
                else:
                    # 味方が別の駒で攻めた（スルーした）ため、連打ストップ
                    score -= self.CONTINUOUS_ATTACK_BONUS
                    score -= 300.0  # 危険ペナルティ
            else:
                # 味方にターンが回っていない場合は通常の連打
                score += self.CONTINUOUS_ATTACK_BONUS

        # ★ 味方への「遅延かかりごたえ（2枚目の攻めでのシグナル返し）」
        if tr is not None:
            ally_first = tr.get("ally_first_attack")
            if ally_first is not None and attack == ally_first:
                is_unreasonable_block = (action_type == "attack_after_block" and block in ("8", "9"))
                if not is_unreasonable_block:
                    if tr.get("my_attack_count", 0) == 1:
                        # 自分の2回目の攻めで、味方の1枚目（本命）に応える場合は特大ボーナス！
                        score += self.KAKARI_GOTAE_BONUS * 10.0
                    else:
                        score += self.KAKARI_GOTAE_BONUS
            elif attack in tr.get("ally_past_attacks", set()):
                # 味方の2枚目以降の攻めに対する通常のかかりごたえ
                is_unreasonable_block = (action_type == "attack_after_block" and block in ("8", "9"))
                if not is_unreasonable_block:
                    score += self.KAKARI_GOTAE_BONUS

        # 盾割り（Tatewari）＆ 絶対安全（Absolute Safe）の判定
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
                        score += self.TATEWARI_BONUS # 敵の王・玉あぶり出し

        if state.attacker is None and state.current_attack is None and attack == "1":
            score -= 100.0

        if attack in ("9", "8") and has_non_king_attack_option:
            score -= self.KING_ATTACK_PENALTY

        if tr is not None and tr.get("shi_plan_active", False) and attack == "1":
            score += self.SHI_PLAN_ATTACK_FORCE

        score += POINTS.get(attack, 0) / 10.0

        # 伏せ札の評価ロジック
        if action_type == "attack_after_block" and block is not None:
            penalty_table = {"9": 10, "8": 10, "7": 4, "6": 4, "5": 4, "4": 4, "3": 3, "2": 8, "1": 1}
            base_penalty = float(penalty_table.get(block, 0))
            context_penalty = 0.0
            
            if tr is not None:
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
                
            # ★ 新規追加：中盤の強制詰めごいた判定（手札4枚時、反撃が絶対安全なら味方の攻めでも受けて勝つ）
            tr = self._track.get(id(state))
            if tr is not None and len(state.hands[player]) == 4:
                try:
                    s = self._apply_action_on_copy(state, player, (action_type, block, None))
                    next_actions = s.legal_actions(player)
                    for (nt, nb, na) in next_actions:
                        if nt in ("attack", "attack_after_block") and na is not None:
                            if self._is_absolute_safe_for_tsume(state, player, na, tr):
                                return 1e8  # 詰めろ確定！味方の攻めであっても強引に割り込んで勝利を掴む
                except Exception:
                    pass

            if state.attacker is not None and self._same_team(state.attacker, player):
                return -100.0
            base = 1.0 if block in ("8", "9") else 5.0

        tr = self._track.get(id(state))
        if tr is None:
            return base

        if action_type == "receive" and block == "1" and tr.get("shi_plan_active", False):
            base += self.SHI_PLAN_RECEIVE_FORCE

        enemy_attack_turn = (
            state.phase == "receive"
            and state.current_attack is not None
            and state.attacker is not None
            and (not self._same_team(state.attacker, player))
        )

        if enemy_attack_turn and (not tr["first_enemy_attack_seen"]):
            if state.current_attack == "1":
                ones = state.hands[player].count("1")
                strong = self._strong_initial_hand(state)

                is_receive_1 = (action_type == "receive" and block == "1")
                is_receive_not1 = (action_type == "receive" and block != "1")
                if is_receive_not1:
                    return -1e18

                if ones >= 2:
                    return base + (self.FIRST_ENEMY_SHI_FORCE if is_receive_1 else -self.FIRST_ENEMY_SHI_FORCE)

                if ones == 1:
                    if strong:
                        return base + (self.FIRST_ENEMY_SHI_FORCE if is_receive_1 else -self.FIRST_ENEMY_SHI_FORCE)
                    else:
                        if not tr["first_enemy_attack_skipped"]:
                            return base + (self.FIRST_ENEMY_SHI_FORCE if action_type == "pass" else -self.FIRST_ENEMY_SHI_FORCE)
                        return base

            strong = self._strong_initial_hand(state)
            receiving_with_king = (action_type == "receive" and block in ("8", "9"))
            prefer_skip_once = (not strong) or receiving_with_king

            if prefer_skip_once:
                if not tr["first_enemy_attack_skipped"]:
                    base += self.FIRST_ENEMY_PASS_BONUS if action_type == "pass" else -self.FIRST_ENEMY_PASS_BONUS
            else:
                base += -self.FIRST_ENEMY_RECEIVE_BONUS if action_type == "pass" else self.FIRST_ENEMY_RECEIVE_BONUS

        return base

    def select_action(self, state, player: str, actions: List[Action]) -> Action:
        if self.me is None:
            self.me = player
        elif self.me != player:
            raise ValueError(f"{self.name}: bound to me={self.me}, cannot play for {player}")

        self._ensure_trackers(state)
        tr = self._track.get(id(state))

        # ★ 新規追加：王・玉が受けや伏せで失われた場合、だまだまコンボを動的に強制終了
        if tr is not None and tr.get("kg_plan_active"):
            kings_in_hand = state.hands[player].count("8") + state.hands[player].count("9")
            kings_in_past = 1 if "8" in tr.get("my_past_attacks", set()) else 0
            kings_in_past += 1 if "9" in tr.get("my_past_attacks", set()) else 0
            if kings_in_hand + kings_in_past < 2:
                tr["kg_plan_active"] = False

        # ★ パーフェクトゲーム（確定上がり）の計画と実行
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

        has_non_king_attack_option = any(
            (t in ("attack", "attack_after_block")) and (a is not None) and (a not in ("8", "9"))
            for (t, _b, a) in actions
        )

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

        shi_mode = False
        if tr is not None:
            shi_mode = (
                tr.get("shi_plan_active", False)
                or tr.get("shi_signal", 0.0) >= self.SHI_SIGNAL_ACTIVATE
                or state.hands[player].count("1") >= 4
            )

        if tr is not None and shi_mode:
            ally = tr["ally"]
            enemy_attack_turn = (
                state.phase == "receive"
                and state.current_attack is not None
                and state.attacker is not None
                and (not self._same_team(state.attacker, player))
            )

            if enemy_attack_turn and (not tr.get("first_enemy_attack_seen", False)):
                for act in actions:
                    if act[0] == "pass":
                        tr["first_enemy_attack_seen"] = True
                        tr["first_enemy_attack_skipped"] = True
                        return act

            if enemy_attack_turn and tr.get("first_enemy_attack_seen", False):
                recv1 = [act for act in actions if act[0] == "receive" and act[1] == "1"]
                if recv1:
                    chosen = recv1[0]
                    tr["shi_plan_active"] = True
                    tr["shi_signal"] = max(tr.get("shi_signal", 0.0), self.SHI_SIGNAL_ACTIVATE)
                    return chosen

            if state.phase == "receive" and state.current_attack == "1" and state.attacker == ally and tr.get("shi_chain_passed", False):
                my_shi = state.hands[player].count("1")
                if my_shi <= 2:
                    cands = [act for act in actions if act[0] == "attack_after_block" and act[1] == "1" and act[2] is not None and act[2] != "1"]
                    if cands:
                        has_non_king = any((c[2] is not None) and (c[2] not in ("8", "9")) for c in cands)
                        best = cands[0]
                        best_score = -1e18
                        for (t, b, a) in cands:
                            sc = self._score_attack_phase(state, player, t, b, a, has_non_king_attack_option=has_non_king)
                            if sc > best_score:
                                best_score = sc
                                best = (t, b, a)
                        tr["shi_plan_active"] = True
                        tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                        return best
                    for act in actions:
                        if act[0] == "receive" and act[1] == "1":
                            tr["shi_plan_active"] = True
                            return act

                if my_shi == 3:
                    for act in actions:
                        if act[0] == "pass":
                            return act

                if my_shi >= 4:
                    for act in actions:
                        if act[0] == "attack_after_block" and act[1] == "1" and act[2] == "1":
                            tr["shi_plan_active"] = True
                            tr["shi_message_sent"] = True
                            tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                            return act
                    for act in actions:
                        if act[0] == "receive" and act[1] == "1":
                            tr["shi_plan_active"] = True
                            return act

        attack_actions = [(t, b, a) for (t, b, a) in actions if t in ("attack", "attack_after_block") and a is not None]

        if tr is not None and self.KEEP_KING_GYOKU_FOR_LAST_WHEN_TWO_LEFT and attack_actions:
            if len(state.hands[player]) == 2 and any(x in state.hands[player] for x in ("8", "9")):
                non_king_attack_actions = [act for act in attack_actions if act[2] not in ("8", "9")]
                if non_king_attack_actions:
                    best = non_king_attack_actions[0]
                    best_score = -1e18
                    for (t, b, a) in non_king_attack_actions:
                        sc = self._score_attack_phase(state, player, t, b, a, has_non_king_attack_option=True)
                        if sc > best_score:
                            best_score = sc
                            best = (t, b, a)
                    tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                    return best

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
                            if sc > best_score:
                                best_score = sc
                                best = (t, b, a)
                        tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                        return best

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

        if tr is not None and shi_mode and (not tr.get("kg_plan_active", False)) and attack_actions:
            shi_cands = [act for act in attack_actions if act[2] == "1"]
            if shi_cands:
                chosen = shi_cands[0]
                best_score = -1e18
                for (t, b, a) in shi_cands:
                    sc = self._score_attack_phase(state, player, t, b, a, has_non_king_attack_option=has_non_king_attack_option)
                    if sc > best_score:
                        best_score = sc
                        chosen = (t, b, a)
                tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                return chosen

        best_action = actions[0]
        best_score = -1e18

        for (t, block, attack) in actions:
            if t in ("attack", "attack_after_block"):
                score = self._score_attack_phase(state, player, t, block, attack, has_non_king_attack_option=has_non_king_attack_option)
            else:
                score = self._score_receive_phase(state, player, t, block)

            if score > best_score:
                best_score = score
                best_action = (t, block, attack)

        if tr is not None:
            enemy_attack_turn = (
                state.phase == "receive"
                and state.current_attack is not None
                and state.attacker is not None
                and (not self._same_team(state.attacker, player))
            )
            if enemy_attack_turn and (not tr.get("first_enemy_attack_seen", False)):
                tr["first_enemy_attack_seen"] = True
                if best_action[0] == "pass":
                    tr["first_enemy_attack_skipped"] = True

            if best_action[0] in ("attack", "attack_after_block"):
                tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                if tr.get("kg_plan_active") and tr["my_attack_count"] == 2 and best_action[2] in ("8", "9") and tr.get("kg_second") is None:
                    tr["kg_second"] = best_action[2]
                if tr.get("kg_plan_active") and tr["my_attack_count"] >= 3:
                    tr["kg_plan_active"] = False

        return best_action
# goita_ai2/agents/rule_based.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from collections import Counter
import copy

from goita_ai2.state import POINTS  # 基本点（9=50, ... ,1=10）

Action = Tuple[str, Optional[str], Optional[str]]  # (action_type, block, attack)
TARGET_X = ("2", "3", "4", "5")  # 「かかり」対象（4枚駒）
TARGET_LAST1 = ("2", "3", "4", "5", "6", "7")      # ★残り1枚狙い対象


class RuleBasedAgent:
    def __init__(self, name: str = "RuleBased"):
        self.name = name
        self.me: Optional[str] = None

        # 対局(state)ごとのトラッカー
        self._track: Dict[int, dict] = {}
        # 初期手札（占有率用）
        self._initial_hands_by_state_id: Dict[int, Dict[str, List[str]]] = {}

        # ③ 上がり読み（1手先だけ）
        self.WIN_NOW_BONUS = 10_000.0            # この手で上がり
        self.WIN_AFTER_RECEIVE_BONUS = 9_000.0  # 受け→次の攻めで1手上がり

        # ② 9・8（王・玉）の使いどころ：攻めで出すのを遅らせる
        self.KING_ATTACK_PENALTY = 300.0        # 代替攻めがあるなら 8/9 攻めを強く避ける

        # ★最初の「敵の攻め」に対する受け方針
        self.FIRST_ENEMY_RECEIVE_BONUS = 500.0   # 強手札：最初の敵攻めは受けを強く後押し
        self.FIRST_ENEMY_PASS_BONUS = 500.0      # 弱手札 or 8/9受け：最初の敵攻めは1回だけパスを強く後押し

        # ★追加：残り1枚（2-7）を自分が握っているなら攻めで出す
        #   - かかりSTRONG(+120) より下、占有率(3枚=+55) より上くらい
        self.LAST_ONE_BONUS = 65.0

    # ★追加：席を固定する（最初に1回）
    def bind_player(self, player: str) -> None:
        if self.me is None:
            self.me = player
        elif self.me != player:
            raise ValueError(f"{self.name}: already bound to {self.me}, cannot bind to {player}")

    # ----------------------------
    # 基本ユーティリティ
    # ----------------------------
    def _same_team(self, p1: str, p2: str) -> bool:
        return (
            (p1 in ("A", "C") and p2 in ("A", "C")) or
            (p1 in ("B", "D") and p2 in ("B", "D"))
        )

    def _ally_of(self, me: str) -> str:
        return "C" if me == "A" else "A" if me == "C" else "D" if me == "B" else "B"

    def _get_initial_hand(self, state, player: str) -> List[str]:
        sid = id(state)
        if sid not in self._initial_hands_by_state_id:
            self._initial_hands_by_state_id[sid] = {
                p: list(state.hands[p]) for p in ("A", "B", "C", "D")
            }
        return self._initial_hands_by_state_id[sid][player]

    def _ensure_trackers(self, state) -> None:
        sid = id(state)
        if sid in self._track:
            return
        if self.me is None:
            return

        init_hand = self._get_initial_hand(state, self.me)
        cnt_all = Counter(init_hand)

        kakari = {x: "UNCERTAIN" for x in TARGET_X}   # "UNCERTAIN" / "STRONG" / "DEAD"
        enemy_revealed = {x: False for x in TARGET_X} # 相手がXを公開した（bool）
        miss = {x: 0 for x in TARGET_X}               # 味方の最初の攻め機会で外した回数
        supported = {x: False for x in TARGET_X}      # 味方がXで攻めた（強証拠）

        pending_axis: Optional[str] = None            # 自分がX軸を提示した
        pending_ally_received = {"A": None, "B": None, "C": None, "D": None}

        my_init_count = {x: cnt_all.get(x, 0) for x in TARGET_X}

        # ★味方の直近X攻め（かかりごたえ）
        ally_axis_pending: Optional[str] = None

        # ★公開情報として見えた駒の枚数（受けで表／攻めで表）
        public_seen_counts = {str(i): 0 for i in range(1, 10)}

        # 自分が4枚(100%)なら最初からSTRONG
        for x in TARGET_X:
            if my_init_count[x] == 4:
                kakari[x] = "STRONG"

        self._track[sid] = dict(
            kakari=kakari,
            enemy_revealed=enemy_revealed,
            miss=miss,
            supported=supported,
            pending_axis=pending_axis,
            pending_ally_received=pending_ally_received,
            my_init_count=my_init_count,
            init_count_all=cnt_all,
            ally=self._ally_of(self.me),
            ally_axis_pending=ally_axis_pending,

            # ★最初の「敵の攻め」に遭遇したか／スルー済みか
            first_enemy_attack_seen=False,
            first_enemy_attack_skipped=False,

            # ★追加：公開済み枚数
            public_seen_counts=public_seen_counts,
        )

    def _strong_initial_hand(self, state) -> bool:
        """
        初期手札が強いかどうか（ユーザー指定の定義）
          - 2-5 のどれか4枚
          - 6/7 のどちらか2枚
          - 2-5 のどれか3枚
        """
        tr = self._track.get(id(state))
        if tr is None:
            return False

        c_x = tr["my_init_count"]     # 2-5の枚数
        c_all = tr["init_count_all"]  # 全駒の枚数（初期手札）

        for x in ("2", "3", "4", "5"):
            if c_x.get(x, 0) == 4:
                return True
        for x in ("6", "7"):
            if c_all.get(x, 0) == 2:
                return True
        for x in ("2", "3", "4", "5"):
            if c_x.get(x, 0) == 3:
                return True

        return False

    # ★追加：公開情報的に「残り1枚」を自分が握っているなら加点
    def _last_one_remaining_bonus(self, state, player: str, attack: Optional[str]) -> float:
        if attack is None or attack not in TARGET_LAST1:
            return 0.0

        tr = self._track.get(id(state))
        if tr is None:
            return 0.0

        # 自分の現在手札にその駒が無いなら対象外
        if attack not in state.hands[player]:
            return 0.0

        # 総枚数（初期配分）
        total = 4 if attack in ("2", "3", "4", "5") else 2

        seen = tr["public_seen_counts"].get(attack, 0)

        # 公開済みが total-1（=残り1枚が未公開）で、その未公開が自分手札にある
        # → 公開情報的に「残り1枚を自分が握っている」と判断できる
        if seen == total - 1:
            return self.LAST_ONE_BONUS

        return 0.0

    # ----------------------------
    # ③ 上がり読み（1手先だけ）
    # ----------------------------
    def _apply_action_on_copy(self, state, player: str, action: Action):
        """stateをdeepcopyして action を1回だけ適用したstateを返す。"""
        s = copy.deepcopy(state)
        t, block, attack = action

        if t == "pass":
            s.apply_pass(player)
        elif t == "receive":
            if block is None:
                raise ValueError("receive requires block")
            s.apply_receive(player, block)
        elif t == "attack":
            if attack is None:
                raise ValueError("attack requires attack")
            s.apply_attack(player, attack)
        elif t == "attack_after_block":
            if block is None or attack is None:
                raise ValueError("attack_after_block requires block and attack")
            s.apply_attack_after_block(player, block, attack)
        else:
            raise ValueError(f"unknown action type: {t}")

        return s

    def _win_now_bonus(self, state, player: str, action: Action) -> float:
        """この手で上がれるなら超加点。"""
        try:
            s = self._apply_action_on_copy(state, player, action)
        except Exception:
            return 0.0
        return self.WIN_NOW_BONUS if (s.finished and s.winner == player) else 0.0

    def _win_after_receive_bonus(self, state, player: str, action: Action) -> float:
        """
        受けたあと、次の自分の攻めで1手上がりできるなら加点。
        """
        t, block, _ = action
        if t != "receive" or block is None:
            return 0.0

        try:
            s = self._apply_action_on_copy(state, player, action)
        except Exception:
            return 0.0

        try:
            next_actions = s.legal_actions(player)
        except Exception:
            return 0.0

        for (nt, nb, na) in next_actions:
            if nt not in ("attack", "attack_after_block"):
                continue
            try:
                s2 = self._apply_action_on_copy(s, player, (nt, nb, na))
            except Exception:
                continue
            if s2.finished and s2.winner == player:
                return self.WIN_AFTER_RECEIVE_BONUS

        return 0.0

    # ----------------------------
    # 公開情報イベント（simulate.py から呼ぶ）
    # ----------------------------
    def on_public_action(self, state, player: str, action: Action) -> None:
        if self.me is None:
            return

        self._ensure_trackers(state)
        tr = self._track.get(id(state))
        if tr is None:
            return

        action_type, block, attack = action
        ally = tr["ally"]

        # ★公開済み枚数カウント（受けは block が表）
        if action_type == "receive" and block is not None:
            if block in tr["public_seen_counts"]:
                tr["public_seen_counts"][block] += 1

            if (not self._same_team(player, self.me)) and block in TARGET_X:
                tr["enemy_revealed"][block] = True
            if self._same_team(player, self.me) and block in TARGET_X:
                tr["pending_ally_received"][player] = block

        # ★攻めで表に出た駒カウント（attack は表）
        if action_type in ("attack", "attack_after_block") and attack is not None:
            if attack in tr["public_seen_counts"]:
                tr["public_seen_counts"][attack] += 1

            if (not self._same_team(player, self.me)) and attack in TARGET_X:
                tr["enemy_revealed"][attack] = True

            if player == self.me and attack in TARGET_X:
                tr["pending_axis"] = attack

            if player == ally and attack in TARGET_X:
                tr["ally_axis_pending"] = attack

            if player == self.me and tr.get("ally_axis_pending") == attack:
                tr["ally_axis_pending"] = None

            if player == ally:
                pend_recv = tr["pending_ally_received"].get(player)
                if pend_recv in TARGET_X:
                    if attack != pend_recv:
                        tr["miss"][pend_recv] += 1
                    tr["pending_ally_received"][player] = None

            if player == ally and tr["pending_axis"] in TARGET_X:
                x = tr["pending_axis"]
                if attack == x:
                    tr["supported"][x] = True
                else:
                    tr["miss"][x] += 1
                tr["pending_axis"] = None

        for x in TARGET_X:
            if tr["kakari"][x] in ("STRONG", "DEAD"):
                continue

            if tr["supported"][x]:
                tr["kakari"][x] = "STRONG"
                continue

            if tr["my_init_count"][x] == 3 and tr["enemy_revealed"][x]:
                tr["kakari"][x] = "STRONG"
                continue

            if not tr["enemy_revealed"][x]:
                threshold = 1 if tr["my_init_count"][x] == 2 else 2 if tr["my_init_count"][x] == 3 else 999
                if tr["miss"][x] >= threshold:
                    tr["kakari"][x] = "DEAD"

    def _kakari_score(self, state, attack: Optional[str]) -> float:
        if self.me is None or attack is None or attack not in TARGET_X:
            return 0.0
        tr = self._track.get(id(state))
        if tr is None:
            return 0.0

        st = tr["kakari"].get(attack, "UNCERTAIN")
        if st == "STRONG":
            return 120.0
        if st == "DEAD":
            return -120.0
        if tr["miss"].get(attack, 0) == 1:
            return -30.0
        return 0.0

    def _occupancy_priority_bonus(self, state, attack: str) -> float:
        tr = self._track.get(id(state))
        if tr is None:
            return 0.0

        c_x = tr["my_init_count"]
        c_all = tr["init_count_all"]

        if attack in ("2", "3", "4", "5") and c_x.get(attack, 0) == 4:
            return 80.0
        if attack in ("6", "7") and c_all.get(attack, 0) == 2:
            return 70.0
        if attack in ("2", "3", "4", "5") and c_x.get(attack, 0) == 3:
            return 55.0
        if attack in ("2", "3", "4", "5") and c_x.get(attack, 0) == 2:
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

        score += self._kakari_score(state, attack)

        tr = self._track.get(id(state))
        if tr is not None:
            ax = tr.get("ally_axis_pending")
            if ax in TARGET_X and attack == ax:
                score += 90.0

        # ★追加：盤面（公開情報）上「残り1枚」を自分が握っているなら、それを攻めで出す
        score += self._last_one_remaining_bonus(state, player, attack)

        score += self._occupancy_priority_bonus(state, attack)

        if state.attacker is None and state.current_attack is None:
            if attack == "1":
                score -= 100.0

        # ② 王・玉(9/8)は「攻め」で温存（ただし上がりは別枠で最優先）
        if attack in ("9", "8") and has_non_king_attack_option:
            score -= self.KING_ATTACK_PENALTY

        score += POINTS.get(attack, 0) / 10.0

        if action_type == "attack_after_block" and block is not None:
            penalty_table = {"9": 10, "8": 10, "7": 8, "6": 8, "5": 6, "4": 6, "3": 4, "2": 4, "1": 1}
            score -= float(penalty_table.get(block, 0))

        # ★ここを「self.me」ではなく必ず player で判定（席混線に強くする）
        score += self._win_now_bonus(state, player, (action_type, block, attack))

        return score

    def _score_receive_phase(self, state, player: str, action_type: str, block: Optional[str]) -> float:
        """
        受け戦略（今回のシンプル版 + 条件追加）
          - 初期手札が強い かつ 8/9受けでない：最初の「敵の攻め」は受けを強く推奨
          - 初期手札が強くない OR 8/9で受ける場合：最初の「敵の攻め」は1回だけスルー（パス）を強く推奨
          - 味方の攻めは基本止めない（既存 -100 を維持）
          - 受け→次攻めで1手上がりが見えるなら最優先
        """
        # まず、既存のベーススコア
        if action_type == "pass":
            base = 0.0
        else:
            if action_type != "receive" or block is None:
                return -1e18

            # ★ここも player で判定
            bonus = self._win_after_receive_bonus(state, player, (action_type, block, None))
            if bonus > 0:
                return 1e9

            # 味方の攻めは止めない（強ペナルティ）
            if state.attacker is not None and self._same_team(state.attacker, player):
                return -100.0

            # 受け駒の大雑把な好み（既存維持）
            base = 1.0 if block in ("8", "9") else 5.0

        # ★ここから「最初の敵の攻め」だけ特別ルール
        tr = self._track.get(id(state))
        if tr is None:
            return base

        enemy_attack_turn = (
            state.phase == "receive" and
            state.current_attack is not None and
            state.attacker is not None and
            (not self._same_team(state.attacker, player))
        )

        if enemy_attack_turn and (not tr["first_enemy_attack_seen"]):
            strong = self._strong_initial_hand(state)
            # ★追加条件：8/9で受ける場合も「スルー側」に寄せる
            receiving_with_king = (action_type == "receive" and block in ("8", "9"))
            prefer_skip_once = (not strong) or receiving_with_king

            if prefer_skip_once:
                # 弱手札 or 8/9受け：最初の敵攻めは1回だけスルーしたい
                if not tr["first_enemy_attack_skipped"]:
                    if action_type == "pass":
                        base += self.FIRST_ENEMY_PASS_BONUS
                    else:
                        base -= self.FIRST_ENEMY_PASS_BONUS
            else:
                # 強手札（かつ 8/9受けでない）：最初の敵攻めは即受け（パスを避ける）
                if action_type == "pass":
                    base -= self.FIRST_ENEMY_RECEIVE_BONUS
                else:
                    base += self.FIRST_ENEMY_RECEIVE_BONUS

        return base

    def select_action(self, state, player: str, actions: List[Action]) -> Action:
        # ★席が混ざったら即気づけるようにする
        if self.me is None:
            self.me = player
        elif self.me != player:
            raise ValueError(
                f"{self.name}: called with player={player} but this agent is bound to me={self.me}. "
                f"Use separate RuleBasedAgent instances per seat."
            )

        self._ensure_trackers(state)

        has_non_king_attack_option = any(
            (t in ("attack", "attack_after_block")) and (a is not None) and (a not in ("8", "9"))
            for (t, _b, a) in actions
        )

        best_action = actions[0]
        best_score = -1e18

        for (t, block, attack) in actions:
            if t in ("attack", "attack_after_block"):
                score = self._score_attack_phase(
                    state, player, t, block, attack,
                    has_non_king_attack_option=has_non_king_attack_option,
                )
            else:
                score = self._score_receive_phase(state, player, t, block)

            if score > best_score:
                best_score = score
                best_action = (t, block, attack)

        # ★「最初の敵の攻め」遭遇フラグを、選択結果で確定更新
        tr = self._track.get(id(state))
        if tr is not None:
            enemy_attack_turn = (
                state.phase == "receive" and
                state.current_attack is not None and
                state.attacker is not None and
                (not self._same_team(state.attacker, player))
            )
            if enemy_attack_turn and (not tr["first_enemy_attack_seen"]):
                tr["first_enemy_attack_seen"] = True
                if best_action[0] == "pass":
                    tr["first_enemy_attack_skipped"] = True

        return best_action

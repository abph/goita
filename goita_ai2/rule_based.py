from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from collections import Counter
import copy

from goita_ai2.state import POINTS  # 基本点（9=50, ... ,1=10）

Action = Tuple[str, Optional[str], Optional[str]]  # (action_type, block, attack)
TARGET_X = ("2", "3", "4", "5")  # 「かかり」対象（4枚駒）


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
        enemy_revealed = {x: False for x in TARGET_X} # 相手がXを公開した
        miss = {x: 0 for x in TARGET_X}               # 味方の最初の攻め機会で外した回数
        supported = {x: False for x in TARGET_X}      # 味方がXで攻めた（強証拠）

        pending_axis: Optional[str] = None            # 自分がX軸を提示した
        pending_ally_received = {"A": None, "B": None, "C": None, "D": None}

        my_init_count = {x: cnt_all.get(x, 0) for x in TARGET_X}

        # ★味方の直近X攻め（かかりごたえ）
        ally_axis_pending: Optional[str] = None

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
        )

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

        if action_type == "receive" and block is not None:
            if (not self._same_team(player, self.me)) and block in TARGET_X:
                tr["enemy_revealed"][block] = True
            if self._same_team(player, self.me) and block in TARGET_X:
                tr["pending_ally_received"][player] = block

        if action_type in ("attack", "attack_after_block") and attack is not None:
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

    # ----------------------------
    # 受け戦略のための補助（今回追加）
    # ----------------------------
    def _hand_strength01(self, state, player: str) -> float:
        """
        手札の強さを 0〜1 に正規化した指標（大雑把でOK）。
        - 2-5の占有を強く評価
        - 6/7の2枚を評価
        - 1(し)が多いのは微減点
        """
        hand = state.hands[player]
        c = Counter(hand)

        raw = 0.0
        for x in ("2", "3", "4", "5"):
            raw += {4: 3.0, 3: 2.0, 2: 1.0}.get(c.get(x, 0), 0.0)

        for x in ("6", "7"):
            if c.get(x, 0) >= 2:
                raw += 1.2

        raw -= 0.15 * max(0, c.get("1", 0) - 2)

        # だいたい 0〜1 に収める（上限は粗くてOK）
        return max(0.0, min(1.0, raw / 7.0))

    def _ally_block_penalty(self, hs01: float) -> float:
        """
        味方の攻めを止めることへのペナルティ。
        手札が弱いほど「味方を通す」価値が高い。
        """
        if hs01 < 0.45:   # 弱い
            return 600.0
        if hs01 < 0.65:   # 普通
            return 300.0
        return 120.0      # 強い

    def _scarce_receive(self, state, player: str) -> bool:
        """
        「受けられる駒が少ない」= いまの攻めを同駒で受ける手段が乏しい。
        ※8/9万能受けはここでは含めない（温存したいので最後の手段扱い）。
        """
        atk = state.current_attack
        if atk is None:
            return False
        if atk in ("1", "2", "3", "4", "5", "6", "7"):
            return state.hands[player].count(atk) <= 1
        return False

    def _best_attack_score_after_receive(self, s, player: str) -> float:
        """
        受けた後の自分の攻めの最大スコア（=受けで主導権を取る価値）。
        """
        try:
            actions = s.legal_actions(player)
        except Exception:
            return -1e18

        has_non_king = any(
            (t in ("attack", "attack_after_block")) and (a is not None) and (a not in ("8", "9"))
            for (t, _b, a) in actions
        )

        best = -1e18
        for (t, b, a) in actions:
            if t not in ("attack", "attack_after_block"):
                continue
            best = max(
                best,
                self._score_attack_phase(
                    s, player, t, b, a,
                    has_non_king_attack_option=has_non_king,
                )
            )
        return best

    def _score_receive_phase(self, state, player: str, action_type: str, block: Optional[str]) -> float:
        # pass
        if action_type == "pass":
            return 0.0

        # receive
        if action_type != "receive" or block is None:
            return -1e18

        # ★ここも player で判定：受け→次の攻めで上がりが見えるなら最優先
        bonus = self._win_after_receive_bonus(state, player, (action_type, block, None))
        if bonus > 0:
            return 1e9

        hs = self._hand_strength01(state, player)

        # 味方の攻めを止めるのは原則NG（手札が弱いほどNGを強める）
        ally_pen = 0.0
        if state.attacker is not None and self._same_team(state.attacker, player):
            ally_pen = self._ally_block_penalty(hs)

            # 例外：手札が強いのに、受けられる駒が少ない局面は
            # 「今逃すと攻め権を取りにくい」ので止めても受けを許す
            if hs >= 0.65 and self._scarce_receive(state, player):
                ally_pen *= 0.25

        # 受け札のコスト（温存重視：弱い手札ほど強く効かせる）
        cost = 0.0
        if block in ("2", "3", "4", "5"):
            cost += 35.0
        elif block in ("6", "7"):
            cost += 25.0
        elif block in ("8", "9"):
            cost += 45.0
        elif block == "1":
            cost += 5.0

        if hs < 0.45:
            cost *= 1.35  # 弱い→温存重視
        elif hs >= 0.65:
            cost *= 0.75  # 強い→攻め権重視

        # 受けた後の「攻めの強さ」を評価に入れる
        try:
            s = self._apply_action_on_copy(state, player, ("receive", block, None))
        except Exception:
            return -1e18
        future_attack = self._best_attack_score_after_receive(s, player)

        # 合成：係数0.08は「攻め評価のスケールが大きい」前提の控えめ係数
        return 0.08 * future_attack - cost - ally_pen

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

        return best_action

# goita_ai2/agents/rule_based.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from collections import Counter
import copy

from goita_ai2.state import POINTS  # 基本点（9=50, ... ,1=10）

Action = Tuple[str, Optional[str], Optional[str]]  # (action_type, block, attack)

TARGET_X = ("2", "3", "4", "5")                 # 「かかり」対象（4枚駒）
TARGET_LAST1 = ("2", "3", "4", "5", "6", "7")   # 公開情報的に「残り1枚」を狙う対象


class RuleBasedAgent:
    """
    ルールベースAI（攻め/受け）

    重要: 3枚目の攻めで 8/9 が攻めとして合法なら基本は 8/9 を出す。
          ただし「確定で通る」非8/9（次の受け手が“どの受け駒でも受けられない”）がある場合はそれを優先。
          なお、8/9(王玉)は 1/2 以外を受けられるため、確定判定はそれを考慮する。
    """

    def __init__(self, name: str = "RuleBased"):
        self.name = name
        self.me: Optional[str] = None

        # 対局(state)ごとのトラッカー
        self._track: Dict[int, dict] = {}
        # 初期手札（占有率用）
        self._initial_hands_by_state_id: Dict[int, Dict[str, List[str]]] = {}

        # --- 上がり読み（1手先だけ） ---
        self.WIN_NOW_BONUS = 10_000.0            # この手で上がり
        self.WIN_AFTER_RECEIVE_BONUS = 9_000.0  # 受け→次の攻めで1手上がり

        # --- 王・玉（8/9）の扱い ---
        self.KING_ATTACK_PENALTY = 300.0        # 代替攻めがあるなら 8/9 攻めを避ける（通常時）
        self.KING_GYOKU_FORCE_ORDER = True      # 初期に8/9両方持ちなら、2枚目→3枚目で順に出す

        # 3枚目の攻め：8/9強制（ただし真の確定手があればそちらを優先）
        self.FORCE_KING_GYOKU_ON_THIRD_ATTACK = True
        self.PREFER_TRULY_UNRECEIVABLE_ON_THIRD_ATTACK = True

        # 残り2枚で 8/9 が含まれるなら最後まで温存（即上がりは例外）
        self.KEEP_KING_GYOKU_FOR_LAST_WHEN_TWO_LEFT = True

        # --- 最初の敵の攻め（受け） ---
        self.FIRST_ENEMY_RECEIVE_BONUS = 500.0
        self.FIRST_ENEMY_PASS_BONUS = 500.0
        self.FIRST_ENEMY_SHI_FORCE = 800.0

        # --- 公開情報 残り1枚ボーナス ---
        self.LAST_ONE_BONUS = 65.0

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
        enemy_revealed = {x: False for x in TARGET_X}
        miss = {x: 0 for x in TARGET_X}
        supported = {x: False for x in TARGET_X}

        pending_axis: Optional[str] = None
        pending_ally_received = {"A": None, "B": None, "C": None, "D": None}
        my_init_count = {x: cnt_all.get(x, 0) for x in TARGET_X}

        ally_axis_pending: Optional[str] = None

        public_seen_counts = {str(i): 0 for i in range(1, 10)}

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

            first_enemy_attack_seen=False,
            first_enemy_attack_skipped=False,

            public_seen_counts=public_seen_counts,

            # 攻め回数は on_public_action に依存せず select_action 内で更新する
            my_attack_count=0,

            # 初期に8/9両方持ちなら 2→3枚目プランON
            kg_plan_active=(("9" in init_hand) and ("8" in init_hand)),
            kg_second=None,
        )

    def _strong_initial_hand(self, state) -> bool:
        tr = self._track.get(id(state))
        if tr is None:
            return False

        c_x = tr["my_init_count"]
        c_all = tr["init_count_all"]

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

    # ----------------------------
    # 公開情報：残り1枚ボーナス
    # ----------------------------
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

    # ----------------------------
    # 1手読み（上がり）
    # ----------------------------
    def _apply_action_on_copy(self, state, player: str, action: Action):
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
        try:
            s = self._apply_action_on_copy(state, player, action)
        except Exception:
            return 0.0
        return self.WIN_NOW_BONUS if (s.finished and s.winner == player) else 0.0

    def _win_after_receive_bonus(self, state, player: str, action: Action) -> float:
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
    # 公開情報イベント
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

        # 受けで表に出た駒
        if action_type == "receive" and block is not None:
            if block in tr["public_seen_counts"]:
                tr["public_seen_counts"][block] += 1

            if (not self._same_team(player, self.me)) and block in TARGET_X:
                tr["enemy_revealed"][block] = True
            if self._same_team(player, self.me) and block in TARGET_X:
                tr["pending_ally_received"][player] = block

        # 攻めで表に出た駒
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

    # ----------------------------
    # 攻め評価
    # ----------------------------
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

        # ユーザー定義：初期手札の強さを攻め優先に反映
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

        score += self._last_one_remaining_bonus(state, player, attack)
        score += self._occupancy_priority_bonus(state, attack)

        # 初手の攻めで「し」は避ける
        if state.attacker is None and state.current_attack is None:
            if attack == "1":
                score -= 100.0

        # 王玉は通常時は温存（ただし select_action の強制が優先）
        if attack in ("9", "8") and has_non_king_attack_option:
            score -= self.KING_ATTACK_PENALTY

        # 点数ちょい加点
        score += POINTS.get(attack, 0) / 10.0

        # ブロック消費ペナルティ（軽く）
        if action_type == "attack_after_block" and block is not None:
            penalty_table = {"9": 10, "8": 10, "7": 8, "6": 8, "5": 6, "4": 6, "3": 4, "2": 4, "1": 1}
            score -= float(penalty_table.get(block, 0))

        # 上がり超優先
        score += self._win_now_bonus(state, player, (action_type, block, attack))

        return score

    # ----------------------------
    # 受け評価
    # ----------------------------
    def _score_receive_phase(self, state, player: str, action_type: str, block: Optional[str]) -> float:
        if action_type == "pass":
            base = 0.0
        else:
            if action_type != "receive" or block is None:
                return -1e18

            bonus = self._win_after_receive_bonus(state, player, (action_type, block, None))
            if bonus > 0:
                return 1e9

            # 味方の攻めは止めない（例外は今は入れない）
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

        # 最初の敵の攻め ルール
        if enemy_attack_turn and (not tr["first_enemy_attack_seen"]):
            # し(1)専用
            if state.current_attack == "1":
                ones = state.hands[player].count("1")
                strong = self._strong_initial_hand(state)

                is_receive_1 = (action_type == "receive" and block == "1")
                if action_type == "receive" and block != "1":
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

    # ----------------------------
    # 「真に確定で通る」判定
    # ----------------------------
    def _defender_can_receive_attack(self, defender_hand: List[str], attack: str) -> bool:
        """
        defender_hand が attack を何かしらで受けられるなら True。

        仕様:
          - 同種があれば受けられる
          - 8/9(王玉)は 1/2 以外を受けられる
        """
        if attack in defender_hand:
            return True
        if attack not in ("1", "2") and ("8" in defender_hand or "9" in defender_hand):
            return True
        return False

    def _best_truly_unreceivable_attack_action(self, state, player: str, attack_actions: List[Action]) -> Optional[Action]:
        """
        次の受け手が“どの受け駒でも受けられない”攻めがあればそれを返す。
        8/9 が受けに使える点を考慮した「真の確定」。
        """
        # next_player の定義揺れに耐える
        try:
            defender = state.next_player(player)  # type: ignore
        except TypeError:
            defender = type(state).next_player(player)

        defender_hand = state.hands[defender]

        cands: List[Action] = []
        for act in attack_actions:
            a = act[2]
            if a is None:
                continue
            if self._defender_can_receive_attack(defender_hand, a):
                continue
            cands.append(act)

        if not cands:
            return None

        # 8/9は温存したいので、可能なら非8/9を優先
        non_king = [x for x in cands if x[2] not in ("8", "9")]
        pool = non_king if non_king else cands

        # 同種が多い & 点が高いを優先（簡易）
        def key(act: Action):
            a = act[2]
            return (state.hands[player].count(a), POINTS.get(a, 0))

        return sorted(pool, key=key, reverse=True)[0]

    # ----------------------------
    # 行動選択
    # ----------------------------
    def select_action(self, state, player: str, actions: List[Action]) -> Action:
        # 席固定
        if self.me is None:
            self.me = player
        elif self.me != player:
            raise ValueError(
                f"{self.name}: called with player={player} but this agent is bound to me={self.me}. "
                f"Use separate RuleBasedAgent instances per seat."
            )

        self._ensure_trackers(state)
        tr = self._track.get(id(state))

        has_non_king_attack_option = any(
            (t in ("attack", "attack_after_block")) and (a is not None) and (a not in ("8", "9"))
            for (t, _b, a) in actions
        )

        # 0) 即上がり最優先
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

        # 攻め候補
        attack_actions = [(t, b, a) for (t, b, a) in actions if t in ("attack", "attack_after_block") and a is not None]

        # 1) 残り2枚で8/9含むなら 8/9を温存（非8/9が出せるならそれ）
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
                    chosen = best
                    tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                    return chosen

        # 次の攻めが何枚目か
        next_attack_no = int(tr.get("my_attack_count", 0)) + 1 if tr is not None else 999

        # 2) 初期8/9両方持ちなら 2→3枚目で順に出す
        if tr is not None and tr.get("kg_plan_active") and self.KING_GYOKU_FORCE_ORDER and attack_actions:
            if next_attack_no in (2, 3):
                hand = state.hands[player]
                has9 = "9" in hand
                has8 = "8" in hand

                if has8 or has9:
                    if next_attack_no == 2:
                        for p in ["9", "8"]:
                            if p == "9" and not has9:
                                continue
                            if p == "8" and not has8:
                                continue
                            for act in attack_actions:
                                if act[2] == p:
                                    chosen = act
                                    tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                                    if chosen[2] in ("8", "9") and tr.get("kg_second") is None:
                                        tr["kg_second"] = chosen[2]
                                    return chosen

                    if next_attack_no == 3:
                        second = tr.get("kg_second")
                        want = "8" if second == "9" else "9" if second == "8" else None
                        if want is not None:
                            for act in attack_actions:
                                if act[2] == want:
                                    chosen = act
                                    tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                                    tr["kg_plan_active"] = False
                                    return chosen

                        # 2枚目で出せてない場合：出せる方
                        for p in ["9", "8"]:
                            for act in attack_actions:
                                if act[2] == p:
                                    chosen = act
                                    tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                                    tr["kg_plan_active"] = False
                                    return chosen

        # 3) ★3枚目：8/9が出せるなら基本は 8→9
        #    ただし「真の確定で通る」非8/9があるならそれを優先
        if tr is not None and next_attack_no == 3 and attack_actions:
            has_king_attack = any(act[2] in ("8", "9") for act in attack_actions)

            if has_king_attack:
                if self.PREFER_TRULY_UNRECEIVABLE_ON_THIRD_ATTACK:
                    unrecv = self._best_truly_unreceivable_attack_action(state, player, attack_actions)
                    if unrecv is not None and unrecv[2] not in ("8", "9"):
                        chosen = unrecv
                        tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                        return chosen

                if self.FORCE_KING_GYOKU_ON_THIRD_ATTACK:
                    for p in ["8", "9"]:
                        for act in attack_actions:
                            if act[2] == p:
                                chosen = act
                                tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                                if tr.get("kg_plan_active"):
                                    tr["kg_plan_active"] = False
                                return chosen

        # 4) 通常スコアリング
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

        # 最初の敵の攻めフラグ更新
        if tr is not None:
            enemy_attack_turn = (
                state.phase == "receive" and
                state.current_attack is not None and
                state.attacker is not None and
                (not self._same_team(state.attacker, player))
            )
            if enemy_attack_turn and (not tr.get("first_enemy_attack_seen", False)):
                tr["first_enemy_attack_seen"] = True
                if best_action[0] == "pass":
                    tr["first_enemy_attack_skipped"] = True

            # 攻め回数を内部更新
            if best_action[0] in ("attack", "attack_after_block"):
                tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                if tr.get("kg_plan_active") and tr["my_attack_count"] == 2 and best_action[2] in ("8", "9") and tr.get("kg_second") is None:
                    tr["kg_second"] = best_action[2]
                if tr.get("kg_plan_active") and tr["my_attack_count"] >= 3:
                    tr["kg_plan_active"] = False

        return best_action

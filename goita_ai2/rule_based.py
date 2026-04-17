# goita_ai2/agents/rule_based.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from collections import Counter
import copy

from goita_ai2.state import POINTS  # 基本点（9=50, ... ,1=10）

Action = Tuple[str, Optional[str], Optional[str]]  # (action_type, block, attack)
TARGET_X = ("2", "3", "4", "5")  # 「かかり」対象（4枚駒）
TARGET_LAST1 = ("2", "3", "4", "5", "6", "7")      # 残り1枚狙い対象


class RuleBasedAgent:
    def __init__(self, name: str = "RuleBased"):
        self.name = name
        self.me: Optional[str] = None

        self._track: Dict[int, dict] = {}
        self._initial_hands_by_state_id: Dict[int, Dict[str, List[str]]] = {}

        self.WIN_NOW_BONUS = 10_000.0
        self.WIN_AFTER_RECEIVE_BONUS = 9_000.0

        self.KING_ATTACK_PENALTY = 300.0

        self.FIRST_ENEMY_RECEIVE_BONUS = 500.0
        self.FIRST_ENEMY_PASS_BONUS = 500.0

        self.LAST_ONE_BONUS = 65.0
        self.FIRST_ENEMY_SHI_FORCE = 800.0

        self.KING_GYOKU_FORCE_ORDER = True
        self.FORCE_KING_GYOKU_ON_THIRD_ATTACK = True

        # ★NEW：3枚目8/9強制よりも優先する「確定で通る攻め」があればそっち
        self.PREFER_UNRECEIVABLE_ON_THIRD_ATTACK = True

        # ★NEW：残り2枚で8/9が含まれるなら、8/9は最後まで温存（即上がりは例外）
        self.KEEP_KING_GYOKU_FOR_LAST_WHEN_TWO_LEFT = True

        # ===== "し"(=駒"1") 攻め戦略 =====
        # しプラン資格：自分＋ペアの「し」合計がこの枚数以上なら、し攻めが理論上成立
        self.SHI_PLAN_MIN_TEAM_COUNT = 6

        # しプラン中の行動優先（強めに効かせる）
        self.SHI_PLAN_ATTACK_FORCE = 2_000.0   # しで攻める（プラン中）
        self.SHI_PLAN_RECEIVE_FORCE = 2_000.0  # しで受ける（プラン中）

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

        ally_player = self._ally_of(self.me)
        ally_init_hand = self._get_initial_hand(state, ally_player)
        ally_cnt_all = Counter(ally_init_hand)

        shi_team_total = int(cnt_all.get("1", 0) + ally_cnt_all.get("1", 0))
        shi_plan_eligible = (shi_team_total >= self.SHI_PLAN_MIN_TEAM_COUNT)

        kakari = {x: "UNCERTAIN" for x in TARGET_X}
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
            ally=ally_player,
            ally_axis_pending=ally_axis_pending,
            first_enemy_attack_seen=False,
            first_enemy_attack_skipped=False,
            public_seen_counts=public_seen_counts,

            # ★自分の攻め回数
            # ===== "し"(=1) 攻め戦略のトラッキング =====
            shi_team_total=shi_team_total,
            shi_plan_eligible=shi_plan_eligible,
            shi_plan_active=False,          # 発火条件で True にする
            shi_message_sent=False,         # 「本気でし攻め」メッセージ送信済み
            shi_chain_attacker=None,        # 直近の「し攻め」をしたプレイヤー
            shi_chain_passed=False,         # 直近の「し攻め」に対して誰かがパスしたか
            shi_chain_first_passer=None,    # 最初にパスしたプレイヤー

            # （agentが選択した攻め回数）
            my_attack_count=0,

            # ★初期に8/9両方持ちのときだけ2→3枚目プランON
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

    def on_public_action(self, state, player: str, action: Action) -> None:
        if self.me is None:
            return
        self._ensure_trackers(state)
        tr = self._track.get(id(state))
        if tr is None:
            return

        action_type, block, attack = action
        ally = tr["ally"]

        # ===== "し"(=1) 攻め戦略：公開情報からのトラッキング =====
        if action_type in ("attack", "attack_after_block") and attack == "1":
            tr["shi_chain_attacker"] = player
            tr["shi_chain_passed"] = False
            tr["shi_chain_first_passer"] = None
            # 味方が「し」で攻め始めたら、しプランを共有してONにする（資格がある局のみ）
            if tr.get("shi_plan_eligible") and self._same_team(player, self.me):
                tr["shi_plan_active"] = True

            # 「しで受けて、さらにしで攻めた」= 強いメッセージ
            if action_type == "attack_after_block" and block == "1" and tr.get("shi_plan_eligible") and self._same_team(player, self.me):
                tr["shi_message_sent"] = True

        if action_type == "pass":
            # 直近の「し攻め」に対して誰かがパスした（=し消耗戦が効いているシグナル）
            if state.current_attack == "1" and tr.get("shi_chain_attacker") is not None and tr["shi_chain_attacker"] != player:
                tr["shi_chain_passed"] = True
                if tr.get("shi_chain_first_passer") is None:
                    tr["shi_chain_first_passer"] = player

        if action_type == "receive" and block is not None:
            if block in tr["public_seen_counts"]:
                tr["public_seen_counts"][block] += 1
            if (not self._same_team(player, self.me)) and block in TARGET_X:
                tr["enemy_revealed"][block] = True
            if self._same_team(player, self.me) and block in TARGET_X:
                tr["pending_ally_received"][player] = block

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

        score += self._last_one_remaining_bonus(state, player, attack)
        score += self._occupancy_priority_bonus(state, attack)

        if state.attacker is None and state.current_attack is None and attack == "1":
            score -= 100.0

        if attack in ("9", "8") and has_non_king_attack_option:
            score -= self.KING_ATTACK_PENALTY

        score += POINTS.get(attack, 0) / 10.0

        if action_type == "attack_after_block" and block is not None:
            penalty_table = {"9": 10, "8": 10, "7": 8, "6": 8, "5": 6, "4": 6, "3": 4, "2": 4, "1": 1}
            score -= float(penalty_table.get(block, 0))

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


    # ★NEW：次プレイヤー（次の受け手）がその攻めを「何でも受けられない」= 確定で通る攻めを選ぶ
    # 注意：受け手は同種の駒だけでなく、王/玉(9/8)でも（攻めが1/2以外なら）受けられるため、
    #       「同種を持っていない = 確定」とはならない。
    def _defender_can_receive_attack(self, defender_hand: List[str], attack: str) -> bool:
        """defender_hand が attack を何かしらで受けられるなら True"""
        # 同種で受けられる
        if attack in defender_hand:
            return True
        # 王/玉(9/8)は 1/2 以外を受けられる（state.py の受け条件に合わせる）
        if attack not in ("1", "2") and ("8" in defender_hand or "9" in defender_hand):
            return True
        return False

    def _next_player(self, state, player: str) -> str:
        """state.next_player の実装差（staticmethod/classmethod/instance method）を吸収して次手番を返す。"""
        try:
            return state.next_player(player)  # type: ignore[attr-defined]
        except TypeError:
            return type(state).next_player(player)  # type: ignore[attr-defined]

    def _best_unreceivable_attack_action(self, state, player: str, attack_actions: List[Action]) -> Optional[Action]:
        """次の受け手が“何でも受けられない”攻め（=確定で通る）を選ぶ。なければ None。"""
        defender = self._next_player(state, player)
        defender_hand = state.hands[defender]

        cands: List[Action] = []
        for act in attack_actions:
            a = act[2]
            if a is None:
                continue
            if self._defender_can_receive_attack(defender_hand, a):
                continue  # 受けられるので「確定で通る」ではない
            cands.append(act)

        if not cands:
            return None

        # 優先：8/9は温存したいので除外できるなら除外（ただし候補がそれしか無いなら許す）
        non_king = [x for x in cands if x[2] not in ("8", "9")]
        pool = non_king if non_king else cands

        # スコア：手札に同種が多いほど良い / 点が高いほど良い（シンプル決め打ち）
        def key(act: Action):
            a = act[2]
            return (state.hands[player].count(a), POINTS.get(a, 0))

        return sorted(pool, key=key, reverse=True)[0]

    def select_action(self, state, player: str, actions: List[Action]) -> Action:
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

        # 0) この手で上がれるなら最優先
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

        # ===== "し"(=1) 攻め戦略：局中の強制判断（資格がある局のみ） =====
        if tr is not None and tr.get("shi_plan_eligible", False):
            ally = tr["ally"]

            enemy_attack_turn = (
                state.phase == "receive"
                and state.current_attack is not None
                and state.attacker is not None
                and (not self._same_team(state.attacker, player))
            )

            # (S1) 相手チームが親の初回攻め：一度パスして「相手の伏せ」にしを消費させる
            if (not tr.get("shi_plan_active", False)) and enemy_attack_turn and (not tr.get("first_enemy_attack_seen", False)):
                # 「初回はパス」が目的なので、相手の攻め駒が何であっても基本パス（※即上がり系は上で処理済）
                for act in actions:
                    if act[0] == "pass":
                        # ★FIX: ここで即 return すると first_enemy_attack_seen が立たず、以降ずっと『初回扱い』になってパスし続ける事故が起きる。
                        tr["first_enemy_attack_seen"] = True
                        tr["first_enemy_attack_skipped"] = True
                        return act

            # (S2) 相手の攻めを一度見た後：しで受けてプラン発火（「伏せがし」になりやすい読み）
            if (not tr.get("shi_plan_active", False)) and enemy_attack_turn and tr.get("first_enemy_attack_seen", False):
                recv1 = [act for act in actions if act[0] in ("receive", "attack_after_block") and act[1] == "1"]
                if recv1:
                    # 受け（または受けて即攻め）で「し」を出した時点で、しプランON
                    chosen = sorted(recv1, key=lambda a: 0 if a[0] == "receive" else 1)[0]
                    tr["shi_plan_active"] = True
                    if chosen[0] in ("attack", "attack_after_block"):
                        tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                    return chosen

            # (S3) 連携：味方が「し」で攻め、直前の受け手がパス → 自分（ペア）の受け番
            if state.phase == "receive" and state.current_attack == "1" and state.attacker == ally and tr.get("shi_chain_passed", False):
                my_shi = state.hands[player].count("1")

                # し<=2：しでは攻め切れない → しで受けて、強い駒で返す（可能なら attack_after_block）
                if my_shi <= 2:
                    cands = [act for act in actions if act[0] == "attack_after_block" and act[1] == "1" and act[2] is not None and act[2] != "1"]
                    if cands:
                        # 強い駒で返す：通常の攻めスコアで最大を選ぶ（し以外）
                        has_non_king = any((c[2] is not None) and (c[2] not in ("8", "9")) for c in cands)
                        best = cands[0]
                        best_score = -1e18
                        for (t, b, a) in cands:
                            sc = self._score_attack_phase(state, player, t, b, a, has_non_king_attack_option=has_non_king)
                            if sc > best_score:
                                best_score = sc
                                best = (t, b, a)
                        tr["shi_plan_active"] = True  # 受けでしを出す＝しプラン共有
                        tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                        return best

                    # attack_after_block が無い場合は、とにかく「し」で受ける
                    for act in actions:
                        if act[0] == "receive" and act[1] == "1":
                            tr["shi_plan_active"] = True
                            return act

                # し==3：そのままパス（「しで攻めることは可能」というメッセージ）
                if my_shi == 3:
                    for act in actions:
                        if act[0] == "pass":
                            return act

                # し>=4：しで受けて、しで返す（強いメッセージ）
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

        # ★B：残り2枚で8/9が含まれるなら、8/9は最後まで温存（今は出さない）
        if tr is not None and self.KEEP_KING_GYOKU_FOR_LAST_WHEN_TWO_LEFT and attack_actions:
            # 攻めに出す局面の「自分の残り手札枚数」で判定（攻め直前の手札）
            if len(state.hands[player]) == 2 and any(x in state.hands[player] for x in ("8", "9")):
                # 8/9以外の攻めが合法なら、それを優先して選ばせる（※即上がりは上で処理済）
                non_king_attack_actions = [act for act in attack_actions if act[2] not in ("8", "9")]
                if non_king_attack_actions:
                    # 通常評価で非8/9から選ぶ
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
                # 非8/9が無いなら仕方ない（8/9を出すしかない）→下へ

        # 1) 初期に両方持ち：2枚目→3枚目の強制
        if tr is not None and tr.get("kg_plan_active") and self.KING_GYOKU_FORCE_ORDER:
            next_attack_no = int(tr.get("my_attack_count", 0)) + 1
            if attack_actions and next_attack_no in (2, 3):
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
                        for p in ["9", "8"]:
                            for act in attack_actions:
                                if act[2] == p:
                                    chosen = act
                                    tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                                    tr["kg_plan_active"] = False
                                    return chosen

        # ★A：3枚目の攻めで8/9を出せる局面でも「確定で通る非8/9」があるならそっち
        if tr is not None and self.PREFER_UNRECEIVABLE_ON_THIRD_ATTACK and attack_actions:
            next_attack_no = int(tr.get("my_attack_count", 0)) + 1
            if next_attack_no == 3:
                # 3枚目で 8/9 も出せる状況か？
                has_king_attack = any(act[2] in ("8", "9") for act in attack_actions)
                if has_king_attack:
                    unrecv = self._best_unreceivable_attack_action(state, player, attack_actions)
                    if unrecv is not None and unrecv[2] not in ("8", "9"):
                        chosen = unrecv
                        tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                        return chosen

        # 2) 汎用：3枚目の攻めで8/9が出せるなら必ず出す（8→9）
        if tr is not None and self.FORCE_KING_GYOKU_ON_THIRD_ATTACK and attack_actions:
            next_attack_no = int(tr.get("my_attack_count", 0)) + 1
            if next_attack_no == 3:
                for p in ["8", "9"]:
                    for act in attack_actions:
                        if act[2] == p:
                            chosen = act
                            tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                            if tr.get("kg_plan_active"):
                                tr["kg_plan_active"] = False
                            return chosen


        # ===== "し"(=1) 攻め戦略：攻め番での貫徹（kg_plan_active と味方軸は優先） =====
        if tr is not None and tr.get("shi_plan_active", False) and (not tr.get("kg_plan_active", False)) and attack_actions:
            # (S4) 味方の2〜5軸合わせは「し」より優先
            ax = tr.get("ally_axis_pending")
            if ax in TARGET_X:
                ax_cands = [act for act in attack_actions if act[2] == ax]
                if ax_cands:
                    chosen = ax_cands[0]
                    best_score = -1e18
                    for (t, b, a) in ax_cands:
                        sc = self._score_attack_phase(state, player, t, b, a, has_non_king_attack_option=has_non_king_attack_option)
                        if sc > best_score:
                            best_score = sc
                            chosen = (t, b, a)
                    tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                    return chosen

            # (S5) それ以外は、原則「し」で攻める
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


        # 3) 通常スコアリング
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

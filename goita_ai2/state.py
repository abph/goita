# goita_ai2/state.py
from typing import Dict, List, Tuple, Optional


POINTS: Dict[str, int] = {
    "9": 50,  # 王
    "8": 50,  # 玉
    "7": 40,  # 飛
    "6": 40,  # 角
    "5": 30,  # 金
    "4": 30,  # 銀
    "3": 20,  # 馬
    "2": 20,  # 香
    "1": 10,  # し
}


class GoitaState:
    """
    ごいた1局分のゲーム状態を表すクラス。
    ルールは goita.jp 第2版 PDF に準拠（あなたとすり合わせた内容）。
    """

    def __init__(self, hands: Dict[str, List[str]], dealer: str = "A"):
        """
        hands: {"A": [...], "B": [...], "C": [...], "D": [...]}, 各 8 枚
               駒は "1"〜"9" の文字列
        dealer: 親プレイヤー（"A"〜"D"）
        """
        self.hands: Dict[str, List[str]] = {
            "A": list(hands["A"]),
            "B": list(hands["B"]),
            "C": list(hands["C"]),
            "D": list(hands["D"]),
        }


        # 「両王持ち」フラグ：配牌時点で 8(玉) と 9(王) を両方持っていたか
        # いったん両方持っていれば、その後どちらかを使って片方だけになっても
        # もう片方の 8/9 を攻めに使える扱いにする（期待挙動の修正）。
        self.had_both_kings: Dict[str, bool] = {
            p: (("8" in self.hands[p]) and ("9" in self.hands[p]))
            for p in ("A", "B", "C", "D")
        }

        # 伏せ札（中身は内部でのみ使用、後でobservation側で隠す）
        self.face_down_hidden: Dict[str, List[str]] = {
            "A": [],
            "B": [],
            "C": [],
            "D": [],
        }

        # 現在場に出ている攻めの駒（なければ None）
        self.current_attack: Optional[str] = None

        # その攻めを出したプレイヤー（なければ None）
        self.attacker: Optional[str] = None

        # フェーズ: "receive" or "attack"
        # 親の最初は攻めフェーズ（伏せ→攻め）なので "attack"
        self.phase: str = "attack"

        # 現在の手番プレイヤー
        self.turn: str = dealer

        # ダブル判定用：「最後に伏せた駒」と「そのプレイヤー」
        self.last_block: Optional[str] = None
        self.last_block_player: Optional[str] = None

        # 「受けとして 8/9 が表で出た回数」
        self.king_block_used: int = 0

        # 終局フラグと勝者
        self.finished: bool = False
        self.winner: Optional[str] = None

        # チーム得点（AC チーム / BD チーム）
        self.team_score: Dict[str, int] = {"AC": 0, "BD": 0}

    # ================================
    # ユーティリティ
    # ================================
    @staticmethod
    def next_player(p: str) -> str:
        order = ["A", "B", "C", "D"]
        i = order.index(p)
        return order[(i + 1) % 4]

    # ================================
    # 受け判定
    # ================================
    def can_receive(self, player: str, block: str) -> bool:
        """
        player が block で current_attack を受けられるかどうか。
        8/9 は 1,2 以外を全て受けられる。
        それ以外の駒は同種でのみ受けられる。
        """
        if self.current_attack is None:
            return False

        attack = self.current_attack

        if block in ("8", "9"):
            # 8/9 は 1,2 以外の攻めを受けられる
            return attack not in ("1", "2")

        return block == attack

    # ================================
    # 行動適用メソッド
    # ================================
    def apply_receive(self, player: str, block: str) -> None:
        """player が block で current_attack を受ける。"""
        if self.phase != "receive":
            raise ValueError("Receive is only allowed in receive phase")
        if player != self.turn:
            raise ValueError(f"It is not {player}'s turn")
        if not self.can_receive(player, block):
            raise ValueError(f"{player} cannot receive {self.current_attack} with {block}")

        hand = self.hands[player]
        if block not in hand:
            raise ValueError(f"{player} does not have block piece {block}")
        hand.remove(block)

        # 受けとして 8/9 が出たらフラグ更新
        if block in ("8", "9"):
            self.king_block_used += 1

        # 攻めをクリア
        self.current_attack = None

        # 攻撃元は受けたプレイヤーに移動
        self.attacker = player
        self.phase = "attack"
        self.turn = player  # 受けた人が続けて攻める

    def apply_pass(self, player: str) -> None:
        """player がパスする。"""
        if self.phase != "receive":
            raise ValueError("Pass is only allowed in receive phase")
        if player != self.turn:
            raise ValueError(f"It is not {player}'s turn")

        next_p = self.next_player(player)
        self.turn = next_p

        # まだ攻撃元に戻っていない → 次の人の受けフェーズが続く
        if self.attacker is None or next_p != self.attacker:
            return

        # 攻撃元に戻った = 誰も受けなかった
        # → 攻撃元が伏せ→攻めを行う attack フェーズへ
        self.phase = "attack"
        self.turn = self.attacker

    def finish_round(self, player: str, attack: str) -> None:
        """上がり時の共通処理。"""
        self.finished = True
        self.winner = player

        score, team = self.calculate_score(player, attack)
        self.team_score[team] += score

    def apply_attack(self, player: str, attack: str) -> None:
        """受け直後の攻め（伏せなし攻め）。"""
        if self.phase != "attack":
            raise ValueError("Attack is only allowed in attack phase")
        if player != self.turn:
            raise ValueError(f"It is not {player}'s turn")

        hand = self.hands[player]
        if attack not in hand:
            raise ValueError(f"{player} does not have attack piece {attack}")

        if not self._can_attack_without_block(player, attack):
            raise ValueError(f"{player} cannot attack with {attack} in this state")

        hand.remove(attack)

        self.current_attack = attack
        self.attacker = player
        self.turn = self.next_player(player)
        self.phase = "receive"

        # 上がり判定
        if len(hand) == 0:
            self.finish_round(player, attack)

    def apply_attack_after_block(self, player: str, block: str, attack: str) -> None:
        """
        誰も受けずに攻撃元に戻ったときの攻め（必ず伏せ→攻め）。
        親の最初の手番（attacker=None, current_attack=None）もここを通す。
        """
        if self.phase != "attack":
            raise ValueError("This attack requires attack phase")
        if player != self.turn:
            raise ValueError(f"It is not {player}'s turn")

        hand = self.hands[player]

        if block not in hand or attack not in hand:
            raise ValueError(f"{player} does not have required pieces")

        if not self._can_attack_with_block(player, block, attack):
            raise ValueError(f"{player} cannot attack with block={block}, attack={attack}")

        # 伏せ
        hand.remove(block)
        self.face_down_hidden[player].append(block)
        self.last_block = block
        self.last_block_player = player

        # 伏せは「受け」ではないので king_block_used は増やさない

        # 攻め
        hand.remove(attack)
        self.current_attack = attack
        self.attacker = player
        self.turn = self.next_player(player)
        self.phase = "receive"

        # 上がり判定
        if len(hand) == 0:
            self.finish_round(player, attack)

    # ================================
    # 8/9 の攻め条件ヘルパー
    # ================================
    def _can_attack_without_block(self, player: str, attack: str) -> bool:
        """
        伏せなし攻め（受け直後など）で attack を出せるか。
        8/9 の条件①②（受けで8/9が出ている / 両王持ち）のみを考慮し、
        最後の一手条件③はここでは使わない。
        """
        hand = self.hands[player]
        if attack not in hand:
            return False

        # 1〜7 は常に攻め可能
        if attack in ("1", "2", "3", "4", "5", "6", "7"):
            return True

        if attack in ("8", "9"):
            both_kings = self.had_both_kings.get(player, False)
            already_king_used = (self.king_block_used > 0)
            # 伏せなし攻めでは「最後の一手条件」は使わない
            last_finish = (len(hand) == 1)
            # 伏せなし攻めでも「最後の一手」は許可する（8枚目に8/9が出せない問題の修正）
            return both_kings or already_king_used or last_finish
        return False

    def _can_attack_with_block(self, player: str, block: str, attack: str) -> bool:
        """
        伏せ→攻め のときに attack を出せるか。
        8/9 の条件①②③（受け8/9 / 両王持ち / 最後の一手）をすべて考慮。
        """
        hand = self.hands[player]

        if block not in hand or attack not in hand:
            return False

        # block と attack が同じ駒なら 2枚必要
        if block == attack and hand.count(block) < 2:
            return False

        # 1〜7 は常に攻め可能
        if attack in ("1", "2", "3", "4", "5", "6", "7"):
            return True

        if attack in ("8", "9"):
            both_kings = self.had_both_kings.get(player, False)
            already_king_used = (self.king_block_used > 0)
            # この手で上がる（残り2枚で block+attack を使い切る）
            last_finish = (len(hand) == 2)

            return both_kings or already_king_used or last_finish

        return False

    # ================================
    # 得点計算
    # ================================
    def calculate_score(self, player: str, attack: str) -> Tuple[int, str]:
        """
        上がり時の得点計算（基本点＋ダブル）。
        戻り値: (score, team)  例: (60, "AC")
        """
        base = POINTS[attack]


        # 王玉上がり（直前の伏せ＋最後の攻め が 8/9 の組）
        # 例：7枚目が伏せで8、8枚目が9（攻め） → 100
        #     7枚目が伏せで9、8枚目が8（攻め） → 100
        hidden = self.face_down_hidden.get(player, [])
        if hidden and set([hidden[-1], attack]) == {"8", "9"}:
            base = 100
        # ダブル判定：
        # ・最後に伏せた駒(last_block)が上がった駒(attack)と同じ
        # ・それを伏せたのが上がったプレイヤー
        is_double = (
            self.last_block_player == player and
            self.last_block == attack
        )
        score = base * (2 if is_double else 1)

        team = "AC" if player in ("A", "C") else "BD"
        return score, team

    # ================================
    # 合法手列挙
    # ================================
    def legal_actions(self, player: str) -> List[Tuple[str, Optional[str], Optional[str]]]:
        """
        現在のプレイヤーが選べる合法手一覧を返す。
        各要素は (action_type, block, attack)。
          action_type:
            - "pass"
            - "receive"
            - "attack"
            - "attack_after_block"
        """
        actions: List[Tuple[str, Optional[str], Optional[str]]] = []

        if self.finished:
            return actions

        hand = self.hands[player]

        # 受けフェーズ
        if self.phase == "receive":
            # パスは常に可能
            actions.append(("pass", None, None))

            # 場に攻めがない（親の初手など）の場合、受けはそもそも発生しない
            if self.current_attack is None:
                return actions

            # 受けに使える駒を列挙
            for block in sorted(set(hand)):
                if self.can_receive(player, block):
                    actions.append(("receive", block, None))

            return actions

        # 攻めフェーズ
        if self.phase == "attack":
            if player != self.turn:
                return actions

            # 親の初手：attacker=None, current_attack=None
            # このときは必ず「伏せ→攻め」
            if self.attacker is None and self.current_attack is None:
                for block in sorted(set(hand)):
                    for attack in sorted(set(hand)):
                        if block == attack and hand.count(block) < 2:
                            continue
                        if self._can_attack_with_block(player, block, attack):
                            actions.append(("attack_after_block", block, attack))
                return actions

            # 受けた直後の攻め：current_attack=None かつ attacker==player
            if self.current_attack is None and self.attacker == player:
                for attack in sorted(set(hand)):
                    if self._can_attack_without_block(player, attack):
                        actions.append(("attack", None, attack))
                return actions

            # 誰も受けずに攻撃元に戻ったケースなど：
            # attack フェーズで current_attack が残っている状態は
            # 「伏せ→攻め」が必要な状況として扱う
            for block in sorted(set(hand)):
                for attack in sorted(set(hand)):
                    if block == attack and hand.count(block) < 2:
                        continue
                    if self._can_attack_with_block(player, block, attack):
                        actions.append(("attack_after_block", block, attack))
            return actions

        return actions


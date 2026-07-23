"""配られた手駒そのものの強さを評価します。
攻めタイプ、受けられる駒の種類、絶対ランク・相対ランクなどを計算します。
2香や中駒ペアと大駒を組み合わせた、特殊な攻め順の判定にも使用します。
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from goita_ai2.constants import POINTS

Action = Tuple[str, Optional[str], Optional[str]]


class HandEvaluationMixin:
    """Evaluates hand strength, attack types, and planned attack shapes."""

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

    def _two_kyosha_single_big_attack_plan(self, counts: Counter) -> Optional[List[str]]:
        if counts.get("2", 0) != 2:
            return None
        if any(counts.get(piece, 0) >= 2 for piece in ("3", "4", "5")):
            return None

        if any(counts.get(piece, 0) >= 2 for piece in ("6", "7")):
            return None
        big_pieces = [piece for piece in ("7", "6") if counts.get(piece, 0) == 1]
        if not big_pieces:
            return None

        royal_count = counts.get("8", 0) + counts.get("9", 0)
        big_piece = big_pieces[0]
        if royal_count == 0:
            return [big_piece, "2", "2"]
        if royal_count == 1 and len(big_pieces) == 1:
            return ["2", big_piece, "2"]
        return None

    def _two_kyosha_gold_pair_attack_plan(self, counts: Counter) -> Optional[List[str]]:
        attack_type = self._classify_attack_type(counts)
        if attack_type.get("label") != "two_kyosha":
            return None
        if counts.get("5", 0) != 2:
            return None

        royal_count = counts.get("8", 0) + counts.get("9", 0)
        if royal_count not in (0, 1):
            return None
        return ["5", "2", "2"]

    def _two_kyosha_middle_pair_royal_attack_plan(self, counts: Counter) -> Optional[List[str]]:
        attack_type = self._classify_attack_type(counts)
        if attack_type.get("label") != "two_kyosha":
            return None
        if counts.get("2", 0) != 2:
            return None
        if counts.get("8", 0) + counts.get("9", 0) != 1:
            return None
        middle_pairs = [piece for piece in ("5", "4", "3") if counts.get(piece, 0) == 2]
        if len(middle_pairs) != 1:
            return None
        return ["2", middle_pairs[0]]

    def _middle_pair_single_big_attack_plan(self, counts: Counter) -> Optional[List[str]]:
        attack_type = self._classify_attack_type(counts)
        if attack_type.get("label") != "two_middle":
            return None

        middle_pairs = [piece for piece in ("5", "4", "3") if counts.get(piece, 0) == 2]
        if len(middle_pairs) != 1:
            return None

        big_pieces = [piece for piece in ("7", "6") if counts.get(piece, 0) == 1]
        if len(big_pieces) != 1 or sum(counts.get(piece, 0) for piece in ("6", "7")) != 1:
            return None

        royal_count = counts.get("8", 0) + counts.get("9", 0)
        middle_piece = middle_pairs[0]
        big_piece = big_pieces[0]
        if royal_count == 0:
            return [middle_piece, middle_piece, big_piece]
        if royal_count == 1:
            return [middle_piece, big_piece, middle_piece]
        return None

    def _special_attack_sequence_plan(self, counts: Counter) -> Optional[Dict[str, object]]:
        plan = self._two_kyosha_middle_pair_royal_attack_plan(counts)
        if plan is not None:
            return {"label": "two_kyosha_middle_pair_royal", "sequence": plan}

        plan = self._two_kyosha_gold_pair_attack_plan(counts)
        if plan is not None:
            return {"label": "two_kyosha_gold_pair", "sequence": plan}

        plan = self._two_kyosha_single_big_attack_plan(counts)
        if plan is not None:
            return {"label": "two_kyosha_single_big", "sequence": plan}

        plan = self._middle_pair_single_big_attack_plan(counts)
        if plan is not None:
            return {"label": "middle_pair_single_big", "sequence": plan}
        return None

    def _special_attack_sequence_action(
        self,
        state,
        player: str,
        actions: List[Action],
        *,
        has_non_king_attack_option: bool,
    ) -> Optional[Action]:
        tr = self._track.get(id(state))
        if tr is None or state.phase != "attack":
            return None

        plan_info = tr.get("special_attack_plan")
        if not isinstance(plan_info, dict):
            return None
        plan = plan_info.get("sequence")
        if not isinstance(plan, list) or not plan:
            return None

        step = 0
        for piece in tr.get("my_attack_history", []):
            if step < len(plan) and piece == plan[step]:
                step += 1
        if step >= len(plan):
            return None

        expected_attack = plan[step]
        required_after = Counter(plan[step + 1:])
        candidates: List[Tuple[float, Action]] = []
        for action_type, block, attack in actions:
            if action_type not in ("attack", "attack_after_block") or attack != expected_attack:
                continue

            remaining = list(state.hands[player])
            if block is not None:
                if block not in remaining:
                    continue
                remaining.remove(block)
            if attack not in remaining:
                continue
            remaining.remove(attack)
            remaining_counts = Counter(remaining)
            if any(remaining_counts.get(piece, 0) < needed for piece, needed in required_after.items()):
                continue

            score = self._score_attack_phase(
                state,
                player,
                action_type,
                block,
                attack,
                has_non_king_attack_option=has_non_king_attack_option,
            )
            if action_type == "attack_after_block":
                score += self._score_receive_phase(state, player, "receive", block)
            candidates.append((score, (action_type, block, attack)))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

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
            and self._is_kakarigotae_piece(attack)
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

    def _load_relative_hand_rank_table(self) -> Dict[str, Dict[str, str]]:
        if self._relative_hand_rank_table is not None:
            return self._relative_hand_rank_table

        path = Path(__file__).resolve().parents[2] / "results" / "relative_hand_all_4745_500_combined.csv"
        table: Dict[str, Dict[str, str]] = {}
        if path.exists():
            with path.open(encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    table[row["hand_digits"]] = row
        self._relative_hand_rank_table = table
        return table

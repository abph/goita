"""攻め駒と伏せ駒の選び方を評価します。
親の初手、連続攻め、し攻め、かかりごたえ、王玉を使わせる攻めなどを扱います。
攻め筋を残すための伏せ方や、複数ある攻め駒の優先順位もここで計算します。
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional, Tuple

from goita_ai2.constants import POINTS

Action = Tuple[str, Optional[str], Optional[str]]
TARGET_LAST1 = ("2", "3", "4", "5", "6", "7")


class AttackStrategyMixin:
    """Scores attack pieces and chooses which piece to hide."""

    @staticmethod
    def _is_kakarigotae_piece(piece: Optional[str]) -> bool:
        return piece in ("2", "3", "4", "5")

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

    def _dealer_opening_attack_plan_pieces(
        self,
        state,
        player: str,
        attack: Optional[str] = None,
    ) -> set[str]:
        tr = self._track.get(id(state))
        if tr is None:
            return set()

        if attack is not None:
            profile = self._attack_shape_profile_for_piece(tr["my_init_count"], attack)
            if profile is None:
                return set()
            return {attack}

        profile = self._classify_attack_type(tr["my_init_count"])
        attack_type = int(profile["type"])
        if attack_type in (1, 2, 3, 4, 5, 6, 7, 8):
            return set(str(p) for p in profile["pieces"])
        return set()

    def _can_block_surplus_four_middle(
        self,
        state,
        player: str,
        block: Optional[str],
        attack: Optional[str],
    ) -> bool:
        if block is None or attack is None or block != attack:
            return False
        if attack not in ("3", "4", "5"):
            return False
        return state.hands[player].count(attack) >= 4

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

        plan_pieces = self._dealer_opening_attack_plan_pieces(state, player, attack)
        if not plan_pieces:
            return 0.0

        value = 0.0
        if attack in plan_pieces:
            value += self.DEALER_OPENING_PLAN_ATTACK_BONUS
        if block in plan_pieces:
            if self._can_block_surplus_four_middle(state, player, block, attack):
                value += self.DEALER_SURPLUS_FOUR_MIDDLE_BLOCK_BONUS
            elif not (block == "1" and attack == "1" and state.hands[player].count("1") >= 4):
                value -= self.DEALER_OPENING_PLAN_BLOCK_PENALTY
            else:
                value += self.DEALER_FOUR_SHI_BLOCK_SHI_BONUS
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

        mode_active = bool(tr.get("shi_attack_mode")) and state.hands[player].count("1") > 0
        if mode_active and not self._has_strong_repeat_attack(counts):
            value = self.SHI_ATTACK_MODE_BONUS
            ally_signal = str(tr.get("ally_shi_signal", "unknown"))
            if ally_signal == "passed":
                value += 120.0
            elif ally_signal == "returned_shi":
                value += 260.0
            if tr.get("inherit_ally_shi_attack"):
                value += 280.0
            return value

        axes = self._initial_hand_axes_for_state(state, player)
        rank = self._strategy_rank_from_axes(axes)
        four_shi_signal_attack = (
            counts.get("1", 0) >= 4
            and state.hands[player].count("1") >= 3
            and rank not in ("S", "A", "B")
            and not self._has_strong_repeat_attack(counts)
        )
        if rank not in ("C", "D") and not tr.get("inherit_ally_shi_attack") and not four_shi_signal_attack:
            return 0.0

        if self._has_strong_repeat_attack(counts):
            return 0.0

        if int(axes.get("receive_type", 0)) < 5 and not four_shi_signal_attack:
            return 0.0

        if "1" in tr.get("enemy_past_attacks", set()) and not tr.get("inherit_ally_shi_attack"):
            return 0.0

        already_used_shi_attack = "1" in tr.get("my_past_attacks", set())
        ally_signal = str(tr.get("ally_shi_signal", "unknown"))
        public_seen = sum(int(v) for v in tr.get("public_seen_counts", {}).values())
        is_early = public_seen <= 4

        if already_used_shi_attack and ally_signal not in ("passed", "returned_shi"):
            return 0.0
        if not already_used_shi_attack and not is_early and not tr.get("inherit_ally_shi_attack") and not four_shi_signal_attack:
            return 0.0

        ally = tr.get("ally")
        if self._public_hand_strength(tr, ally) >= 10.0:
            return 0.0

        value = self.WEAK_SHI_ATTACK_BONUS
        if four_shi_signal_attack:
            value += 120.0
        if ally_signal == "passed":
            value += 100.0
        elif ally_signal == "returned_shi":
            value += 220.0
        if tr.get("inherit_ally_shi_attack"):
            value += 280.0
        return value

    def _single_middle_over_four_shi_signal_penalty(
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
            or state.hands[player].count(attack) != 1
            or state.hands[player].count("1") < 3
            or tr.get("my_init_count", Counter()).get("1", 0) < 4
            or self._weak_shi_attack_plan_value(state, player) <= 0
        ):
            return 0.0
        return self.SINGLE_MIDDLE_OVER_FOUR_SHI_SIGNAL_PENALTY

    def _multi_shi_after_big_receive_first_attack_bonus(self, state, player: str) -> float:
        tr = self._track.get(id(state))
        hand = state.hands[player]
        if (
            tr is None
            or int(tr.get("my_attack_count", 0)) != 0
            or tr.get("my_last_receive_piece") not in ("6", "7")
            or hand.count("1") < 3
            or hand.count("8") + hand.count("9") > 1
            or any(hand.count(piece) >= 2 for piece in ("2", "3", "4", "5", "6", "7"))
        ):
            return 0.0
        return self.FOUR_SHI_AFTER_BIG_RECEIVE_FIRST_ATTACK_BONUS

    def _four_shi_receive_return_action(
        self,
        state,
        player: str,
        actions: List[Action],
    ) -> Optional[Action]:
        tr = self._track.get(id(state))
        hand = state.hands[player]
        if (
            tr is None
            or state.phase != "attack"
            or state.turn != player
            or int(tr.get("my_attack_count", 0)) != 0
            or tr.get("my_last_receive_piece") != "1"
            or int(tr.get("my_init_count", Counter()).get("1", 0)) < 4
            or hand.count("1") < 3
            or any(hand.count(piece) >= 2 for piece in ("2", "3", "4", "5", "6", "7"))
        ):
            return None

        return next(
            (
                action
                for action in actions
                if action[0] in ("attack", "attack_after_block") and action[2] == "1"
            ),
            None,
        )

    def _four_shi_after_big_receive_first_attack_bonus(self, state, player: str) -> float:
        if state.hands[player].count("1") < 4:
            return 0.0
        return self._multi_shi_after_big_receive_first_attack_bonus(state, player)

    def _shi_attack_score_adjustment(self, state, player: str) -> float:
        tr = self._track.get(id(state))
        if tr is None:
            return -self.NON_WEAK_SHI_ATTACK_PENALTY

        multi_shi_bonus = self._multi_shi_after_big_receive_first_attack_bonus(state, player)
        if multi_shi_bonus > 0:
            return multi_shi_bonus

        exhaust_bonus = self._shi_exhaust_attack_bonus(state, player)
        if exhaust_bonus > 0:
            return exhaust_bonus

        sashikomi_bonus = self._shi_sashikomi_attack_bonus(state, player)
        if sashikomi_bonus > 0:
            return sashikomi_bonus

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

    def _weak_shi_fallback_high_point_attack_bonus(
        self,
        state,
        player: str,
        action_type: str,
        attack: Optional[str],
    ) -> float:
        if action_type not in ("attack", "attack_after_block") or attack is None:
            return 0.0
        if attack in ("1", "8", "9"):
            return 0.0

        tr = self._track.get(id(state))
        if tr is None:
            return 0.0
        if self._weak_shi_attack_plan_value(state, player) > 0:
            return 0.0

        initial_shi = tr["my_init_count"].get("1", 0)
        current_shi = state.hands[player].count("1")
        if initial_shi > 2 and current_shi > 2:
            return 0.0

        return max(0.0, float(POINTS.get(attack, 0) - POINTS.get("2", 20))) * self.WEAK_SHI_FALLBACK_HIGH_POINT_WEIGHT

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
            if (
                attack == "1"
                and self._is_dealer_opening_attack(state, player)
                and hand.count("1") >= 4
            ):
                return self.DEALER_FOUR_SHI_BLOCK_SHI_BONUS
            if remaining_len == 2 and remaining_shi == 1:
                return self.WEAK_SHI_ENDGAME_MIXED_BLOCK_BONUS
            return 60.0 if shi_is_surplus else -700.0
        if block in ("2", "8", "9"):
            return -260.0
        return 120.0

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

    def _is_fourth_middle_attack(self, state, player: str, attack: Optional[str]) -> bool:
        if attack not in ("2", "3", "4", "5"):
            return False
        tr = self._track.get(id(state))
        if tr is None:
            return False
        if state.hands[player].count(attack) != 1:
            return False
        seen_and_mine = int(tr.get("public_seen_counts", {}).get(attack, 0)) + state.hands[player].count(attack)
        return seen_and_mine >= self._piece_total(attack)

    def _fourth_middle_early_attack_delay_penalty(
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
            or int(tr.get("my_attack_count", 0)) >= 2
            or not self._is_fourth_middle_attack(state, player, attack)
        ):
            return 0.0
        return self.FOURTH_MIDDLE_EARLY_ATTACK_DELAY_PENALTY

    def _fourth_middle_third_attack_bonus(
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
            or int(tr.get("my_attack_count", 0)) != 2
            or not self._is_fourth_middle_attack(state, player, attack)
        ):
            return 0.0
        return self.FOURTH_MIDDLE_THIRD_ATTACK_BONUS

    def _second_kyosha_single_shi_block_adjustment(
        self,
        state,
        player: str,
        action_type: str,
        block: Optional[str],
        attack: Optional[str],
    ) -> float:
        tr = self._track.get(id(state))
        hand = state.hands[player]
        if (
            tr is None
            or action_type != "attack_after_block"
            or attack != "2"
            or block is None
            or int(tr.get("my_attack_count", 0)) != 1
            or tr.get("my_last_attack") != "2"
            or hand.count("2") < 2
            or hand.count("1") != 1
            or not any(piece in ("8", "9") for piece in hand)
        ):
            return 0.0

        singleton_middle = [
            piece
            for piece in ("3", "4", "5")
            if hand.count(piece) == 1
        ]
        if not singleton_middle:
            return 0.0
        lowest_middle = min(
            singleton_middle,
            key=lambda piece: (POINTS.get(piece, 0), piece),
        )

        if block == "1":
            return -self.SECOND_KYOSHA_KEEP_SINGLE_SHI_BLOCK_PENALTY
        if block == lowest_middle:
            return self.SECOND_KYOSHA_LOW_MIDDLE_BLOCK_BONUS
        return 0.0

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

    def _enemy_shi_attack_threat_for_fuse(self, tr: Optional[dict], player: str) -> bool:
        if tr is None:
            return False
        enemies = [p for p in ("A", "B", "C", "D") if p != player and not self._same_team(p, player)]
        models = tr.get("public_hand_models", {})
        enemy_shi_attacks = 0
        enemy_shi_attackers = 0
        for enemy in enemies:
            model = models.get(enemy, {})
            n = int(model.get("attacks", Counter()).get("1", 0))
            if n > 0:
                enemy_shi_attackers += 1
                enemy_shi_attacks += n

        if enemy_shi_attackers >= 2 or enemy_shi_attacks >= 2:
            return True

        if enemy_shi_attacks == 1 and tr.get("enemy_team_rejected_shi_attack"):
            return False

        ally = self._ally_of(player)
        ally_pass_count = int(models.get(ally, {}).get("pass_count", 0))
        return enemy_shi_attacks > 0 and ally_pass_count > 0

    def _next_enemy_likely_shi_exhausted(self, state, player: str) -> bool:
        tr = self._track.get(id(state))
        if tr is None:
            return False
        if "1" not in tr.get("my_past_attacks", set()):
            return False
        if state.hands[player].count("1") <= 0:
            return False

        next_enemy = state.next_player(player)
        if self._same_team(next_enemy, player):
            return False

        models = tr.get("public_hand_models", {})
        model = models.get(next_enemy, {})
        attacks = model.get("attacks", Counter())
        blocks = model.get("blocks", Counter())
        visible_shi_spent = int(attacks.get("1", 0)) + int(blocks.get("1", 0))
        hidden_blocks = int(tr.get("hidden_block_counts", {}).get(next_enemy, 0))

        if visible_shi_spent <= 0 or hidden_blocks < 2:
            return False

        first_attack = model.get("first_attack")
        if first_attack == "1" and int(attacks.get("1", 0)) >= 1:
            return False

        return visible_shi_spent + min(hidden_blocks, 2) >= 3

    def _shi_exhaust_attack_bonus(self, state, player: str) -> float:
        if not self._next_enemy_likely_shi_exhausted(state, player):
            return 0.0
        return self.SHI_EXHAUST_ATTACK_BONUS

    def _fuse_strategy_hidden_block_adjustment(
        self,
        state,
        player: str,
        action_type: str,
        block: Optional[str],
        attack: Optional[str],
    ) -> float:
        if action_type != "attack_after_block" or block is None:
            return 0.0

        tr = self._track.get(id(state))
        hand = list(state.hands[player])
        after_hand = list(hand)
        if block in after_hand:
            after_hand.remove(block)
        if attack in after_hand:
            after_hand.remove(attack)

        block_no = len(getattr(state, "face_down_hidden", {}).get(player, [])) + 1
        value = 0.0

        # 香と王玉は、点数以上に受け・終盤待ちとしての価値が高い。
        if block == "2":
            value -= self.FUSE_KYOSHA_BLOCK_PENALTY
            if tr is not None and self._opponents_piece_pressure(tr, player, "2") > 0:
                value -= 60.0
        elif block in ("8", "9"):
            value -= self.FUSE_KING_BLOCK_PENALTY

        # 金銀は実戦で受けに使う機会が多いため、大駒より少し残したい。
        if block in ("4", "5"):
            value -= self.FUSE_KEEP_KIN_GIN_RECEIVE_BONUS

        # しは基本的には伏せやすい。ただし敵方のし攻めが濃い時と、
        # 2枚目以降に最後のしを消す時は守備価値が上がる。
        if block == "1":
            remaining_shi = after_hand.count("1")
            if remaining_shi < 2 and self._enemy_shi_attack_threat_for_fuse(tr, player):
                value -= self.FUSE_ENEMY_SHI_THREAT_BLOCK_PENALTY
            has_king_after = any(p in ("8", "9") for p in after_hand)
            if block_no >= 2 and remaining_shi <= 0 and not has_king_after:
                value -= self.FUSE_KEEP_LAST_SHI_PENALTY
            if block_no >= 3 and has_king_after:
                value += self.FUSE_THIRD_BLOCK_KING_SHI_BONUS

        # 攻め駒が飽和している場合は、強い駒を1枚伏せてもよい。
        # 例: 角角飛飛、馬馬飛飛、銀銀飛飛など。
        if block in ("3", "4", "5", "6", "7") and hand.count(block) >= 2:
            before_pairs = self._strong_attack_pair_pieces(hand)
            after_pairs = self._strong_attack_pair_pieces(after_hand)
            if len(before_pairs) >= 2 and after_pairs:
                value += self.FUSE_ATTACK_SATURATION_BLOCK_BONUS
                if block in ("6", "7"):
                    value += 8.0
                elif block == "3":
                    value += 4.0

        return value

    def _piece_count_kakari_adjustment(self, state, player: str, attack: str) -> float:
        tr = self._track.get(id(state))
        if tr is None or not self._is_kakarigotae_piece(attack):
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

    def _kakari_saturation_followup_pieces(self, hand: List[str]) -> List[str]:
        counts = Counter(hand)
        profiles = [
            profile
            for profile in self._attack_shape_profiles(counts)
            if int(profile.get("value", 0)) >= 3
        ]
        pieces = {
            str(piece)
            for profile in profiles
            for piece in profile.get("pieces", [])
        }
        return sorted(pieces, key=lambda piece: (POINTS.get(piece, 0), piece), reverse=True)

    def _kakari_saturation_attack_bonus(self, state, player: str, attack: Optional[str]) -> float:
        if not self._is_kakarigotae_piece(attack):
            return 0.0
        tr = self._track.get(id(state))
        if tr is None:
            return 0.0
        if self._is_fourth_middle_attack(state, player, attack):
            return 0.0
        if not (attack == tr.get("ally_first_attack") or attack in tr.get("ally_past_attacks", set())):
            return 0.0
        if attack not in state.hands[player]:
            return 0.0

        after_attack = list(state.hands[player])
        after_attack.remove(attack)
        followups = self._kakari_saturation_followup_pieces(after_attack)
        if not followups:
            return 0.0

        bonus = self.KAKARI_SATURATION_ATTACK_BONUS
        bonus += max(float(POINTS.get(piece, 0)) for piece in followups) / 3.0
        ally = self._ally_of(player)
        remaining_min, remaining_max = self._estimate_remaining_range(tr, ally, attack)
        if remaining_min > 0 or remaining_max > 0:
            bonus += self.KAKARI_SATURATION_ALLY_REMAINING_BONUS
        return bonus

    def _ally_force_king_attack_bonus(
        self,
        state,
        player: str,
        action_type: str,
        attack: Optional[str],
    ) -> float:
        tr = self._track.get(id(state))
        if (
            tr is not None
            and action_type == "attack"
            and attack is not None
            and attack == tr.get("pending_ally_force_king_attack_piece")
        ):
            return self.ALLY_FORCE_KING_ATTACK_BONUS
        return 0.0

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

        is_kakari = self._is_kakarigotae_piece(attack) and (
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
        guaranteed_score = self._guaranteed_finish_score_after_attack_action(
            state,
            player,
            (action_type, block, attack),
        )
        if guaranteed_score is not None:
            score += 1e8 + guaranteed_score

        score += self._last_one_remaining_bonus(state, player, attack)
        score += self._occupancy_priority_bonus(state, attack)
        score += self._public_attack_safety_bonus(state, player, attack)
        score += self._attack_strategy_bonus(state, player, attack)
        score += self._multi_attack_shape_plan_adjustment(state, player, action_type, block, attack)
        score += self._weak_shi_fallback_high_point_attack_bonus(state, player, action_type, attack)
        score += self._piece_count_attack_adjustment(state, player, attack)
        score += self._kakari_saturation_attack_bonus(state, player, attack)
        score += self._fourth_middle_third_attack_bonus(state, player, action_type, attack)
        score += self._ally_force_king_attack_bonus(state, player, action_type, attack)
        score += self._endgame_remaining_pair_adjustment(state, player, block, attack)
        score += self._future_attack_plan_adjustment(
            state,
            player,
            action_type,
            block,
            attack,
        )
        score += self._second_kyosha_single_shi_block_adjustment(
            state,
            player,
            action_type,
            block,
            attack,
        )
        score += self._conditional_shi_royal_finish_adjustment(
            state,
            player,
            action_type,
            block,
            attack,
        )
        score += self._dealer_opening_plan_adjustment(state, player, action_type, block, attack)

        if tr is not None and tr.get("my_attack_count", 0) == 0 and state.hands[player].count(attack) == 1:
            if attack not in ("8", "9", "1"):
                score -= 30.0
        score -= self._single_middle_after_big_receive_first_attack_penalty(
            state,
            player,
            action_type,
            attack,
        )
        score -= self._single_middle_over_four_shi_signal_penalty(
            state,
            player,
            action_type,
            attack,
        )
        score -= self._fourth_middle_early_attack_delay_penalty(
            state,
            player,
            action_type,
            attack,
        )

        if tr is not None and attack != "1" and attack == tr.get("my_last_attack"):
            if tr.get("ally_attacked_since_my_last_attack"):
                if tr.get("ally_last_attack") == attack:
                    score += self.CONTINUOUS_ATTACK_BONUS * 1.5
                else:
                    score -= self.CONTINUOUS_ATTACK_BONUS
                    score -= 300.0
            else:
                score += self.CONTINUOUS_ATTACK_BONUS

        if tr is not None and self._is_kakarigotae_piece(attack):
            ally_first = tr.get("ally_first_attack")
            is_fourth_middle = self._is_fourth_middle_attack(state, player, attack)
            if ally_first is not None and attack == ally_first:
                is_unreasonable_block = is_fourth_middle or (action_type == "attack_after_block" and block in ("8", "9"))
                if not is_unreasonable_block:
                    if tr.get("my_attack_count", 0) == 1:
                        score += self.KAKARI_GOTAE_BONUS * 10.0
                    else:
                        score += self.KAKARI_GOTAE_BONUS
            elif attack in tr.get("ally_past_attacks", set()):
                is_unreasonable_block = is_fourth_middle or (action_type == "attack_after_block" and block in ("8", "9"))
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
        if action_type in ("attack", "attack_after_block") and attack == "1":
            score += self._shi_attack_score_adjustment(state, player)

        if attack in ("9", "8") and has_non_king_attack_option:
            score -= self.KING_ATTACK_PENALTY

        score += POINTS.get(attack, 0) / 10.0

        if action_type in ("attack", "attack_after_block") and block is not None:
            penalty_table = {"9": 120, "8": 120, "7": 4, "6": 4, "5": 7, "4": 7, "3": 3, "2": 18, "1": 1}
            base_penalty = float(penalty_table.get(block, 0))
            score += self._shi_attack_hidden_block_adjustment(state, player, action_type, block, attack)
            score += self._piece_count_hidden_block_adjustment(state, player, block)
            score += self._fuse_strategy_hidden_block_adjustment(state, player, action_type, block, attack)
            score -= self._same_piece_pair_spend_penalty(state, player, action_type, block, attack)

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

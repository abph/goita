"""AIが実際に選ぶ最終行動を決定します。
合法手に対して各戦略モジュールの評価を集め、確定上がりなどの優先手順も反映します。
選んだ行動と判断理由を記録し、棋譜やデバッグ表示で確認できる形にします。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

Action = Tuple[str, Optional[str], Optional[str]]


class DecisionMixin:
    """Combines strategy scores and selects the final legal action."""

    def _set_decision_reason(self, reason: str) -> None:
        self.last_decision_reason = reason

    def _set_score_fallback_detail(self, detail: str) -> None:
        self.last_score_fallback_detail = detail

    def _classify_score_fallback(
        self,
        state,
        player: str,
        action: Action,
        *,
        has_non_king_attack_option: bool,
    ) -> str:
        action_type, block, attack = action
        tr = self._track.get(id(state))

        enemy_attack_turn = (
            state.phase == "receive"
            and state.current_attack is not None
            and state.attacker is not None
            and (not self._same_team(state.attacker, player))
        )

        if action_type == "pass":
            if enemy_attack_turn and tr is not None:
                if self._preserve_current_attack_for_win_value(state, player) > 0:
                    return "pass_preserve_win_attack"
                if self._piece_count_receive_adjustment(state, player, action_type, block) > 0:
                    return "pass_piece_count_inference"
                attacker_count = tr.get("enemy_attack_counts", {}).get(state.attacker, 1)
                if attacker_count == 1:
                    policy = getattr(self, "ENEMY_FIRST_ATTACK_POLICY", "strict_pass")
                    if policy == "hand_strength":
                        axes = self._initial_hand_axes_for_state(state, player)
                        return (
                            f"enemy_first_pass_hand_strength_"
                            f"abs{axes['absolute_rank']}_rel{axes['relative_rank']}_"
                            f"{axes['total_score']}_"
                            f"atk{axes['attack_score']}_rcv{axes['receive_score']}"
                        )
                    return "enemy_first_pass"
                return "enemy_later_pass"
            return "pass_base"

        if action_type == "receive":
            if block is not None and self._win_after_receive_bonus(state, player, action) > 0:
                return "receive_win_after"
            if block is not None and self._score_receive_phase(state, player, action_type, block) >= 1e8:
                return "receive_tsume_after"
            if enemy_attack_turn and tr is not None:
                if self._preserve_current_attack_for_win_value(state, player) > 0:
                    return "receive_spends_win_attack"
                if self._early_enemy_first_king_receive_penalty(state, player, action_type, block) > 0:
                    return "enemy_first_receive_king_reserved"
                if self._piece_count_receive_adjustment(state, player, action_type, block) < 0:
                    return "receive_piece_count_risk"
                attacker_count = tr.get("enemy_attack_counts", {}).get(state.attacker, 1)
                if attacker_count == 1:
                    policy = getattr(self, "ENEMY_FIRST_ATTACK_POLICY", "strict_pass")
                    if policy == "hand_strength":
                        axes = self._initial_hand_axes_for_state(state, player)
                        return (
                            f"enemy_first_{action_type}_hand_strength_"
                            f"abs{axes['absolute_rank']}_rel{axes['relative_rank']}_"
                            f"{axes['total_score']}_"
                            f"atk{axes['attack_score']}_rcv{axes['receive_score']}"
                        )
                    return f"enemy_first_receive_{policy}"
                if block in ("8", "9"):
                    return "enemy_later_receive_king"
                return "enemy_later_receive"
            if state.attacker is not None and self._same_team(state.attacker, player):
                if self._ally_shi_exhaust_receive_bonus(state, player, block) > 0:
                    return "ally_shi_exhaust_receive"
                if self._ally_force_king_receive_bonus(state, player, action_type, block) > 0:
                    return "ally_force_king_receive"
                if self._ally_strong_followup_receive_bonus(state, player, action_type, block) > 0:
                    return "ally_strong_followup_receive"
                if self._kakari_saturation_receive_bonus(state, player, block) > 0:
                    return "ally_attack_receive_saturation"
                return "ally_attack_receive"
            if block in ("8", "9"):
                return "receive_king_base"
            return "receive_base"

        if action_type not in ("attack", "attack_after_block") or attack is None:
            return "other"

        if self._win_now_bonus(state, player, action) > 0:
            return "attack_win_now_score"

        if tr is not None:
            is_safe = self._is_absolute_safe_for_tsume(state, player, attack, tr)
            is_agari = (len(state.hands[player]) <= 2)
            if is_safe or is_agari:
                temp_hand = list(state.hands[player])
                if block is not None and block in temp_hand:
                    temp_hand.remove(block)
                if attack in temp_hand:
                    temp_hand.remove(attack)
                if len(temp_hand) == 0:
                    return "attack_agari_score"
                if self._max_tsume_score(temp_hand, state, player, tr) >= 0:
                    return "attack_tsume_score"

            ally_first = tr.get("ally_first_attack")
            if self._kakari_saturation_attack_bonus(state, player, attack) > 0:
                return "attack_kakari_saturation"
            if self._ally_force_king_attack_bonus(state, player, action_type, attack) > 0:
                return "attack_force_enemy_king"
            if self._can_block_surplus_four_middle(state, player, block, attack):
                return "block_surplus_four_middle"
            conditional_finish_score = self._conditional_shi_royal_finish_score(
                state,
                player,
                action_type,
                block,
                attack,
            )
            if conditional_finish_score is not None:
                return f"attack_conditional_shi_royal_finish_{int(conditional_finish_score)}"
            if self._fourth_middle_early_attack_delay_penalty(state, player, action_type, attack) > 0:
                return "attack_delay_fourth_middle"
            if self._fourth_middle_third_attack_bonus(state, player, action_type, attack) > 0:
                return "attack_fourth_middle_third"
            if (
                self._second_kyosha_single_shi_block_adjustment(
                    state,
                    player,
                    action_type,
                    block,
                    attack,
                )
                >= self.SECOND_KYOSHA_LOW_MIDDLE_BLOCK_BONUS
            ):
                return "block_low_middle_keep_single_shi"
            if self._is_kakarigotae_piece(attack) and (
                attack == ally_first or attack in tr.get("ally_past_attacks", set())
            ):
                if not self._is_fourth_middle_attack(state, player, attack):
                    return "attack_kakari_score"

            if attack != "1" and attack == tr.get("my_last_attack"):
                return "attack_continuous_score"

            dealer_plan_adjustment = self._dealer_opening_plan_adjustment(state, player, action_type, block, attack)
            if dealer_plan_adjustment >= self.DEALER_OPENING_PLAN_ATTACK_BONUS:
                return "dealer_opening_primary_attack"
            if dealer_plan_adjustment <= -self.DEALER_OPENING_PLAN_BLOCK_PENALTY:
                return "block_dealer_opening_primary_attack"

            multi_attack_adjustment = self._multi_attack_shape_plan_adjustment(state, player, action_type, block, attack)
            if multi_attack_adjustment >= self.LOWER_ATTACK_SHAPE_BLOCK_BONUS:
                return "block_lower_attack_shape"
            if multi_attack_adjustment <= -self.LOWER_ATTACK_SHAPE_ATTACK_PENALTY:
                return "attack_lower_attack_shape"
            if multi_attack_adjustment <= -self.TOP_ATTACK_SHAPE_BLOCK_PENALTY:
                return "block_top_attack_shape"

            visible_kings = (
                tr["public_seen_counts"].get("8", 0)
                + tr["public_seen_counts"].get("9", 0)
                + state.hands[player].count("8")
                + state.hands[player].count("9")
            )
            total_p = 4 if attack in ("2", "3", "4", "5") else 2 if attack in ("6", "7") else 10 if attack == "1" else 1
            seen_and_mine = tr["public_seen_counts"].get(attack, 0) + state.hands[player].count(attack)
            if seen_and_mine == total_p:
                if attack == "2" or (attack not in ("1", "8", "9") and visible_kings == 2):
                    return "attack_absolute_safe"
                if attack not in ("1", "8", "9"):
                    return "attack_tatewari"

            if self._last_one_remaining_bonus(state, player, attack) > 0:
                return "attack_last_one"

            if self._same_piece_pair_spend_penalty(state, player, action_type, block, attack) > 0:
                return "block_spends_attack_pair"

            if (
                attack == "1"
                and self._four_shi_after_big_receive_first_attack_bonus(state, player) > 0
            ):
                return "attack_four_shi_over_single_middle"

            if self._single_middle_over_four_shi_signal_penalty(state, player, action_type, attack) > 0:
                return "attack_avoid_single_middle_over_four_shi"

            if (
                attack == "2"
                and self._kyosha_probe_expected_score_after_attack_action(state, player, action, tr) is not None
            ):
                return "attack_kyosha_probe_high_score"

            if attack == "1" and self._shi_exhaust_attack_bonus(state, player) > 0:
                return "attack_shi_exhaust_enemy"

            if self._weak_shi_fallback_high_point_attack_bonus(state, player, action_type, attack) > 0:
                return "attack_high_point_after_weak_shi"

            if attack == "1" and self._shi_sashikomi_attack_bonus(state, player) > 0:
                return "attack_shi_sashikomi"

            if self._shi_sashikomi_wait_bonus(state, player, block, attack) > 0:
                return "attack_keep_shi_sashikomi"

            if self._occupancy_priority_bonus(state, attack) > 0:
                return "attack_occupancy"

            if self._endgame_remaining_pair_adjustment(state, player, block, attack) >= 50.0:
                return "attack_endgame_high_score_pair"

            piece_count_adjustment = self._piece_count_attack_adjustment(state, player, attack)
            if piece_count_adjustment >= self.INFER_ATTACK_EXHAUSTED_BONUS:
                return "attack_piece_count_clear"
            if piece_count_adjustment <= -self.INFER_ATTACK_OVERLAP_PENALTY:
                return "attack_avoid_enemy_piece_count"

            kakari_adjustment = self._piece_count_kakari_adjustment(state, player, attack)
            if kakari_adjustment > 0:
                return "attack_kakari_piece_count_clear"
            if kakari_adjustment < 0:
                return "attack_kakari_piece_count_blocked"

            if self._public_attack_safety_bonus(state, player, attack) >= self.PUBLIC_SAFE_ATTACK_BONUS_MID:
                return "attack_public_safe"

            strategy_bonus = self._attack_strategy_bonus(state, player, attack)
            if strategy_bonus > 0:
                profile = self._classify_attack_type(tr["my_init_count"])
                return f"attack_strategy_type_{profile['type']}_{profile['label']}"
            if strategy_bonus < 0:
                return "attack_receive_keep_penalty"

        if attack in ("8", "9") and has_non_king_attack_option:
            return "attack_king_penalty_context"

        if action_type == "attack_after_block" and block is not None:
            fuse_adjustment = self._fuse_strategy_hidden_block_adjustment(state, player, action_type, block, attack)
            if fuse_adjustment >= self.FUSE_ATTACK_SATURATION_BLOCK_BONUS:
                return "block_attack_saturation"
            if fuse_adjustment <= -self.FUSE_KYOSHA_BLOCK_PENALTY:
                return "block_fuse_keep_key_piece"
            if block == "1" and fuse_adjustment <= -self.FUSE_KEEP_LAST_SHI_PENALTY:
                return "block_keep_shi_defense"
            if self._piece_count_hidden_block_adjustment(state, player, block) < 0:
                return "block_keep_piece_count"
            if block in ("8", "9"):
                return "block_king_penalty"
            if block == "1":
                return "block_shi_context"
            return "block_piece_penalty"

        return "attack_piece_value"

    def select_action(self, state, player: str, actions: List[Action]) -> Action:
        self._set_decision_reason("")
        self._set_score_fallback_detail("")

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

        if tr is not None and tr.get("pending_ally_force_king_attack_piece"):
            attack_actions = [
                (t, b, a)
                for (t, b, a) in actions
                if t in ("attack", "attack_after_block") and a is not None
            ]
            win_now_actions = [
                (self._win_now_bonus(state, player, act), act)
                for act in attack_actions
                if self._win_now_bonus(state, player, act) > 0
            ]
            if win_now_actions:
                win_now_actions.sort(key=lambda x: x[0], reverse=True)
                chosen = win_now_actions[0][1]
                tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                tr["pending_ally_force_king_attack_piece"] = None
                self._set_decision_reason("win_now")
                return chosen

            pending_piece = str(tr.get("pending_ally_force_king_attack_piece"))
            for act in attack_actions:
                if act[2] == pending_piece:
                    tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                    tr["pending_ally_force_king_attack_piece"] = None
                    self._set_decision_reason("score_fallback")
                    self._set_score_fallback_detail("attack_force_enemy_king")
                    return act

        reach_avoidance_tsume = self._reach_avoidance_conditional_tsume_action(
            state,
            player,
            actions,
            has_non_king_attack_option=has_non_king_attack_option,
        )
        if reach_avoidance_tsume is not None:
            chosen, receive_risk, _risk_gap = reach_avoidance_tsume
            if tr is not None:
                tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                if tr.get("kg_plan_active") and tr["my_attack_count"] >= 3:
                    tr["kg_plan_active"] = False
            next_enemy = state.next_player(player)
            self._set_decision_reason("conditional_tsume")
            self._set_score_fallback_detail(
                f"reach_avoid_next_{next_enemy}_piece_{chosen[2]}_risk_{int(round(receive_risk * 100))}"
            )
            return chosen

        # A forced scoring route outranks kakarigotae and other attack signals.
        high_score_tsume = self._high_score_tsume_action(
            state,
            player,
            actions,
            has_non_king_attack_option=has_non_king_attack_option,
        )
        if high_score_tsume is not None:
            chosen, route_score, immediate = high_score_tsume
            if tr is not None and chosen[0] in ("attack", "attack_after_block"):
                tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                if tr.get("kg_plan_active") and tr["my_attack_count"] == 2 and chosen[2] in ("8", "9") and tr.get("kg_second") is None:
                    tr["kg_second"] = chosen[2]
                if tr.get("kg_plan_active") and tr["my_attack_count"] >= 3:
                    tr["kg_plan_active"] = False
            self._set_decision_reason("win_now" if immediate else "tsume")
            self._set_score_fallback_detail(f"high_score_{int(route_score)}")
            return chosen

        kakari_actions: List[Tuple[float, Action]] = []
        if tr is not None:
            ally_first = tr.get("ally_first_attack")
            ally_past = tr.get("ally_past_attacks", set())
            for (t, b, a) in actions:
                if (
                    t in ("attack", "attack_after_block")
                    and self._is_kakarigotae_piece(a)
                ):
                    is_unreasonable_block = (
                        self._is_fourth_middle_attack(state, player, a)
                        or (t == "attack_after_block" and b in ("8", "9"))
                    )
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
            self._set_decision_reason("kakari")
            return chosen

        responded_actions: List[Tuple[float, Action]] = []
        if tr is not None:
            for (t, b, a) in actions:
                if t in ("attack", "attack_after_block") and a is not None:
                    if a in ("1", "2", "3", "4", "5") and a in tr.get("ally_responded_to_my_attacks", set()):
                        is_unreasonable_block = (
                            self._is_fourth_middle_attack(state, player, a)
                            and int(tr.get("my_attack_count", 0)) < 2
                        ) or (t == "attack_after_block" and b in ("8", "9"))
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
            self._set_decision_reason("responded")
            return chosen

        give_way_action = self._give_way_to_ally_guaranteed_win_action(state, player, actions)
        if give_way_action is not None:
            return give_way_action

        guaranteed_finish_receive = self._guaranteed_finish_receive_action(state, player, actions)
        if guaranteed_finish_receive is not None:
            return guaranteed_finish_receive

        inferred_endgame_result = self._inferred_endgame_team_result_action(
            state,
            player,
            actions,
        )
        if inferred_endgame_result is not None:
            chosen, winner, score = inferred_endgame_result
            self._set_decision_reason("inferred_endgame")
            if self._same_team(winner, player):
                winner_role = "self" if winner == player else "ally"
                self._set_score_fallback_detail(
                    f"inferred_endgame_{winner_role}_win_{winner}_{score}"
                )
            else:
                self._set_score_fallback_detail(
                    f"inferred_endgame_min_loss_{winner}_{score}"
                )
            return chosen

        royal_reserve_pass = self._enemy_second_attack_royal_reserve_pass_action(
            state,
            player,
            actions,
        )
        if royal_reserve_pass is not None:
            return royal_reserve_pass

        if state.phase == "receive" and state.current_attack in ("1", "2"):
            rank_policy_action = self._enemy_first_same_piece_rank_policy_action(state, player, actions)
            if rank_policy_action is not None:
                return rank_policy_action

        # A stale ally-response estimate must not hide a guaranteed scoring route.
        prefilter_high_score_tsume = self._high_score_tsume_action(
            state,
            player,
            actions,
            has_non_king_attack_option=has_non_king_attack_option,
        )
        protected_tsume_action = prefilter_high_score_tsume[0] if prefilter_high_score_tsume is not None else None

        inferred_shi_sashikomi = None
        if protected_tsume_action is None:
            inferred_shi_sashikomi = self._inferred_ally_shi_sashikomi_finish_action(
                state,
                player,
                actions,
            )
        if inferred_shi_sashikomi is not None:
            if tr is not None:
                tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
            self._set_decision_reason("score_fallback")
            self._set_score_fallback_detail("attack_inferred_ally_shi_sashikomi_win")
            return inferred_shi_sashikomi

        filtered_actions = []
        if tr is not None:
            ignored = tr.get("ally_ignored_my_attacks", set())
            for act in actions:
                t, b, a = act
                if t in ("attack", "attack_after_block") and a is not None:
                    if a in ignored and act != protected_tsume_action:
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

        early_big_receive_action = self._early_big_piece_same_receive_action(state, player, actions)
        if early_big_receive_action is not None:
            return early_big_receive_action

        rank_policy_action = self._enemy_first_same_piece_rank_policy_action(state, player, actions)
        if rank_policy_action is not None:
            return rank_policy_action

        kg_keep_width_action = self._king_gyoku_opening_keep_receive_width_action(state, player, actions)
        if kg_keep_width_action is not None:
            return kg_keep_width_action

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
                        self._set_decision_reason("perfect_plan")
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
            self._set_decision_reason("win_now")
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
            self._set_decision_reason("tsume")
            return chosen

        special_sequence_action = self._special_attack_sequence_action(
            state,
            player,
            actions,
            has_non_king_attack_option=has_non_king_attack_option,
        )
        if special_sequence_action is not None:
            if tr is not None:
                tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
            self._set_decision_reason("score_fallback")
            action_type, block, attack = special_sequence_action
            conditional_finish_score = self._conditional_shi_royal_finish_score(
                state,
                player,
                action_type,
                block,
                attack,
            )
            if conditional_finish_score is not None:
                self._set_score_fallback_detail(
                    f"attack_conditional_shi_royal_finish_{int(conditional_finish_score)}"
                )
            else:
                plan_label = tr.get("special_attack_plan", {}).get("label", "special") if tr is not None else "special"
                self._set_score_fallback_detail(f"attack_sequence_{plan_label}")
            return special_sequence_action

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
                            self._set_decision_reason("forced_king_third")
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
                                    self._set_decision_reason("king_order")
                                    return act
                    if next_attack_no == 3:
                        second = tr.get("kg_second")
                        want = "8" if second == "9" else "9" if second == "8" else None
                        if want is not None:
                            for act in attack_actions:
                                if act[2] == want:
                                    tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                                    tr["kg_plan_active"] = False
                                    self._set_decision_reason("king_order")
                                    return act
                        for p in ["9", "8"]:
                            for act in attack_actions:
                                if act[2] == p:
                                    tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                                    tr["kg_plan_active"] = False
                                    self._set_decision_reason("king_order")
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
                        self._set_decision_reason("safe_nonking_third")
                        return best

        # --- 第8位：味方の「し」攻めに対するレスポンス（しシグナルへの返答） ---
        if tr is not None:
            ally = tr["ally"]
            if state.phase == "receive" and state.current_attack == "1" and state.attacker == ally:
                initial_shis = tr["my_init_count"].get("1", 0)
                current_shis = state.hands[player].count("1")
                can_show_four_shi_signal = (
                    current_shis >= 4
                    or (current_shis >= 3 and "1" in tr.get("my_past_attacks", set()))
                    or (current_shis >= 2 and tr.get("shi_attack_mode"))
                )

                # 1. 「現在の手札」に「し」が4枚以上ある場合（し受け・し攻め）
                if can_show_four_shi_signal:
                    for act in actions:
                        if act[0] == "attack_after_block" and act[1] == "1" and act[2] == "1":
                            tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                            tr["shi_attack_mode"] = True
                            tr["shi_attack_mode_source"] = "ally_signal"
                            if tr.get("kg_plan_active") and tr["my_attack_count"] == 2 and act[2] in ("8", "9") and tr.get("kg_second") is None:
                                tr["kg_second"] = act[2]
                            if tr.get("kg_plan_active") and tr["my_attack_count"] >= 3:
                                tr["kg_plan_active"] = False
                            self._set_decision_reason("shi_signal")
                            return act
                    # 「し受け・し攻め」が物理的にできない場合は、とりあえず「し」で受ける
                    for act in actions:
                        if act[0] == "receive" and act[1] == "1":
                            tr["shi_attack_mode"] = True
                            tr["shi_attack_mode_source"] = "ally_signal_receive"
                            self._set_decision_reason("shi_signal")
                            return act

                # 2. 「配牌時の手札」に「し」が3枚だった場合（パス）
                elif initial_shis == 3:
                    for act in actions:
                        if act[0] == "pass":
                            self._set_decision_reason("shi_signal")
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
                        self._set_decision_reason("shi_signal")
                        return best

                    # 別の駒で攻められない場合（残り1枚など）はとりあえず「し」で受ける
                    for act in actions:
                        if act[0] == "receive" and act[1] == "1":
                            self._set_decision_reason("shi_signal")
                            return act


        # --- 第9位：総合スコア評価（通常時の最適解計算） ---
        if (
            tr is not None
            and state.phase == "receive"
            and state.attacker is not None
            and not self._same_team(state.attacker, player)
            and self._preserve_current_attack_for_win_value(state, player) > 0
        ):
            immediate_receive_win = any(
                act[0] == "receive" and self._win_after_receive_bonus(state, player, act) > 0
                for act in actions
            )
            if not immediate_receive_win:
                for act in actions:
                    if act[0] == "pass":
                        self._set_decision_reason("score_fallback")
                        self._set_score_fallback_detail("pass_preserve_win_attack")
                        return act

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

        score_fallback_detail = self._classify_score_fallback(
            state,
            player,
            best_action,
            has_non_king_attack_option=has_non_king_attack_option,
        )

        if tr is not None:
            if best_action[0] == "receive":
                tr["pending_ally_force_king_attack_piece"] = self._ally_force_king_attack_piece_after_receive(
                    state,
                    player,
                    best_action[0],
                    best_action[1],
                )
            if best_action[0] in ("attack", "attack_after_block"):
                tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                if tr.get("kg_plan_active") and tr["my_attack_count"] == 2 and best_action[2] in ("8", "9") and tr.get("kg_second") is None:
                    tr["kg_second"] = best_action[2]
                if tr.get("kg_plan_active") and tr["my_attack_count"] >= 3:
                    tr["kg_plan_active"] = False

        self._set_decision_reason("score_fallback")
        self._set_score_fallback_detail(score_fallback_detail)
        return best_action

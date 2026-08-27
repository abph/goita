"""AIが実際に選ぶ最終行動を決定します。
合法手に対して各戦略モジュールの評価を集め、確定上がりなどの優先手順も反映します。
選んだ行動と判断理由を記録し、棋譜やデバッグ表示で確認できる形にします。
"""

from __future__ import annotations

import copy
from collections import Counter
from typing import List, Optional, Tuple

from goita_ai2.constants import PIECE_TOTALS, POINTS

Action = Tuple[str, Optional[str], Optional[str]]


class DecisionMixin:
    """Combines strategy scores and selects the final legal action."""

    def _receive_before_unproven_enemy_third_attack(
        self,
        state,
        player: str,
        actions: List[Action],
        *,
        baseline_detail: str,
        search_result,
    ) -> Optional[Action]:
        """Spend the matching piece unless waiting for attack three is proven safe."""
        if (
            baseline_detail != "pass_preserve_win_attack"
            or state.phase != "receive"
            or state.current_attack is None
            or state.attacker is None
            or self._same_team(state.attacker, player)
        ):
            return None

        tracker = self._track.get(id(state))
        if tracker is None:
            return None
        if int(tracker.get("enemy_attack_counts", {}).get(state.attacker, 0)) != 2:
            return None

        wait_is_safe = bool(
            search_result is not None
            and search_result.action[0] == "pass"
            and getattr(search_result, "enemy_third_attack_wait", False)
        )
        if wait_is_safe:
            return None

        return next(
            (
                action
                for action in actions
                if action[0] == "receive"
                and action[1] == state.current_attack
                and action[1] not in ("8", "9")
            ),
            None,
        )

    def _adopt_rule_preview(self, preview) -> None:
        """Adopt preview mutations without invalidating live tracker references."""
        current_track = self._track
        for state_id in list(current_track):
            if state_id not in preview._track:
                del current_track[state_id]
        for state_id, preview_tracker in preview._track.items():
            if state_id in current_track:
                current_track[state_id].clear()
                current_track[state_id].update(preview_tracker)
            else:
                current_track[state_id] = preview_tracker

        preview_values = dict(preview.__dict__)
        preview_values.pop("_track", None)
        for performance_key in (
            "performance_totals",
            "last_performance_metrics",
            "_active_performance_metrics",
            "_active_precomputed_inference_seconds",
            "_pending_inference_seconds",
            "_time_search_cache",
            "_time_search_effective_budget",
            "last_time_search_budget",
            "_background_search_controller",
            "_time_search_cancel_event",
            "last_time_search_cache_hit",
            "last_time_search_cache_key",
            "last_time_search_cache_source",
            "last_time_search_cached_compute_ms",
            "last_time_search_cache_branch_kind",
            "last_time_search_cache_branch_context",
            "last_prediction_cache_hit",
            "last_prediction_cache_key",
            "last_prediction_cache_samples",
            "_prediction_rollforward_states",
            "_prediction_rollforward_key",
            "_prediction_cache_rollforward_enabled",
        ):
            preview_values.pop(performance_key, None)
        self.__dict__.update(preview_values)
        self._track = current_track

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
                and self._multi_shi_after_big_receive_first_attack_bonus(state, player) > 0
            ):
                if state.hands[player].count("1") == 3:
                    return "attack_three_shi_after_big_receive"
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
        """Compare the rule-based choice with a public-information search."""
        self.last_attack_candidate_scores = []
        self.last_attack_candidate_snapshot = {}
        started = self._begin_performance_decision()
        try:
            chosen = self._select_action_with_measurement(state, player, actions)
            self._finalize_attack_candidate_snapshot(actions, chosen)
            return chosen
        finally:
            self._finish_performance_decision(started)

    def _finalize_attack_candidate_snapshot(
        self,
        actions: List[Action],
        chosen: Action,
    ) -> None:
        """Keep a compact, decision-time comparison for later explanations."""
        if chosen[0] not in ("attack", "attack_after_block") or chosen[2] is None:
            return

        score_by_action = {}
        for item in self.last_attack_candidate_scores:
            action = tuple(item.get("action", ()))
            score = item.get("score")
            if len(action) == 3 and isinstance(score, (int, float)):
                score_by_action[action] = float(score)

        by_piece = {}
        piece_order = []
        for action in actions:
            action_type, block, attack = action
            if action_type not in ("attack", "attack_after_block") or attack is None:
                continue
            score = score_by_action.get(tuple(action))
            candidate = {
                "attack": str(attack),
                "score": round(score, 1) if score is not None else None,
            }
            if attack not in by_piece:
                by_piece[attack] = candidate
                piece_order.append(attack)
            elif score is not None:
                previous_score = by_piece[attack].get("score")
                if previous_score is None or score > float(previous_score):
                    by_piece[attack] = candidate

        chosen_score = score_by_action.get(tuple(chosen))
        alternatives = [
            dict(by_piece[piece])
            for piece in piece_order
            if piece != chosen[2]
        ]
        if any(item.get("score") is not None for item in alternatives):
            alternatives.sort(
                key=lambda item: (
                    item.get("score") is not None,
                    float(item["score"])
                    if item.get("score") is not None
                    else -1e18,
                ),
                reverse=True,
            )

        compact_alternatives = []
        for item in alternatives[:3]:
            alternative_score = item.get("score")
            score_gap = None
            if chosen_score is not None and alternative_score is not None:
                score_gap = round(float(chosen_score) - float(alternative_score), 1)
            compact_alternatives.append({
                "attack": item["attack"],
                "score": alternative_score,
                "score_gap": score_gap,
            })

        block_alternatives = []
        if chosen[0] == "attack_after_block" and chosen[1] is not None:
            by_block = {}
            block_order = []
            for action in actions:
                action_type, block, attack = action
                if (
                    action_type != "attack_after_block"
                    or block is None
                    or attack != chosen[2]
                    or block == chosen[1]
                ):
                    continue
                score = score_by_action.get(tuple(action))
                candidate = {
                    "block": str(block),
                    "score": round(score, 1) if score is not None else None,
                }
                if block not in by_block:
                    by_block[block] = candidate
                    block_order.append(block)
                elif score is not None:
                    previous_score = by_block[block].get("score")
                    if previous_score is None or score > float(previous_score):
                        by_block[block] = candidate

            ordered_blocks = [dict(by_block[block]) for block in block_order]
            if any(item.get("score") is not None for item in ordered_blocks):
                ordered_blocks.sort(
                    key=lambda item: (
                        item.get("score") is not None,
                        float(item["score"])
                        if item.get("score") is not None
                        else -1e18,
                    ),
                    reverse=True,
                )
            for item in ordered_blocks[:3]:
                alternative_score = item.get("score")
                score_gap = None
                if chosen_score is not None and alternative_score is not None:
                    score_gap = round(float(chosen_score) - float(alternative_score), 1)
                block_alternatives.append({
                    "block": item["block"],
                    "score": alternative_score,
                    "score_gap": score_gap,
                })

        has_hidden_block = (
            chosen[0] == "attack_after_block" and chosen[1] is not None
        )
        if not compact_alternatives and not block_alternatives and not has_hidden_block:
            return
        self.last_attack_candidate_snapshot = {
            "version": 2,
            "chosen": {
                "block": str(chosen[1]) if chosen[1] is not None else None,
                "attack": str(chosen[2]),
                "score": round(chosen_score, 1) if chosen_score is not None else None,
            },
            "alternatives": compact_alternatives,
            "block_alternatives": block_alternatives,
            "decision_reason": str(self.last_decision_reason or ""),
            "decision_detail": str(self.last_score_fallback_detail or ""),
        }

    def _select_action_with_measurement(
        self,
        state,
        player: str,
        actions: List[Action],
    ) -> Action:
        self.last_time_search_cache_hit = False
        self.last_time_search_cache_key = None
        self.last_time_search_cache_source = None
        self.last_time_search_cached_compute_ms = 0.0
        self.last_time_search_cache_branch_kind = None
        self.last_time_search_cache_branch_context = None
        self.last_prediction_cache_hit = False
        self.last_prediction_cache_key = None
        self.last_prediction_cache_samples = 0
        self.last_information_set_search = None
        self.last_branched_attack_metrics = {}
        if self.me is None:
            self.me = player
        elif self.me != player:
            raise ValueError(f"{self.name}: bound to me={self.me}, cannot play for {player}")
        self._ensure_trackers(state)

        # Preview on a clone because the established policy records plan state
        # while choosing. We only adopt those mutations when its move survives.
        with self._measure_performance("rule_based"):
            prediction_states = getattr(self, "_prediction_rollforward_states", [])
            preview = copy.deepcopy(self, {id(prediction_states): []})
            preview._prediction_cache_rollforward_enabled = False
            preview._prediction_rollforward_key = None
            baseline_action = preview._select_rule_based_action(state, player, actions)
        baseline_reason = str(preview.last_decision_reason or "")
        baseline_detail = str(preview.last_score_fallback_detail or "")
        protected = (
            baseline_reason in (
                "win_now",
                "tsume",
                "inferred_endgame",
                "conditional_tsume",
                "kakari",
                "shi_signal",
                "responded",
                "forced_king_third",
                "king_order",
                "safe_nonking_third",
            )
            or baseline_detail in ("receive_win_after", "receive_tsume_after")
            or baseline_detail.startswith("attack_sequence_")
            or baseline_detail.startswith("pass_full_receive_cover_")
            or baseline_detail.startswith("pass_ally_guaranteed_win_")
            or baseline_detail.startswith("pass_ally_kyosha_continuation")
            or baseline_detail.startswith("pass_royal_reserve_")
            or baseline_detail.startswith("receive_no_shi_royal_")
        )

        low_reentry_receive_search = (
            not protected
            and baseline_action[0] == "pass"
            and self._should_deep_search_low_reentry_receive(
                state,
                player,
                actions,
            )
        )
        weak_first_receive_search = (
            not protected
            and not low_reentry_receive_search
            and self._should_deep_search_weak_first_receive(
                state,
                player,
                actions,
            )
        )
        deep_receive_search = (
            low_reentry_receive_search or weak_first_receive_search
        )

        search_result = None
        if not protected:
            previous_profile = str(
                getattr(self, "_time_search_profile", "default")
            )
            if low_reentry_receive_search:
                search_profile = "low_reentry_receive"
            elif weak_first_receive_search:
                search_profile = "weak_first_receive"
            else:
                search_profile = "default"
            self._time_search_profile = search_profile
            if deep_receive_search:
                budget_plan = self._prepare_time_search_budget(
                    state,
                    player,
                    actions,
                    configured_seconds=self.WEAK_FIRST_RECEIVE_SEARCH_MAX_SECONDS,
                    configured_samples=self.WEAK_FIRST_RECEIVE_SEARCH_SAMPLE_COUNT,
                    configured_depth=self.WEAK_FIRST_RECEIVE_SEARCH_MAX_DEPTH,
                    configured_nodes=self.WEAK_FIRST_RECEIVE_SEARCH_MAX_NODES,
                    adaptive_enabled=False,
                )
            else:
                budget_plan = self._prepare_time_search_budget(state, player, actions)
            try:
                search_result = self._time_limited_search_action(
                    state,
                    player,
                    actions,
                    baseline_action,
                    cancel_event=getattr(self, "_time_search_cancel_event", None),
                )
            finally:
                self._finish_time_search_budget(budget_plan, search_result)
                self._time_search_profile = previous_profile

        if search_result is not None:
            search_snapshot = search_result.as_dict()
            search_snapshot["cache_hit"] = bool(self.last_time_search_cache_hit)
            search_snapshot["cache_key"] = self.last_time_search_cache_key
            search_snapshot["cache_source"] = self.last_time_search_cache_source
            search_snapshot["cache_branch_kind"] = (
                self.last_time_search_cache_branch_kind
            )
            search_snapshot["cache_branch_context"] = (
                self.last_time_search_cache_branch_context
            )
            search_snapshot["budget"] = dict(self.last_time_search_budget or {})
            search_snapshot["prediction_cache_hit"] = bool(
                self.last_prediction_cache_hit
            )
            search_snapshot["prediction_cache_key"] = self.last_prediction_cache_key
            search_snapshot["prediction_cache_samples"] = int(
                self.last_prediction_cache_samples
            )
            preview_tracker = preview._track.get(id(state))
            if preview_tracker is not None:
                preview_tracker["last_time_limited_search"] = dict(search_snapshot)
            tracker = self._track.get(id(state))
            if tracker is not None:
                tracker["last_time_limited_search"] = dict(search_snapshot)

        if (
            search_result is not None
            and search_result.decisive
            and search_result.action != baseline_action
        ):
            self.last_attack_candidate_scores = copy.deepcopy(
                getattr(preview, "last_attack_candidate_scores", [])
            )
            self._commit_timed_search_action(state, player, search_result.action)
            if (
                low_reentry_receive_search
                and search_result.action[0] == "receive"
            ):
                tracker = self._track.get(id(state))
                if tracker is not None:
                    tracker["pending_low_reentry_attack_piece"] = (
                        self._low_reentry_followup_piece(
                            state,
                            player,
                            str(search_result.action[1]),
                        )
                    )
            self._set_decision_reason("time_search")
            if getattr(search_result, "enemy_third_attack_wait", False):
                self._set_score_fallback_detail(
                    f"wait_enemy_third_guaranteed_win_"
                    f"depth_{search_result.depth}_samples_{search_result.samples}_"
                    f"agreement_{int(round(search_result.agreement * 100))}"
                )
            else:
                self._set_score_fallback_detail(
                    f"{'low_reentry_receive_' if low_reentry_receive_search else ''}"
                    f"{'weak_first_receive_' if weak_first_receive_search else ''}"
                    f"{'cache_' if self.last_time_search_cache_hit else ''}"
                    f"depth_{search_result.depth}_samples_{search_result.samples}_"
                    f"agreement_{int(round(search_result.agreement * 100))}"
                )
            return search_result.action

        if (
            search_result is not None
            and baseline_action[0] == "pass"
            and getattr(search_result, "enemy_third_attack_wait", False)
        ):
            self._adopt_rule_preview(preview)
            self._set_decision_reason("time_search")
            self._set_score_fallback_detail(
                f"wait_enemy_third_guaranteed_win_"
                f"depth_{search_result.depth}_samples_{search_result.samples}_"
                f"agreement_{int(round(search_result.agreement * 100))}"
            )
            return baseline_action

        receive_before_third = self._receive_before_unproven_enemy_third_attack(
            state,
            player,
            actions,
            baseline_detail=baseline_detail,
            search_result=search_result,
        )
        if receive_before_third is not None:
            self.last_attack_candidate_scores = copy.deepcopy(
                getattr(preview, "last_attack_candidate_scores", [])
            )
            self._set_decision_reason("score_fallback")
            self._set_score_fallback_detail(
                "receive_before_unproven_enemy_third_attack"
            )
            return receive_before_third

        self._adopt_rule_preview(preview)
        return baseline_action

    def _low_reentry_followup_piece(
        self,
        state,
        player: str,
        receive_piece: str,
    ) -> Optional[str]:
        """Choose the strongest public-information attack after receiving."""
        tracker = self._track.get(id(state))
        if tracker is None:
            return None

        remaining = Counter(state.hands[player])
        if remaining.get(receive_piece, 0) <= 0:
            return None
        remaining[receive_piece] -= 1
        public_seen = tracker.get("public_seen_counts", {})

        ranked = []
        for piece, count in remaining.items():
            if count <= 0 or piece in ("8", "9"):
                continue
            outside = max(
                0,
                int(PIECE_TOTALS[piece])
                - int(public_seen.get(piece, 0))
                - int(count),
            )
            is_strong = outside <= 1 or count >= 3
            if not is_strong:
                continue
            ranked.append(
                (
                    1 if outside == 0 else 0,
                    -outside,
                    int(POINTS[piece]),
                    int(count),
                    piece,
                )
            )

        if not ranked:
            return None
        ranked.sort(reverse=True)
        return str(ranked[0][-1])

    def _should_deep_search_low_reentry_receive(
        self,
        state,
        player: str,
        actions: List[Action],
    ) -> bool:
        """Search when passing as the last responder may end participation."""
        if (
            state.phase != "receive"
            or state.current_attack is None
            or state.attacker is None
            or self._same_team(state.attacker, player)
            or state.next_player(player) != state.attacker
        ):
            return False

        current_attack = str(state.current_attack)
        if not any(action[0] == "pass" for action in actions):
            return False
        if not any(
            action[0] == "receive" and action[1] == current_attack
            for action in actions
        ):
            return False

        hand = state.hands[player]
        if "1" in hand or "8" in hand or "9" in hand:
            return False

        tracker = self._track.get(id(state))
        if tracker is None:
            return False
        public_seen = tracker.get("public_seen_counts", {})
        remaining = Counter(hand)
        remaining[current_attack] -= 1

        future_receive_types = 0
        for piece, count in remaining.items():
            if count <= 0 or piece in (current_attack, "8", "9"):
                continue
            outside = max(
                0,
                int(PIECE_TOTALS[piece])
                - int(public_seen.get(piece, 0))
                - int(count),
            )
            if outside > 0:
                future_receive_types += 1

        return (
            future_receive_types <= 2
            and self._low_reentry_followup_piece(
                state,
                player,
                current_attack,
            )
            is not None
        )

    def _should_deep_search_weak_first_receive(
        self,
        state,
        player: str,
        actions: List[Action],
    ) -> bool:
        """Use a deeper pass-vs-receive comparison for weak first responses."""
        if (
            state.phase != "receive"
            or state.current_attack in (None, "1", "2")
            or state.attacker is None
            or self._same_team(state.attacker, player)
        ):
            return False

        tracker = self._track.get(id(state))
        if tracker is None:
            return False
        if int(tracker.get("enemy_attack_counts", {}).get(state.attacker, 1)) != 1:
            return False
        if player != state.next_player(state.attacker):
            return False

        current_attack = str(state.current_attack)
        has_pass = any(action[0] == "pass" for action in actions)
        has_same_receive = any(
            action[0] == "receive" and action[1] == current_attack
            for action in actions
        )
        if not (has_pass and has_same_receive):
            return False

        axes = self._initial_hand_axes_for_state(state, player)
        absolute_rank = str(axes.get("absolute_rank", axes.get("rank", "D")))
        return absolute_rank in ("D", "E", "F", "X")

    def _select_rule_based_action(self, state, player: str, actions: List[Action]) -> Action:
        self._set_decision_reason("")
        self._set_score_fallback_detail("")
        self.last_attack_candidate_scores = []

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

        # Compare every publicly proven finish before applying local attack plans.
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
                tr["pending_weak_hand_shi_signal"] = False
                tr["pending_ally_force_king_attack_piece"] = None
                tr["pending_inferred_endgame_attack"] = None
                if (
                    tr.get("kg_plan_active")
                    and tr["my_attack_count"] == 2
                    and chosen[2] in ("8", "9")
                    and tr.get("kg_second") is None
                ):
                    tr["kg_second"] = chosen[2]
                if tr.get("kg_plan_active") and tr["my_attack_count"] >= 3:
                    tr["kg_plan_active"] = False
            self._set_decision_reason("win_now" if immediate else "tsume")
            self._set_score_fallback_detail(f"high_score_{int(route_score)}")
            return chosen

        royal_bridge = self._royal_bridge_finish_action(state, player, actions)
        if royal_bridge is not None:
            chosen, finish_score = royal_bridge
            if tr is not None:
                tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                tr["pending_weak_hand_shi_signal"] = False
                tr["pending_ally_force_king_attack_piece"] = None
                tr["pending_inferred_endgame_attack"] = None
            self._set_decision_reason("tsume")
            self._set_score_fallback_detail(
                f"royal_bridge_high_score_{int(finish_score)}"
            )
            return chosen

        if tr is not None and tr.get("pending_inferred_endgame_attack") is not None:
            planned_attack = tr.get("pending_inferred_endgame_attack")
            tr["pending_inferred_endgame_attack"] = None
            if planned_attack in actions:
                tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                self._set_decision_reason("inferred_endgame")
                self._set_score_fallback_detail("inferred_endgame_followup_attack")
                return planned_attack

        if tr is not None and tr.get("pending_low_reentry_attack_piece") is not None:
            planned_piece = str(tr.get("pending_low_reentry_attack_piece"))
            tr["pending_low_reentry_attack_piece"] = None
            planned_attack = next(
                (
                    action
                    for action in actions
                    if action[0] in ("attack", "attack_after_block")
                    and action[2] == planned_piece
                ),
                None,
            )
            if planned_attack is not None:
                tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                self._set_decision_reason("time_search")
                self._set_score_fallback_detail("low_reentry_followup_attack")
                return planned_attack

        if tr is not None and tr.get("pending_weak_hand_shi_signal"):
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
                win_now_actions.sort(key=lambda item: item[0], reverse=True)
                chosen = win_now_actions[0][1]
                tr["pending_weak_hand_shi_signal"] = False
                tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                self._set_decision_reason("win_now")
                return chosen

            shi_attack = next((act for act in attack_actions if act[2] == "1"), None)
            tr["pending_weak_hand_shi_signal"] = False
            if shi_attack is not None:
                tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                self._set_decision_reason("score_fallback")
                self._set_score_fallback_detail("attack_weak_hand_shi_signal")
                return shi_attack

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

        four_shi_return = self._four_shi_receive_return_action(
            state,
            player,
            actions,
        )
        if four_shi_return is not None:
            if tr is not None:
                tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
                tr["shi_attack_mode"] = True
                tr["shi_attack_mode_source"] = "four_shi_receive_return"
            self._set_decision_reason("score_fallback")
            self._set_score_fallback_detail("attack_four_shi_receive_return")
            return four_shi_return

        give_way_action = self._give_way_to_ally_guaranteed_win_action(state, player, actions)
        if give_way_action is not None:
            return give_way_action

        full_receive_cover_wait = self._full_receive_cover_royal_wait_pass_action(
            state,
            player,
            actions,
        )
        if full_receive_cover_wait is not None:
            return full_receive_cover_wait

        guaranteed_finish_receive = self._guaranteed_finish_receive_action(state, player, actions)
        if guaranteed_finish_receive is not None:
            # A compact exact endgame may expose a safe wait whose team score is
            # higher than the direct self-finish. Compare that route before
            # committing to the guaranteed receive; otherwise keep the proven
            # self-finish as the fallback.
            guaranteed_result = self._forced_win_result_after_receive_action(
                state,
                player,
                guaranteed_finish_receive,
            )
            higher_score_endgame = self._inferred_endgame_team_result_action(
                state,
                player,
                actions,
            )
            guaranteed_score = float(guaranteed_result.minimum_score or 0.0)
            if higher_score_endgame is not None:
                chosen, winner, score = higher_score_endgame
                if (
                    self._same_team(winner, player)
                    and float(score) > guaranteed_score
                ):
                    self._set_decision_reason("inferred_endgame")
                    winner_role = "self" if winner == player else "ally"
                    self._set_score_fallback_detail(
                        f"inferred_endgame_{winner_role}_win_{winner}_{score}"
                    )
                    return chosen
            if tr is not None:
                tr["pending_inferred_endgame_attack"] = None
            return guaranteed_finish_receive

        no_shi_royal_commit = self._no_shi_royal_endgame_commit_action(
            state,
            player,
            actions,
        )
        if no_shi_royal_commit is not None:
            return no_shi_royal_commit

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

        ally_kyosha_continuation = self._ally_kyosha_continuation_pass_action(
            state,
            player,
            actions,
        )
        if ally_kyosha_continuation is not None:
            return ally_kyosha_continuation

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

        # Guaranteed routes were already resolved at the top of this turn.
        # From here on, every action is a non-proven strategic alternative.
        protected_tsume_action = None
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

        enemy_team_shi_pressure = self._inferred_enemy_team_shi_attack_action(
            state,
            player,
            actions,
            has_non_king_attack_option=has_non_king_attack_option,
        )
        if enemy_team_shi_pressure is not None:
            chosen, pressure = enemy_team_shi_pressure
            if tr is not None:
                tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
            self._set_decision_reason("score_fallback")
            self._set_score_fallback_detail(
                f"attack_enemy_team_shi_remaining_{int(pressure['level'])}"
            )
            return chosen

        branched_choice = self._production_branched_attack_action(
            state,
            player,
            actions,
        )
        if branched_choice is not None:
            source = branched_choice.active.plan.source
            action_type, block, attack = branched_choice.action
            conditional_finish_score = self._conditional_shi_royal_finish_score(
                state,
                player,
                action_type,
                block,
                attack,
            )
            classified_detail = self._classify_score_fallback(
                state,
                player,
                branched_choice.action,
                has_non_king_attack_option=has_non_king_attack_option,
            )
            if conditional_finish_score is not None:
                self._set_decision_reason("score_fallback")
                self._set_score_fallback_detail(
                    f"attack_conditional_shi_royal_finish_"
                    f"{int(conditional_finish_score)}"
                )
            elif source.startswith("representative:"):
                template_id = source.split(":", 1)[1]
                self._set_decision_reason("score_fallback")
                if (
                    template_id.startswith("fourth_middle_finisher_")
                    and classified_detail not in (
                        "attack_piece_value",
                        "block_piece_penalty",
                    )
                ):
                    self._set_score_fallback_detail(classified_detail)
                else:
                    self._set_score_fallback_detail(f"attack_sequence_{template_id}")
            elif classified_detail not in ("attack_piece_value", "block_piece_penalty"):
                self._set_decision_reason("score_fallback")
                self._set_score_fallback_detail(classified_detail)
            else:
                evaluation = branched_choice.active.plan.evaluation
                self._set_decision_reason("branched_plan")
                self._set_score_fallback_detail(
                    f"attack_branched_plan_"
                    f"{'continue' if branched_choice.continued else 'new'}_"
                    f"min_{int(evaluation.minimum_score)}_"
                    f"risk_{int(round(evaluation.failure_risk * 100))}"
                )
            self._commit_branched_attack_choice(state, branched_choice)
            return branched_choice.action

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

        shallow_eight_card = self._eight_card_shallow_plan_action(
            state,
            player,
            actions,
            has_non_king_attack_option=has_non_king_attack_option,
        )
        if shallow_eight_card is not None:
            chosen, plan = shallow_eight_card
            if tr is not None:
                tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
            self._set_decision_reason("score_fallback")
            self._set_score_fallback_detail(
                f"attack_shallow_eight_card_plan_{int(float(plan['finish_score']))}"
            )
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
                if tr.get("my_shi_approval_sent"):
                    pass_action = next(
                        (act for act in actions if act[0] == "pass"),
                        None,
                    )
                    if pass_action is not None:
                        self._set_decision_reason("shi_signal")
                        self._set_score_fallback_detail(
                            "ally_shi_approval_already_sent_pass"
                        )
                        return pass_action
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
                            tr["my_shi_approval_pending"] = True
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
        scored_actions = []

        for (t, block, attack) in actions:
            if t == "attack_after_block":
                score = self._score_receive_phase(state, player, "receive", block)
                score += self._score_attack_phase(state, player, t, block, attack, has_non_king_attack_option=has_non_king_attack_option)
            elif t == "attack":
                score = self._score_attack_phase(state, player, t, block, attack, has_non_king_attack_option=has_non_king_attack_option)
            else:
                score = self._score_receive_phase(state, player, t, block)

            scored_actions.append({
                "action": (t, block, attack),
                "score": float(score),
            })

            if score > best_score:
                best_score = score
                best_action = (t, block, attack)

        self.last_attack_candidate_scores = scored_actions

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

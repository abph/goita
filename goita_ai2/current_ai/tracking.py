"""対局中に公開された行動と駒を記録します。
各プレイヤーの攻め回数、受け、パス、公開済みの駒、し攻めの合図などを保存します。
行動のたびにinference.pyを更新し、以後の判断で最新の推定を利用できるようにします。
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional, Tuple

Action = Tuple[str, Optional[str], Optional[str]]


class TrackingMixin:
    """Stores public actions and refreshes inference state after each move."""

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

        public_seen_counts = {str(i): 0 for i in range(1, 10)}

        self._track[sid] = dict(
            my_init_count=cnt_all,
            ally=ally_player,
            public_seen_counts=public_seen_counts,

            my_attack_count=0,
            my_attack_history=[],
            special_attack_plan=self._special_attack_sequence_plan(cnt_all),
            shallow_eight_card_plan=None,
            kg_plan_active=(("9" in init_hand) and ("8" in init_hand)),
            kg_second=None,

            my_past_attacks=set(),
            ally_past_attacks=set(),
            enemy_past_attacks=set(),
            enemy_attack_counts={},
            hidden_block_counts={p: 0 for p in ("A", "B", "C", "D")},
            other_first_attack_strategy_by_player={},
            other_piece_count_estimates={},
            pending_partner_first_strategy_reactions={},
            current_piece_count_caps={},
            piece_pass_evidence={p: {} for p in ("A", "B", "C", "D")},
            estimated_current_hands={},
            joint_hand_inference={},
            unknown_piece_pool={},
            piece_inference_revision=0,
            last_piece_inference_reason="initial_hand",

            ally_responded_to_my_attacks=set(),
            ally_ignored_my_attacks=set(),
            ally_pending_response_piece=None,
            ally_passed_my_shi_count=0,
            enemy_passed_my_shi_count=0,
            ally_shi_signal="unknown",
            shi_attack_mode=False,
            shi_attack_mode_source=None,
            my_shi_approval_pending=False,
            my_shi_approval_sent=False,
            i_passed_ally_shi=False,
            inherit_ally_shi_attack=False,
            my_open_shi_attack_pending=False,
            ally_open_shi_attack_pending=False,
            ally_shi_passed_by_enemy=False,
            ally_shi_sashikomi_candidate=False,
            ally_consumed_count=0,
            ally_passed_enemy_dealer_first_attack=False,
            ally_passed_enemy_dealer_first_attack_piece=None,
            ally_passed_enemy_first_attack=False,
            ally_passed_enemy_first_attack_attacker=None,
            ally_passed_enemy_first_attack_piece=None,
            pending_ally_force_king_attack_piece=None,
            pending_weak_hand_shi_signal=False,
            pending_inferred_endgame_attack=None,
            last_forced_win_score_plan=None,
            my_last_receive_piece=None,
            enemy_pending_shi_receive_players=set(),
            enemy_team_rejected_shi_attack=False,
            active_attack_context=None,
            public_hand_models={
                p: dict(
                    strength=0.0,
                    estimated_rank_score=5.0,
                    estimated_rank="D",
                    rank_confidence=0.0,
                    rank_evidence_weight=0.0,
                    rank_history=[],
                    attack_count=0,
                    receive_count=0,
                    pass_count=0,
                    attacks=Counter(),
                    blocks=Counter(),
                    first_attack=None,
                    inferred_attack_strategy=None,
                    inferred_attack_strategy_active=False,
                    strategy_broken=False,
                    partner_first_strategy_rejected=False,
                    partner_first_strategy_reaction=None,
                )
                for p in ("A", "B", "C", "D")
            },

            perfect_plan=None,
            perfect_plan_step=0,
        )
        self._refresh_public_piece_inference(
            state,
            self._track[sid],
            reason="initial_hand",
        )

    def on_public_action(self, state, player: str, action: Action) -> None:
        if self.me is None:
            return
        self._ensure_trackers(state)
        tr = self._track.get(id(state))
        if tr is None:
            return

        action_type, block, attack = action
        visible_block = block
        if action_type == "attack_after_block" and player != self.me:
            visible_block = None

        self._update_public_hand_model(state, tr, player, action_type, visible_block, attack)

        if action_type == "attack_after_block":
            hidden_counts = tr.setdefault("hidden_block_counts", {p: 0 for p in ("A", "B", "C", "D")})
            hidden_counts[player] = int(hidden_counts.get(player, 0)) + 1

        if player == tr.get("ally"):
            consumed = 0
            if action_type == "attack_after_block":
                consumed = 2
            elif action_type in ("receive", "attack"):
                consumed = 1
            if consumed:
                tr["ally_consumed_count"] = min(8, int(tr.get("ally_consumed_count", 0)) + consumed)

        if (
            action_type == "pass"
            and player == tr.get("ally")
            and state.attacker == self.me
            and state.current_attack == "1"
        ):
            tr["ally_passed_my_shi_count"] = int(tr.get("ally_passed_my_shi_count", 0)) + 1
            if tr.get("ally_shi_signal") == "unknown":
                tr["ally_shi_signal"] = "passed"

        if (
            action_type == "pass"
            and player == self.me
            and state.attacker == tr.get("ally")
            and state.current_attack == "1"
        ):
            tr["i_passed_ally_shi"] = True

        if (
            action_type == "pass"
            and player != self.me
            and not self._same_team(player, self.me)
            and state.attacker == self.me
            and state.current_attack == "1"
        ):
            tr["enemy_passed_my_shi_count"] = int(tr.get("enemy_passed_my_shi_count", 0)) + 1

        if (
            action_type == "pass"
            and player != self.me
            and not self._same_team(player, self.me)
            and state.attacker == tr.get("ally")
            and state.current_attack == "1"
        ):
            tr["ally_shi_passed_by_enemy"] = True

        if (
            action_type == "pass"
            and player == tr.get("ally")
            and state.attacker is not None
            and state.current_attack is not None
            and not self._same_team(state.attacker, self.me)
            and tr.get("enemy_attack_counts", {}).get(state.attacker, 1) == 1
            and player == state.next_player(state.attacker)
        ):
            tr["ally_passed_enemy_first_attack"] = True
            tr["ally_passed_enemy_first_attack_attacker"] = state.attacker
            tr["ally_passed_enemy_first_attack_piece"] = str(state.current_attack)

            dealer = getattr(state, "dealer", None)
            if dealer is not None and state.attacker == dealer:
                tr["ally_passed_enemy_dealer_first_attack"] = True
                tr["ally_passed_enemy_dealer_first_attack_piece"] = str(state.current_attack)

        dealer = getattr(state, "dealer", None)
        if (
            action_type == "pass"
            and dealer is not None
            and player == tr.get("ally")
            and state.attacker == dealer
            and state.current_attack is not None
            and not self._same_team(state.attacker, self.me)
            and tr.get("enemy_attack_counts", {}).get(state.attacker, 1) == 1
        ):
            tr["ally_passed_enemy_dealer_first_attack"] = True
            tr["ally_passed_enemy_dealer_first_attack_piece"] = str(state.current_attack)

        if player == self.me and action_type in ("attack", "attack_after_block"):
            if tr.get("my_shi_approval_pending"):
                tr["my_shi_approval_sent"] = (attack == "1")
                tr["my_shi_approval_pending"] = False
            tr["pending_ally_force_king_attack_piece"] = None
            tr["my_last_receive_piece"] = None

        if action_type in ("receive", "attack_after_block") and visible_block is not None:
            if visible_block in tr["public_seen_counts"]:
                tr["public_seen_counts"][visible_block] += 1
            if player == self.me and action_type == "receive":
                tr["my_last_receive_piece"] = visible_block
            if player == tr.get("ally") and action_type == "receive":
                if visible_block in tr.get("my_past_attacks", set()):
                    tr["ally_pending_response_piece"] = visible_block
                else:
                    tr["ally_pending_response_piece"] = None
            if (
                player is not None
                and player != self.me
                and not self._same_team(player, self.me)
                and action_type == "receive"
                and visible_block == "1"
            ):
                tr.setdefault("enemy_pending_shi_receive_players", set()).add(player)
            if (
                visible_block == "1"
                and state.current_attack == "1"
                and player is not None
                and not self._same_team(player, self.me)
                and tr.get("i_passed_ally_shi")
            ):
                tr["inherit_ally_shi_attack"] = True

        if action_type in ("attack", "attack_after_block") and attack is not None:
            if attack in tr["public_seen_counts"]:
                tr["public_seen_counts"][attack] += 1

            if player == self.me:
                tr["my_past_attacks"].add(attack)
                tr.setdefault("my_attack_history", []).append(attack)
                tr["my_last_attack"] = attack
                tr["ally_attacked_since_my_last_attack"] = False
                if attack == "1":
                    tr["my_open_shi_attack_pending"] = True
                    if tr.get("my_init_count", Counter()).get("1", 0) >= 3:
                        tr["shi_attack_mode"] = True
                        tr["shi_attack_mode_source"] = "self"
                    if tr.get("inherit_ally_shi_attack"):
                        tr["inherit_ally_shi_attack"] = False
                else:
                    tr["my_open_shi_attack_pending"] = False
                    if tr.get("shi_attack_mode") and state.hands[player].count("1") > 0:
                        tr["shi_attack_mode"] = False
                        tr["shi_attack_mode_source"] = None
            elif self._same_team(player, self.me):
                if not tr["ally_past_attacks"]:
                    tr["ally_first_attack"] = attack
                tr["ally_past_attacks"].add(attack)
                tr["ally_last_attack"] = attack
                tr["ally_attacked_since_my_last_attack"] = True
                # An ally replaying one of my earlier attack pieces is a response
                # (including kakarigotae), even when it follows another receive.
                if attack in tr["my_past_attacks"]:
                    tr["ally_responded_to_my_attacks"].add(attack)
                    tr["ally_ignored_my_attacks"].discard(attack)
                if attack == "1":
                    tr["ally_open_shi_attack_pending"] = True
                else:
                    tr["ally_open_shi_attack_pending"] = False

                pending_response = tr.get("ally_pending_response_piece")
                if pending_response is not None and action_type == "attack":
                    if pending_response == "1" and "1" in tr["my_past_attacks"]:
                        if attack == "1":
                            tr["ally_shi_signal"] = "returned_shi"
                            tr["shi_attack_mode"] = True
                            tr["ally_shi_sashikomi_candidate"] = False
                        else:
                            if int(tr.get("enemy_passed_my_shi_count", 0)) > 0:
                                tr["ally_shi_signal"] = "sashikomi"
                                tr["ally_shi_sashikomi_candidate"] = True
                            else:
                                tr["ally_shi_signal"] = "weak"
                            tr["shi_attack_mode"] = False
                            tr["shi_attack_mode_source"] = None
                    if attack == pending_response and attack in tr["my_past_attacks"]:
                        tr["ally_responded_to_my_attacks"].add(attack)
                    elif pending_response in tr["my_past_attacks"] and pending_response not in tr["ally_responded_to_my_attacks"]:
                        tr["ally_ignored_my_attacks"].add(pending_response)
                    if pending_response == "1":
                        tr["my_open_shi_attack_pending"] = False
                    tr["ally_pending_response_piece"] = None

                if action_type == "attack_after_block":
                    if block == "1" and tr.get("my_open_shi_attack_pending"):
                        if attack == "1":
                            tr["ally_shi_signal"] = "returned_shi"
                            tr["ally_shi_sashikomi_candidate"] = False
                        elif int(tr.get("enemy_passed_my_shi_count", 0)) > 0:
                            tr["ally_shi_signal"] = "sashikomi"
                            tr["ally_shi_sashikomi_candidate"] = True
                        else:
                            tr["ally_shi_signal"] = "weak"
                        if attack == "1":
                            tr["shi_attack_mode"] = True
                        else:
                            tr["shi_attack_mode"] = False
                            tr["shi_attack_mode_source"] = None
                        tr["my_open_shi_attack_pending"] = False
                    if attack in tr["my_past_attacks"]:
                        tr["ally_responded_to_my_attacks"].add(attack)

                    for past_attack in tr["my_past_attacks"]:
                        if past_attack != attack and past_attack not in tr["ally_responded_to_my_attacks"]:
                            tr["ally_ignored_my_attacks"].add(past_attack)
            else:
                if action_type in ("receive", "attack_after_block") and block == "1" and tr.get("my_open_shi_attack_pending"):
                    tr["my_open_shi_attack_pending"] = False

                pending_shi_receivers = tr.setdefault("enemy_pending_shi_receive_players", set())
                if player in pending_shi_receivers:
                    if attack != "1":
                        tr["enemy_team_rejected_shi_attack"] = True
                    pending_shi_receivers.discard(player)

                tr["enemy_past_attacks"].add(attack)
                if "enemy_attack_counts" not in tr:
                    tr["enemy_attack_counts"] = {}
                previous_count = tr["enemy_attack_counts"].get(player, 0)
                tr["enemy_attack_counts"][player] = previous_count + 1

            self._infer_missing_kings_from_third_visible_attack(
                state,
                tr,
                player,
                action_type,
                attack,
            )

        visible_evidence: List[str] = []
        if action_type == "receive" and visible_block is not None:
            visible_evidence.append(f"receive:{visible_block}")
        if action_type in ("attack", "attack_after_block") and attack is not None:
            visible_evidence.append(f"attack:{attack}")
        # My original hand is already excluded from the unknown pool, so my own
        # action cannot change the three-player allocation seen by this agent.
        if player != self.me:
            evidence_label = ",".join(visible_evidence) if visible_evidence else action_type
            self._refresh_public_piece_inference(
                state,
                tr,
                reason=f"{player}:{evidence_label}",
            )

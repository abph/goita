"""Communicate limited shi support by taking over with a viable attack.

This is a public tactical signal, not a claim that the partner has zero shi.
Only Experimental AI2 uses it; saved AI profiles keep their old conventions.
"""
from __future__ import annotations

import copy
from collections import Counter


class ShiSupportChangeMixin:
    def _intercepted_shi_source(self, state, receiver, incoming):
        """Find shi -> enemy receive -> different attack, allowing passes."""
        history = self._neural_public_history_by_state_id.get(id(state), ())
        end = len(history)
        while end and history[end - 1]["action"][0] == "pass":
            end -= 1
        if end < 3:
            return None
        receive, attack = history[end - 2:end]
        source_index = end - 3
        while source_index >= 0 and history[source_index]["action"][0] == "pass":
            source_index -= 1
        if source_index < 0:
            return None
        source = history[source_index]
        enemy = attack["player"]
        if (
            source["action"][0] not in ("attack", "attack_after_block")
            or source["action"][2] != "1"
            or receive["player"] != enemy
            or tuple(receive["action"]) != ("receive", "1", None)
            or attack["action"][0] != "attack" or attack["action"][2] != incoming
            or self._same_team(enemy, receiver)
            or source["player"] != self._ally_of(receiver)
        ):
            return None
        return {"source": source["player"], "interceptor": enemy,
                "received_piece": incoming}

    def _limited_shi_support_context(self, state, player, actions):
        tracker = self._track.get(id(state), {})
        incoming = state.current_attack
        receive = ("receive", incoming, None)
        if (
            state.phase != "receive" or incoming not in ("3", "4", "5", "6", "7")
            or receive not in actions or len(state.hands[player]) < 6
            or state.hands[player].count("1") > 1
            or not tracker.get("ally_open_shi_attack_pending")
            or tracker.get("shi_attack_mode") or tracker.get("inherit_ally_shi_attack")
            or tracker.get("my_shi_approval_sent")
            or tracker.get("my_limited_shi_support_sent")
            or any(len(state.hands[seat]) <= 2 for seat in "ABCD" if seat != player)
        ):
            return None
        source = self._intercepted_shi_source(state, player, incoming)
        if source is None or source["interceptor"] != state.attacker:
            return None
        return dict(source, own_shi=state.hands[player].count("1"),
                    attack_count_before=int(tracker.get("my_attack_count", 0)))

    def _limited_shi_support_plan(self, state, player, actions):
        context = self._limited_shi_support_context(state, player, actions)
        if context is None:
            return None

        prediction_states = getattr(self, "_prediction_rollforward_states", [])
        preview = copy.deepcopy(self, {id(prediction_states): []})
        after = copy.deepcopy(state)
        preview._track[id(after)] = copy.deepcopy(self._track[id(state)])
        preview._my_initial_hands_by_state_id[id(after)] = list(self._get_my_initial_hand(state))
        preview._neural_public_history_by_state_id[id(after)] = copy.deepcopy(
            self._neural_public_history_by_state_id.get(id(state), [])
        )
        preview.TIME_SEARCH_ENABLED = False
        preview.TIME_SEARCH_BACKGROUND_ENABLED = False
        preview._prediction_cache_rollforward_enabled = False
        after.apply_receive(player, context["received_piece"])
        preview.on_public_action(after, player, ("receive", context["received_piece"], None))
        follow_actions = after.legal_actions(player)
        before_choice = copy.deepcopy(preview._track[id(after)])
        established = preview._select_rule_based_action(after, player, follow_actions)
        preview._track[id(after)].clear()
        preview._track[id(after)].update(before_choice)
        counts = Counter(after.hands[player])
        candidates = []
        for action in follow_actions:
            piece = action[2]
            if action[0] != "attack" or piece not in ("2", "3", "4", "5", "6", "7"):
                continue
            # A singleton middle piece without a concrete purpose is not a
            # useful replacement signal. Do not echo an enemy's reserved piece.
            if piece in "2345" and counts[piece] < 2 and not preview._singleton_first_attack_has_concrete_purpose(
                after, player, *action,
            ):
                continue
            if preview._enemy_used_attack_reuse_penalty(after, player, piece) > 0:
                continue
            future = preview._future_attack_plan_for_action(after, player, action)
            if future is None or not future.get("steps") or future.get("finish_score", 0) <= 0:
                continue
            score = preview._score_attack_phase(after, player, *action, has_non_king_attack_option=True)
            candidates.append((action == established, score, action, future))
        if not candidates:
            return None
        _established, _score, attack, future = max(candidates, key=lambda row: row[:2])
        return dict(context, followup=attack[2],
                    expected_hand_after_receive=sorted(after.hands[player]),
                    future_attacks=[attack[2], *future["attacks"]],
                    finish_score=future["finish_score"], received=False,
                    evidence="own_hand_and_public_sequence", proven=False)

    def _select_rule_based_action(self, state, player, actions):
        self.last_shi_support_change = None
        self._ensure_trackers(state)
        tracker = self._track.get(id(state), {})
        pending = tracker.get("limited_shi_support_plan")
        before = copy.deepcopy(tracker) if pending and state.phase == "attack" else None
        chosen = super()._select_rule_based_action(state, player, actions)
        tracker = self._track.get(id(state), {})
        authority = self._rule_search_authority(self.last_decision_reason, self.last_score_fallback_detail)
        protected = authority == "proven" or self.last_decision_reason in (
            "win_now", "tsume", "conditional_tsume", "inferred_endgame", "upside_finish",
        )
        if pending and state.phase == "attack":
            followup = ("attack", None, pending["followup"])
            valid = (
                pending.get("received")
                and sorted(state.hands[player]) == pending["expected_hand_after_receive"]
                and before.get("my_attack_count", 0) == pending["attack_count_before"]
                and followup in actions
            )
            if valid and not protected:
                tracker.clear()
                tracker.update(before)
                tracker["my_attack_count"] = pending["attack_count_before"] + 1
                self.last_attack_candidate_scores = []
                self._set_decision_reason("shi_signal")
                self._set_score_fallback_detail(f"attack_limited_shi_support_{pending['followup']}")
                self.last_shi_support_change = copy.deepcopy(pending)
                return followup
            tracker.pop("limited_shi_support_plan", None)
        if chosen == ("pass", None, None) and authority == "ordinary" and not protected:
            plan = self._limited_shi_support_plan(state, player, actions)
            if plan is not None:
                tracker["limited_shi_support_plan"] = plan
                self.last_attack_candidate_scores = []
                self._set_decision_reason("shi_signal")
                self._set_score_fallback_detail(
                    f"receive_limited_shi_support_{plan['own_shi']}_followup_{plan['followup']}"
                )
                self.last_shi_support_change = copy.deepcopy(plan)
                return ("receive", plan["received_piece"], None)
        return chosen

    def _lookup_conditional_response_plan(self, state, player, actions, baseline_action):
        # A cached ordinary pass must not silently erase this new purpose.
        if baseline_action[0] == "receive" and self._limited_shi_support_context(state, player, actions):
            return None
        return super()._lookup_conditional_response_plan(state, player, actions, baseline_action)

    def on_public_action(self, state, player, action):
        tracker = self._track.get(id(state))
        pending = tracker.get("limited_shi_support_plan") if tracker else None
        reply = None
        if (tracker and player == tracker.get("ally") and action[0] == "receive"
                and action[1] in ("3", "4", "5", "6", "7")):
            reply = self._intercepted_shi_source(state, player, action[1])
            if reply and reply["source"] != self.me:
                reply = None
        super().on_public_action(state, player, action)
        tracker = self._track.get(id(state))
        if tracker is None:
            return
        if pending:
            if (player == self.me and action == ("receive", pending["received_piece"], None)
                    and sorted(state.hands[player]) == pending["expected_hand_after_receive"]):
                pending["received"] = True
            else:
                tracker.pop("limited_shi_support_plan", None)
                if (player == self.me and pending.get("received")
                        and action[0] == "attack" and action[2] != "1"):
                    tracker["my_limited_shi_support_sent"] = True
        if reply is not None:
            tracker["pending_ally_shi_support_response"] = reply
        elif player == tracker.get("ally") and action[0] in ("attack", "attack_after_block"):
            observed = tracker.pop("pending_ally_shi_support_response", None)
            if observed is not None and action[2] != "1":
                tracker["ally_limited_shi_support"] = dict(
                    observed, returned_piece=action[2], is_hard_count_bound=False,
                )
                tracker["ally_shi_signal"] = "weak"
                tracker["shi_attack_mode"] = False
                tracker["shi_attack_mode_source"] = None
                tracker["my_open_shi_attack_pending"] = False
            elif action[2] == "1" and tracker.pop("ally_limited_shi_support", None):
                tracker["ally_shi_signal"] = "returned_shi"
                tracker["shi_attack_mode"] = True

    def _inferred_enemy_team_shi_pressure(self, state, player, *, shi_spend=1):
        tracker = self._track.get(id(state), {})
        if tracker.get("ally_limited_shi_support"):
            summary = self._opponents_piece_count_summary(tracker, player, "1")
            if summary.get("map_count", 1) > 0 or summary.get("expected", 1) > 0.75:
                return None
        return super()._inferred_enemy_team_shi_pressure(state, player, shi_spend=shi_spend)

    def _finalize_attack_candidate_snapshot(self, actions, chosen):
        super()._finalize_attack_candidate_snapshot(actions, chosen)
        plan = getattr(self, "last_shi_support_change", None)
        if plan and self.last_score_fallback_detail.startswith("attack_limited_shi_support_"):
            self.last_attack_candidate_snapshot["shi_support_change"] = copy.deepcopy(plan)

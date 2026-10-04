"""Connect an opening hand outline to a receive-and-attack route.

The outline is a hypothesis, not an order to play fixed pieces.  A lance
receive is compared with passing only after its legal follow-up attacks are
included in the same public-information search.
"""

from __future__ import annotations

import copy
import time
from collections import Counter
from typing import Optional

from goita_ai2.intermediate_middle3.information_set_policy import InformationSetPolicy
from goita_ai2.intermediate_middle3.information_set_search import (
    InformationSetSearchCancelled,
    InformationSetSearchDeadline,
    InformationSetSearchWorld,
)


class RoundStrategyMixin:
    ROUND_ROUTE_SAMPLES = 16
    ROUND_ROUTE_DEPTH = 8
    ROUND_ROUTE_MAX_SECONDS = 6.0
    ROUND_ROUTE_MIN_MARGIN = 80.0

    def _initial_round_strategy(self, state, player: str) -> dict:
        """Keep a revisable three-attack outline derived from our own deal."""
        hand = list(self._get_my_initial_hand(state))
        counts = Counter(hand)
        plan = self._search_future_attack_plan(state, player, hand, None, 1)
        if counts.get("1", 0) >= 3:
            goal = "shi_continuation"
        elif counts.get("8", 0) or counts.get("9", 0):
            goal = "self_finish_or_force_enemy_royal"
        else:
            goal = "self_finish_or_ally_handoff"
        return {
            "goal": goal,
            "opening_attacks": list(plan.get("attacks", ()))[:3],
            "status": "provisional",
        }

    def _kyosha_round_followup_rule(
        self,
        state,
        after_receive,
        player: str,
        attacks,
    ) -> Optional[dict]:
        """Preview the established attack policy after receiving kyosha."""
        tracker = self._track.get(id(state))
        if tracker is None:
            return None

        preview = copy.deepcopy(self)
        preview._track[id(after_receive)] = copy.deepcopy(tracker)
        initial_hand = preview._my_initial_hands_by_state_id.get(id(state))
        if initial_hand is not None:
            preview._my_initial_hands_by_state_id[id(after_receive)] = list(initial_hand)
        preview.on_public_action(
            after_receive,
            player,
            ("receive", "2", None),
        )
        two_shi_first_attack_signal_risk = bool(
            preview._two_shi_first_attack_signal_risk(
                after_receive,
                player,
            )
        )
        action = preview._select_rule_based_action(
            after_receive,
            player,
            list(attacks),
        )
        if action not in attacks:
            return None
        reason = str(preview.last_decision_reason or "")
        detail = str(preview.last_score_fallback_detail or "")
        return {
            "action": action,
            "reason": reason,
            "detail": detail,
            "authority": preview._rule_search_authority(reason, detail),
            "two_shi_first_attack_signal_risk": bool(
                action[2] != "1"
                and two_shi_first_attack_signal_risk
            ),
        }

    def _compare_kyosha_round_routes(self, state, player: str) -> Optional[dict]:
        """Compare pass with receive plus every legal attack in shared worlds."""
        tracker = self._track.get(id(state))
        if not tracker or not self.TIME_SEARCH_ENABLED:
            return None
        pass_action = ("pass", None, None)
        receive_action = ("receive", "2", None)
        legal = state.legal_actions(player)
        if pass_action not in legal or receive_action not in legal:
            return None
        after_receive = self._timed_search_apply(state, player, receive_action)
        attacks = [
            action for action in after_receive.legal_actions(player)
            if action[0] in ("attack", "attack_after_block")
        ]
        if not attacks:
            return None

        started = time.perf_counter()
        deadline = started + min(
            float(self.ROUND_ROUTE_MAX_SECONDS),
            float(getattr(self, "TIME_SEARCH_HARD_MAX_SECONDS", 20.0)),
        )
        cancel_event = getattr(self, "_time_search_cancel_event", None)
        try:
            samples = self._timed_search_sample_states(
                state, player, tracker, self.ROUND_ROUTE_SAMPLES
            )
            information_set, worlds = self._information_set_search_worlds(
                state, player, tracker, samples
            )
            if not worlds:
                return None
            baseline_scores = {
                team: int(state.team_score.get(team, 0))
                for team in ("AC", "BD")
            }
            stats = {"nodes": 0, "max_nodes": int(self.KYOSHA_PASS_COMPARE_MAX_NODES)}
            routes = [(pass_action,), *[(receive_action, attack) for attack in attacks]]
            values = {}
            for route in routes:
                child_worlds = []
                for world in worlds:
                    child = world.state
                    for action in route:
                        child = self._timed_search_apply(child, player, action)
                    child_worlds.append(InformationSetSearchWorld(
                        world.index, child, world.probability, world.confidence
                    ))
                history = tuple(
                    self._information_set_observed_action(player, player, action)
                    for action in route
                )
                outcome = self._information_set_search_bundle(
                    state, tuple(child_worlds), player, information_set,
                    baseline_scores, max(0, self.ROUND_ROUTE_DEPTH - len(route)),
                    deadline, stats, InformationSetPolicy(), tracker, history,
                    cancel_event,
                )
                values[route] = float(outcome.value)
        except (InformationSetSearchDeadline, InformationSetSearchCancelled,
                TypeError, ValueError):
            # An incomplete comparison cannot rank routes fairly.
            return None

        search_best_receive = max(
            (route for route in values if len(route) == 2),
            key=lambda route: values[route],
        )
        best_receive = search_best_receive
        followup_rule = self._kyosha_round_followup_rule(
            state,
            after_receive,
            player,
            attacks,
        )
        followup_override_blocked = False
        followup_value_gap = 0.0
        if followup_rule is not None:
            rule_route = (receive_action, followup_rule["action"])
            if rule_route in values and rule_route != search_best_receive:
                followup_value_gap = (
                    values[search_best_receive] - values[rule_route]
                )
                authority = str(followup_rule.get("authority") or "ordinary")
                two_shi_signal_risk = bool(
                    followup_rule.get("two_shi_first_attack_signal_risk")
                    and search_best_receive[1][2] == "1"
                )
                protect_followup = authority == "proven" or (
                    authority == "strong"
                    or two_shi_signal_risk
                ) and followup_value_gap < float(
                    self.TIME_SEARCH_STRONG_RULE_OVERRIDE_MARGIN
                )
                if protect_followup:
                    best_receive = rule_route
                    followup_override_blocked = True
        pass_value = values[(pass_action,)]
        receive_value = values[best_receive]
        receiver_position = self._kyosha_pass_compare_receiver_position(
            state, player, legal
        )
        minimum_margin = max(
            float(self.ROUND_ROUTE_MIN_MARGIN),
            float(self.KYOSHA_PASS_COMPARE_LATER_MIN_MARGIN)
            if receiver_position == "later" else 0.0,
        )
        chosen = best_receive if receive_value > pass_value + minimum_margin else (pass_action,)
        return {
            "chosen": chosen,
            "pass_value": round(pass_value, 2),
            "receive_value": round(receive_value, 2),
            "attack_values": {
                str(route[1][2]): round(value, 2)
                for route, value in values.items() if len(route) == 2
            },
            "search_best_attack": str(search_best_receive[1][2]),
            "followup_rule_action": (
                list(followup_rule["action"])
                if followup_rule is not None else None
            ),
            "followup_rule_reason": (
                str(followup_rule.get("reason") or "")
                if followup_rule is not None else ""
            ),
            "followup_rule_detail": (
                str(followup_rule.get("detail") or "")
                if followup_rule is not None else ""
            ),
            "followup_rule_authority": (
                str(followup_rule.get("authority") or "ordinary")
                if followup_rule is not None else "ordinary"
            ),
            "followup_two_shi_signal_risk": bool(
                followup_rule
                and followup_rule.get("two_shi_first_attack_signal_risk")
                and search_best_receive[1][2] == "1"
            ),
            "followup_override_blocked": followup_override_blocked,
            "followup_value_gap": round(followup_value_gap, 2),
            "depth": self.ROUND_ROUTE_DEPTH,
            "samples": len(worlds),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }

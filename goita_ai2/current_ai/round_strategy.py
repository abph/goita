"""Connect an opening hand outline to a receive-and-attack route.

The outline is a hypothesis, not an order to play fixed pieces.  A lance
receive is compared with passing only after its legal follow-up attacks are
included in the same public-information search.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Optional

from goita_ai2.current_ai.information_set_policy import InformationSetPolicy
from goita_ai2.current_ai.information_set_search import (
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

        best_receive = max(
            (route for route in values if len(route) == 2),
            key=lambda route: values[route],
        )
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
            "depth": self.ROUND_ROUTE_DEPTH,
            "samples": len(worlds),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }

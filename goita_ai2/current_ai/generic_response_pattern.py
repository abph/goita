"""Builds generalized keys for pass-versus-receive positions.
The key keeps tactical public features while omitting exact opponent hands,
seat names, room data, and exact own-hand identities that share one role.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, Optional, Tuple

from goita_ai2.constants import ALL_SEATS, PIECE_TOTALS, POINTS
from goita_ai2.current_ai.generic_response_store import (
    generic_response_pattern_store,
    human_root_pair_is_focus_eligible,
    medium_response_pattern_payload,
    tactical_response_pattern_payload,
)
from goita_ai2.current_ai.human_response_dictionary import (
    human_response_dictionary,
)
from goita_ai2.current_ai.search_cache import _digest_payload


Action = Tuple[str, Optional[str], Optional[str]]


class GenericResponsePatternMixin:
    """Converts a visible receive position into a reusable tactical shape."""

    GENERIC_RESPONSE_PATTERN_VERSION = 1
    GENERIC_RESPONSE_PATTERN_MIN_DEPTH = 5
    GENERIC_RESPONSE_PATTERN_MIN_AGREEMENT = 0.60
    GENERIC_RESPONSE_PATTERN_MIN_CONFIDENCE = 0.45
    GENERIC_RESPONSE_SHADOW_MIN_OBSERVATIONS = 5
    GENERIC_RESPONSE_SHADOW_MIN_DOMINANCE = 0.60
    GENERIC_RESPONSE_TACTICAL_SHADOW_MIN_OBSERVATIONS = 8
    GENERIC_RESPONSE_TACTICAL_SHADOW_MIN_DOMINANCE = 0.70
    GENERIC_RESPONSE_TACTICAL_PRIORITY_ENABLED = False
    GENERIC_RESPONSE_TACTICAL_PAIRED_COMPARISON_ENABLED = False
    GENERIC_RESPONSE_HUMAN_TARGETED_PRIORITY_ENABLED = False
    GENERIC_RESPONSE_HUMAN_PAIRED_COMPARISON_ENABLED = False
    GENERIC_RESPONSE_HUMAN_ROOT_COMPARISON_SECONDS = 5.0
    GENERIC_RESPONSE_HUMAN_ROOT_COMPARISON_MIN_DEPTH = 5
    GENERIC_RESPONSE_HUMAN_ROOT_COMPARISON_NODE_MULTIPLIER = 1.5
    GENERIC_RESPONSE_TACTICAL_PRIORITY_MIN_OBSERVATIONS = 15
    GENERIC_RESPONSE_TACTICAL_PRIORITY_MIN_DOMINANCE = 0.90
    GENERIC_RESPONSE_MEDIUM_SHADOW_MIN_OBSERVATIONS = 5
    GENERIC_RESPONSE_MEDIUM_PRIORITY_MIN_OBSERVATIONS = 10
    GENERIC_RESPONSE_MEDIUM_MIN_DOMINANCE = 0.70
    GENERIC_RESPONSE_PRIORITY_ENABLED = True
    GENERIC_RESPONSE_NARROWING_ENABLED = False
    GENERIC_RESPONSE_NARROWING_DEPTH = 3
    GENERIC_RESPONSE_NARROWING_MIN_CONFIDENCE = 0.45
    GENERIC_RESPONSE_NARROWING_MIN_EXCLUDED_MARGIN = 450.0
    GENERIC_RESPONSE_NARROWING_MIN_CONTINUATION_SECONDS = 1.0
    GENERIC_RESPONSE_NARROWING_NODE_EXTENSION_RATIO = 0.25

    @staticmethod
    def _generic_count_bucket(value: int) -> str:
        value = max(0, int(value))
        return str(value) if value <= 2 else "3+"

    @staticmethod
    def _generic_hand_size_bucket(value: int) -> str:
        value = max(0, int(value))
        if value <= 2:
            return "reach"
        if value <= 4:
            return "late"
        if value <= 6:
            return "middle"
        return "early"

    @staticmethod
    def _generic_confidence_bucket(value: float) -> str:
        value = max(0.0, min(1.0, float(value)))
        if value < 0.35:
            return "low"
        if value < 0.65:
            return "middle"
        return "high"

    @staticmethod
    def _generic_piece_family(piece: Optional[str]) -> str:
        if piece == "1":
            return "shi"
        if piece == "2":
            return "kyosha"
        if piece in ("3", "4", "5"):
            return "middle"
        if piece in ("6", "7"):
            return "big"
        if piece in ("8", "9"):
            return "royal"
        return "none"

    def _generic_seat_relation(
        self,
        player: str,
        other: Optional[str],
    ) -> str:
        if other is None:
            return "none"
        if other == player:
            return "self"
        if self._same_team(player, other):
            return "ally"
        return "enemy"

    @staticmethod
    def _generic_receive_distance(
        player: str,
        attacker: Optional[str],
    ) -> int:
        if attacker not in ALL_SEATS:
            return 0
        return (ALL_SEATS.index(player) - ALL_SEATS.index(attacker)) % 4

    def _generic_hand_shape(
        self,
        hand: Iterable[str],
        current_attack: Optional[str],
    ) -> Dict[str, object]:
        counts = Counter(str(piece) for piece in hand)
        middle_pairs = sum(counts[piece] >= 2 for piece in ("3", "4", "5"))
        big_pairs = sum(counts[piece] >= 2 for piece in ("6", "7"))
        middle_singletons = sum(counts[piece] == 1 for piece in ("3", "4", "5"))
        big_singletons = sum(counts[piece] == 1 for piece in ("6", "7"))
        return {
            "size": self._generic_hand_size_bucket(sum(counts.values())),
            "shi": self._generic_count_bucket(counts["1"]),
            "kyosha": self._generic_count_bucket(counts["2"]),
            "royals": self._generic_count_bucket(counts["8"] + counts["9"]),
            "same_piece": self._generic_count_bucket(
                counts.get(str(current_attack), 0)
            ),
            "middle_pairs": self._generic_count_bucket(middle_pairs),
            "big_pairs": self._generic_count_bucket(big_pairs),
            "middle_singletons": self._generic_count_bucket(middle_singletons),
            "big_singletons": self._generic_count_bucket(big_singletons),
        }

    def _generic_followup_shape(
        self,
        hand: Iterable[str],
        public_seen: Dict[str, int],
        current_attack: Optional[str],
    ) -> Dict[str, object]:
        counts = Counter(str(piece) for piece in hand)
        distinct_reentry = 0
        pair_followups = 0
        scarce_followups = 0
        fourth_followups = 0
        families = set()
        best_points = 0

        for piece, count in counts.items():
            if count <= 0:
                continue
            best_points = max(best_points, int(POINTS.get(piece, 0)))
            if piece in ("8", "9"):
                continue
            outside = max(
                0,
                int(PIECE_TOTALS[piece])
                - int(public_seen.get(piece, 0))
                - int(count),
            )
            if piece != current_attack and outside > 0:
                distinct_reentry += 1
            if count >= 2:
                pair_followups += 1
            if outside <= 1:
                scarce_followups += 1
                families.add(self._generic_piece_family(piece))
            if outside == 0:
                fourth_followups += 1

        return {
            "reentry_width": self._generic_count_bucket(distinct_reentry),
            "pair_followups": self._generic_count_bucket(pair_followups),
            "scarce_followups": self._generic_count_bucket(scarce_followups),
            "fourth_followups": self._generic_count_bucket(fourth_followups),
            "strong_families": tuple(sorted(families)),
            "best_points": best_points,
        }

    def _generic_rank_summary(
        self,
        tracker: dict,
        seat: Optional[str],
    ) -> Dict[str, str]:
        model = tracker.get("public_hand_models", {}).get(seat, {})
        return {
            "rank": str(model.get("estimated_rank", "D")),
            "confidence": self._generic_confidence_bucket(
                float(model.get("rank_confidence", 0.0))
            ),
        }

    def _generic_score_context(self, state, player: str) -> Dict[str, str]:
        own_team = "AC" if player in ("A", "C") else "BD"
        enemy_team = "BD" if own_team == "AC" else "AC"
        own_score = int(state.team_score.get(own_team, 0))
        enemy_score = int(state.team_score.get(enemy_team, 0))
        lead = own_score - enemy_score
        if lead <= -50:
            lead_bucket = "far_behind"
        elif lead < 0:
            lead_bucket = "behind"
        elif lead == 0:
            lead_bucket = "tied"
        elif lead < 50:
            lead_bucket = "ahead"
        else:
            lead_bucket = "far_ahead"
        return {
            "lead": lead_bucket,
            "own_match_point": "near" if own_score >= 120 else "normal",
            "enemy_match_point": "near" if enemy_score >= 120 else "normal",
        }

    def _generic_response_pattern_payload(
        self,
        state,
        player: str,
        actions: Iterable[Action],
        baseline_action: Action,
    ) -> Dict[str, object]:
        """Return generalized features built only from visible information."""
        if state.phase != "receive" or state.current_attack is None:
            raise ValueError("A generic response pattern requires receive phase")

        tracker = self._track.get(id(state)) or {}
        actions_list = list(actions)
        current_attack = str(state.current_attack)
        attacker = state.attacker
        ally = self._ally_of(player)
        next_receiver = state.next_player(player)
        public_seen = {
            str(piece): int(count)
            for piece, count in tracker.get("public_seen_counts", {}).items()
        }
        hand = list(state.hands[player])

        receive_modes = set()
        for action_type, block, _attack in actions_list:
            if action_type != "receive" or block is None:
                continue
            if block == current_attack:
                receive_modes.add("same")
            elif block in ("8", "9"):
                receive_modes.add("royal")
            else:
                receive_modes.add("other")

        hand_after_same_receive = list(hand)
        if "same" in receive_modes and current_attack in hand_after_same_receive:
            hand_after_same_receive.remove(current_attack)

        models = tracker.get("public_hand_models", {})
        attacker_attack_count = int(
            models.get(attacker, {}).get("attack_count", 0)
        )
        unseen_current = max(
            0,
            int(PIECE_TOTALS[current_attack])
            - int(public_seen.get(current_attack, 0))
            - hand.count(current_attack),
        )
        hidden_counts = tracker.get("hidden_block_counts", {})

        return {
            "version": int(self.GENERIC_RESPONSE_PATTERN_VERSION),
            "context": {
                "attacker_relation": self._generic_seat_relation(player, attacker),
                "dealer_relation": self._generic_seat_relation(player, state.dealer),
                "receive_distance": self._generic_receive_distance(player, attacker),
                "attack_piece": current_attack,
                "attack_family": self._generic_piece_family(current_attack),
                "attacker_attack_count": self._generic_count_bucket(
                    attacker_attack_count
                ),
                "my_attack_count": self._generic_count_bucket(
                    int(tracker.get("my_attack_count", 0))
                ),
                "attacker_hand": self._generic_hand_size_bucket(
                    len(state.hands.get(attacker, ()))
                ),
                "ally_hand": self._generic_hand_size_bucket(
                    len(state.hands.get(ally, ()))
                ),
                "next_receiver_hand": self._generic_hand_size_bucket(
                    len(state.hands.get(next_receiver, ()))
                ),
                "next_receiver_hidden": self._generic_count_bucket(
                    int(hidden_counts.get(next_receiver, 0))
                ),
                "king_blocks_used": self._generic_count_bucket(
                    int(state.king_block_used)
                ),
                "unseen_current_piece": self._generic_count_bucket(unseen_current),
            },
            "legal": {
                "can_pass": any(action[0] == "pass" for action in actions_list),
                "receive_modes": tuple(sorted(receive_modes)),
                "baseline_root": str(baseline_action[0]),
            },
            "hand": self._generic_hand_shape(hand, current_attack),
            "after_same_receive": {
                "hand": self._generic_hand_shape(
                    hand_after_same_receive,
                    current_attack,
                ),
                "followup": self._generic_followup_shape(
                    hand_after_same_receive,
                    public_seen,
                    current_attack,
                ),
            },
            "signals": {
                "ally_shi": str(tracker.get("ally_shi_signal", "unknown")),
                "shi_attack_mode": bool(tracker.get("shi_attack_mode", False)),
                "enemy_rejected_shi": bool(
                    tracker.get("enemy_team_rejected_shi_attack", False)
                ),
                "attacker_rank": self._generic_rank_summary(tracker, attacker),
                "ally_rank": self._generic_rank_summary(tracker, ally),
            },
            "score": self._generic_score_context(state, player),
        }

    def _generic_response_pattern_key(
        self,
        state,
        player: str,
        actions: Iterable[Action],
        baseline_action: Action,
    ) -> str:
        return _digest_payload(
            self._generic_response_pattern_payload(
                state,
                player,
                actions,
                baseline_action,
            )
        )

    def _medium_response_pattern_payload(
        self,
        state,
        player: str,
        actions: Iterable[Action],
        baseline_action: Action,
    ) -> Dict[str, object]:
        detailed = self._generic_response_pattern_payload(
            state,
            player,
            actions,
            baseline_action,
        )
        return medium_response_pattern_payload(detailed)

    def _tactical_response_pattern_payload(
        self,
        state,
        player: str,
        actions: Iterable[Action],
        baseline_action: Action,
    ) -> Dict[str, object]:
        detailed = self._generic_response_pattern_payload(
            state,
            player,
            actions,
            baseline_action,
        )
        return tactical_response_pattern_payload(detailed)

    def _tactical_response_pattern_key(
        self,
        state,
        player: str,
        actions: Iterable[Action],
        baseline_action: Action,
    ) -> str:
        return _digest_payload(
            self._tactical_response_pattern_payload(
                state,
                player,
                actions,
                baseline_action,
            )
        )

    def _medium_response_pattern_key(
        self,
        state,
        player: str,
        actions: Iterable[Action],
        baseline_action: Action,
    ) -> str:
        return _digest_payload(
            self._medium_response_pattern_payload(
                state,
                player,
                actions,
                baseline_action,
            )
        )

    def _generic_response_action_label(
        self,
        action: Action,
        current_attack: Optional[str],
    ) -> str:
        if action[0] == "pass":
            return "pass"
        if action[0] != "receive":
            return "other"
        if action[1] == current_attack:
            return "receive_same"
        if action[1] in ("8", "9"):
            return "receive_royal"
        return "receive_other"

    def _generic_followup_label(
        self,
        state,
        player: str,
        receive_action: Action,
        followup_piece: Optional[str],
    ) -> str:
        if receive_action[0] != "receive" or followup_piece is None:
            return "none"

        hand = list(state.hands[player])
        try:
            hand.remove(str(receive_action[1]))
        except ValueError:
            return "none"
        piece = str(followup_piece)
        counts = Counter(hand)
        if counts[piece] <= 0:
            return "none"

        tracker = self._track.get(id(state)) or {}
        public_seen = tracker.get("public_seen_counts", {})
        outside = max(
            0,
            int(PIECE_TOTALS[piece])
            - int(public_seen.get(piece, 0))
            - int(counts[piece]),
        )
        family = self._generic_piece_family(piece)
        if outside == 0:
            return f"fourth_{family}"
        if counts[piece] >= 2:
            return f"{family}_pair"
        if outside <= 1:
            return f"scarce_{family}"
        if family in ("shi", "kyosha", "royal"):
            return family
        return f"{family}_single"

    def _record_generic_response_evidence(
        self,
        state,
        player: str,
        actions: Iterable[Action],
        baseline_action: Action,
        selected_action: Action,
        *,
        followup_piece: Optional[str],
        source: str,
        depth: int,
        agreement: float,
        information_confidence: float,
        margin: float,
    ) -> bool:
        """Aggregate one reliable adopted search decision without exact hands."""
        store = generic_response_pattern_store()
        if state.phase != "receive" or selected_action[0] not in (
            "pass",
            "receive",
        ):
            return False
        if int(depth) < int(self.GENERIC_RESPONSE_PATTERN_MIN_DEPTH):
            store.reject("depth")
            return False
        if float(agreement) < float(
            self.GENERIC_RESPONSE_PATTERN_MIN_AGREEMENT
        ):
            store.reject("agreement")
            return False
        if float(information_confidence) < float(
            self.GENERIC_RESPONSE_PATTERN_MIN_CONFIDENCE
        ):
            store.reject("confidence")
            return False

        actions_list = list(actions)
        payload = self._generic_response_pattern_payload(
            state,
            player,
            actions_list,
            baseline_action,
        )
        key = _digest_payload(payload)
        store.record(
            pattern_key=key,
            features=payload,
            action_label=self._generic_response_action_label(
                selected_action,
                state.current_attack,
            ),
            followup_label=self._generic_followup_label(
                state,
                player,
                selected_action,
                followup_piece,
            ),
            source=str(source or "foreground"),
            depth=int(depth),
            agreement=float(agreement),
            confidence=float(information_confidence),
            margin=float(margin),
        )
        return True

    def _compare_generic_response_shadow(
        self,
        state,
        player: str,
        actions: Iterable[Action],
        baseline_action: Action,
        actual_action: Action,
    ) -> dict:
        """Measure a generic recommendation while leaving the action untouched."""
        if (
            getattr(self, "_time_search_cancel_event", None) is not None
            or state.phase != "receive"
            or actual_action[0] not in ("pass", "receive")
        ):
            return {"status": "not_applicable"}
        actions_list = list(actions)
        root_choices = {
            action[0]
            for action in actions_list
            if action[0] in ("pass", "receive")
        }
        if len(root_choices) < 2:
            return {"status": "not_applicable"}
        key = self._generic_response_pattern_key(
            state,
            player,
            actions_list,
            baseline_action,
        )
        medium_key = self._medium_response_pattern_key(
            state,
            player,
            actions_list,
            baseline_action,
        )
        result = generic_response_pattern_store().compare_shadow(
            pattern_key=key,
            medium_pattern_key=medium_key,
            actual_action=self._generic_response_action_label(
                actual_action,
                state.current_attack,
            ),
            min_observations=self.GENERIC_RESPONSE_SHADOW_MIN_OBSERVATIONS,
            min_dominance=self.GENERIC_RESPONSE_SHADOW_MIN_DOMINANCE,
            medium_min_observations=(
                self.GENERIC_RESPONSE_MEDIUM_SHADOW_MIN_OBSERVATIONS
            ),
            medium_min_dominance=(
                self.GENERIC_RESPONSE_MEDIUM_MIN_DOMINANCE
            ),
        )
        self.last_generic_response_shadow = dict(result)
        tactical_payload = self._tactical_response_pattern_payload(
            state,
            player,
            actions_list,
            baseline_action,
        )
        store = generic_response_pattern_store()
        tactical_result = store.compare_tactical_shadow(
            pattern_key=_digest_payload(tactical_payload),
            actual_action=self._generic_response_action_label(
                actual_action,
                state.current_attack,
            ),
            min_observations=(
                self.GENERIC_RESPONSE_TACTICAL_SHADOW_MIN_OBSERVATIONS
            ),
            min_dominance=(
                self.GENERIC_RESPONSE_TACTICAL_SHADOW_MIN_DOMINANCE
            ),
        )
        self.last_generic_response_tactical_shadow = dict(tactical_result)
        actual_label = self._generic_response_action_label(
            actual_action,
            state.current_attack,
        )
        human_result = store.compare_human_shadow(
            recommendation=human_response_dictionary().recommendation(
                tactical_payload
            ),
            actual_action=actual_label,
        )
        self.last_generic_response_human_shadow = dict(human_result)
        return result

    def _human_response_comparison_actions(
        self,
        state,
        player: str,
        actions: Iterable[Action],
        baseline_action: Action,
    ) -> Tuple[dict, Tuple[Action, ...]]:
        """Return legal root actions represented by the aggregate human hint."""
        if state.phase != "receive":
            return {"status": "not_applicable"}, tuple()
        actions_list = list(actions)
        tactical_payload = self._tactical_response_pattern_payload(
            state,
            player,
            actions_list,
            baseline_action,
        )
        recommendation = human_response_dictionary().recommendation(
            tactical_payload
        )
        if recommendation.get("status") != "recommended":
            return recommendation, tuple()
        recommended_label = str(
            recommendation.get("recommended_action", "other")
        )
        candidates = self._generic_response_actions_for_label(
            state,
            actions_list,
            recommended_label,
        )
        return recommendation, candidates

    def _generic_response_actions_for_label(
        self,
        state,
        actions: Iterable[Action],
        label: str,
    ) -> Tuple[Action, ...]:
        """Return every legal concrete action represented by one route label."""
        return tuple(
            action
            for action in actions
            if self._generic_response_action_label(
                action,
                state.current_attack,
            ) == str(label)
        )

    def _generic_response_priority_action(
        self,
        state,
        player: str,
        actions: Iterable[Action],
        baseline_action: Action,
    ) -> Optional[Action]:
        """Return a generic move-order hint, never a final action decision."""
        if (
            not self.GENERIC_RESPONSE_PRIORITY_ENABLED
            or state.phase != "receive"
        ):
            return None
        actions_list = list(actions)
        root_choices = {
            action[0]
            for action in actions_list
            if action[0] in ("pass", "receive")
        }
        if len(root_choices) < 2:
            return None

        key = self._generic_response_pattern_key(
            state,
            player,
            actions_list,
            baseline_action,
        )
        store = generic_response_pattern_store()
        recommendation = store.recommendation(
            key,
            granularity="detailed",
            min_observations=self.GENERIC_RESPONSE_SHADOW_MIN_OBSERVATIONS,
            min_dominance=self.GENERIC_RESPONSE_SHADOW_MIN_DOMINANCE,
        )
        if recommendation.get("status") != "recommended":
            medium_key = self._medium_response_pattern_key(
                state,
                player,
                actions_list,
                baseline_action,
            )
            medium_recommendation = store.recommendation(
                medium_key,
                granularity="medium",
                min_observations=(
                    self.GENERIC_RESPONSE_MEDIUM_PRIORITY_MIN_OBSERVATIONS
                ),
                min_dominance=self.GENERIC_RESPONSE_MEDIUM_MIN_DOMINANCE,
            )
            if medium_recommendation.get("status") == "recommended":
                recommendation = medium_recommendation
        label = str(recommendation.get("recommended_action", ""))

        def matches(action: Action) -> bool:
            if label == "pass":
                return action[0] == "pass"
            if action[0] != "receive":
                return False
            if label == "receive_same":
                return action[1] == state.current_attack
            if label == "receive_royal":
                return action[1] in ("8", "9")
            if label == "receive_other":
                return action[1] not in (state.current_attack, "8", "9")
            return False

        candidates = [action for action in actions_list if matches(action)]
        if baseline_action in candidates:
            priority = baseline_action
        elif candidates:
            priority = max(
                candidates,
                key=lambda action: self._timed_search_rule_prior(
                    state,
                    player,
                    action,
                ),
            )
        else:
            priority = None

        if not bool(
            getattr(self, "_suppress_response_dictionary_metrics", False)
        ):
            store.record_priority_query(
                recommendation if priority is not None else None
            )
        self.last_generic_response_priority = {
            **recommendation,
            "priority_action": priority,
            "used": priority is not None,
        }
        return priority

    def _tactical_response_priority_action(
        self,
        state,
        player: str,
        actions: Iterable[Action],
        baseline_action: Action,
    ) -> Optional[Action]:
        """Offer a high-support tactical receive as a debug-only search hint."""
        if (
            not self.GENERIC_RESPONSE_TACTICAL_PRIORITY_ENABLED
            or state.phase != "receive"
        ):
            return None
        actions_list = list(actions)
        root_choices = {
            action[0]
            for action in actions_list
            if action[0] in ("pass", "receive")
        }
        if len(root_choices) < 2:
            return None

        store = generic_response_pattern_store()
        tactical_payload = self._tactical_response_pattern_payload(
            state,
            player,
            actions_list,
            baseline_action,
        )
        recommendation = store.recommendation(
            _digest_payload(tactical_payload),
            granularity="tactical",
            min_observations=(
                self.GENERIC_RESPONSE_TACTICAL_PRIORITY_MIN_OBSERVATIONS
            ),
            min_dominance=(
                self.GENERIC_RESPONSE_TACTICAL_PRIORITY_MIN_DOMINANCE
            ),
        )
        if recommendation.get("status") != "recommended":
            store.record_tactical_priority_query("no_recommendation")
            return None
        if recommendation.get("recommended_action") != "receive_same":
            store.record_tactical_priority_query("rejected_action")
            return None
        if tactical_payload.get("same_piece") == "none":
            store.record_tactical_priority_query("rejected_context")
            return None

        priority = next(
            (
                action
                for action in actions_list
                if action[0] == "receive"
                and action[1] == state.current_attack
            ),
            None,
        )
        if priority is None:
            store.record_tactical_priority_query("rejected_context")
            return None
        try:
            after_receive = self._timed_search_apply(state, player, priority)
            followups = after_receive.legal_actions(player)
        except (AttributeError, TypeError, ValueError):
            store.record_tactical_priority_query("no_legal_followup")
            return None
        if not any(action[0] == "attack" for action in followups):
            store.record_tactical_priority_query("no_legal_followup")
            return None

        store.record_tactical_priority_query("offered")
        self.last_generic_response_tactical_priority = {
            **recommendation,
            "priority_action": priority,
            "used": True,
            "anonymous_context": tactical_payload,
        }
        return priority

    def _human_targeted_response_priority_action(
        self,
        state,
        player: str,
        actions: Iterable[Action],
        baseline_action: Action,
    ) -> Optional[Action]:
        """Offer a reviewed human-kifu receive as a debug-only search hint."""
        if (
            not self.GENERIC_RESPONSE_HUMAN_TARGETED_PRIORITY_ENABLED
            or state.phase != "receive"
        ):
            return None
        actions_list = list(actions)
        root_choices = {
            action[0]
            for action in actions_list
            if action[0] in ("pass", "receive")
        }
        if len(root_choices) < 2:
            return None

        store = generic_response_pattern_store()
        tactical_payload = self._tactical_response_pattern_payload(
            state,
            player,
            actions_list,
            baseline_action,
        )
        recommendation = (
            human_response_dictionary().targeted_priority_recommendation(
                tactical_payload
            )
        )
        status = str(recommendation.get("status", "no_pattern"))
        if status in ("no_pattern", "load_error"):
            store.record_human_targeted_priority_query("no_recommendation")
            return None
        if status == "not_targeted":
            store.record_human_targeted_priority_query("not_targeted")
            return None
        if status == "unsupported_action":
            store.record_human_targeted_priority_query("rejected_action")
            return None
        if status != "recommended":
            store.record_human_targeted_priority_query("rejected_context")
            return None

        priority = next(
            (
                action
                for action in actions_list
                if action[0] == "receive"
                and action[1] == state.current_attack
            ),
            None,
        )
        if priority is None:
            store.record_human_targeted_priority_query("rejected_context")
            return None
        try:
            after_receive = self._timed_search_apply(state, player, priority)
            followups = after_receive.legal_actions(player)
        except (AttributeError, TypeError, ValueError):
            store.record_human_targeted_priority_query("no_legal_followup")
            return None
        if not any(action[0] == "attack" for action in followups):
            store.record_human_targeted_priority_query("no_legal_followup")
            return None

        store.record_human_targeted_priority_query("offered")
        self.last_generic_response_human_targeted_priority = {
            **recommendation,
            "priority_action": priority,
            "used": True,
            "anonymous_context": tactical_payload,
        }
        return priority

    def _record_human_targeted_response_priority_effect(
        self,
        *,
        reordered: bool,
        baseline_disagreed: bool,
        selected: bool,
        completed_depth: int,
    ) -> None:
        generic_response_pattern_store().record_human_targeted_priority_effect(
            reordered=reordered,
            baseline_disagreed=baseline_disagreed,
            selected=selected,
            completed_depth=completed_depth,
        )

    def _record_tactical_response_priority_effect(
        self,
        *,
        reordered: bool,
        baseline_disagreed: bool,
        selected: bool,
        completed_depth: int,
    ) -> None:
        generic_response_pattern_store().record_tactical_priority_effect(
            reordered=reordered,
            baseline_disagreed=baseline_disagreed,
            selected=selected,
            completed_depth=completed_depth,
        )

    def _record_tactical_response_paired_comparison(
        self,
        *,
        comparison_complete: bool,
        action_matched: bool = False,
        with_priority_selected: bool = False,
        without_priority_selected: bool = False,
        with_depth: int = 0,
        without_depth: int = 0,
        with_elapsed_seconds: float = 0.0,
        without_elapsed_seconds: float = 0.0,
        with_nodes: int = 0,
        without_nodes: int = 0,
        value_delta: float = 0.0,
        margin_delta: float = 0.0,
    ) -> None:
        generic_response_pattern_store().record_tactical_priority_pair(
            comparison_complete=comparison_complete,
            action_matched=action_matched,
            with_priority_selected=with_priority_selected,
            without_priority_selected=without_priority_selected,
            with_depth=with_depth,
            without_depth=without_depth,
            with_elapsed_seconds=with_elapsed_seconds,
            without_elapsed_seconds=without_elapsed_seconds,
            with_nodes=with_nodes,
            without_nodes=without_nodes,
            value_delta=value_delta,
            margin_delta=margin_delta,
        )

    def _record_human_response_paired_comparison(
        self,
        *,
        comparison_complete: bool,
        selected_side: str = "other",
        normal_depth: int = 0,
        priority_depth: int = 0,
        normal_elapsed_seconds: float = 0.0,
        priority_elapsed_seconds: float = 0.0,
        normal_nodes: int = 0,
        priority_nodes: int = 0,
        value_delta: float = 0.0,
    ) -> None:
        generic_response_pattern_store().record_human_priority_pair(
            comparison_complete=comparison_complete,
            selected_side=selected_side,
            normal_depth=normal_depth,
            priority_depth=priority_depth,
            normal_elapsed_seconds=normal_elapsed_seconds,
            priority_elapsed_seconds=priority_elapsed_seconds,
            normal_nodes=normal_nodes,
            priority_nodes=priority_nodes,
            value_delta=value_delta,
        )

    def _record_human_response_root_comparison(
        self,
        *,
        comparison_complete: bool,
        pattern_key: str = "",
        incomplete_reason: str = "",
        selected_side: str = "other",
        common_depth: int = 0,
        ai_depth: int = 0,
        human_depth: int = 0,
        ai_elapsed_seconds: float = 0.0,
        human_elapsed_seconds: float = 0.0,
        ai_nodes: int = 0,
        human_nodes: int = 0,
        value_delta: float = 0.0,
        ai_terminal_win_rate: float = 0.0,
        human_terminal_win_rate: float = 0.0,
        ai_terminal_loss_rate: float = 0.0,
        human_terminal_loss_rate: float = 0.0,
        ai_terminal_point_swing: float = 0.0,
        human_terminal_point_swing: float = 0.0,
        ai_stop_reason: str = "",
        human_stop_reason: str = "",
        human_action_label: str = "other",
        ai_action_label: str = "other",
        mean_value_delta: float = 0.0,
        terminal_outcome_delta: float = 0.0,
        terminal_score_delta: float = 0.0,
        nonterminal_delta: float = 0.0,
    ) -> None:
        """Record an equal-budget, root-constrained diagnostic comparison."""
        generic_response_pattern_store().record_human_root_pair(
            comparison_complete=comparison_complete,
            pattern_key=pattern_key,
            incomplete_reason=incomplete_reason,
            selected_side=selected_side,
            common_depth=common_depth,
            ai_depth=ai_depth,
            human_depth=human_depth,
            ai_elapsed_seconds=ai_elapsed_seconds,
            human_elapsed_seconds=human_elapsed_seconds,
            ai_nodes=ai_nodes,
            human_nodes=human_nodes,
            value_delta=value_delta,
            ai_terminal_win_rate=ai_terminal_win_rate,
            human_terminal_win_rate=human_terminal_win_rate,
            ai_terminal_loss_rate=ai_terminal_loss_rate,
            human_terminal_loss_rate=human_terminal_loss_rate,
            ai_terminal_point_swing=ai_terminal_point_swing,
            human_terminal_point_swing=human_terminal_point_swing,
            ai_stop_reason=ai_stop_reason,
            human_stop_reason=human_stop_reason,
            human_action_label=human_action_label,
            ai_action_label=ai_action_label,
            mean_value_delta=mean_value_delta,
            terminal_outcome_delta=terminal_outcome_delta,
            terminal_score_delta=terminal_score_delta,
            nonterminal_delta=nonterminal_delta,
        )

    def _human_response_root_focus_eligible(
        self,
        human_action_label: str,
        ai_action_label: str,
    ) -> bool:
        return human_root_pair_is_focus_eligible(
            human_action_label,
            ai_action_label,
        )

    def _record_human_response_root_focus(self, *, eligible: bool) -> None:
        generic_response_pattern_store().record_human_root_focus(
            eligible=eligible
        )

    def _record_generic_response_priority_effect(
        self,
        *,
        reordered: bool,
        beam_preserved: bool,
        comparison_complete: bool,
        recommended_selected: bool,
        action_changed: bool,
        with_depth: int,
        without_depth: int,
        with_elapsed_seconds: float,
        without_elapsed_seconds: float,
        value_delta: float,
    ) -> None:
        """Record aggregate-only evidence about one priority hint's effect."""
        generic_response_pattern_store().record_priority_effect(
            reordered=reordered,
            beam_preserved=beam_preserved,
            comparison_complete=comparison_complete,
            recommended_selected=recommended_selected,
            action_changed=action_changed,
            with_depth=with_depth,
            without_depth=without_depth,
            with_elapsed_seconds=with_elapsed_seconds,
            without_elapsed_seconds=without_elapsed_seconds,
            value_delta=value_delta,
        )

    def _record_generic_response_narrowing_shadow(
        self,
        *,
        status: str,
        matched: bool = False,
        priority_selected: bool = False,
        full_candidates: int = 0,
        kept_candidates: int = 0,
        depth: int = 0,
        actual_elapsed_seconds: float = 0.0,
        estimated_elapsed_seconds: float = 0.0,
        value_loss: float = 0.0,
    ) -> None:
        """Record an aggregate-only hypothetical depth-three narrowing."""
        generic_response_pattern_store().record_narrowing_shadow(
            status=status,
            matched=matched,
            priority_selected=priority_selected,
            full_candidates=full_candidates,
            kept_candidates=kept_candidates,
            depth=depth,
            actual_elapsed_seconds=actual_elapsed_seconds,
            estimated_elapsed_seconds=estimated_elapsed_seconds,
            value_loss=value_loss,
        )

    def _record_generic_response_active_narrowing(
        self,
        *,
        status: str,
        full_candidates: int = 0,
        kept_candidates: int = 0,
        completed_depth: int = 0,
        elapsed_seconds: float = 0.0,
        priority_selected: bool = False,
        continuation_extension_seconds: float = 0.0,
        added_nodes: int = 0,
        no_deepening_reason: str = "",
    ) -> None:
        """Record aggregate-only results from live candidate narrowing."""
        generic_response_pattern_store().record_active_narrowing(
            status=status,
            full_candidates=full_candidates,
            kept_candidates=kept_candidates,
            completed_depth=completed_depth,
            elapsed_seconds=elapsed_seconds,
            priority_selected=priority_selected,
            continuation_extension_seconds=continuation_extension_seconds,
            added_nodes=added_nodes,
            no_deepening_reason=no_deepening_reason,
        )

    def _record_generic_response_search_result(
        self,
        state,
        player: str,
        actions: Iterable[Action],
        baseline_action: Action,
        selected_action: Action,
        search_result,
        *,
        source: str,
    ) -> bool:
        """Record a foreground search only when its root action was adopted."""
        if getattr(self, "_time_search_cancel_event", None) is not None:
            return False
        if selected_action != getattr(search_result, "action", None):
            generic_response_pattern_store().reject("not_adopted")
            return False
        followup = None
        if selected_action[0] == "receive":
            followup = self._low_reentry_followup_piece(
                state,
                player,
                str(selected_action[1]),
            )
        return self._record_generic_response_evidence(
            state,
            player,
            actions,
            baseline_action,
            selected_action,
            followup_piece=followup,
            source=source,
            depth=int(getattr(search_result, "depth", 0)),
            agreement=float(getattr(search_result, "agreement", 0.0)),
            information_confidence=float(
                getattr(search_result, "information_confidence", 0.0)
            ),
            margin=float(getattr(search_result, "margin", 0.0)),
        )

    def _record_generic_response_plan_reuse(
        self,
        state,
        player: str,
        actions: Iterable[Action],
        baseline_action: Action,
        plan,
    ) -> bool:
        """Record a previously searched exact plan when a real turn reuses it."""
        if getattr(self, "_time_search_cancel_event", None) is not None:
            return False
        return self._record_generic_response_evidence(
            state,
            player,
            actions,
            baseline_action,
            plan.action,
            followup_piece=plan.followup_attack_piece,
            source=f"response_dictionary_{plan.cache_source}",
            depth=int(plan.depth),
            agreement=float(plan.agreement),
            information_confidence=float(plan.information_confidence),
            margin=float(plan.margin),
        )

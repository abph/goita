"""Searches ordinary positions under a strict thinking-time budget.
The engine samples hidden hands from public inference, compares every root move
shallowly, then deepens the most promising moves without reading actual hands.
"""

from __future__ import annotations

import copy
import hashlib
import random
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from goita_ai2.constants import ALL_SEATS, PIECE_TOTALS, POINTS
from goita_ai2.current_ai.information_set_search import (
    InformationSetSearchCancelled,
    InformationSetSearchDeadline,
)
from goita_ai2.current_ai.prediction_cache import PredictionSample


Action = Tuple[str, Optional[str], Optional[str]]


class _SearchDeadline(Exception):
    """Stops an unfinished iteration without discarding earlier results."""


class _SearchCancelled(Exception):
    """Stops speculative work after its projected public branch changed."""


@dataclass(frozen=True)
class TimedSearchResult:
    """One fully completed iterative-deepening result."""

    action: Action
    depth: int
    samples: int
    nodes: int
    elapsed_seconds: float
    value: float
    margin: float
    agreement: float
    decisive: bool
    information_set: bool = False
    candidate_count: int = 0
    information_confidence: float = 0.0
    policy_decisions: int = 0
    enemy_third_attack_wait: bool = False

    def as_dict(self) -> Dict[str, object]:
        result = {
            "action": self.action,
            "depth": self.depth,
            "samples": self.samples,
            "nodes": self.nodes,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "value": round(self.value, 2),
            "margin": round(self.margin, 2),
            "agreement": round(self.agreement, 3),
            "decisive": self.decisive,
        }
        if self.information_set:
            result.update({
                "information_set": True,
                "candidate_count": self.candidate_count,
                "information_confidence": round(self.information_confidence, 3),
                "policy_decisions": self.policy_decisions,
            })
        if self.enemy_third_attack_wait:
            result["enemy_third_attack_wait"] = True
        return result


class TimedSearchMixin:
    """Adds public-information sampling and time-limited coalition search."""

    @staticmethod
    def _timed_search_prioritize_root_actions(
        actions: Sequence[Action],
        priority_action: Optional[Action],
    ) -> List[Action]:
        """Move one hint first without removing or scoring any legal action."""
        ordered = list(actions)
        if priority_action in ordered:
            ordered.remove(priority_action)
            ordered.insert(0, priority_action)
        return ordered

    @staticmethod
    def _timed_search_safe_generic_narrowing(
        actions: Sequence[Action],
        aggregate: Dict[Action, float],
        priority_action: Optional[Action],
        *,
        enabled: bool,
        search_profile: str,
        information_enabled: bool,
        information_confidence: float,
        minimum_confidence: float,
        minimum_excluded_margin: float,
    ) -> Tuple[Optional[List[Action]], str]:
        """Keep a learned move and its best rival only when depth three is clear."""
        if not enabled:
            return None, "disabled"
        if priority_action is None or priority_action not in aggregate:
            return None, "incomplete"
        if len(actions) <= 2:
            return None, "no_reduction"
        if search_profile != "default":
            return None, "specialized_profile"
        if not information_enabled:
            return None, "information_unavailable"
        if float(information_confidence) < float(minimum_confidence):
            return None, "low_confidence"

        ranked = sorted(
            (action for action in actions if action in aggregate),
            key=lambda action: aggregate[action],
            reverse=True,
        )
        if len(ranked) != len(actions) or priority_action not in ranked[:2]:
            return None, "priority_not_top_two"
        strongest_rival = next(
            action for action in ranked if action != priority_action
        )
        kept = [priority_action, strongest_rival]
        best_excluded = max(
            aggregate[action]
            for action in ranked
            if action not in kept
        )
        weakest_kept = min(aggregate[action] for action in kept)
        if weakest_kept - best_excluded < float(minimum_excluded_margin):
            return None, "insufficient_margin"
        return kept, "applied"

    @staticmethod
    def _timed_search_extend_narrowing_deadline(
        deadline: float,
        hard_deadline: float,
        now: float,
        minimum_continuation_seconds: float,
    ) -> Tuple[float, float]:
        """Reserve post-narrowing time without exceeding the hard limit."""
        extended = min(
            float(hard_deadline),
            max(
                float(deadline),
                float(now) + max(0.0, float(minimum_continuation_seconds)),
            ),
        )
        return extended, max(0.0, extended - float(deadline))

    @staticmethod
    def _timed_search_extend_narrowing_nodes(
        maximum_nodes: int,
        extension_ratio: float,
    ) -> Tuple[int, int]:
        """Add a bounded percentage of nodes after safe narrowing."""
        base = max(1, int(maximum_nodes))
        added = max(0, int(round(base * max(0.0, float(extension_ratio)))))
        return base + added, added

    def _timed_search_kyosha_pass_compare_is_decisive(
        self,
        *,
        baseline_action: Action,
        best_action: Action,
        completed_depth: int,
        agreement: float,
        margin: float,
        information_enabled: bool,
        information_confidence: float,
    ) -> bool:
        """Adopt either better depth-seven branch in a lance comparison."""
        compared_actions = {baseline_action[0], best_action[0]}
        receive_action = (
            baseline_action
            if baseline_action[0] == "receive"
            else best_action
        )
        return bool(
            getattr(self, "_time_search_profile", "default")
            == "kyosha_pass_compare"
            and baseline_action != best_action
            and compared_actions == {"pass", "receive"}
            and receive_action[1] == "2"
            and information_enabled
            and completed_depth
            >= int(self.KYOSHA_PASS_COMPARE_TARGET_DEPTH)
            and agreement >= float(self.KYOSHA_PASS_COMPARE_MIN_AGREEMENT)
            and margin > float(self.KYOSHA_PASS_COMPARE_MIN_MARGIN)
            and information_confidence
            >= float(self.KYOSHA_PASS_COMPARE_MIN_CONFIDENCE)
        )

    def _timed_search_weak_first_receive_is_decisive(
        self,
        *,
        baseline_action: Action,
        best_action: Action,
        completed_depth: int,
        agreement: float,
        margin: float,
        information_enabled: bool,
        information_confidence: float,
        zero_shi_stop_signal: bool = False,
    ) -> bool:
        """Allow a robust receive result to replace a weak-hand pass."""
        profile = getattr(self, "_time_search_profile", "default")
        if (
            profile not in ("weak_first_receive", "low_reentry_receive")
            or baseline_action[0] != "pass"
            or best_action[0] != "receive"
            or not information_enabled
        ):
            return False
        if profile == "low_reentry_receive":
            return (
                completed_depth
                >= int(self.LOW_REENTRY_RECEIVE_SEARCH_MIN_OVERRIDE_DEPTH)
                and agreement
                >= float(self.LOW_REENTRY_RECEIVE_SEARCH_OVERRIDE_AGREEMENT)
                and margin
                >= float(self.LOW_REENTRY_RECEIVE_SEARCH_OVERRIDE_MARGIN)
                and information_confidence
                >= float(self.LOW_REENTRY_RECEIVE_SEARCH_MIN_CONFIDENCE)
            )
        if zero_shi_stop_signal:
            return (
                completed_depth
                >= int(self.WEAK_FIRST_RECEIVE_SEARCH_MIN_OVERRIDE_DEPTH)
                and agreement
                >= float(self.ZERO_SHI_STOP_SIGNAL_OVERRIDE_AGREEMENT)
                and margin >= float(self.ZERO_SHI_STOP_SIGNAL_OVERRIDE_MARGIN)
                and information_confidence
                >= float(self.ZERO_SHI_STOP_SIGNAL_MIN_CONFIDENCE)
            )
        return (
            completed_depth
            >= int(self.WEAK_FIRST_RECEIVE_SEARCH_MIN_OVERRIDE_DEPTH)
            and agreement
            >= float(self.WEAK_FIRST_RECEIVE_SEARCH_OVERRIDE_AGREEMENT)
            and margin >= float(self.WEAK_FIRST_RECEIVE_SEARCH_OVERRIDE_MARGIN)
            and information_confidence
            >= float(self.WEAK_FIRST_RECEIVE_SEARCH_MIN_CONFIDENCE)
        )

    def _timed_search_zero_shi_stop_signal_context(
        self,
        state,
        player: str,
        tracker: dict,
        *,
        baseline_action: Action,
        best_action: Action,
    ) -> bool:
        """Detect a zero-shi partner response that asks the ally to stop shi pressure."""
        return bool(
            state.phase == "receive"
            and state.current_attack not in (None, "1")
            and state.attacker is not None
            and not self._same_team(state.attacker, player)
            and baseline_action[0] == "pass"
            and best_action[0] == "receive"
            and best_action[1] == state.current_attack
            and state.hands[player].count("1") == 0
            and tracker.get("ally_open_shi_attack_pending")
            and tracker.get("enemy_team_rejected_shi_attack")
        )

    def _timed_search_enemy_third_attack_wait_is_safe(
        self,
        state,
        player: str,
        tracker: dict,
        *,
        baseline_action: Action,
        best_action: Action,
        completed_depth: int,
        agreement: float,
        information_enabled: bool,
        information_confidence: float,
        best_minimum: float,
        baseline_minimum: float,
        margin: float,
    ) -> bool:
        """Accept a second-attack pass only when every inferred world still wins."""
        if (
            not information_enabled
            or state.phase != "receive"
            or state.attacker is None
            or self._same_team(state.attacker, player)
            or best_action[0] != "pass"
            or baseline_action[0] not in ("pass", "receive")
            or int(tracker.get("enemy_attack_counts", {}).get(state.attacker, 0)) != 2
        ):
            return False

        minimum_depth = max(5, min(7, int(self.TIME_SEARCH_MAX_DEPTH)))
        if completed_depth < minimum_depth:
            return False

        # Terminal team wins score at 100000 or more. Staying above 50000 in
        # every sampled world excludes attractive but unproven horizon values.
        if best_minimum < 50000.0 or best_minimum < baseline_minimum:
            return False

        return (
            agreement >= max(0.80, float(self.TIME_SEARCH_OVERRIDE_AGREEMENT))
            and information_confidence >= 0.60
            and margin >= max(150.0, float(self.TIME_SEARCH_OVERRIDE_MARGIN) * 0.5)
        )

    def _timed_search_public_seed(self, state, player: str, tr: dict) -> int:
        estimates = tr.get("estimated_current_hands", {})
        estimate_key = []
        for seat in ALL_SEATS:
            if seat == player:
                continue
            for piece in (str(i) for i in range(1, 10)):
                item = estimates.get(seat, {}).get(piece, {})
                estimate_key.append(
                    (
                        seat,
                        piece,
                        int(item.get("min", 0)),
                        int(item.get("max", PIECE_TOTALS[piece])),
                        float(item.get("expected", 0.0)),
                    )
                )
        public_key = (
            player,
            tuple(sorted(state.hands[player])),
            tuple((seat, len(state.hands[seat])) for seat in ALL_SEATS),
            tuple(
                (seat, int(tr.get("hidden_block_counts", {}).get(seat, 0)))
                for seat in ALL_SEATS
            ),
            tuple(
                (piece, int(tr.get("public_seen_counts", {}).get(piece, 0)))
                for piece in (str(i) for i in range(1, 10))
            ),
            state.phase,
            state.turn,
            state.current_attack,
            state.attacker,
            int(state.king_block_used),
            tuple(estimate_key),
        )
        digest = hashlib.sha256(repr(public_key).encode("ascii")).digest()
        return int.from_bytes(digest[:8], "big")

    def _timed_search_prediction_cache_key(
        self,
        state,
        player: str,
        tr: dict,
    ) -> str:
        estimates = tr.get("estimated_current_hands", {})
        estimate_key = []
        for seat in ALL_SEATS:
            if seat == player:
                continue
            for piece in (str(index) for index in range(1, 10)):
                item = estimates.get(seat, {}).get(piece, {})
                estimate_key.append(
                    (
                        seat,
                        piece,
                        int(item.get("min", 0)),
                        int(item.get("max", PIECE_TOTALS[piece])),
                        round(float(item.get("expected", 0.0)), 6),
                    )
                )
        public_key = (
            "prediction-v1",
            player,
            tuple(sorted(state.hands[player])),
            tuple(sorted(state.face_down_hidden[player])),
            tuple((seat, len(state.hands[seat])) for seat in ALL_SEATS),
            tuple(
                (seat, int(tr.get("hidden_block_counts", {}).get(seat, 0)))
                for seat in ALL_SEATS
            ),
            tuple(
                (piece, int(tr.get("unknown_piece_pool", {}).get(piece, 0)))
                for piece in (str(index) for index in range(1, 10))
            ),
            tuple(
                (
                    seat,
                    self._observed_piece_count_for_player(tr, seat, "8"),
                    self._observed_piece_count_for_player(tr, seat, "9"),
                )
                for seat in ALL_SEATS
                if seat != player
            ),
            state.dealer,
            state.turn,
            state.phase,
            state.current_attack,
            state.attacker,
            state.last_block_player,
            state.last_block if state.last_block_player == player else None,
            int(state.king_block_used),
            tuple(estimate_key),
        )
        return hashlib.sha256(repr(public_key).encode("ascii")).hexdigest()

    @staticmethod
    def _timed_search_prediction_snapshot(
        sampled,
        player: str,
    ) -> PredictionSample:
        opponents = [seat for seat in ALL_SEATS if seat != player]
        return PredictionSample(
            opponent_hands=tuple(
                (seat, tuple(sorted(sampled.hands[seat])))
                for seat in opponents
            ),
            opponent_hidden=tuple(
                (seat, tuple(sorted(sampled.face_down_hidden[seat])))
                for seat in opponents
            ),
            opponent_had_both_kings=tuple(
                (seat, bool(sampled.had_both_kings.get(seat, False)))
                for seat in opponents
            ),
            last_block=sampled.last_block,
        )

    @staticmethod
    def _timed_search_materialize_prediction(
        state,
        player: str,
        prediction: PredictionSample,
    ):
        sampled = copy.copy(state)
        sampled.hands = {
            seat: list(state.hands[seat])
            for seat in ALL_SEATS
        }
        sampled.face_down_hidden = {
            seat: list(state.face_down_hidden[seat])
            for seat in ALL_SEATS
        }
        sampled.had_both_kings = dict(state.had_both_kings)
        sampled.team_score = dict(state.team_score)
        for seat, hand in prediction.opponent_hands:
            sampled.hands[seat] = list(hand)
        for seat, hidden in prediction.opponent_hidden:
            sampled.face_down_hidden[seat] = list(hidden)
        for seat, had_both in prediction.opponent_had_both_kings:
            sampled.had_both_kings[seat] = bool(had_both)
        sampled.hands[player] = list(state.hands[player])
        sampled.face_down_hidden[player] = list(state.face_down_hidden[player])
        sampled.last_block = prediction.last_block
        return sampled

    @staticmethod
    def _timed_search_prediction_state_key(sampled, player: str) -> tuple:
        return tuple(
            (
                seat,
                tuple(sorted(sampled.hands[seat])),
                tuple(sorted(sampled.face_down_hidden[seat])),
            )
            for seat in ALL_SEATS
            if seat != player
        )

    def _timed_search_remember_prediction_states(
        self,
        cache_key: Optional[str],
        states: Sequence[object],
    ) -> None:
        if not getattr(self, "_prediction_cache_rollforward_enabled", True):
            return
        self._prediction_rollforward_key = cache_key
        self._prediction_rollforward_states = list(states)

    def _timed_search_prediction_matches_public(
        self,
        sampled,
        public_state,
        player: str,
        tr: dict,
    ) -> bool:
        if (
            sampled.phase != public_state.phase
            or sampled.turn != public_state.turn
            or sampled.current_attack != public_state.current_attack
            or sampled.attacker != public_state.attacker
            or bool(sampled.finished) != bool(public_state.finished)
            or sampled.winner != public_state.winner
            or int(sampled.king_block_used) != int(public_state.king_block_used)
        ):
            return False
        hidden_counts = tr.get("hidden_block_counts", {})
        estimates = tr.get("estimated_current_hands", {})
        for seat in ALL_SEATS:
            if len(sampled.hands[seat]) != len(public_state.hands[seat]):
                return False
            expected_hidden = (
                len(public_state.face_down_hidden[seat])
                if seat == player
                else int(hidden_counts.get(seat, 0))
            )
            if len(sampled.face_down_hidden[seat]) != expected_hidden:
                return False
            if seat == player:
                if sorted(sampled.hands[seat]) != sorted(public_state.hands[seat]):
                    return False
                continue
            hand_count = Counter(sampled.hands[seat])
            for piece in (str(index) for index in range(1, 10)):
                item = estimates.get(seat, {}).get(piece, {})
                minimum = int(item.get("min", 0))
                maximum = int(item.get("max", PIECE_TOTALS[piece]))
                if not minimum <= int(hand_count.get(piece, 0)) <= maximum:
                    return False
        return True

    def _advance_prediction_cache_for_public_action(
        self,
        state,
        actor: str,
        action: Action,
        tr: dict,
    ) -> None:
        """Roll inferred deals forward using only the publicly visible action."""
        if (
            not getattr(self, "TIME_SEARCH_PREDICTION_CACHE_ENABLED", True)
            or not getattr(self, "_prediction_cache_rollforward_enabled", True)
        ):
            return
        roots = list(getattr(self, "_prediction_rollforward_states", ()))
        if not roots or self.me is None:
            return

        action_type, block, attack = action
        advanced = []
        seen = set()
        for sampled in roots:
            legal = sampled.legal_actions(actor)
            if action_type == "attack_after_block" and actor != self.me:
                candidates = [
                    candidate
                    for candidate in legal
                    if candidate[0] == "attack_after_block"
                    and candidate[2] == attack
                ]
            else:
                candidates = [candidate for candidate in legal if candidate == action]
            if not candidates:
                continue
            chosen = max(
                candidates,
                key=lambda candidate: (
                    self._timed_search_action_priority(sampled, actor, candidate),
                    candidate,
                ),
            )
            try:
                chosen_type, chosen_block, chosen_attack = chosen
                if chosen_type == "pass":
                    sampled.apply_pass(actor)
                elif chosen_type == "receive":
                    sampled.apply_receive(actor, chosen_block)
                elif chosen_type == "attack":
                    sampled.apply_attack(actor, chosen_attack)
                else:
                    sampled.apply_attack_after_block(
                        actor,
                        chosen_block,
                        chosen_attack,
                    )
                next_sampled = sampled
            except (ValueError, KeyError):
                continue
            if not self._timed_search_prediction_matches_public(
                next_sampled,
                state,
                self.me,
                tr,
            ):
                continue
            sample_key = self._timed_search_prediction_state_key(
                next_sampled,
                self.me,
            )
            if sample_key in seen:
                continue
            seen.add(sample_key)
            advanced.append(next_sampled)

        if not advanced:
            self._prediction_rollforward_key = None
            self._prediction_rollforward_states = []
            return
        next_key = self._timed_search_prediction_cache_key(state, self.me, tr)
        if state.turn == self.me and not state.finished:
            snapshots = tuple(
                self._timed_search_prediction_snapshot(sampled, self.me)
                for sampled in advanced
            )
            self._prediction_cache_store_rollforward(next_key, snapshots)
        self._timed_search_remember_prediction_states(next_key, advanced)

    @staticmethod
    def _timed_search_weighted_piece(
        rng: random.Random,
        candidates: Sequence[str],
        weights: Sequence[float],
    ) -> str:
        total = sum(max(0.0, value) for value in weights)
        if total <= 0:
            return candidates[rng.randrange(len(candidates))]
        point = rng.random() * total
        running = 0.0
        for piece, weight in zip(candidates, weights):
            running += max(0.0, weight)
            if point <= running:
                return piece
        return candidates[-1]

    def _timed_search_sample_state(
        self,
        state,
        player: str,
        tr: dict,
        rng: random.Random,
    ):
        """Create one hidden-hand deal using public ranges, never real contents."""
        other_players = [seat for seat in ALL_SEATS if seat != player]
        hidden_counts = tr.get("hidden_block_counts", {})
        pool = Counter(
            {
                piece: int(tr.get("unknown_piece_pool", {}).get(piece, 0))
                for piece in (str(i) for i in range(1, 10))
            }
        )
        required_slots = sum(
            len(state.hands[seat]) + int(hidden_counts.get(seat, 0))
            for seat in other_players
        )
        if sum(pool.values()) != required_slots:
            return None

        estimates = tr.get("estimated_current_hands", {})
        for _attempt in range(24):
            available = pool.copy()
            assigned = {seat: Counter() for seat in other_players}
            valid = True

            # Satisfy every inferred lower bound before filling uncertain slots.
            for seat in other_players:
                seat_estimates = estimates.get(seat, {})
                for piece in (str(i) for i in range(1, 10)):
                    minimum = int(seat_estimates.get(piece, {}).get("min", 0))
                    if minimum > int(available.get(piece, 0)):
                        valid = False
                        break
                    assigned[seat][piece] += minimum
                    available[piece] -= minimum
                if sum(assigned[seat].values()) > len(state.hands[seat]):
                    valid = False
                if not valid:
                    break
            if not valid:
                continue

            fill_order: List[str] = []
            for seat in other_players:
                fill_order.extend(
                    [seat] * (len(state.hands[seat]) - sum(assigned[seat].values()))
                )
            rng.shuffle(fill_order)

            for seat in fill_order:
                seat_estimates = estimates.get(seat, {})
                candidates: List[str] = []
                weights: List[float] = []
                for piece in (str(i) for i in range(1, 10)):
                    count_left = int(available.get(piece, 0))
                    maximum = int(
                        seat_estimates.get(piece, {}).get("max", PIECE_TOTALS[piece])
                    )
                    if count_left <= 0 or assigned[seat][piece] >= maximum:
                        continue
                    expected = float(seat_estimates.get(piece, {}).get("expected", 0.0))
                    need = max(0.0, expected - assigned[seat][piece])
                    candidates.append(piece)
                    weights.append(count_left * (0.2 + need))
                if not candidates:
                    valid = False
                    break
                piece = self._timed_search_weighted_piece(rng, candidates, weights)
                assigned[seat][piece] += 1
                available[piece] -= 1
            if not valid:
                continue

            hidden_bag: List[str] = []
            for piece in (str(i) for i in range(1, 10)):
                hidden_bag.extend([piece] * int(available.get(piece, 0)))
            expected_hidden = sum(int(hidden_counts.get(seat, 0)) for seat in other_players)
            if len(hidden_bag) != expected_hidden:
                continue
            rng.shuffle(hidden_bag)

            sampled = copy.deepcopy(state)
            sampled.hands[player] = list(state.hands[player])
            sampled.face_down_hidden[player] = list(state.face_down_hidden[player])
            hidden_offset = 0
            for seat in other_players:
                sampled.hands[seat] = sorted(assigned[seat].elements())
                hidden_size = int(hidden_counts.get(seat, 0))
                sampled.face_down_hidden[seat] = sorted(
                    hidden_bag[hidden_offset:hidden_offset + hidden_size]
                )
                hidden_offset += hidden_size

                observed_eight = self._observed_piece_count_for_player(tr, seat, "8")
                observed_nine = self._observed_piece_count_for_player(tr, seat, "9")
                original_eight = (
                    sampled.hands[seat].count("8")
                    + sampled.face_down_hidden[seat].count("8")
                    + observed_eight
                )
                original_nine = (
                    sampled.hands[seat].count("9")
                    + sampled.face_down_hidden[seat].count("9")
                    + observed_nine
                )
                sampled.had_both_kings[seat] = original_eight > 0 and original_nine > 0

            if sampled.last_block_player in other_players:
                hidden = sampled.face_down_hidden[sampled.last_block_player]
                sampled.last_block = hidden[-1] if hidden else None
            return sampled
        return None

    def _timed_search_generate_sample_states(
        self,
        state,
        player: str,
        tr: dict,
        count: int,
        *,
        seed_salt: int = 0,
    ) -> List[object]:
        seed = self._timed_search_public_seed(state, player, tr)
        rng = random.Random(seed ^ (int(seed_salt) * 0x9E3779B97F4A7C15))
        samples = []
        seen = set()
        attempts = max(count * 5, 12)
        for _ in range(attempts):
            sampled = self._timed_search_sample_state(state, player, tr, rng)
            if sampled is None:
                continue
            key = tuple(
                (seat, tuple(sorted(sampled.hands[seat])))
                for seat in ALL_SEATS
                if seat != player
            )
            if key in seen and len(seen) >= max(2, count // 2):
                continue
            seen.add(key)
            samples.append(sampled)
            if len(samples) >= count:
                break
        return samples

    def _timed_search_sample_states(
        self,
        state,
        player: str,
        tr: dict,
        count: int,
    ) -> List[object]:
        requested = max(1, int(count))
        self.last_prediction_cache_hit = False
        self.last_prediction_cache_samples = 0
        if not getattr(self, "TIME_SEARCH_PREDICTION_CACHE_ENABLED", True):
            self.last_prediction_cache_key = None
            generated = self._timed_search_generate_sample_states(
                state,
                player,
                tr,
                requested,
            )
            self._timed_search_remember_prediction_states(None, generated)
            return generated

        cache_key = self._timed_search_prediction_cache_key(state, player, tr)
        self.last_prediction_cache_key = cache_key
        cached = self._prediction_cache_get(cache_key, requested)
        if cached is not None:
            self.last_prediction_cache_hit = True
            self.last_prediction_cache_samples = len(cached)
            materialized = [
                self._timed_search_materialize_prediction(state, player, prediction)
                for prediction in cached
            ]
            self._timed_search_remember_prediction_states(cache_key, materialized)
            return materialized

        partial_cached = self._prediction_cache_get_available(cache_key, requested)

        owner, event = self._prediction_cache_claim(cache_key)
        if not owner:
            wait_seconds = min(
                float(getattr(self, "TIME_SEARCH_PREDICTION_CACHE_WAIT_SECONDS", 0.12)),
                float(
                    self._effective_time_search_setting(
                        "effective_seconds",
                        self.TIME_SEARCH_MAX_SECONDS,
                    )
                ),
            )
            cached = self._prediction_cache_wait(
                cache_key,
                event,
                requested,
                wait_seconds,
                cancel_event=getattr(self, "_time_search_cancel_event", None),
            )
            if cached is not None:
                self.last_prediction_cache_hit = True
                self.last_prediction_cache_samples = len(cached)
                materialized = [
                    self._timed_search_materialize_prediction(
                        state,
                        player,
                        prediction,
                    )
                    for prediction in cached
                ]
                self._timed_search_remember_prediction_states(
                    cache_key,
                    materialized,
                )
                return materialized
            waited_partial = self._prediction_cache_get_available(
                cache_key,
                requested,
            )
            if waited_partial is not None:
                partial_cached = waited_partial
            owner, event = self._prediction_cache_claim(cache_key)

        try:
            combined = []
            seen = set()
            for prediction in partial_cached or ():
                sampled = self._timed_search_materialize_prediction(
                    state,
                    player,
                    prediction,
                )
                sample_key = self._timed_search_prediction_state_key(sampled, player)
                if sample_key in seen:
                    continue
                seen.add(sample_key)
                combined.append(sampled)
            reused_count = len(combined)
            salt = 0 if not combined else len(combined) + 1
            generation_round = 0
            while len(combined) < requested and generation_round < 4:
                needed = requested - len(combined)
                generated = self._timed_search_generate_sample_states(
                    state,
                    player,
                    tr,
                    max(needed, min(requested, needed * 2)),
                    seed_salt=salt + generation_round,
                )
                for sampled in generated:
                    sample_key = self._timed_search_prediction_state_key(
                        sampled,
                        player,
                    )
                    if sample_key in seen:
                        continue
                    seen.add(sample_key)
                    combined.append(sampled)
                    if len(combined) >= requested:
                        break
                generation_round += 1
        except Exception:
            if owner:
                self._prediction_cache_finish(cache_key, tuple())
            raise
        if owner:
            snapshots = tuple(
                self._timed_search_prediction_snapshot(sampled, player)
                for sampled in combined
            )
            self._prediction_cache_finish(
                cache_key,
                snapshots,
                generated_count=max(0, len(combined) - reused_count),
            )
        self.last_prediction_cache_hit = bool(partial_cached)
        self.last_prediction_cache_samples = len(partial_cached or ())
        self._timed_search_remember_prediction_states(cache_key, combined)
        return combined

    def _timed_search_apply(self, state, player: str, action: Action):
        next_state = copy.deepcopy(state)
        action_type, block, attack = action
        if action_type == "pass":
            next_state.apply_pass(player)
        elif action_type == "receive":
            next_state.apply_receive(player, block)
        elif action_type == "attack":
            next_state.apply_attack(player, attack)
        else:
            next_state.apply_attack_after_block(player, block, attack)
        return next_state

    def _timed_search_action_priority(self, state, actor: str, action: Action) -> float:
        action_type, block, attack = action
        if action_type == "pass":
            receive_count = sum(1 for item in state.legal_actions(actor) if item[0] == "receive")
            return 30.0 if receive_count == 0 else -15.0
        if action_type == "receive":
            score = 22.0
            if block in ("8", "9"):
                score -= 35.0
            if state.attacker is not None and self._same_team(state.attacker, actor):
                score -= 28.0
            return score

        if attack is None:
            return -1000.0
        hand = state.hands[actor]
        score = float(POINTS.get(attack, 0)) * 0.35
        score += max(0, hand.count(attack) - 1) * 24.0
        if attack in ("8", "9") and len(hand) > (2 if block is not None else 1):
            score -= 45.0
        if attack == "1" and hand.count("1") < 3:
            score -= 12.0
        if attack == "1":
            enemy_shi = sum(
                state.hands[seat].count("1")
                for seat in ALL_SEATS
                if seat != actor and not self._same_team(seat, actor)
            )
            if enemy_shi <= 1:
                score += 85.0
            elif enemy_shi == 2:
                score += 50.0
        if block is not None:
            score -= float(POINTS.get(block, 0)) * 0.12

        can_be_received = False
        for seat in ALL_SEATS:
            if seat == actor:
                continue
            for receive_piece in set(state.hands[seat]):
                if receive_piece == attack or (
                    receive_piece in ("8", "9") and attack not in ("1", "2")
                ):
                    can_be_received = True
                    break
            if can_be_received:
                break
        if not can_be_received:
            score += 75.0

        try:
            after = self._timed_search_apply(state, actor, action)
            if after.finished and after.winner == actor:
                score += 10000.0
        except Exception:
            return -10000.0
        return score

    def _timed_search_ordered_actions(self, state, beam_width: int) -> List[Action]:
        actor = state.turn
        actions = state.legal_actions(actor)
        compact_endgame = len(state.hands.get(actor, ())) <= 4
        if len(actions) <= beam_width or state.phase == "receive" or compact_endgame:
            return sorted(
                actions,
                key=lambda action: self._timed_search_action_priority(state, actor, action),
                reverse=True,
            )
        immediate = []
        scored = []
        for action in actions:
            priority = self._timed_search_action_priority(state, actor, action)
            if priority >= 9000:
                immediate.append(action)
            else:
                scored.append((priority, action))
        scored.sort(key=lambda item: item[0], reverse=True)
        chosen = immediate + [action for _score, action in scored[:beam_width]]
        return chosen[: max(beam_width, len(immediate))]

    @staticmethod
    def _timed_search_team(player: str) -> str:
        return "AC" if player in ("A", "C") else "BD"

    def _timed_search_terminal_value(
        self,
        state,
        root_player: str,
        baseline_scores: Dict[str, int],
    ) -> float:
        root_team = self._timed_search_team(root_player)
        enemy_team = "BD" if root_team == "AC" else "AC"
        root_gain = int(state.team_score.get(root_team, 0)) - int(baseline_scores[root_team])
        enemy_gain = int(state.team_score.get(enemy_team, 0)) - int(baseline_scores[enemy_team])
        if state.winner is not None and self._same_team(state.winner, root_player):
            return 100000.0 + root_gain * 500.0
        return -100000.0 - enemy_gain * 500.0

    def _timed_search_static_value(
        self,
        state,
        root_player: str,
        baseline_scores: Dict[str, int],
    ) -> float:
        if state.finished:
            return self._timed_search_terminal_value(state, root_player, baseline_scores)

        allies = [seat for seat in ALL_SEATS if self._same_team(seat, root_player)]
        enemies = [seat for seat in ALL_SEATS if not self._same_team(seat, root_player)]
        ally_cards = sum(len(state.hands[seat]) for seat in allies)
        enemy_cards = sum(len(state.hands[seat]) for seat in enemies)
        value = float(enemy_cards - ally_cards) * 145.0

        root_team = self._timed_search_team(root_player)
        enemy_team = "BD" if root_team == "AC" else "AC"
        root_gain = int(state.team_score.get(root_team, 0)) - int(baseline_scores[root_team])
        enemy_gain = int(state.team_score.get(enemy_team, 0)) - int(baseline_scores[enemy_team])
        value += float(root_gain - enemy_gain) * 350.0

        for seat in ALL_SEATS:
            sign = 1.0 if self._same_team(seat, root_player) else -1.0
            counts = Counter(state.hands[seat])
            shape = 0.0
            for piece, count in counts.items():
                if count >= 2:
                    shape += (count - 1) * (9.0 + POINTS.get(piece, 0) * 0.18)
            receive_types = sum(1 for piece in counts if piece not in ("8", "9"))
            receive_types += 2 * (counts.get("8", 0) + counts.get("9", 0))
            shape += receive_types * 3.0
            value += sign * shape

        if state.attacker is not None and state.current_attack is not None:
            pressure = 18.0 + max(0, 4 - len(state.hands[state.attacker])) * 28.0
            value += pressure if self._same_team(state.attacker, root_player) else -pressure
            if state.current_attack == "1":
                defending_shi = sum(
                    state.hands[seat].count("1")
                    for seat in ALL_SEATS
                    if not self._same_team(seat, state.attacker)
                )
                shi_pressure = 360.0 if defending_shi <= 1 else 210.0 if defending_shi == 2 else 0.0
                if self._same_team(state.attacker, root_player):
                    value += shi_pressure
                else:
                    value -= shi_pressure
        return value

    @staticmethod
    def _timed_search_state_key(state, depth: int) -> tuple:
        return (
            depth,
            tuple((seat, tuple(sorted(state.hands[seat]))) for seat in ALL_SEATS),
            state.phase,
            state.turn,
            state.current_attack,
            state.attacker,
            state.last_block,
            state.last_block_player,
            int(state.king_block_used),
            tuple((seat, bool(state.had_both_kings.get(seat, False))) for seat in ALL_SEATS),
        )

    def _timed_search_minimax(
        self,
        state,
        root_player: str,
        baseline_scores: Dict[str, int],
        depth: int,
        alpha: float,
        beta: float,
        deadline: float,
        stats: Dict[str, int],
        memo: Dict[tuple, float],
        cancel_event=None,
    ) -> float:
        if cancel_event is not None and cancel_event.is_set():
            raise _SearchCancelled()
        maximum_nodes = int(stats.get("max_nodes", self.TIME_SEARCH_MAX_NODES))
        if time.perf_counter() >= deadline:
            raise _SearchDeadline("time_limit")
        if stats["nodes"] >= maximum_nodes:
            raise _SearchDeadline("node_limit")
        stats["nodes"] += 1
        if state.finished or depth <= 0:
            return self._timed_search_static_value(state, root_player, baseline_scores)

        key = self._timed_search_state_key(state, depth)
        if key in memo:
            return memo[key]
        actions = self._timed_search_ordered_actions(state, self.TIME_SEARCH_BRANCH_BEAM)
        if not actions:
            return self._timed_search_static_value(state, root_player, baseline_scores)

        maximizing = self._same_team(state.turn, root_player)
        value = -float("inf") if maximizing else float("inf")
        complete = True
        for action in actions:
            try:
                next_state = self._timed_search_apply(state, state.turn, action)
                child = self._timed_search_minimax(
                    next_state,
                    root_player,
                    baseline_scores,
                    depth - 1,
                    alpha,
                    beta,
                    deadline,
                    stats,
                    memo,
                    cancel_event,
                )
            except (_SearchDeadline, _SearchCancelled):
                raise
            except Exception:
                complete = False
                continue
            if maximizing:
                value = max(value, child)
                alpha = max(alpha, value)
            else:
                value = min(value, child)
                beta = min(beta, value)
            if beta <= alpha:
                complete = False
                break

        if value in (-float("inf"), float("inf")):
            value = self._timed_search_static_value(state, root_player, baseline_scores)
        if complete:
            memo[key] = value
        return value

    def _timed_search_rule_prior(self, state, player: str, action: Action) -> float:
        action_type, block, attack = action
        try:
            if action_type == "attack_after_block":
                return float(self._score_receive_phase(state, player, "receive", block)) + float(
                    self._score_attack_phase(
                        state,
                        player,
                        action_type,
                        block,
                        attack,
                        has_non_king_attack_option=any(
                            item[0] in ("attack", "attack_after_block")
                            and item[2] not in (None, "8", "9")
                            for item in state.legal_actions(player)
                        ),
                    )
                )
            if action_type == "attack":
                return float(
                    self._score_attack_phase(
                        state,
                        player,
                        action_type,
                        block,
                        attack,
                        has_non_king_attack_option=any(
                            item[0] in ("attack", "attack_after_block")
                            and item[2] not in (None, "8", "9")
                            for item in state.legal_actions(player)
                        ),
                    )
                )
            return float(self._score_receive_phase(state, player, action_type, block))
        except Exception:
            return 0.0

    def _time_limited_search_action(
        self,
        state,
        player: str,
        actions: List[Action],
        baseline_action: Action,
        cancel_event=None,
    ) -> Optional[TimedSearchResult]:
        tr = self._track.get(id(state))
        if (
            not self.TIME_SEARCH_ENABLED
            or tr is None
            or len(actions) < 2
            or state.finished
        ):
            return None

        self.last_time_search_cache_hit = False
        self.last_time_search_cache_source = None
        self.last_time_search_cached_compute_ms = 0.0
        self.last_time_search_cache_branch_kind = None
        self.last_time_search_cache_branch_context = None
        cache_key = None
        cache_owner = False
        if self.TIME_SEARCH_CACHE_ENABLED:
            with self._measure_performance("cache"):
                cache_key = self._timed_search_cache_key(
                    state,
                    player,
                    tr,
                    actions,
                    baseline_action,
                )
                self.last_time_search_cache_key = cache_key.digest
                cached = self._get_cached_timed_search(cache_key)
            if cached is not None:
                self.last_time_search_cache_hit = True
                return cached

            with self._measure_performance("cache"):
                cache_owner, inflight_event = self._claim_timed_search_compute(cache_key)
            if not cache_owner:
                with self._measure_performance("cache"):
                    completed, cached = self._wait_for_timed_search_compute(
                        cache_key,
                        inflight_event,
                        float(
                            self._effective_time_search_setting(
                                "effective_seconds",
                                self.TIME_SEARCH_MAX_SECONDS,
                            )
                        ),
                        cancel_event=cancel_event,
                    )
                if cached is not None:
                    self.last_time_search_cache_hit = True
                    return cached
                if not completed or (cancel_event is not None and cancel_event.is_set()):
                    return None
                with self._measure_performance("cache"):
                    cache_owner, _inflight_event = self._claim_timed_search_compute(cache_key)
                if not cache_owner:
                    return None

        result = None
        foreground_registered = False
        try:
            if cancel_event is None:
                from goita_ai2.current_ai.background_search import (
                    background_search_foreground_started,
                )

                background_search_foreground_started()
                foreground_registered = True
            with self._measure_performance("sample_generation"):
                samples = self._timed_search_sample_states(
                    state,
                    player,
                    tr,
                    int(
                        self._effective_time_search_setting(
                            "effective_samples",
                            self.TIME_SEARCH_SAMPLE_COUNT,
                        )
                    ),
                )
            if not samples or (cancel_event is not None and cancel_event.is_set()):
                return None

            with self._measure_performance("search"):
                actual_context: Dict[str, object] = {}
                result = self._time_limited_search_from_samples(
                    state,
                    player,
                    actions,
                    baseline_action,
                    samples,
                    cancel_event=cancel_event,
                    run_context=actual_context,
                )
                tactical_priority_action = actual_context.get(
                    "tactical_priority_action"
                )
                should_compare_tactical_priority = bool(
                    result is not None
                    and cancel_event is None
                    and getattr(
                        self,
                        "GENERIC_RESPONSE_TACTICAL_PAIRED_COMPARISON_ENABLED",
                        False,
                    )
                    and tactical_priority_action is not None
                    and tactical_priority_action != baseline_action
                )
                if should_compare_tactical_priority:
                    saved_attributes = {}
                    missing = object()
                    for name in (
                        "_suppress_response_dictionary_metrics",
                        "last_generic_response_priority",
                        "last_generic_response_tactical_priority",
                        "last_information_set_search",
                    ):
                        saved_attributes[name] = getattr(self, name, missing)
                    control_result = None
                    try:
                        self._suppress_response_dictionary_metrics = True
                        control_result = self._time_limited_search_from_samples(
                            state,
                            player,
                            actions,
                            baseline_action,
                            samples,
                            cancel_event=None,
                            tactical_priority_enabled=False,
                            record_priority_metrics=False,
                        )
                    except Exception:
                        # A diagnostic comparison must never interrupt the move.
                        control_result = None
                    finally:
                        for name, previous in saved_attributes.items():
                            if previous is missing:
                                try:
                                    delattr(self, name)
                                except AttributeError:
                                    pass
                            else:
                                setattr(self, name, previous)

                    if control_result is None:
                        self._record_tactical_response_paired_comparison(
                            comparison_complete=False,
                        )
                    else:
                        self._record_tactical_response_paired_comparison(
                            comparison_complete=True,
                            action_matched=(
                                result.action == control_result.action
                            ),
                            with_priority_selected=(
                                result.action == tactical_priority_action
                            ),
                            without_priority_selected=(
                                control_result.action
                                == tactical_priority_action
                            ),
                            with_depth=result.depth,
                            without_depth=control_result.depth,
                            with_elapsed_seconds=result.elapsed_seconds,
                            without_elapsed_seconds=(
                                control_result.elapsed_seconds
                            ),
                            with_nodes=result.nodes,
                            without_nodes=control_result.nodes,
                            value_delta=result.value - control_result.value,
                            margin_delta=result.margin - control_result.margin,
                        )
                human_pair_enabled = bool(
                    result is not None
                    and cancel_event is None
                    and getattr(
                        self,
                        "GENERIC_RESPONSE_HUMAN_PAIRED_COMPARISON_ENABLED",
                        False,
                    )
                )
                if human_pair_enabled:
                    human_recommendation, human_actions = (
                        self._human_response_comparison_actions(
                            state,
                            player,
                            actions,
                            baseline_action,
                        )
                    )
                else:
                    human_recommendation, human_actions = ({}, tuple())
                human_label = str(
                    human_recommendation.get("recommended_action", "other")
                )
                result_label = self._generic_response_action_label(
                    result.action,
                    state.current_attack,
                ) if result is not None else "other"
                should_compare_human_priority = bool(
                    human_pair_enabled
                    and human_actions
                    and human_label != result_label
                )
                if should_compare_human_priority:
                    ai_actions = self._generic_response_actions_for_label(
                        state,
                        actions,
                        result_label,
                    )
                    saved_attributes = {}
                    missing = object()
                    for name in (
                        "_suppress_response_dictionary_metrics",
                        "last_generic_response_priority",
                        "last_generic_response_tactical_priority",
                        "last_information_set_search",
                        "GENERIC_RESPONSE_NARROWING_ENABLED",
                    ):
                        saved_attributes[name] = getattr(self, name, missing)
                    human_result = None
                    ai_result = None
                    human_context: Dict[str, object] = {}
                    ai_context: Dict[str, object] = {}
                    try:
                        self._suppress_response_dictionary_metrics = True
                        self.GENERIC_RESPONSE_NARROWING_ENABLED = False
                        comparison_seconds = float(
                            getattr(
                                self,
                                "GENERIC_RESPONSE_HUMAN_ROOT_COMPARISON_SECONDS",
                                5.0,
                            )
                        )
                        if ai_actions:
                            ai_baseline = (
                                result.action
                                if result.action in ai_actions
                                else ai_actions[0]
                            )
                            ai_result = self._time_limited_search_from_samples(
                                state,
                                player,
                                list(ai_actions),
                                ai_baseline,
                                samples,
                                cancel_event=None,
                                tactical_priority_enabled=False,
                                record_priority_metrics=False,
                                run_context=ai_context,
                                forced_priority_action=ai_baseline,
                                max_seconds_override=comparison_seconds,
                            )
                        human_baseline = (
                            baseline_action
                            if baseline_action in human_actions
                            else max(
                                human_actions,
                                key=lambda action: self._timed_search_rule_prior(
                                    state,
                                    player,
                                    action,
                                ),
                            )
                        )
                        human_result = self._time_limited_search_from_samples(
                            state,
                            player,
                            list(human_actions),
                            human_baseline,
                            samples,
                            cancel_event=None,
                            tactical_priority_enabled=False,
                            record_priority_metrics=False,
                            run_context=human_context,
                            forced_priority_action=human_baseline,
                            max_seconds_override=comparison_seconds,
                        )
                    except Exception:
                        # Comparison-only work must never interrupt the move.
                        human_result = None
                        ai_result = None
                    finally:
                        for name, previous in saved_attributes.items():
                            if previous is missing:
                                try:
                                    delattr(self, name)
                                except AttributeError:
                                    pass
                            else:
                                setattr(self, name, previous)

                    comparison_complete = bool(
                        human_result is not None
                        and ai_result is not None
                    )
                    if not comparison_complete:
                        self._record_human_response_root_comparison(
                            comparison_complete=False,
                        )
                    else:
                        human_value = float(
                            human_context.get(
                                "evaluation_value",
                                human_result.value,
                            )
                        )
                        ai_value = float(
                            ai_context.get(
                                "evaluation_value",
                                ai_result.value,
                            )
                        )
                        value_delta = human_value - ai_value
                        if value_delta > 1e-6:
                            selected_side = "human"
                        elif value_delta < -1e-6:
                            selected_side = "ai"
                        else:
                            selected_side = "other"
                        self._record_human_response_root_comparison(
                            comparison_complete=True,
                            selected_side=selected_side,
                            ai_depth=ai_result.depth,
                            human_depth=human_result.depth,
                            ai_elapsed_seconds=ai_result.elapsed_seconds,
                            human_elapsed_seconds=(
                                human_result.elapsed_seconds
                            ),
                            ai_nodes=ai_result.nodes,
                            human_nodes=human_result.nodes,
                            value_delta=value_delta,
                            ai_terminal_win_rate=float(
                                ai_context.get("terminal_win_rate", 0.0)
                            ),
                            human_terminal_win_rate=float(
                                human_context.get("terminal_win_rate", 0.0)
                            ),
                            ai_terminal_loss_rate=float(
                                ai_context.get("terminal_loss_rate", 0.0)
                            ),
                            human_terminal_loss_rate=float(
                                human_context.get("terminal_loss_rate", 0.0)
                            ),
                            ai_terminal_point_swing=float(
                                ai_context.get("terminal_point_swing", 0.0)
                            ),
                            human_terminal_point_swing=float(
                                human_context.get(
                                    "terminal_point_swing",
                                    0.0,
                                )
                            ),
                        )
            return result
        finally:
            try:
                if cache_key is not None and cache_owner:
                    with self._measure_performance("cache"):
                        active_metrics = self._active_performance_metrics or {}
                        compute_seconds = float(
                            active_metrics.get("sample_generation", 0.0)
                        ) + (
                            float(result.elapsed_seconds)
                            if result is not None
                            else float(active_metrics.get("search", 0.0))
                        )
                        self._finish_timed_search_compute(
                            cache_key,
                            result,
                            source="background" if cancel_event is not None else "foreground",
                            compute_seconds=compute_seconds,
                            branch_kind=(
                                getattr(self, "_time_search_background_branch_kind", None)
                                if cancel_event is not None
                                else None
                            ),
                            branch_context=(
                                getattr(
                                    self,
                                    "_time_search_background_branch_context",
                                    None,
                                )
                                if cancel_event is not None
                                else None
                            ),
                        )
            finally:
                if foreground_registered:
                    from goita_ai2.current_ai.background_search import (
                        background_search_foreground_finished,
                    )

                    background_search_foreground_finished()

    def _time_limited_search_from_samples(
        self,
        state,
        player: str,
        actions: List[Action],
        baseline_action: Action,
        samples: Sequence[object],
        cancel_event=None,
        *,
        tactical_priority_enabled: bool = True,
        record_priority_metrics: bool = True,
        run_context: Optional[Dict[str, object]] = None,
        forced_priority_action: Optional[Action] = None,
        max_seconds_override: Optional[float] = None,
    ) -> Optional[TimedSearchResult]:
        start = time.perf_counter()
        if max_seconds_override is None:
            effective_seconds = float(
                self._effective_time_search_setting(
                    "effective_seconds",
                    self.TIME_SEARCH_MAX_SECONDS,
                )
            )
        else:
            effective_seconds = max(0.01, float(max_seconds_override))
        hard_deadline = start + float(
            getattr(self, "TIME_SEARCH_HARD_MAX_SECONDS", 20.0)
        )
        deadline = start + min(
            hard_deadline - start,
            max(0.01, effective_seconds),
        )
        baseline_scores = {
            "AC": int(state.team_score.get("AC", 0)),
            "BD": int(state.team_score.get("BD", 0)),
        }
        stats = {
            "nodes": 0,
            "max_nodes": int(
                self._effective_time_search_setting(
                    "effective_nodes",
                    self.TIME_SEARCH_MAX_NODES,
                )
            ),
        }
        information_set = None
        information_worlds = tuple()
        information_tracker = self._track.get(id(state))
        if (
            getattr(self, "TIME_SEARCH_INFORMATION_SET_ENABLED", True)
            and information_tracker is not None
        ):
            try:
                information_set, information_worlds = self._information_set_search_worlds(
                    state,
                    player,
                    information_tracker,
                    samples,
                )
            except (TypeError, ValueError):
                information_set = None
                information_worlds = tuple()
        information_enabled = bool(information_set is not None and information_worlds)
        if information_enabled:
            self.last_information_set_search = {
                "key": information_set.key.digest,
                "candidates": len(information_worlds),
                "observations": information_set.total_observations,
                "rejected_observations": information_set.rejected_observations,
                "confidence": information_set.confidence,
                "normalized_entropy": information_set.normalized_entropy,
            }

        control_root_actions = list(actions)
        root_actions = list(control_root_actions)
        tactical_priority_action = None
        if tactical_priority_enabled and forced_priority_action is None:
            tactical_priority_action = self._tactical_response_priority_action(
                state,
                player,
                root_actions,
                baseline_action,
            )
        if run_context is not None:
            run_context["tactical_priority_action"] = tactical_priority_action
        generic_priority_action = forced_priority_action
        if generic_priority_action is None:
            generic_priority_action = tactical_priority_action
        if generic_priority_action is None:
            generic_priority_action = self._generic_response_priority_action(
                state,
                player,
                root_actions,
                baseline_action,
            )
        priority_is_tactical = tactical_priority_action is not None
        root_actions = self._timed_search_prioritize_root_actions(
            root_actions,
            generic_priority_action,
        )
        priority_reordered = bool(
            generic_priority_action is not None
            and control_root_actions
            and control_root_actions[0] != generic_priority_action
        )
        control_active_actions = list(control_root_actions)
        priority_beam_preserved = False
        priority_comparison_complete = True
        narrowing_shadow_status = "insufficient_depth"
        narrowing_shadow_actions: List[Action] = []
        narrowing_shadow_full_candidates = 0
        narrowing_shadow_elapsed_through_depth_three = 0.0
        active_narrowing_status: Optional[str] = (
            "insufficient_depth"
            if generic_priority_action is not None
            and bool(getattr(self, "GENERIC_RESPONSE_NARROWING_ENABLED", False))
            else None
        )
        active_narrowing_full_candidates = 0
        active_narrowing_kept_candidates = 0
        active_narrowing_continuation_extension_seconds = 0.0
        active_narrowing_added_nodes = 0
        active_narrowing_stop_reason = ""
        active_narrowing_depth = int(
            getattr(self, "GENERIC_RESPONSE_NARROWING_DEPTH", 3)
        )
        completed_values: Optional[Dict[Action, List[float]]] = None
        completed_world_values: Optional[Dict[Action, Dict[int, float]]] = None
        completed_robust_values: Optional[Dict[Action, float]] = None
        completed_policy_decisions: Optional[Dict[Action, int]] = None
        completed_depth = 0
        stable_best: Optional[Action] = None
        stable_count = 0

        rule_scores = {
            action: self._timed_search_rule_prior(state, player, action)
            for action in root_actions
        }
        ranked_rules = sorted(root_actions, key=lambda action: rule_scores[action], reverse=True)
        rule_rank_bonus = {
            action: self.TIME_SEARCH_RULE_PRIOR_WEIGHT
            * (len(ranked_rules) - index)
            / max(1, len(ranked_rules))
            for index, action in enumerate(ranked_rules)
        }
        search_profile = getattr(self, "_time_search_profile", "default")
        if search_profile not in ("kyosha_pass_compare", "shi_insertion"):
            rule_rank_bonus[baseline_action] = (
                rule_rank_bonus.get(baseline_action, 0.0)
                + self.TIME_SEARCH_BASELINE_PRIOR
            )

        maximum_depth = int(
            self._effective_time_search_setting(
                "effective_depth",
                self.TIME_SEARCH_MAX_DEPTH,
            )
        )
        minimum_override_depth = 5
        weak_first_receive_profile = search_profile == "weak_first_receive"
        if search_profile == "kyosha_pass_compare":
            maximum_depth = min(
                maximum_depth,
                int(self.KYOSHA_PASS_COMPARE_TARGET_DEPTH),
            )
            minimum_override_depth = min(
                maximum_depth,
                int(self.KYOSHA_PASS_COMPARE_TARGET_DEPTH),
            )
        elif search_profile == "low_reentry_receive":
            minimum_override_depth = min(
                maximum_depth,
                int(self.LOW_REENTRY_RECEIVE_SEARCH_TARGET_DEPTH),
            )
        elif search_profile == "shi_insertion":
            maximum_depth = min(
                maximum_depth,
                int(self.SHI_INSERTION_SEARCH_TARGET_DEPTH),
            )
            minimum_override_depth = min(
                maximum_depth,
                int(self.SHI_INSERTION_SEARCH_TARGET_DEPTH),
            )
        elif weak_first_receive_profile:
            minimum_override_depth = min(
                maximum_depth,
                int(self.WEAK_FIRST_RECEIVE_SEARCH_TARGET_DEPTH),
            )
        elif len(state.hands[player]) >= 7:
            maximum_depth = min(maximum_depth, 7)
            minimum_override_depth = min(
                maximum_depth,
                int(self.TIME_SEARCH_EARLY_OVERRIDE_MIN_DEPTH),
            )
        for depth in range(1, maximum_depth + 1, 2):
            iteration: Dict[Action, List[float]] = {action: [] for action in root_actions}
            iteration_world_values: Dict[Action, Dict[int, float]] = {}
            iteration_robust_values: Dict[Action, float] = {}
            iteration_policy_decisions: Dict[Action, int] = {}
            try:
                if information_enabled:
                    for action in root_actions:
                        if cancel_event is not None and cancel_event.is_set():
                            raise InformationSetSearchCancelled()
                        if time.perf_counter() >= deadline:
                            raise InformationSetSearchDeadline("time_limit")
                        outcome = self._information_set_search_root_action(
                            state,
                            information_worlds,
                            player,
                            action,
                            information_set,
                            baseline_scores,
                            depth,
                            deadline,
                            stats,
                            information_tracker,
                            cancel_event,
                        )
                        world_values = outcome.values_dict()
                        iteration_world_values[action] = world_values
                        iteration[action] = [
                            world_values[world.index]
                            for world in information_worlds
                            if world.index in world_values
                        ]
                        iteration_robust_values[action] = outcome.value
                        iteration_policy_decisions[action] = outcome.policy_decisions
                else:
                    for sampled in samples:
                        if cancel_event is not None and cancel_event.is_set():
                            raise _SearchCancelled()
                        memo: Dict[tuple, float] = {}
                        for action in root_actions:
                            if cancel_event is not None and cancel_event.is_set():
                                raise _SearchCancelled()
                            if time.perf_counter() >= deadline:
                                raise _SearchDeadline("time_limit")
                            next_state = self._timed_search_apply(sampled, player, action)
                            value = self._timed_search_minimax(
                                next_state,
                                player,
                                baseline_scores,
                                depth - 1,
                                -float("inf"),
                                float("inf"),
                                deadline,
                                stats,
                                memo,
                                cancel_event,
                            )
                            iteration[action].append(value)
            except (_SearchCancelled, InformationSetSearchCancelled):
                return None
            except (_SearchDeadline, InformationSetSearchDeadline) as exc:
                active_narrowing_stop_reason = (
                    str(exc.args[0]) if exc.args else "other"
                )
                break

            completed_values = iteration
            completed_world_values = iteration_world_values if information_enabled else None
            completed_robust_values = iteration_robust_values if information_enabled else None
            completed_policy_decisions = (
                iteration_policy_decisions if information_enabled else None
            )
            completed_depth = depth
            aggregate = {
                action: (
                    (
                        iteration_robust_values[action]
                        if information_enabled
                        else sum(values) / len(values)
                    )
                    + rule_rank_bonus.get(action, 0.0)
                )
                for action, values in iteration.items()
                if values
            }
            if not aggregate:
                break
            best = max(aggregate, key=aggregate.get)
            if best == stable_best:
                stable_count += 1
            else:
                stable_best = best
                stable_count = 1

            # Compare, and optionally activate, a two-candidate continuation.
            if depth == active_narrowing_depth and generic_priority_action is not None:
                narrowing_shadow_full_candidates = len(root_actions)
                narrowing_shadow_elapsed_through_depth_three = (
                    time.perf_counter() - start
                )
                if generic_priority_action not in aggregate:
                    narrowing_shadow_status = "incomplete"
                elif len(root_actions) <= 2:
                    narrowing_shadow_status = "no_reduction"
                else:
                    strongest_rival = max(
                        (
                            action
                            for action in root_actions
                            if action != generic_priority_action
                        ),
                        key=lambda action: aggregate.get(
                            action,
                            -float("inf"),
                        ),
                    )
                    narrowing_shadow_actions = [
                        generic_priority_action,
                        strongest_rival,
                    ]
                    narrowing_shadow_status = "compared"

                if bool(
                    getattr(self, "GENERIC_RESPONSE_NARROWING_ENABLED", False)
                ):
                    active_narrowing_full_candidates = len(root_actions)
                    if maximum_depth <= depth:
                        active_narrowing_status = "insufficient_depth"
                    else:
                        narrowed_actions, active_narrowing_status = (
                            self._timed_search_safe_generic_narrowing(
                                root_actions,
                                aggregate,
                                generic_priority_action,
                                enabled=True,
                                search_profile=search_profile,
                                information_enabled=information_enabled,
                                information_confidence=(
                                    float(information_set.confidence)
                                    if information_enabled
                                    else 0.0
                                ),
                                minimum_confidence=float(
                                    getattr(
                                        self,
                                        "GENERIC_RESPONSE_NARROWING_MIN_CONFIDENCE",
                                        0.45,
                                    )
                                ),
                                minimum_excluded_margin=float(
                                    getattr(
                                        self,
                                        "GENERIC_RESPONSE_NARROWING_MIN_EXCLUDED_MARGIN",
                                        450.0,
                                    )
                                ),
                            )
                        )
                        if narrowed_actions is not None:
                            root_actions = narrowed_actions
                            active_narrowing_kept_candidates = len(root_actions)
                            deadline, extension = (
                                self._timed_search_extend_narrowing_deadline(
                                    deadline,
                                    hard_deadline,
                                    time.perf_counter(),
                                    float(
                                        getattr(
                                            self,
                                            "GENERIC_RESPONSE_NARROWING_MIN_CONTINUATION_SECONDS",
                                            1.0,
                                        )
                                    ),
                                )
                            )
                            active_narrowing_continuation_extension_seconds += (
                                extension
                            )
                            extended_nodes, added_nodes = (
                                self._timed_search_extend_narrowing_nodes(
                                    int(stats["max_nodes"]),
                                    float(
                                        getattr(
                                            self,
                                            "GENERIC_RESPONSE_NARROWING_NODE_EXTENSION_RATIO",
                                            0.25,
                                        )
                                    ),
                                )
                            )
                            stats["max_nodes"] = extended_nodes
                            active_narrowing_added_nodes += added_nodes

            if depth == 1 and len(root_actions) > self.TIME_SEARCH_ROOT_BEAM:
                narrowed = sorted(
                    root_actions,
                    key=lambda action: aggregate.get(action, -float("inf")),
                    reverse=True,
                )[: self.TIME_SEARCH_ROOT_BEAM]
                if baseline_action not in narrowed:
                    narrowed[-1] = baseline_action
                if (
                    generic_priority_action is not None
                    and generic_priority_action not in narrowed
                ):
                    replaceable = [
                        action
                        for action in reversed(narrowed)
                        if action != baseline_action
                    ]
                    if replaceable:
                        narrowed[narrowed.index(replaceable[0])] = (
                            generic_priority_action
                        )
                control_narrowed = sorted(
                    control_root_actions,
                    key=lambda action: aggregate.get(action, -float("inf")),
                    reverse=True,
                )[: self.TIME_SEARCH_ROOT_BEAM]
                if baseline_action not in control_narrowed:
                    control_narrowed[-1] = baseline_action
                control_active_actions = control_narrowed
                priority_beam_preserved = bool(
                    generic_priority_action is not None
                    and generic_priority_action in narrowed
                    and generic_priority_action not in control_narrowed
                )
                priority_comparison_complete = (
                    set(narrowed) == set(control_narrowed)
                )
                root_actions = narrowed
            if depth >= minimum_override_depth and stable_count >= 2:
                ordered = sorted(aggregate.values(), reverse=True)
                if len(ordered) == 1 or ordered[0] - ordered[1] >= self.TIME_SEARCH_STABLE_MARGIN:
                    break

        if completed_values is None or completed_depth <= 0:
            return None

        aggregate = {}
        for action, values in completed_values.items():
            if not values:
                continue
            if information_enabled and completed_robust_values is not None:
                aggregate[action] = (
                    completed_robust_values[action]
                    + rule_rank_bonus.get(action, 0.0)
                )
            else:
                ordered_values = sorted(values)
                lower_index = max(0, (len(ordered_values) - 1) // 4)
                lower_quartile = ordered_values[lower_index]
                mean_value = sum(values) / len(values)
                aggregate[action] = (
                    mean_value * 0.78
                    + lower_quartile * 0.22
                    + rule_rank_bonus.get(action, 0.0)
                )
        if not aggregate:
            return None
        if run_context is not None:
            run_context["root_values"] = dict(aggregate)
            run_context["completed_depth"] = completed_depth
        ordered_actions = sorted(aggregate, key=aggregate.get, reverse=True)
        best_action = ordered_actions[0]
        best_value = aggregate[best_action]
        second_value = aggregate[ordered_actions[1]] if len(ordered_actions) > 1 else best_value
        margin = best_value - second_value

        if run_context is not None:
            run_context["evaluation_value"] = (
                best_value - rule_rank_bonus.get(best_action, 0.0)
            )
            if information_enabled and completed_world_values is not None:
                best_world_values = completed_world_values.get(
                    best_action,
                    {},
                )
                weighted_values = [
                    (
                        float(best_world_values[world.index]),
                        float(world.probability),
                    )
                    for world in information_worlds
                    if world.index in best_world_values
                ]
            else:
                best_values = completed_values.get(best_action, [])
                equal_weight = 1.0 / max(1, len(best_values))
                weighted_values = [
                    (float(value), equal_weight)
                    for value in best_values
                ]
            total_weight = sum(weight for _value, weight in weighted_values)
            if total_weight > 0.0:
                terminal_win_rate = sum(
                    weight
                    for value, weight in weighted_values
                    if value >= 100000.0
                ) / total_weight
                terminal_loss_rate = sum(
                    weight
                    for value, weight in weighted_values
                    if value <= -100000.0
                ) / total_weight
                terminal_point_swing = sum(
                    (
                        (value - 100000.0) / 500.0
                        if value >= 100000.0
                        else (value + 100000.0) / 500.0
                        if value <= -100000.0
                        else 0.0
                    )
                    * weight
                    for value, weight in weighted_values
                ) / total_weight
            else:
                terminal_win_rate = 0.0
                terminal_loss_rate = 0.0
                terminal_point_swing = 0.0
            run_context["terminal_win_rate"] = terminal_win_rate
            run_context["terminal_loss_rate"] = terminal_loss_rate
            run_context["terminal_point_swing"] = terminal_point_swing

        if information_enabled and completed_world_values is not None:
            agreement_mass = 0.0
            available_mass = 0.0
            for world in information_worlds:
                available = [
                    action
                    for action, values in completed_world_values.items()
                    if world.index in values
                ]
                if not available:
                    continue
                sample_best = max(
                    available,
                    key=lambda action: completed_world_values[action][world.index]
                    + rule_rank_bonus.get(action, 0.0),
                )
                available_mass += world.probability
                if sample_best == best_action:
                    agreement_mass += world.probability
            agreement = agreement_mass / max(available_mass, 1e-12)
            sample_total = int(information_set.total_observations)
        else:
            agreements = 0
            sample_total = len(next(iter(completed_values.values())))
            for sample_index in range(sample_total):
                available = [
                    action
                    for action in completed_values
                    if len(completed_values[action]) > sample_index
                ]
                if not available:
                    continue
                sample_best = max(
                    available,
                    key=lambda action: completed_values[action][sample_index]
                    + rule_rank_bonus.get(action, 0.0),
                )
                if sample_best == best_action:
                    agreements += 1
            agreement = agreements / max(1, sample_total)
        best_minimum = min(completed_values[best_action])
        baseline_values = completed_values.get(baseline_action, [])
        baseline_minimum = min(baseline_values) if baseline_values else -float("inf")
        enemy_third_attack_wait = self._timed_search_enemy_third_attack_wait_is_safe(
            state,
            player,
            information_tracker,
            baseline_action=baseline_action,
            best_action=best_action,
            completed_depth=completed_depth,
            agreement=agreement,
            information_enabled=information_enabled,
            information_confidence=(
                float(information_set.confidence) if information_enabled else 0.0
            ),
            best_minimum=best_minimum,
            baseline_minimum=baseline_minimum,
            margin=margin,
        )
        weak_first_receive_decisive = self._timed_search_weak_first_receive_is_decisive(
            baseline_action=baseline_action,
            best_action=best_action,
            completed_depth=completed_depth,
            agreement=agreement,
            margin=margin,
            information_enabled=information_enabled,
            information_confidence=(
                float(information_set.confidence) if information_enabled else 0.0
            ),
            zero_shi_stop_signal=self._timed_search_zero_shi_stop_signal_context(
                state,
                player,
                information_tracker,
                baseline_action=baseline_action,
                best_action=best_action,
            ),
        )
        kyosha_pass_compare_decisive = (
            self._timed_search_kyosha_pass_compare_is_decisive(
                baseline_action=baseline_action,
                best_action=best_action,
                completed_depth=completed_depth,
                agreement=agreement,
                margin=margin,
                information_enabled=information_enabled,
                information_confidence=(
                    float(information_set.confidence)
                    if information_enabled
                    else 0.0
                ),
            )
        )
        if search_profile == "kyosha_pass_compare":
            decisive = kyosha_pass_compare_decisive
        else:
            decisive = (
                best_action != baseline_action
                and best_minimum >= 50000.0
                and baseline_minimum < 50000.0
            ) or (
                best_action != baseline_action
                and completed_depth >= minimum_override_depth
                and agreement >= self.TIME_SEARCH_OVERRIDE_AGREEMENT
                and margin >= self.TIME_SEARCH_OVERRIDE_MARGIN
            ) or enemy_third_attack_wait or weak_first_receive_decisive
        elapsed_seconds = time.perf_counter() - start
        if generic_priority_action is not None and record_priority_metrics:
            control_candidates = [
                action
                for action in control_active_actions
                if action in aggregate
            ]
            comparison_complete = bool(
                priority_comparison_complete
                and len(control_candidates) == len(control_active_actions)
            )
            control_best_action = (
                max(control_candidates, key=lambda action: aggregate[action])
                if comparison_complete and control_candidates
                else None
            )
            control_value = (
                aggregate[control_best_action]
                if control_best_action is not None
                else best_value
            )
            if priority_is_tactical:
                self._record_tactical_response_priority_effect(
                    reordered=priority_reordered,
                    baseline_disagreed=(
                        generic_priority_action != baseline_action
                    ),
                    selected=best_action == generic_priority_action,
                    completed_depth=completed_depth,
                )
            else:
                self._record_generic_response_priority_effect(
                    reordered=priority_reordered,
                    beam_preserved=priority_beam_preserved,
                    comparison_complete=comparison_complete,
                    recommended_selected=best_action == generic_priority_action,
                    action_changed=(
                        control_best_action is not None
                        and best_action != control_best_action
                    ),
                    with_depth=completed_depth,
                    without_depth=(completed_depth if comparison_complete else 0),
                    with_elapsed_seconds=elapsed_seconds,
                    without_elapsed_seconds=(
                        elapsed_seconds if comparison_complete else 0.0
                    ),
                    value_delta=best_value - control_value,
                )
            if (
                narrowing_shadow_status == "compared"
                and active_narrowing_status != "applied"
            ):
                shadow_candidates = [
                    action
                    for action in narrowing_shadow_actions
                    if action in aggregate
                ]
                if len(shadow_candidates) != len(narrowing_shadow_actions):
                    narrowing_shadow_status = "incomplete"
                else:
                    shadow_best_action = max(
                        shadow_candidates,
                        key=lambda action: aggregate[action],
                    )
                    deeper_elapsed = max(
                        0.0,
                        elapsed_seconds
                        - narrowing_shadow_elapsed_through_depth_three,
                    )
                    kept_ratio = (
                        len(shadow_candidates)
                        / max(1, narrowing_shadow_full_candidates)
                    )
                    estimated_elapsed = (
                        narrowing_shadow_elapsed_through_depth_three
                        + deeper_elapsed * kept_ratio
                    )
                    self._record_generic_response_narrowing_shadow(
                        status="compared",
                        matched=shadow_best_action == best_action,
                        priority_selected=(
                            best_action == generic_priority_action
                        ),
                        full_candidates=narrowing_shadow_full_candidates,
                        kept_candidates=len(shadow_candidates),
                        depth=completed_depth,
                        actual_elapsed_seconds=elapsed_seconds,
                        estimated_elapsed_seconds=estimated_elapsed,
                        value_loss=(
                            best_value - aggregate[shadow_best_action]
                        ),
                    )
            if (
                narrowing_shadow_status != "compared"
                and active_narrowing_status != "applied"
            ):
                self._record_generic_response_narrowing_shadow(
                    status=narrowing_shadow_status,
                )
            if active_narrowing_status is not None:
                live_status = active_narrowing_status
                if live_status == "applied":
                    live_status = (
                        "deepened"
                        if completed_depth > active_narrowing_depth
                        else "no_deepening"
                    )
                self._record_generic_response_active_narrowing(
                    status=live_status,
                    full_candidates=active_narrowing_full_candidates,
                    kept_candidates=active_narrowing_kept_candidates,
                    completed_depth=completed_depth,
                    elapsed_seconds=elapsed_seconds,
                    priority_selected=(best_action == generic_priority_action),
                    continuation_extension_seconds=(
                        active_narrowing_continuation_extension_seconds
                    ),
                    added_nodes=active_narrowing_added_nodes,
                    no_deepening_reason=(
                        active_narrowing_stop_reason
                        if live_status == "no_deepening"
                        else ""
                    ),
                )

        return TimedSearchResult(
            action=best_action,
            depth=completed_depth,
            samples=sample_total,
            nodes=stats["nodes"],
            elapsed_seconds=elapsed_seconds,
            value=best_value,
            margin=margin,
            agreement=agreement,
            decisive=decisive,
            information_set=information_enabled,
            candidate_count=len(information_worlds) if information_enabled else 0,
            information_confidence=(
                float(information_set.confidence) if information_enabled else 0.0
            ),
            policy_decisions=(
                int((completed_policy_decisions or {}).get(best_action, 0))
                if information_enabled
                else 0
            ),
            enemy_third_attack_wait=enemy_third_attack_wait,
        )

    def _commit_timed_search_action(self, state, player: str, action: Action) -> None:
        tr = self._track.get(id(state))
        if tr is None or action[0] not in ("attack", "attack_after_block"):
            return
        tr["my_attack_count"] = int(tr.get("my_attack_count", 0)) + 1
        tr["pending_weak_hand_shi_signal"] = False
        tr["pending_ally_force_king_attack_piece"] = None
        tr["pending_inferred_endgame_attack"] = None
        if (
            tr.get("kg_plan_active")
            and tr["my_attack_count"] == 2
            and action[2] in ("8", "9")
            and tr.get("kg_second") is None
        ):
            tr["kg_second"] = action[2]
        if tr.get("kg_plan_active") and tr["my_attack_count"] >= 3:
            tr["kg_plan_active"] = False

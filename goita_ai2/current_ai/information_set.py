"""Represents one Goita position using only information visible to one player.

The key deliberately excludes real opposing hands, while candidate deals keep
their normalized probability and evidence confidence for later shared search.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

from goita_ai2.constants import ALL_SEATS, PIECE_TOTALS
from goita_ai2.current_ai.prediction_cache import PredictionSample


PIECES = tuple(str(index) for index in range(1, 10))

# These fields are all derived from the observer's hand and public actions.
# Local attack plans and previous search outputs are intentionally excluded.
_PUBLIC_TRACKER_FIELDS = (
    "ally",
    "public_seen_counts",
    "my_attack_count",
    "my_attack_history",
    "my_past_attacks",
    "ally_past_attacks",
    "enemy_past_attacks",
    "enemy_attack_counts",
    "hidden_block_counts",
    "other_first_attack_strategy_by_player",
    "other_piece_count_estimates",
    "current_piece_count_caps",
    "piece_pass_evidence",
    "estimated_current_hands",
    "joint_hand_inference",
    "unknown_piece_pool",
    "public_hand_models",
    "ally_responded_to_my_attacks",
    "ally_ignored_my_attacks",
    "ally_shi_signal",
    "shi_attack_mode",
    "enemy_team_rejected_shi_attack",
    "active_attack_context",
)


def _freeze(value):
    """Convert public tracker data into a stable, hashable representation."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return repr(value)


@dataclass(frozen=True)
class PublicInformationSetKey:
    """Stable identity for everything one player currently knows."""

    digest: str
    player: str
    phase: str
    turn: str
    hand_size: int
    hidden_candidate_slots: int
    inference_revision: int = field(compare=False)

    def as_dict(self) -> Dict[str, object]:
        return {
            "digest": self.digest,
            "player": self.player,
            "phase": self.phase,
            "turn": self.turn,
            "hand_size": self.hand_size,
            "hidden_candidate_slots": self.hidden_candidate_slots,
            "inference_revision": self.inference_revision,
        }


@dataclass(frozen=True)
class InformationSetCandidate:
    """One hidden deal compatible with a public information-set key."""

    prediction: PredictionSample
    observations: int
    relative_likelihood: float
    probability: float
    confidence: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "opponent_hands": {
                seat: list(hand)
                for seat, hand in self.prediction.opponent_hands
            },
            "opponent_hidden": {
                seat: list(hand)
                for seat, hand in self.prediction.opponent_hidden
            },
            "observations": self.observations,
            "relative_likelihood": round(self.relative_likelihood, 6),
            "probability": round(self.probability, 6),
            "confidence": round(self.confidence, 6),
        }


@dataclass(frozen=True)
class InformationSet:
    """A public position and the weighted hidden deals compatible with it."""

    key: PublicInformationSetKey
    candidates: Tuple[InformationSetCandidate, ...]
    total_observations: int
    rejected_observations: int
    confidence: float
    normalized_entropy: float

    @property
    def effective_candidate_count(self) -> float:
        denominator = sum(candidate.probability ** 2 for candidate in self.candidates)
        return 0.0 if denominator <= 0.0 else 1.0 / denominator

    def as_dict(self) -> Dict[str, object]:
        return {
            "key": self.key.as_dict(),
            "candidate_count": len(self.candidates),
            "total_observations": self.total_observations,
            "rejected_observations": self.rejected_observations,
            "confidence": round(self.confidence, 6),
            "normalized_entropy": round(self.normalized_entropy, 6),
            "effective_candidate_count": round(self.effective_candidate_count, 6),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


class InformationSetMixin:
    """Builds weighted hidden-deal groups without observing real hidden hands."""

    def _information_set_key(
        self,
        state,
        player: str,
        tr: dict,
    ) -> PublicInformationSetKey:
        hidden_counts = tr.get("hidden_block_counts", {})
        tracker_payload = tuple(
            (field, _freeze(tr.get(field)))
            for field in _PUBLIC_TRACKER_FIELDS
        )
        payload = (
            "information-set-v1",
            player,
            tuple(sorted(state.hands[player])),
            tuple(sorted(state.face_down_hidden[player])),
            bool(state.had_both_kings.get(player, False)),
            state.dealer,
            state.turn,
            state.phase,
            state.current_attack,
            state.attacker,
            state.last_block_player,
            state.last_block if state.last_block_player == player else None,
            int(state.king_block_used),
            bool(state.finished),
            state.winner,
            tuple(sorted(state.team_score.items())),
            tuple((seat, len(state.hands[seat])) for seat in ALL_SEATS),
            tuple(
                (
                    seat,
                    len(state.face_down_hidden[seat])
                    if seat == player
                    else int(hidden_counts.get(seat, 0)),
                )
                for seat in ALL_SEATS
            ),
            tracker_payload,
        )
        digest = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()
        hidden_slots = sum(
            len(state.hands[seat]) + int(hidden_counts.get(seat, 0))
            for seat in ALL_SEATS
            if seat != player
        )
        return PublicInformationSetKey(
            digest=digest,
            player=player,
            phase=str(state.phase),
            turn=str(state.turn),
            hand_size=len(state.hands[player]),
            hidden_candidate_slots=hidden_slots,
            inference_revision=int(tr.get("piece_inference_revision", 0)),
        )

    @staticmethod
    def _information_set_prediction(sampled, player: str) -> PredictionSample:
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
    def _candidate_counts(
        prediction: PredictionSample,
    ) -> Tuple[Dict[str, Counter], Dict[str, Counter]]:
        current = {
            seat: Counter(hand)
            for seat, hand in prediction.opponent_hands
        }
        hidden = {
            seat: Counter(hand)
            for seat, hand in prediction.opponent_hidden
        }
        return current, hidden

    def _information_set_candidate_evidence(
        self,
        prediction: PredictionSample,
        player: str,
        tr: dict,
    ) -> Optional[Tuple[float, float]]:
        """Return log likelihood and confidence, or reject an impossible deal."""
        current, hidden = self._candidate_counts(prediction)
        estimates = tr.get("estimated_current_hands", {})
        joint = tr.get("joint_hand_inference", {})
        map_current = joint.get("map_current_counts", {}) if joint.get("feasible") else {}

        log_likelihood = 0.0
        confidence_sum = 0.0
        confidence_fit = 0.0
        evidence_items = 0
        for seat in ALL_SEATS:
            if seat == player:
                continue
            combined = current.get(seat, Counter()) + hidden.get(seat, Counter())
            for count in combined.values():
                log_likelihood -= math.lgamma(int(count) + 1)

            for piece in PIECES:
                item = estimates.get(seat, {}).get(piece, {})
                count = int(current.get(seat, Counter()).get(piece, 0))
                minimum = int(item.get("min", 0))
                maximum = int(item.get("max", PIECE_TOTALS[piece]))
                if count < minimum or count > maximum:
                    return None

                expected = max(float(minimum), min(float(maximum), float(
                    item.get("expected", 0.0)
                )))
                confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
                scale = max(1.0, float(maximum - minimum + 1))
                deviation = abs(float(count) - expected) / scale
                fit = max(0.0, 1.0 - deviation)
                log_likelihood -= confidence * deviation * 2.0
                confidence_sum += confidence
                confidence_fit += confidence * fit
                evidence_items += 1

                map_count = map_current.get(seat, {}).get(piece)
                if map_count is not None:
                    log_likelihood -= 0.12 * abs(count - int(map_count))

        evidence_strength = confidence_sum / max(1, evidence_items)
        match_quality = (
            confidence_fit / confidence_sum
            if confidence_sum > 0.0
            else 0.0
        )
        candidate_confidence = max(
            0.0,
            min(1.0, evidence_strength * match_quality),
        )
        return log_likelihood, candidate_confidence

    def _information_set_candidate_consistent(
        self,
        state,
        player: str,
        tr: dict,
        prediction: PredictionSample,
    ) -> bool:
        """Reject assignments that contradict public counts or royal history."""
        opponents = tuple(seat for seat in ALL_SEATS if seat != player)
        hands = dict(prediction.opponent_hands)
        hidden = dict(prediction.opponent_hidden)
        royal_history = dict(prediction.opponent_had_both_kings)
        if (
            tuple(sorted(hands)) != tuple(sorted(opponents))
            or tuple(sorted(hidden)) != tuple(sorted(opponents))
            or tuple(sorted(royal_history)) != tuple(sorted(opponents))
        ):
            return False

        hidden_counts = tr.get("hidden_block_counts", {})
        combined_pool = Counter()
        for seat in opponents:
            if len(hands[seat]) != len(state.hands[seat]):
                return False
            if len(hidden[seat]) != int(hidden_counts.get(seat, 0)):
                return False
            if any(piece not in PIECES for piece in hands[seat] + hidden[seat]):
                return False
            combined_pool.update(hands[seat])
            combined_pool.update(hidden[seat])
            observed_eight = int(self._observed_piece_count_for_player(tr, seat, "8"))
            observed_nine = int(self._observed_piece_count_for_player(tr, seat, "9"))
            had_both = (
                hands[seat].count("8")
                + hidden[seat].count("8")
                + observed_eight
                > 0
                and hands[seat].count("9")
                + hidden[seat].count("9")
                + observed_nine
                > 0
            )
            if bool(royal_history[seat]) != had_both:
                return False

        expected_pool = Counter({
            piece: int(tr.get("unknown_piece_pool", {}).get(piece, 0))
            for piece in PIECES
        })
        if combined_pool != expected_pool:
            return False

        last_block_player = state.last_block_player
        if last_block_player in opponents:
            if prediction.last_block is None:
                return False
            if prediction.last_block not in hidden[last_block_player]:
                return False
        elif prediction.last_block != state.last_block:
            return False
        return True

    def _build_information_set(
        self,
        state,
        player: str,
        tr: dict,
        sampled_states: Sequence[object],
    ) -> InformationSet:
        """Collapse sampled deals and normalize their posterior-like weights."""
        key = self._information_set_key(state, player, tr)
        grouped: Dict[PredictionSample, Tuple[int, float, float]] = {}
        rejected_observations = 0
        for sampled in sampled_states:
            sampled_key = self._information_set_key(sampled, player, tr)
            if sampled_key.digest != key.digest:
                raise ValueError("candidate state does not share the public information set")
            prediction = self._information_set_prediction(sampled, player)
            if not self._information_set_candidate_consistent(
                state,
                player,
                tr,
                prediction,
            ):
                rejected_observations += 1
                continue
            evidence = self._information_set_candidate_evidence(
                prediction,
                player,
                tr,
            )
            if evidence is None:
                rejected_observations += 1
                continue
            log_likelihood, confidence = evidence
            previous = grouped.get(prediction)
            if previous is None:
                grouped[prediction] = (1, log_likelihood, confidence)
            else:
                grouped[prediction] = (
                    previous[0] + 1,
                    previous[1],
                    previous[2],
                )

        if not grouped:
            raise ValueError("information set has no compatible candidate deals")

        maximum_log = max(item[1] for item in grouped.values())
        weighted = []
        total_weight = 0.0
        for prediction, (observations, log_likelihood, confidence) in grouped.items():
            relative = math.exp(max(-60.0, min(0.0, log_likelihood - maximum_log)))
            weight = float(observations) * relative
            total_weight += weight
            weighted.append((prediction, observations, relative, weight, confidence))

        candidates = tuple(sorted(
            (
                InformationSetCandidate(
                    prediction=prediction,
                    observations=observations,
                    relative_likelihood=relative,
                    probability=weight / total_weight,
                    confidence=confidence,
                )
                for prediction, observations, relative, weight, confidence in weighted
            ),
            key=lambda candidate: (
                -candidate.probability,
                repr(candidate.prediction),
            ),
        ))
        total_observations = sum(candidate.observations for candidate in candidates)
        set_confidence = sum(
            candidate.probability * candidate.confidence
            for candidate in candidates
        )
        if len(candidates) <= 1:
            entropy = 0.0
        else:
            entropy = -sum(
                candidate.probability * math.log(candidate.probability)
                for candidate in candidates
                if candidate.probability > 0.0
            ) / math.log(len(candidates))
        return InformationSet(
            key=key,
            candidates=candidates,
            total_observations=total_observations,
            rejected_observations=rejected_observations,
            confidence=set_confidence,
            normalized_entropy=entropy,
        )

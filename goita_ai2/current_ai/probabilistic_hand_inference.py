"""Builds joint prior and posterior models for opponents' hidden Goita hands.
It samples complete legal deals, then weights them with attacks, receives,
passes, team signals, and known exceptions without reading hidden hands.
"""

from __future__ import annotations

import hashlib
import math
import random
import threading
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from goita_ai2.constants import ALL_SEATS
from goita_ai2.current_ai.information_set import PIECES, PublicInformationSetKey
from goita_ai2.current_ai.prediction_cache import PredictionSample


@dataclass(frozen=True)
class CountProbability:
    """Probability mass assigned to one exact piece count."""

    count: int
    probability: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "count": self.count,
            "probability": round(self.probability, 6),
        }


@dataclass(frozen=True)
class LabelProbability:
    """Probability mass for a rank or structural label."""

    label: str
    probability: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "probability": round(self.probability, 6),
        }


@dataclass(frozen=True)
class PieceProbability:
    """Current and original count distributions for one piece type."""

    piece: str
    current_count_distribution: Tuple[CountProbability, ...]
    original_count_distribution: Tuple[CountProbability, ...]

    @property
    def current_holding_probability(self) -> float:
        return sum(
            item.probability
            for item in self.current_count_distribution
            if item.count > 0
        )

    @property
    def original_holding_probability(self) -> float:
        return sum(
            item.probability
            for item in self.original_count_distribution
            if item.count > 0
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "piece": self.piece,
            "current_holding_probability": round(
                self.current_holding_probability,
                6,
            ),
            "original_holding_probability": round(
                self.original_holding_probability,
                6,
            ),
            "current_count_distribution": [
                item.as_dict() for item in self.current_count_distribution
            ],
            "original_count_distribution": [
                item.as_dict() for item in self.original_count_distribution
            ],
        }


@dataclass(frozen=True)
class HandCandidateProbability:
    """One marginal hand hypothesis for a single player."""

    current_hand: Tuple[str, ...]
    hidden_hand: Tuple[str, ...]
    original_hand: Tuple[str, ...]
    probability: float
    absolute_rank: str
    relative_rank: str
    structure_key: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "current_hand": list(self.current_hand),
            "hidden_hand": list(self.hidden_hand),
            "original_hand": list(self.original_hand),
            "probability": round(self.probability, 6),
            "absolute_rank": self.absolute_rank,
            "relative_rank": self.relative_rank,
            "structure_key": self.structure_key,
        }


@dataclass(frozen=True)
class PlayerHandProbability:
    """All stage-one probability outputs for one other player."""

    seat: str
    current_hand_size: int
    hidden_hand_size: int
    pieces: Tuple[PieceProbability, ...]
    top_hands: Tuple[HandCandidateProbability, ...]
    absolute_rank_distribution: Tuple[LabelProbability, ...]
    relative_rank_distribution: Tuple[LabelProbability, ...]
    structure_distribution: Tuple[LabelProbability, ...]
    confidence: float = 0.0

    def piece(self, piece: str) -> PieceProbability:
        return next(item for item in self.pieces if item.piece == str(piece))

    @property
    def most_likely_hand(self) -> Optional[HandCandidateProbability]:
        return self.top_hands[0] if self.top_hands else None

    def as_dict(self) -> Dict[str, object]:
        return {
            "seat": self.seat,
            "current_hand_size": self.current_hand_size,
            "hidden_hand_size": self.hidden_hand_size,
            "confidence": round(self.confidence, 6),
            "most_likely_hand": (
                self.most_likely_hand.as_dict()
                if self.most_likely_hand is not None
                else None
            ),
            "pieces": [item.as_dict() for item in self.pieces],
            "top_hands": [item.as_dict() for item in self.top_hands],
            "absolute_rank_distribution": [
                item.as_dict() for item in self.absolute_rank_distribution
            ],
            "relative_rank_distribution": [
                item.as_dict() for item in self.relative_rank_distribution
            ],
            "structure_distribution": [
                item.as_dict() for item in self.structure_distribution
            ],
        }


@dataclass(frozen=True)
class ProbabilisticDealCandidate:
    """One complete legal deal with prior and action-weighted posterior mass."""

    prediction: PredictionSample
    observations: int
    prior_probability: float
    relative_likelihood: float = 1.0
    posterior_probability: Optional[float] = None
    confidence: float = 0.0
    evidence: Tuple[str, ...] = ()

    @property
    def probability(self) -> float:
        if self.posterior_probability is None:
            return self.prior_probability
        return self.posterior_probability

    def as_dict(self) -> Dict[str, object]:
        return {
            "prediction": {
                "opponent_hands": {
                    seat: list(hand)
                    for seat, hand in self.prediction.opponent_hands
                },
                "opponent_hidden": {
                    seat: list(hand)
                    for seat, hand in self.prediction.opponent_hidden
                },
                "opponent_had_both_kings": dict(
                    self.prediction.opponent_had_both_kings
                ),
                "last_block": self.prediction.last_block,
            },
            "observations": self.observations,
            "prior_probability": round(self.prior_probability, 6),
            "relative_likelihood": round(self.relative_likelihood, 6),
            "posterior_probability": round(self.probability, 6),
            "confidence": round(self.confidence, 6),
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class InitialProbabilisticHandInference:
    """Stage-one/two prior before action-likelihood updates are introduced."""

    key: PublicInformationSetKey
    observer: str
    requested_samples: int
    accepted_samples: int
    rejected_samples: int
    candidates: Tuple[ProbabilisticDealCandidate, ...]
    players: Tuple[PlayerHandProbability, ...]
    prior_sources: Tuple[str, ...]

    @property
    def effective_candidate_count(self) -> float:
        denominator = sum(
            candidate.probability ** 2
            for candidate in self.candidates
        )
        return 0.0 if denominator <= 0.0 else 1.0 / denominator

    @property
    def normalized_entropy(self) -> float:
        if len(self.candidates) <= 1:
            return 0.0
        return -sum(
            candidate.probability * math.log(candidate.probability)
            for candidate in self.candidates
            if candidate.probability > 0.0
        ) / math.log(len(self.candidates))

    def player(self, seat: str) -> PlayerHandProbability:
        return next(item for item in self.players if item.seat == seat)

    def as_dict(self) -> Dict[str, object]:
        return {
            "stage": "initial_public_prior",
            "key": self.key.as_dict(),
            "observer": self.observer,
            "requested_samples": self.requested_samples,
            "accepted_samples": self.accepted_samples,
            "rejected_samples": self.rejected_samples,
            "distinct_candidate_count": len(self.candidates),
            "effective_candidate_count": round(
                self.effective_candidate_count,
                6,
            ),
            "normalized_entropy": round(self.normalized_entropy, 6),
            "prior_sources": list(self.prior_sources),
            "players": [item.as_dict() for item in self.players],
            "candidates": [item.as_dict() for item in self.candidates],
        }


@dataclass(frozen=True)
class PosteriorProbabilisticHandInference(InitialProbabilisticHandInference):
    """Joint posterior after weighting every legal deal by public actions."""

    evidence_revision: int = 0
    action_evidence_count: int = 0
    evidence_sources: Tuple[str, ...] = ()
    confidence: float = 0.0
    retained_probability_mass: float = 1.0
    elapsed_ms: float = 0.0
    timed_out: bool = False

    def as_dict(self) -> Dict[str, object]:
        value = super().as_dict()
        value.update({
            "stage": "action_weighted_joint_posterior",
            "evidence_revision": self.evidence_revision,
            "action_evidence_count": self.action_evidence_count,
            "evidence_sources": list(self.evidence_sources),
            "confidence": round(self.confidence, 6),
            "retained_probability_mass": round(
                self.retained_probability_mass,
                6,
            ),
            "elapsed_ms": round(self.elapsed_ms, 3),
            "timed_out": self.timed_out,
        })
        return value


class ProbabilisticHandInferenceCache:
    """Thread-safe bounded LRU for immutable posterior distributions."""

    def __init__(self, max_entries: int = 128) -> None:
        self.max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[tuple, PosteriorProbabilisticHandInference] = (
            OrderedDict()
        )
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def __deepcopy__(self, memo):
        return self

    def get(self, key: tuple) -> Optional[PosteriorProbabilisticHandInference]:
        with self._lock:
            value = self._entries.get(key)
            if value is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return value

    def put(
        self,
        key: tuple,
        value: PosteriorProbabilisticHandInference,
    ) -> None:
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self.evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {
                "size": len(self._entries),
                "max_entries": self.max_entries,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
            }


class ProbabilisticHandInferenceMixin:
    """Generates a public-only, uniformly dealt prior for later Bayes updates."""

    def _initialize_probabilistic_hand_inference(self) -> None:
        self._probabilistic_hand_inference_cache = ProbabilisticHandInferenceCache(
            max_entries=int(getattr(
                self,
                "PROBABILISTIC_HAND_CACHE_MAX_ENTRIES",
                128,
            )),
        )

    def _probabilistic_inference_cache(self) -> ProbabilisticHandInferenceCache:
        cache = getattr(self, "_probabilistic_hand_inference_cache", None)
        if cache is None:
            self._initialize_probabilistic_hand_inference()
            cache = self._probabilistic_hand_inference_cache
        return cache

    def probabilistic_hand_inference_cache_snapshot(self) -> Dict[str, object]:
        return self._probabilistic_inference_cache().snapshot()

    def clear_probabilistic_hand_inference_cache(self) -> None:
        self._probabilistic_inference_cache().clear()

    def _probabilistic_posterior_cache_key(
        self,
        state,
        player: str,
        tr: dict,
        *,
        sample_count: int,
        max_candidates: int,
        minimum_probability: float,
    ) -> tuple:
        public_key = self._information_set_key(state, player, tr)
        return (
            "probabilistic_posterior_v2",
            public_key.digest,
            int(tr.get("piece_inference_revision", 0)),
            int(sample_count),
            int(max_candidates),
            round(float(minimum_probability), 9),
        )

    @staticmethod
    def _probabilistic_structure_key(hand: Sequence[str]) -> str:
        counts = Counter(str(piece) for piece in hand)

        def group_shape(pieces: Sequence[str]) -> Tuple[int, int, int, int]:
            values = [counts.get(piece, 0) for piece in pieces]
            return tuple(
                sum(1 for value in values if value == count)
                for count in (1, 2, 3, 4)
            )

        middle = group_shape(("3", "4", "5"))
        big = group_shape(("6", "7"))
        return (
            f"shi{counts.get('1', 0)}|kyo{counts.get('2', 0)}|"
            f"mid_s{middle[0]}_p{middle[1]}_t{middle[2]}_q{middle[3]}|"
            f"big_s{big[0]}_p{big[1]}|"
            f"royal{counts.get('8', 0) + counts.get('9', 0)}"
        )

    def _probabilistic_uniform_prediction(
        self,
        state,
        player: str,
        tr: dict,
        rng: random.Random,
    ) -> Optional[PredictionSample]:
        """Deal the public unknown pool into labeled current/hidden slots."""
        opponents = tuple(seat for seat in ALL_SEATS if seat != player)
        hidden_counts = tr.get("hidden_block_counts", {})
        bag = [
            piece
            for piece in PIECES
            for _ in range(int(tr.get("unknown_piece_pool", {}).get(piece, 0)))
        ]
        required_slots = sum(
            len(state.hands[seat]) + int(hidden_counts.get(seat, 0))
            for seat in opponents
        )
        if len(bag) != required_slots:
            return None
        rng.shuffle(bag)

        offset = 0
        hands = []
        hidden_hands = []
        royal_history = []
        sampled_last_block = state.last_block
        for seat in opponents:
            current_size = len(state.hands[seat])
            hidden_size = int(hidden_counts.get(seat, 0))
            current = tuple(sorted(bag[offset:offset + current_size]))
            offset += current_size
            unsorted_hidden = tuple(bag[offset:offset + hidden_size])
            hidden = tuple(sorted(unsorted_hidden))
            offset += hidden_size
            hands.append((seat, current))
            hidden_hands.append((seat, hidden))

            observed_eight = int(
                self._observed_piece_count_for_player(tr, seat, "8")
            )
            observed_nine = int(
                self._observed_piece_count_for_player(tr, seat, "9")
            )
            had_eight = current.count("8") + hidden.count("8") + observed_eight > 0
            had_nine = current.count("9") + hidden.count("9") + observed_nine > 0
            royal_history.append((seat, had_eight and had_nine))

            if state.last_block_player == seat:
                if not unsorted_hidden:
                    return None
                sampled_last_block = unsorted_hidden[-1]

        return PredictionSample(
            opponent_hands=tuple(hands),
            opponent_hidden=tuple(hidden_hands),
            opponent_had_both_kings=tuple(royal_history),
            last_block=sampled_last_block,
        )

    @staticmethod
    def _probabilistic_prior_seed(state, player: str, tr: dict) -> int:
        """Use physical public facts only, excluding soft behavioral estimates."""
        hidden_counts = tr.get("hidden_block_counts", {})
        payload = (
            "probabilistic-hand-prior-v1",
            player,
            tuple(sorted(state.hands[player])),
            tuple(sorted(state.face_down_hidden[player])),
            state.dealer,
            state.turn,
            state.phase,
            state.current_attack,
            state.attacker,
            state.last_block_player,
            state.last_block if state.last_block_player == player else None,
            int(state.king_block_used),
            tuple((seat, len(state.hands[seat])) for seat in ALL_SEATS),
            tuple(
                (seat, int(hidden_counts.get(seat, 0)))
                for seat in ALL_SEATS
                if seat != player
            ),
            tuple(
                (piece, int(tr.get("unknown_piece_pool", {}).get(piece, 0)))
                for piece in PIECES
            ),
        )
        digest = hashlib.sha256(repr(payload).encode("ascii")).hexdigest()
        return int(digest[:16], 16) ^ 0x50B4B1A57

    @staticmethod
    def _probabilistic_label_distribution(
        masses: Dict[str, float],
    ) -> Tuple[LabelProbability, ...]:
        return tuple(
            LabelProbability(label=label, probability=probability)
            for label, probability in sorted(
                masses.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if probability > 0.0
        )

    @staticmethod
    def _probabilistic_count_distribution(
        masses: Dict[int, float],
    ) -> Tuple[CountProbability, ...]:
        return tuple(
            CountProbability(count=count, probability=probability)
            for count, probability in sorted(masses.items())
            if probability > 0.0
        )

    def _probabilistic_original_hand(
        self,
        tr: dict,
        seat: str,
        current: Sequence[str],
        hidden: Sequence[str],
    ) -> Tuple[str, ...]:
        original = list(current) + list(hidden)
        for piece in PIECES:
            original.extend([
                piece
            ] * int(self._observed_piece_count_for_player(tr, seat, piece)))
        return tuple(sorted(original))

    def _probabilistic_hand_profile(
        self,
        state,
        seat: str,
        original_hand: Tuple[str, ...],
    ) -> Tuple[str, str, str]:
        axes = self._classify_hand_axes(
            list(original_hand),
            is_dealer=seat == state.dealer,
        )
        absolute_rank = str(axes.get("rank", "X"))
        relative = self._relative_hand_info(list(original_hand))
        relative_rank = (
            str(relative.get("relative_rank", absolute_rank))
            if relative is not None
            else absolute_rank
        )
        return (
            absolute_rank,
            relative_rank,
            self._probabilistic_structure_key(original_hand),
        )

    def _probabilistic_player_summaries(
        self,
        state,
        player: str,
        tr: dict,
        candidates: Sequence[ProbabilisticDealCandidate],
    ) -> Tuple[PlayerHandProbability, ...]:
        opponents = tuple(seat for seat in ALL_SEATS if seat != player)
        current_counts = {
            seat: {piece: defaultdict(float) for piece in PIECES}
            for seat in opponents
        }
        original_counts = {
            seat: {piece: defaultdict(float) for piece in PIECES}
            for seat in opponents
        }
        hand_masses = {seat: defaultdict(float) for seat in opponents}
        absolute_masses = {seat: defaultdict(float) for seat in opponents}
        relative_masses = {seat: defaultdict(float) for seat in opponents}
        structure_masses = {seat: defaultdict(float) for seat in opponents}
        profile_cache: Dict[Tuple[str, Tuple[str, ...]], Tuple[str, str, str]] = {}

        for candidate in candidates:
            hands = dict(candidate.prediction.opponent_hands)
            hidden_hands = dict(candidate.prediction.opponent_hidden)
            probability = candidate.probability
            for seat in opponents:
                current = tuple(hands[seat])
                hidden = tuple(hidden_hands[seat])
                original = self._probabilistic_original_hand(
                    tr,
                    seat,
                    current,
                    hidden,
                )
                profile_key = (seat, original)
                profile = profile_cache.get(profile_key)
                if profile is None:
                    profile = self._probabilistic_hand_profile(
                        state,
                        seat,
                        original,
                    )
                    profile_cache[profile_key] = profile
                absolute_rank, relative_rank, structure_key = profile
                for piece in PIECES:
                    current_counts[seat][piece][current.count(piece)] += probability
                    original_counts[seat][piece][original.count(piece)] += probability
                hand_masses[seat][(
                    current,
                    hidden,
                    original,
                    absolute_rank,
                    relative_rank,
                    structure_key,
                )] += probability
                absolute_masses[seat][absolute_rank] += probability
                relative_masses[seat][relative_rank] += probability
                structure_masses[seat][structure_key] += probability

        limit = max(1, int(self.PROBABILISTIC_HAND_TOP_CANDIDATES))
        weighted_candidate_confidence = sum(
            candidate.probability * candidate.confidence
            for candidate in candidates
        )
        has_action_evidence = any(candidate.evidence for candidate in candidates)
        summaries = []
        hidden_counts = tr.get("hidden_block_counts", {})
        for seat in opponents:
            top_hands = tuple(
                HandCandidateProbability(
                    current_hand=key[0],
                    hidden_hand=key[1],
                    original_hand=key[2],
                    probability=probability,
                    absolute_rank=key[3],
                    relative_rank=key[4],
                    structure_key=key[5],
                )
                for key, probability in sorted(
                    hand_masses[seat].items(),
                    key=lambda item: (-item[1], item[0]),
                )[:limit]
            )
            summaries.append(PlayerHandProbability(
                seat=seat,
                current_hand_size=len(state.hands[seat]),
                hidden_hand_size=int(hidden_counts.get(seat, 0)),
                pieces=tuple(
                    PieceProbability(
                        piece=piece,
                        current_count_distribution=self._probabilistic_count_distribution(
                            current_counts[seat][piece]
                        ),
                        original_count_distribution=self._probabilistic_count_distribution(
                            original_counts[seat][piece]
                        ),
                    )
                    for piece in PIECES
                ),
                top_hands=top_hands,
                absolute_rank_distribution=self._probabilistic_label_distribution(
                    absolute_masses[seat]
                ),
                relative_rank_distribution=self._probabilistic_label_distribution(
                    relative_masses[seat]
                ),
                structure_distribution=self._probabilistic_label_distribution(
                    structure_masses[seat]
                ),
                confidence=(
                    min(
                        1.0,
                        0.8 * weighted_candidate_confidence
                        + 0.2 * (top_hands[0].probability if top_hands else 0.0),
                    )
                    if has_action_evidence
                    else 0.0
                ),
            ))
        return tuple(summaries)

    @staticmethod
    def _probabilistic_first_attack_likelihood(
        piece: str,
        original_count: int,
        *,
        strategy_broken: bool,
        my_initial_count: Counter,
    ) -> Tuple[float, str]:
        """Return a soft likelihood for the strongest first-attack signal."""
        count = max(0, int(original_count))
        if piece == "1":
            table = (
                (1.0, 0.58, 0.86, 1.0, 1.0)
                if strategy_broken
                else (1.0, 0.10, 0.30, 0.86, 1.0)
            )
            return table[min(count, 4)], "first_attack_shi"
        if piece == "2":
            if count == 1 and int(my_initial_count.get("2", 0)) >= 3:
                return 1.0, "first_attack_damashi_kyosha_confirmed"
            if (
                count == 1
                and int(my_initial_count.get("2", 0)) == 2
                and int(my_initial_count.get("8", 0))
                + int(my_initial_count.get("9", 0)) == 0
            ):
                return 0.72, "first_attack_damashi_kyosha_suspected"
            if strategy_broken:
                return (0.72 if count == 1 else 1.0), "first_attack_kyosha_broken"
            return (0.22 if count == 1 else 1.0), "first_attack_kyosha_repeat"
        if piece in ("3", "4", "5"):
            if strategy_broken:
                return (0.70 if count == 1 else 1.0), "first_attack_middle_broken"
            return (0.20 if count == 1 else 1.0), "first_attack_middle_repeat"
        if piece in ("6", "7"):
            return (0.72 if count == 1 else 1.0), "first_attack_big_one_or_two"
        return 1.0, "first_attack_royal"

    def _probabilistic_candidate_action_evidence(
        self,
        state,
        player: str,
        tr: dict,
        prediction: PredictionSample,
    ) -> Optional[Tuple[float, float, Tuple[str, ...]]]:
        """Score one complete deal from attacks, receives, passes, and signals."""
        if not self._information_set_candidate_consistent(
            state,
            player,
            tr,
            prediction,
        ):
            return None

        current = {
            seat: Counter(hand)
            for seat, hand in prediction.opponent_hands
        }
        hidden = {
            seat: Counter(hand)
            for seat, hand in prediction.opponent_hidden
        }
        my_initial_count = Counter(tr.get("my_init_count", Counter()))
        log_likelihood = 0.0
        evidence: List[str] = []
        confidence_weight = 0.0
        confidence_fit = 0.0

        def add(label: str, likelihood: float, confidence: float) -> None:
            nonlocal log_likelihood, confidence_weight, confidence_fit
            bounded = max(1e-9, min(1.0, float(likelihood)))
            weight = max(0.0, min(1.0, float(confidence)))
            log_likelihood += math.log(bounded)
            confidence_weight += weight
            confidence_fit += weight * bounded
            evidence.append(label)

        for seat in ALL_SEATS:
            if seat == player:
                continue
            model = tr.get("public_hand_models", {}).get(seat, {})
            original = current.get(seat, Counter()) + hidden.get(seat, Counter())
            for piece in PIECES:
                original[piece] += int(
                    self._observed_piece_count_for_player(tr, seat, piece)
                )
            attacks = model.get("attacks", Counter())
            blocks = model.get("blocks", Counter())

            first_attack = model.get("first_attack")
            if first_attack is not None:
                first_piece = str(first_attack)
                later_switched_from_shi = (
                    first_piece == "1"
                    and int(model.get("attack_count", 0)) >= 2
                    and int(attacks.get("1", 0)) == 1
                )
                likelihood, label = self._probabilistic_first_attack_likelihood(
                    first_piece,
                    int(original.get(first_piece, 0)),
                    strategy_broken=(
                        bool(model.get("strategy_broken"))
                        or later_switched_from_shi
                    ),
                    my_initial_count=my_initial_count,
                )
                if later_switched_from_shi:
                    label = "first_attack_damashi_shi_suspected"
                if first_piece in ("8", "9"):
                    other_royal = "8" if first_piece == "9" else "9"
                    likelihood = 1.0 if original.get(other_royal, 0) > 0 else 0.14
                add(f"{seat}:{label}", likelihood, 0.90)

                if (
                    first_piece in ("6", "7")
                    and int(attacks.get("2", 0)) > 0
                ):
                    add(
                        f"{seat}:big_kyosha_kyosha_shape",
                        1.0 if int(original.get("2", 0)) >= 2 else 0.34,
                        0.74,
                    )

                observed_first = self._observed_piece_count_for_player(
                    tr,
                    seat,
                    first_piece,
                )
                unused_first = max(
                    0,
                    int(original.get(first_piece, 0)) - observed_first,
                )
                reaction = model.get("partner_first_strategy_reaction") or {}
                if bool(model.get("strategy_broken")):
                    add(
                        f"{seat}:first_strategy_broken",
                        1.0 if unused_first == 0 else 0.58,
                        0.78,
                    )
                elif reaction.get("status") == "accepted":
                    reaction_label = str(reaction.get("reason", "accepted"))
                    add(
                        f"{seat}:{reaction_label}",
                        1.0 if unused_first > 0 else 0.78,
                        0.64,
                    )

            for piece in PIECES:
                attack_count = int(attacks.get(piece, 0))
                block_count = int(blocks.get(piece, 0))
                if attack_count >= 2:
                    add(
                        f"{seat}:repeat_attack_{piece}_{attack_count}",
                        1.0,
                        min(1.0, 0.55 + 0.12 * attack_count),
                    )
                if block_count > 0:
                    add(
                        f"{seat}:receive_{piece}_{block_count}",
                        1.0,
                        min(1.0, 0.48 + 0.10 * block_count),
                    )
                if block_count > 0 and attack_count > 0 and piece in ("1", "2", "3", "4", "5"):
                    add(
                        f"{seat}:return_signal_{piece}",
                        1.0,
                        0.72,
                    )

                pass_events = tr.get("piece_pass_evidence", {}).get(seat, {}).get(piece, [])
                if pass_events:
                    pass_likelihood = self._pass_evidence_likelihood(
                        tr,
                        seat,
                        piece,
                        int(original.get(piece, 0)),
                    )
                    add(
                        f"{seat}:pass_{piece}_{len(pass_events)}",
                        pass_likelihood ** 0.85,
                        min(0.94, 0.58 + 0.08 * len(pass_events)),
                    )

            rank = str(model.get("estimated_rank", "D"))
            rank_confidence = max(
                0.0,
                min(1.0, float(model.get("rank_confidence", 0.0))),
            )
            if rank_confidence > 0.0:
                original_hand = tuple(sorted(original.elements()))
                candidate_rank, _relative_rank, _structure = (
                    self._probabilistic_hand_profile(state, seat, original_hand)
                )
                order = {label: index for index, label in enumerate(
                    ("S", "A", "B", "C", "D", "E", "F", "X")
                )}
                distance = abs(order.get(candidate_rank, 7) - order.get(rank, 7))
                add(
                    f"{seat}:public_rank_{rank}",
                    math.exp(-0.30 * rank_confidence * distance),
                    0.45 * rank_confidence,
                )

        coverage = 1.0 - math.exp(-len(evidence) / 5.0)
        mean_fit = (
            confidence_fit / confidence_weight
            if confidence_weight > 0.0
            else 0.0
        )
        confidence = max(0.0, min(1.0, coverage * mean_fit))
        return log_likelihood, confidence, tuple(evidence)

    def _posterior_probabilistic_hand_inference(
        self,
        state,
        player: str,
        *,
        sample_count: Optional[int] = None,
        activate: bool = True,
        max_seconds: Optional[float] = None,
        max_candidates: Optional[int] = None,
        minimum_probability: Optional[float] = None,
        use_cache: bool = True,
    ) -> PosteriorProbabilisticHandInference:
        """Build and optionally retain the action-weighted joint posterior."""
        started = time.perf_counter()
        requested = int(
            self.PROBABILISTIC_HAND_INITIAL_SAMPLE_COUNT
            if sample_count is None
            else sample_count
        )
        requested = max(
            1,
            min(requested, int(self.PROBABILISTIC_HAND_INITIAL_MAX_SAMPLES)),
        )
        retained_limit = max(
            1,
            int(
                getattr(self, "PROBABILISTIC_HAND_MAX_RETAINED_CANDIDATES", 128)
                if max_candidates is None
                else max_candidates
            ),
        )
        retained_floor = max(
            0.0,
            min(
                1.0,
                float(
                    getattr(self, "PROBABILISTIC_HAND_MIN_CANDIDATE_PROBABILITY", 0.0)
                    if minimum_probability is None
                    else minimum_probability
                ),
            ),
        )
        self._ensure_trackers(state)
        tr = self._track.get(id(state))
        if tr is None:
            raise ValueError("public tracker is unavailable")
        cache_key = self._probabilistic_posterior_cache_key(
            state,
            player,
            tr,
            sample_count=requested,
            max_candidates=retained_limit,
            minimum_probability=retained_floor,
        )
        cache = self._probabilistic_inference_cache()
        cached = cache.get(cache_key) if use_cache else None
        if cached is not None:
            if activate:
                tr["probabilistic_hand_inference_active"] = True
                tr["probabilistic_hand_inference_samples"] = requested
                tr["probabilistic_hand_inference"] = cached
                tr["probabilistic_hand_inference_error"] = None
            return cached

        deadline = (
            None
            if max_seconds is None or float(max_seconds) <= 0.0
            else started + float(max_seconds)
        )
        prior = self._initial_probabilistic_hand_inference(
            state,
            player,
            sample_count=requested,
            max_seconds=(
                None
                if deadline is None
                else max(0.001, deadline - time.perf_counter())
            ),
        )

        scored = []
        rejected = prior.rejected_samples
        evidence_sources = set()
        timed_out = prior.accepted_samples < prior.requested_samples
        for candidate in prior.candidates:
            if deadline is not None and time.perf_counter() >= deadline and scored:
                timed_out = True
                break
            result = self._probabilistic_candidate_action_evidence(
                state,
                player,
                tr,
                candidate.prediction,
            )
            if result is None:
                rejected += candidate.observations
                continue
            log_likelihood, confidence, evidence = result
            evidence_sources.update(evidence)
            scored.append((candidate, log_likelihood, confidence, evidence))
        if not scored:
            raise ValueError("probabilistic posterior has no compatible candidate deals")

        maximum_log = max(item[1] for item in scored)
        weighted = []
        total_weight = 0.0
        for candidate, log_likelihood, confidence, evidence in scored:
            relative = math.exp(max(-60.0, min(0.0, log_likelihood - maximum_log)))
            weight = candidate.prior_probability * relative
            total_weight += weight
            weighted.append((candidate, relative, weight, confidence, evidence))
        if total_weight <= 0.0:
            raise ValueError("probabilistic posterior has zero total weight")

        ranked_weighted = sorted(
            weighted,
            key=lambda item: (-item[2], repr(item[0].prediction)),
        )
        minimum_retained = min(
            retained_limit,
            max(1, int(getattr(
                self,
                "PROBABILISTIC_HAND_MIN_RETAINED_CANDIDATES",
                12,
            ))),
        )
        retained_weighted = []
        for item in ranked_weighted:
            normalized_probability = item[2] / total_weight
            if (
                len(retained_weighted) < minimum_retained
                or normalized_probability >= retained_floor
            ) and len(retained_weighted) < retained_limit:
                retained_weighted.append(item)
        retained_weight = sum(item[2] for item in retained_weighted)
        if retained_weight <= 0.0:
            raise ValueError("probabilistic posterior retained zero weight")
        retained_probability_mass = retained_weight / total_weight
        candidates = tuple(
            ProbabilisticDealCandidate(
                prediction=candidate.prediction,
                observations=candidate.observations,
                prior_probability=candidate.prior_probability,
                relative_likelihood=relative,
                posterior_probability=weight / retained_weight,
                confidence=confidence,
                evidence=evidence,
            )
            for candidate, relative, weight, confidence, evidence in retained_weighted
        )
        players = self._probabilistic_player_summaries(
            state,
            player,
            tr,
            candidates,
        )
        if len(candidates) <= 1:
            entropy = 0.0
        else:
            entropy = -sum(
                candidate.probability * math.log(candidate.probability)
                for candidate in candidates
                if candidate.probability > 0.0
            ) / math.log(len(candidates))
        evidence_confidence = sum(
            candidate.probability * candidate.confidence
            for candidate in candidates
        )
        confidence = max(
            0.0,
            min(1.0, evidence_confidence * (0.75 + 0.25 * (1.0 - entropy))),
        )
        posterior = PosteriorProbabilisticHandInference(
            key=prior.key,
            observer=prior.observer,
            requested_samples=prior.requested_samples,
            accepted_samples=sum(candidate.observations for candidate in candidates),
            rejected_samples=rejected,
            candidates=candidates,
            players=players,
            prior_sources=prior.prior_sources,
            evidence_revision=int(tr.get("piece_inference_revision", 0)),
            action_evidence_count=max(
                (len(candidate.evidence) for candidate in candidates),
                default=0,
            ),
            evidence_sources=tuple(sorted(evidence_sources)),
            confidence=confidence,
            retained_probability_mass=retained_probability_mass,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            timed_out=timed_out,
        )
        if use_cache:
            cache.put(cache_key, posterior)
        if activate:
            tr["probabilistic_hand_inference_active"] = True
            tr["probabilistic_hand_inference_samples"] = prior.requested_samples
            tr["probabilistic_hand_inference"] = posterior
            tr["probabilistic_hand_inference_error"] = None
        return posterior

    def _refresh_probabilistic_hand_inference_after_public_action(
        self,
        state,
        action_player: str,
        _action,
        tr: dict,
    ) -> None:
        """Refresh an activated posterior after every subsequent public action."""
        if not bool(getattr(self, "PROBABILISTIC_HAND_AUTO_REFRESH", True)):
            return
        if not bool(tr.get("probabilistic_hand_inference_active")):
            return
        # Our own visible action does not redistribute the unknown opponent pool.
        if str(action_player) == str(self.me):
            return
        tr["probabilistic_hand_changed_players"] = (str(action_player),)
        try:
            self._posterior_probabilistic_hand_inference(
                state,
                str(self.me),
                sample_count=int(tr.get(
                    "probabilistic_hand_inference_samples",
                    self.PROBABILISTIC_HAND_INITIAL_SAMPLE_COUNT,
                )),
                activate=True,
                max_seconds=float(getattr(
                    self,
                    "PROBABILISTIC_HAND_REFRESH_MAX_SECONDS",
                    0.04,
                )),
            )
        except ValueError as exc:
            tr["probabilistic_hand_inference_error"] = str(exc)

    def _initial_probabilistic_hand_inference(
        self,
        state,
        player: str,
        *,
        sample_count: Optional[int] = None,
        max_seconds: Optional[float] = None,
    ) -> InitialProbabilisticHandInference:
        """Create the stage-two legal-deal prior for one observer."""
        self._ensure_trackers(state)
        tr = self._track.get(id(state))
        if tr is None:
            raise ValueError("public tracker is unavailable")
        requested = int(
            self.PROBABILISTIC_HAND_INITIAL_SAMPLE_COUNT
            if sample_count is None
            else sample_count
        )
        requested = max(
            1,
            min(requested, int(self.PROBABILISTIC_HAND_INITIAL_MAX_SAMPLES)),
        )
        key = self._information_set_key(state, player, tr)
        seed = self._probabilistic_prior_seed(state, player, tr)
        rng = random.Random(seed)
        grouped: Counter[PredictionSample] = Counter()
        rejected = 0
        attempts = 0
        accepted_count = 0
        maximum_attempts = max(requested * 4, 16)
        deadline = (
            None
            if max_seconds is None or float(max_seconds) <= 0.0
            else time.perf_counter() + float(max_seconds)
        )
        while accepted_count < requested and attempts < maximum_attempts:
            if deadline is not None and time.perf_counter() >= deadline and accepted_count:
                break
            attempts += 1
            prediction = self._probabilistic_uniform_prediction(
                state,
                player,
                tr,
                rng,
            )
            if prediction is None or not self._information_set_candidate_consistent(
                state,
                player,
                tr,
                prediction,
            ):
                rejected += 1
                continue
            grouped[prediction] += 1
            accepted_count += 1
        accepted = accepted_count
        if accepted <= 0:
            raise ValueError("probabilistic inference has no compatible candidate deals")

        candidates = tuple(
            ProbabilisticDealCandidate(
                prediction=prediction,
                observations=observations,
                prior_probability=observations / accepted,
            )
            for prediction, observations in sorted(
                grouped.items(),
                key=lambda item: (-item[1], repr(item[0])),
            )
        )
        players = self._probabilistic_player_summaries(
            state,
            player,
            tr,
            candidates,
        )
        return InitialProbabilisticHandInference(
            key=key,
            observer=player,
            requested_samples=requested,
            accepted_samples=accepted,
            rejected_samples=rejected,
            candidates=candidates,
            players=players,
            prior_sources=(
                "uniform_public_unknown_pool_deals",
                "natural_multiset_structure_frequency",
                "relative_hand_rank_table_classification",
            ),
        )


__all__ = [
    "CountProbability",
    "HandCandidateProbability",
    "InitialProbabilisticHandInference",
    "LabelProbability",
    "PosteriorProbabilisticHandInference",
    "PieceProbability",
    "PlayerHandProbability",
    "ProbabilisticDealCandidate",
    "ProbabilisticHandInferenceCache",
    "ProbabilisticHandInferenceMixin",
]

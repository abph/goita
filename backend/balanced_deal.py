"""Absolute-rank filtering for deals without extreme hands."""

from __future__ import annotations

from functools import lru_cache
from typing import Mapping, Sequence

from goita_ai2.constants import ALL_SEATS, PIECE_TOTALS
from goita_ai2.current_ai.forced_plans import ForcedPlansMixin
from goita_ai2.current_ai.hand_evaluation import HandEvaluationMixin


BALANCED_ABSOLUTE_RANKS = frozenset({"B", "C", "D", "E", "F"})


class _AbsoluteRankEvaluator(HandEvaluationMixin, ForcedPlansMixin):
    @staticmethod
    def _piece_total(piece: str) -> int:
        return int(PIECE_TOTALS.get(str(piece), 0))


_EVALUATOR = _AbsoluteRankEvaluator()


@lru_cache(maxsize=None)
def _absolute_hand_rank_cached(hand: tuple[str, ...]) -> str:
    axes = _EVALUATOR._classify_hand_axes(list(hand), is_dealer=False)
    return str(axes["rank"])


def absolute_hand_rank(hand: Sequence[str]) -> str:
    """Return the same dealer-independent absolute rank used by the hand list."""
    normalized = tuple(sorted((str(piece) for piece in hand), key=int))
    return _absolute_hand_rank_cached(normalized)


def is_balanced_rank_deal(hands: Mapping[str, Sequence[str]]) -> bool:
    """Accept a deal only when every hand has an absolute rank from B through F."""
    return bool(hands) and all(
        absolute_hand_rank(hand) in BALANCED_ABSOLUTE_RANKS
        for hand in hands.values()
    )


def dealer_unanswerable_attack_count(
    hands: Mapping[str, Sequence[str]],
    dealer: str,
) -> int:
    """Count dealer pieces that neither opponent can receive.

    The dealer can force an uninterrupted opening when at least three of the
    first attacks cannot be received.  The partner may simply pass, leaving
    the dealer to use the remaining two pieces for the finishing block and
    attack.

    Shi and kyosha can only be received by the same piece.  Uma through kaku
    and hisha can also be received by ou/gyoku, so those attacks are counted
    only when the opposing pair owns neither royal.
    """
    dealer = str(dealer)
    if dealer not in ALL_SEATS or any(seat not in hands for seat in ALL_SEATS):
        return 0

    dealer_index = ALL_SEATS.index(dealer)
    opponents = (
        ALL_SEATS[(dealer_index + 1) % len(ALL_SEATS)],
        ALL_SEATS[(dealer_index + 3) % len(ALL_SEATS)],
    )
    opponent_pieces = {
        str(piece)
        for opponent in opponents
        for piece in hands[opponent]
    }
    opponents_have_royal = bool(opponent_pieces.intersection({"8", "9"}))

    unanswerable = 0
    for raw_piece in hands[dealer]:
        piece = str(raw_piece)
        if piece not in {"1", "2", "3", "4", "5", "6", "7"}:
            continue
        if piece in opponent_pieces:
            continue
        if piece not in {"1", "2"} and opponents_have_royal:
            continue
        unanswerable += 1
    return unanswerable


def has_wardna_forced_opening(
    hands: Mapping[str, Sequence[str]],
    dealer: str,
) -> bool:
    """Return whether the dealer has three unstoppable opening attacks."""
    return dealer_unanswerable_attack_count(hands, dealer) >= 3


def is_wardna_balanced_deal(
    hands: Mapping[str, Sequence[str]],
    dealer: str,
) -> bool:
    """Apply the rank filter and remove Wardna-style decided openings."""
    return is_balanced_rank_deal(hands) and not has_wardna_forced_opening(
        hands,
        dealer,
    )

"""Absolute-rank filtering for deals without extreme hands."""

from __future__ import annotations

from functools import lru_cache
from typing import Mapping, Sequence

from goita_ai2.constants import PIECE_TOTALS
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

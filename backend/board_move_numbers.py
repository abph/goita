"""Public attack-order labels, independent of hidden pieces or log truncation."""
from typing import Any, Dict, List, Optional, Sequence


def board_attack_numbers(moves: Sequence[Sequence[Any]]) -> Dict[str, List[Optional[int]]]:
    numbers = {seat: [None] * 4 for seat in "ABCD"}
    counts = {seat: 0 for seat in "ABCD"}
    seats = {str(index): seat for index, seat in enumerate("ABCD")}
    sequence = 0
    for row in moves:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        seat = seats.get(str(row[0]))
        if seat is None or not row[2] or str(row[1]).lower() in {"パス", "pass"}:
            continue
        if counts[seat] >= 4:
            continue
        sequence += 1
        numbers[seat][counts[seat]] = sequence
        counts[seat] += 1
    return numbers

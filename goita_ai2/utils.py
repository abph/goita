# goita_ai2/utils.py
import random
from typing import Dict, List

def create_random_hands() -> Dict[str, List[str]]:
    deck = (
        ["9"] * 1 +
        ["8"] * 1 +
        ["7"] * 2 +
        ["6"] * 2 +
        ["5"] * 4 +
        ["4"] * 4 +
        ["3"] * 4 +
        ["2"] * 4 +
        ["1"] * 10
    )
    assert len(deck) == 32
    random.shuffle(deck)
    return {
        "A": deck[0:8],
        "B": deck[8:16],
        "C": deck[16:24],
        "D": deck[24:32],
    }

"""Build privacy-safe neural policy training data from the private kifu archive.

The exported records contain only information available to the acting player.
Opponent hands and opponent face-down piece identities are used only while
reconstructing the game and never leave this conversion boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

from goita_ai2.constants import ALL_SEATS, PIECE_TOTALS
from goita_ai2.kifu_validation import iter_kifu_decisions


PIECES = tuple(sorted(PIECE_TOTALS))
DEFAULT_OUTPUT = Path("private_data/neural_training/decisions.jsonl")
DEFAULT_SUMMARY = Path("private_data/neural_training/summary.json")


def _stable_token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _split_name(match_id: str) -> str:
    bucket = int(hashlib.sha256(match_id.encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "validation"
    return "train"


def _relative_seat(actor: str, other: Optional[str]) -> str:
    if other not in ALL_SEATS:
        return "none"
    offset = (ALL_SEATS.index(str(other)) - ALL_SEATS.index(actor)) % 4
    return ("self", "next", "partner", "previous")[offset]


def _piece_counts(pieces: Iterable[str]) -> Dict[str, int]:
    counts = Counter(str(piece) for piece in pieces)
    return {piece: int(counts[piece]) for piece in PIECES}


def _action_payload(raw_action: Sequence[Any]) -> Dict[str, Optional[str]]:
    return {
        "type": str(raw_action[0]),
        "block": None if raw_action[1] is None else str(raw_action[1]),
        "attack": None if raw_action[2] is None else str(raw_action[2]),
    }


def _public_history(
    history: Sequence[Mapping[str, Any]],
    actor: str,
) -> list[Dict[str, Any]]:
    public: list[Dict[str, Any]] = []
    for item in history:
        event_actor = str(item["player"])
        raw_action = list(item["action"])
        action_type = str(raw_action[0])
        block = raw_action[1]
        block_is_hidden = action_type == "attack_after_block"
        block_known = not block_is_hidden or event_actor == actor
        public.append({
            "actor": _relative_seat(actor, event_actor),
            "type": action_type,
            "block": str(block) if block is not None and block_known else None,
            "block_known": bool(block is not None and block_known),
            "attack": None if raw_action[2] is None else str(raw_action[2]),
        })
    return public


def _public_counters(
    history: Sequence[Mapping[str, Any]],
    actor: str,
) -> Dict[str, Any]:
    attacks = Counter()
    receives = Counter()
    passes = Counter()
    hidden = Counter()
    played = Counter()
    for item in history:
        relation = _relative_seat(actor, str(item["player"]))
        action = list(item["action"])
        action_type = str(action[0])
        if action_type == "pass":
            passes[relation] += 1
        if action_type == "attack_after_block":
            hidden[relation] += 1
            played[relation] += 2
        elif action_type in {"receive", "attack"}:
            played[relation] += 1
        if action_type == "receive" and action[1] is not None:
            receives[str(action[1])] += 1
        if action[2] is not None:
            attacks[str(action[2])] += 1
    relations = ("self", "next", "partner", "previous")
    return {
        "attack_piece_counts": {piece: int(attacks[piece]) for piece in PIECES},
        "receive_piece_counts": {piece: int(receives[piece]) for piece in PIECES},
        "pass_counts": {relation: int(passes[relation]) for relation in relations},
        "face_down_counts": {relation: int(hidden[relation]) for relation in relations},
        "remaining_hand_sizes": {
            relation: max(0, 8 - int(played[relation])) for relation in relations
        },
    }


def _round_key(case: Mapping[str, Any]) -> tuple[str, int]:
    source = dict(case.get("source", {}) or {})
    return str(source.get("match_id", "")), int(source.get("round_index", 0))


def _excluded_five_shi(case: Mapping[str, Any]) -> bool:
    hands = dict(case.get("initial_hands", {}) or {})
    return any(list(hand).count("1") >= 5 for hand in hands.values())


def _match_players(archive: Mapping[str, Any]) -> Dict[str, Dict[str, str]]:
    result: Dict[str, Dict[str, str]] = {}
    for match in archive.get("matches", []):
        match_id = str(match.get("id", ""))
        raw_players = dict(match.get("players", {}) or {})
        result[match_id] = {
            seat: str(raw_players.get(f"p{index}", ""))
            for index, seat in enumerate(ALL_SEATS)
        }
    return result


def training_record(
    case: Mapping[str, Any],
    *,
    player_name: str,
    target_player: str,
) -> Dict[str, Any]:
    """Convert one reconstructed decision without retaining hidden information."""
    actor = str(case["player"])
    match_id, round_index = _round_key(case)
    position = dict(case.get("position", {}) or {})
    initial_hands = dict(case.get("initial_hands", {}) or {})
    initial_hand = list(initial_hands.get(actor, []))
    history = list(case.get("history", []) or [])
    legal_actions = [list(item) for item in case.get("legal_actions", [])]
    selected = list(case["actual_action"])
    if selected not in legal_actions:
        raise ValueError(f"recorded action is not legal in {case.get('id')}")

    score = dict(case.get("initial_score", {}) or {})
    own_team = "AC" if actor in {"A", "C"} else "BD"
    enemy_team = "BD" if own_team == "AC" else "AC"
    public_history = _public_history(history, actor)
    return {
        "schema_version": 1,
        "profile": "強化中AI2",
        "decision_id": _stable_token(str(case.get("id", ""))),
        "match_group": _stable_token(match_id),
        "round_index": round_index,
        "split": _split_name(match_id),
        "is_target_player": player_name == target_player,
        "category": str(case.get("category", "unknown")),
        "state": {
            "phase": str(position.get("phase", "")),
            "dealer": _relative_seat(actor, str(case.get("dealer", ""))),
            "current_attack": position.get("current_attack"),
            "attacker": _relative_seat(actor, position.get("attacker")),
            "own_score": int(score.get(own_team, 0)),
            "enemy_score": int(score.get(enemy_team, 0)),
            "own_initial_hand": _piece_counts(initial_hand),
            "own_initial_had_both_royals": "8" in initial_hand and "9" in initial_hand,
            "own_hand": _piece_counts(position.get("hand", [])),
            "own_hand_size": int(position.get("hand_size", 0)),
            "public_counters": _public_counters(history, actor),
            "history": public_history,
        },
        "legal_actions": [_action_payload(item) for item in legal_actions],
        "selected_action_index": legal_actions.index(selected),
    }


def iter_training_records(
    kifu_path: Path,
    *,
    target_player: str = "1222",
) -> Iterator[Dict[str, Any]]:
    archive = json.loads(kifu_path.read_text(encoding="utf-8"))
    players = _match_players(archive)
    excluded_rounds: set[tuple[str, int]] = set()
    for case in iter_kifu_decisions(kifu_path):
        round_key = _round_key(case)
        if round_key in excluded_rounds or _excluded_five_shi(case):
            excluded_rounds.add(round_key)
            continue
        if len(case.get("legal_actions", [])) <= 1:
            continue
        match_id, _round_index = round_key
        actor = str(case["player"])
        yield training_record(
            case,
            player_name=players.get(match_id, {}).get(actor, ""),
            target_player=target_player,
        )


def build_training_dataset(
    kifu_path: Path,
    output_path: Path,
    summary_path: Path,
    *,
    target_player: str = "1222",
) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_counts = Counter()
    category_counts = Counter()
    target_count = 0
    record_count = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in iter_training_records(kifu_path, target_player=target_player):
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            record_count += 1
            split_counts[str(record["split"])] += 1
            category_counts[str(record["category"])] += 1
            target_count += int(bool(record["is_target_player"]))
    summary = {
        "schema_version": 1,
        "profile": "強化中AI2",
        "source": str(kifu_path),
        "output": str(output_path),
        "records": record_count,
        "target_player": target_player,
        "target_player_records": target_count,
        "split_counts": dict(sorted(split_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "privacy": {
            "opponent_hands_exported": False,
            "opponent_hidden_piece_identities_exported": False,
            "player_names_exported": False,
            "raw_match_ids_exported": False,
        },
        "filters": {
            "rounds_with_any_five_or_more_shi": "excluded",
            "forced_decisions": "excluded",
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kifu", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--target-player", default="1222")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = build_training_dataset(
        args.kifu,
        args.output,
        args.summary,
        target_player=str(args.target_player),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

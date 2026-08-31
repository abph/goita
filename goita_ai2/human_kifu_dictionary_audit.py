"""Build an anonymous, offline audit of human response patterns from kifu.

The raw archive is private input. The generated report contains only aggregate
tactical features and never embeds names, timestamps, match IDs, hands, or move
histories. It does not affect the live AI.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from goita_ai2.constants import ALL_SEATS, PIECE_TOTALS, POINTS
from goita_ai2.kifu_validation import (
    KANJI_TO_PIECE,
    iter_kifu_decisions,
    replay_validation_case,
    write_json,
)
from goita_ai2.current_ai.human_response_dictionary import (
    build_human_response_dictionary,
)


Action = Tuple[str, Optional[str], Optional[str]]
DEFAULT_RAW_KIFU_PATH = Path("frontend/kifu_data_raw.json")
DEFAULT_AUDIT_PATH = Path("results/kifu_audit/human_response_patterns.json")
DEFAULT_DICTIONARY_PATH = Path(
    "goita_ai2/current_ai/data/human-response-patterns.json"
)


def _digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _team(player: str) -> str:
    return "AC" if player in ("A", "C") else "BD"


def _relation(player: str, other: Optional[str]) -> str:
    if other is None:
        return "none"
    if other == player:
        return "self"
    return "ally" if _team(player) == _team(other) else "enemy"


def _hand_stage(size: int) -> str:
    if size <= 2:
        return "reach"
    if size <= 4:
        return "late"
    if size <= 6:
        return "middle"
    return "early"


def _piece_family(piece: Optional[str]) -> str:
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


def _count_class(value: int) -> str:
    if value <= 0:
        return "none"
    if value == 1:
        return "one"
    return "multiple"


def _score_pressure(state, player: str) -> str:
    own_near = int(state.team_score.get(_team(player), 0)) >= 120
    enemy = "BD" if _team(player) == "AC" else "AC"
    enemy_near = int(state.team_score.get(enemy, 0)) >= 120
    if own_near and enemy_near:
        return "both_near"
    if own_near:
        return "own_near"
    if enemy_near:
        return "enemy_near"
    return "normal"


def _public_history_summary(
    history: Iterable[Mapping[str, object]],
) -> Dict[str, Counter]:
    public_seen: Counter = Counter()
    attack_counts: Counter = Counter()
    hidden_counts: Counter = Counter()
    for item in history:
        player = str(item.get("player", ""))
        raw_action = list(item.get("action", ()))
        if len(raw_action) != 3:
            continue
        action_type, block, attack = raw_action
        if action_type == "receive" and block is not None:
            public_seen[str(block)] += 1
        if action_type in ("attack", "attack_after_block") and attack is not None:
            public_seen[str(attack)] += 1
            attack_counts[player] += 1
        if action_type == "attack_after_block":
            hidden_counts[player] += 1
    return {
        "public_seen": public_seen,
        "attack_counts": attack_counts,
        "hidden_counts": hidden_counts,
    }


def _followup_shape(
    hand: Iterable[str],
    public_seen: Mapping[str, int],
    current_attack: Optional[str],
) -> Tuple[str, str]:
    counts = Counter(str(piece) for piece in hand)
    pair_followups = 0
    scarce_followups = 0
    fourth_followups = 0
    distinct_reentry = 0
    for piece, count in counts.items():
        if count <= 0 or piece in ("8", "9"):
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
        if outside == 0:
            fourth_followups += 1

    if fourth_followups:
        strength = "fourth"
    elif scarce_followups:
        strength = "scarce"
    elif pair_followups:
        strength = "pair"
    else:
        strength = "open"
    if distinct_reentry <= 0:
        reentry = "closed"
    elif distinct_reentry == 1:
        reentry = "narrow"
    else:
        reentry = "wide"
    return strength, reentry


def _response_features(case: Mapping[str, object], state) -> Dict[str, object]:
    player = str(case["player"])
    current_attack = str(state.current_attack or "none")
    hand = list(state.hands[player])
    history_summary = _public_history_summary(case.get("history", ()))
    public_seen = history_summary["public_seen"]
    attack_counts = history_summary["attack_counts"]
    same_count = hand.count(current_attack)
    hand_after_same = list(hand)
    if same_count:
        hand_after_same.remove(current_attack)
    followup_strength, reentry_width = _followup_shape(
        hand_after_same,
        public_seen,
        current_attack,
    )
    legal = state.legal_actions(player)
    next_receiver = state.next_player(player)
    attacker_count = int(attack_counts.get(str(state.attacker), 0))
    return {
        "version": 1,
        "granularity": "human_response_tactical",
        "attacker_relation": _relation(player, state.attacker),
        "attack_piece": current_attack,
        "attack_family": _piece_family(current_attack),
        "attack_stage": "first" if attacker_count <= 1 else "later",
        "hand_stage": _hand_stage(len(hand)),
        "next_receiver_stage": _hand_stage(len(state.hands[next_receiver])),
        "same_piece": _count_class(same_count),
        "royal_receive": any(
            action[0] == "receive" and action[1] in ("8", "9")
            for action in legal
        ),
        "followup_strength": followup_strength,
        "reentry_width": reentry_width,
        "shi_context": (
            "not_shi"
            if current_attack != "1"
            else f"from_{_relation(player, state.attacker)}"
        ),
        "score_pressure": _score_pressure(state, player),
    }


def _response_action_label(action: Action, current_attack: Optional[str]) -> str:
    if action[0] == "pass":
        return "pass"
    if action[0] != "receive":
        return "other"
    if action[1] == current_attack:
        return "receive_same"
    if action[1] in ("8", "9"):
        return "receive_royal"
    return "receive_other"


def _followup_label(
    hand_after_receive: Iterable[str],
    followup_piece: Optional[str],
    public_seen: Mapping[str, int],
) -> str:
    if followup_piece is None:
        return "none"
    piece = str(followup_piece)
    counts = Counter(str(item) for item in hand_after_receive)
    if counts[piece] <= 0:
        return "none"
    family = _piece_family(piece)
    if family == "royal":
        return "royal"
    outside = max(
        0,
        int(PIECE_TOTALS[piece])
        - int(public_seen.get(piece, 0))
        - int(counts[piece]),
    )
    if outside == 0:
        return f"fourth_{family}"
    if counts[piece] >= 2:
        return f"{family}_pair"
    if outside <= 1:
        return f"scarce_{family}"
    return f"{family}_single"


def _round_inventory(archive: Mapping[str, object]) -> Dict[str, object]:
    excluded_five_shi = set()
    invalid_rounds = set()
    all_rounds = set()
    named_players = set()
    for match in archive.get("matches", []):
        match_id = str(match.get("id", "unknown"))
        for value in dict(match.get("players", {}) or {}).values():
            name = str(value or "").strip()
            if name:
                named_players.add(name)
        for round_item in match.get("rounds", []):
            round_index = int(round_item.get("round_index", 0))
            key = (match_id, round_index)
            all_rounds.add(key)
            hands = dict(round_item.get("hand", {}) or {})
            try:
                hand_values = [str(hands[f"p{index}"]) for index in range(4)]
                digits = [KANJI_TO_PIECE[piece] for hand in hand_values for piece in hand]
            except (KeyError, TypeError):
                invalid_rounds.add(key)
                continue
            if any(len(hand) != 8 for hand in hand_values):
                invalid_rounds.add(key)
                continue
            if Counter(digits) != Counter(PIECE_TOTALS):
                invalid_rounds.add(key)
                continue
            if any(hand.count("し") >= 5 for hand in hand_values):
                excluded_five_shi.add(key)
    return {
        "all_rounds": all_rounds,
        "invalid_rounds": invalid_rounds,
        "five_shi_rounds": excluded_five_shi,
        "distinct_named_players": len(named_players),
    }


def audit_human_kifu_archive(
    kifu_path: Path,
    *,
    include_five_shi: bool = False,
    min_observations: int = 15,
    min_matches: int = 8,
    min_dominance: float = 0.70,
    max_patterns: int = 100,
) -> Dict[str, object]:
    """Replay a private archive and return a privacy-safe aggregate report."""
    archive = json.loads(Path(kifu_path).read_text(encoding="utf-8"))
    if not isinstance(archive, dict) or not isinstance(archive.get("matches"), list):
        raise ValueError("kifu archive must contain a matches list")
    inventory = _round_inventory(archive)
    excluded_rounds = set(inventory["invalid_rounds"])
    if not include_five_shi:
        excluded_rounds.update(inventory["five_shi_rounds"])

    patterns: Dict[str, Dict[str, object]] = {}
    decision_counts: Counter = Counter()
    response_counts: Counter = Counter()
    included_rounds = set()
    included_matches = set()
    pending: Optional[Dict[str, object]] = None

    def record_pending(followup_piece: Optional[str] = None) -> None:
        nonlocal pending
        if pending is None:
            return
        pattern_id = str(pending["pattern_id"])
        entry = patterns.setdefault(
            pattern_id,
            {
                "features": copy.deepcopy(pending["features"]),
                "observations": 0,
                "matches": set(),
                "rounds": set(),
                "action_counts": Counter(),
                "followup_counts": Counter(),
                "route_counts": Counter(),
            },
        )
        action_label = str(pending["action_label"])
        followup = _followup_label(
            pending["hand_after_receive"],
            followup_piece,
            pending["public_seen"],
        )
        entry["observations"] = int(entry["observations"]) + 1
        entry["matches"].add(pending["match_id"])
        entry["rounds"].add(pending["round_key"])
        entry["action_counts"][action_label] += 1
        entry["followup_counts"][followup] += 1
        entry["route_counts"][f"{action_label}/{followup}"] += 1
        pending = None

    for case in iter_kifu_decisions(Path(kifu_path)):
        source = dict(case.get("source", {}) or {})
        match_id = str(source.get("match_id", "unknown"))
        round_index = int(source.get("round_index", 0))
        round_key = (match_id, round_index)
        decision_counts["reconstructed"] += 1

        if pending is not None:
            is_immediate_followup = bool(
                pending["match_id"] == match_id
                and pending["round_key"] == round_key
                and int(source.get("decision_index", 0))
                == int(pending["decision_index"]) + 1
                and case["player"] == pending["player"]
                and case["actual_action"][0] == "attack"
            )
            record_pending(
                str(case["actual_action"][2])
                if is_immediate_followup
                else None
            )

        if round_key in excluded_rounds:
            decision_counts["excluded"] += 1
            continue
        decision_counts["included"] += 1
        included_rounds.add(round_key)
        included_matches.add(match_id)
        action: Action = tuple(case["actual_action"])  # type: ignore[assignment]
        decision_counts[action[0]] += 1
        if case.get("position", {}).get("phase") != "receive":
            continue
        if action[0] not in ("pass", "receive"):
            continue

        state = replay_validation_case(case)
        features = _response_features(case, state)
        action_label = _response_action_label(action, state.current_attack)
        response_counts[action_label] += 1
        public_seen = _public_history_summary(case.get("history", ()))[
            "public_seen"
        ]
        hand_after_receive = list(state.hands[str(case["player"])])
        if action[0] == "receive" and action[1] in hand_after_receive:
            hand_after_receive.remove(str(action[1]))
        pending = {
            "pattern_id": _digest(features),
            "features": features,
            "match_id": match_id,
            "round_key": round_key,
            "decision_index": int(source.get("decision_index", 0)),
            "player": str(case["player"]),
            "action_label": action_label,
            "hand_after_receive": hand_after_receive,
            "public_seen": public_seen,
        }
        if action[0] == "pass":
            record_pending()
    record_pending()

    serialized_patterns = []
    reusable_patterns = 0
    for pattern_id, raw in patterns.items():
        observations = int(raw["observations"])
        action_counts = Counter(raw["action_counts"])
        support = max(action_counts.values(), default=0)
        dominance = support / max(1, observations)
        recommended = (
            sorted(action_counts, key=lambda key: (-action_counts[key], key))[0]
            if action_counts
            else "other"
        )
        reusable = bool(
            observations >= max(1, int(min_observations))
            and len(raw["matches"]) >= max(1, int(min_matches))
            and dominance >= max(0.0, min(1.0, float(min_dominance)))
        )
        reusable_patterns += int(reusable)
        serialized_patterns.append({
            "pattern_id": pattern_id[:12],
            "features": raw["features"],
            "observations": observations,
            "distinct_matches": len(raw["matches"]),
            "distinct_rounds": len(raw["rounds"]),
            "recommended_action": recommended,
            "support": support,
            "dominance": round(dominance, 5),
            "reusable_candidate": reusable,
            "action_counts": dict(sorted(action_counts.items())),
            "followup_counts": dict(sorted(raw["followup_counts"].items())),
            "route_counts": dict(sorted(raw["route_counts"].items())),
        })
    serialized_patterns.sort(
        key=lambda item: (
            not bool(item["reusable_candidate"]),
            -int(item["distinct_matches"]),
            -int(item["observations"]),
            str(item["pattern_id"]),
        )
    )

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "offline_human_response_dictionary_audit",
        "live_ai_affected": False,
        "privacy": {
            "raw_archive_is_private": True,
            "names_retained": False,
            "timestamps_retained": False,
            "match_ids_retained": False,
            "hands_retained": False,
            "move_histories_retained": False,
        },
        "source_summary": {
            "declared_matches": int(archive.get("match_count", len(archive["matches"]))),
            "declared_rounds": int(archive.get("round_count", 0)),
            "declared_moves": int(archive.get("move_count", 0)),
            "distinct_player_labels_detected": int(
                inventory["distinct_named_players"]
            ),
            "invalid_rounds": len(inventory["invalid_rounds"]),
            "five_shi_rounds": len(inventory["five_shi_rounds"]),
            "five_shi_included": bool(include_five_shi),
            "included_matches": len(included_matches),
            "included_rounds": len(included_rounds),
        },
        "replay_summary": {
            "reconstructed_decisions": int(decision_counts["reconstructed"]),
            "included_decisions": int(decision_counts["included"]),
            "excluded_decisions": int(decision_counts["excluded"]),
            "inferred_passes": int(decision_counts["pass"]),
            "receives": int(decision_counts["receive"]),
            "attacks": int(decision_counts["attack"]),
            "block_and_attacks": int(decision_counts["attack_after_block"]),
        },
        "response_summary": {
            "observations": sum(response_counts.values()),
            "action_counts": dict(sorted(response_counts.items())),
            "pattern_count": len(patterns),
            "reusable_candidate_count": reusable_patterns,
            "thresholds": {
                "minimum_observations": max(1, int(min_observations)),
                "minimum_distinct_matches": max(1, int(min_matches)),
                "minimum_dominance": max(
                    0.0,
                    min(1.0, float(min_dominance)),
                ),
            },
        },
        "patterns": serialized_patterns[: max(1, int(max_patterns))],
        "patterns_truncated": max(0, len(serialized_patterns) - max(1, int(max_patterns))),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kifu", type=Path, default=DEFAULT_RAW_KIFU_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_AUDIT_PATH)
    parser.add_argument("--dictionary-output", type=Path)
    parser.add_argument("--include-five-shi", action="store_true")
    parser.add_argument("--min-observations", type=int, default=15)
    parser.add_argument("--min-matches", type=int, default=8)
    parser.add_argument("--min-dominance", type=float, default=0.70)
    parser.add_argument("--max-patterns", type=int, default=100)
    args = parser.parse_args(argv)
    report = audit_human_kifu_archive(
        args.kifu,
        include_five_shi=args.include_five_shi,
        min_observations=args.min_observations,
        min_matches=args.min_matches,
        min_dominance=args.min_dominance,
        max_patterns=args.max_patterns,
    )
    write_json(args.output, report)
    dictionary = None
    if args.dictionary_output is not None:
        dictionary = build_human_response_dictionary(report)
        write_json(args.dictionary_output, dictionary)
    print(json.dumps({
        "output": str(args.output),
        "dictionary_output": (
            str(args.dictionary_output)
            if args.dictionary_output is not None else None
        ),
        "dictionary_patterns": (
            int(dictionary["pattern_count"])
            if dictionary is not None else 0
        ),
        "live_ai_affected": report["live_ai_affected"],
        "source_summary": report["source_summary"],
        "replay_summary": report["replay_summary"],
        "response_summary": report["response_summary"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

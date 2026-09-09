"""Read-only, replayable snapshots for human reviews of debug-room decisions."""

import copy
import hashlib
import json
import os
import re
from urllib.parse import unquote
from datetime import datetime, timezone
from pathlib import Path

from goita_ai2.state import GoitaState


ACTION_RE = re.compile(
    r"^([ABCD]): (?:block ([1-9]) -> attack ([1-9])|receive ([1-9])|attack ([1-9])|(pass)(?:\s|$))"
)
SEARCH_FIELDS = (
    "TIME_SEARCH_ENABLED", "TIME_SEARCH_HARD_MAX_SECONDS", "TIME_SEARCH_MAX_SECONDS",
    "TIME_SEARCH_SAMPLE_COUNT", "TIME_SEARCH_MAX_DEPTH", "TIME_SEARCH_MAX_NODES",
    "TIME_SEARCH_ADAPTIVE_BUDGET_ENABLED", "GENERIC_RESPONSE_NARROWING_ENABLED",
)


def parse_action(line):
    match = ACTION_RE.match(str(line))
    if not match:
        return None
    seat, block, attack, receive, solo_attack, passed = match.groups()
    if passed:
        action = ("pass", None, None)
    elif receive:
        action = ("receive", receive, None)
    elif solo_attack:
        action = ("attack", None, solo_attack)
    else:
        action = ("attack_after_block", block, attack)
    return seat, action


def log_turn_numbers(log):
    """One number per turn; a receive and its following attack share a number."""
    numbers = []
    number = 0
    previous = None
    for line in log:
        if str(line).startswith("Game start."):
            number = 0
        parsed = parse_action(line)
        if parsed:
            seat, action = parsed
            follows_receive = (
                previous is not None and previous[0] == seat
                and previous[1][0] == "receive" and action[0] == "attack"
            )
            if not follows_receive:
                number += 1
            numbers.append(number)
        else:
            numbers.append(None)
        previous = parsed
    return numbers


def build_review_snapshot(game, *, apply_action, new_board, update_board):
    """Replay recorded actions only. Never call an AI or invent missing telemetry."""
    log = list(game.get("log", []))
    numbers = log_turn_numbers(log)
    initial_hands = copy.deepcopy(game.get("init_hands", {}))
    if set(initial_hands) != set("ABCD") or any(len(hand) != 8 for hand in initial_hands.values()):
        raise ValueError("初期配牌がありません。対局を開始してから作成してください。")
    state = GoitaState(hands=initial_hands, dealer=game.get("dealer", "A"))
    board = new_board()
    decisions = []
    for index, line in enumerate(log):
        parsed = parse_action(line)
        if not parsed:
            continue
        seat, action = parsed
        if action not in state.legal_actions(seat):
            raise ValueError(f"ログの手番{numbers[index]}を再現できません。")
        before = {
            "turn": state.turn, "phase": state.phase,
            "attacker": state.attacker, "current_attack": state.current_attack,
            "hands": copy.deepcopy(state.hands),
            "board": copy.deepcopy(board),
            "face_down_hidden": copy.deepcopy(state.face_down_hidden),
        }
        reason_match = re.search(r"\[AI:([^\]]+)\]", line)
        candidate_match = re.search(r"\[AI-CANDIDATES:([^\]]+)\]", line)
        candidates = None
        if candidate_match:
            try:
                candidates = json.loads(unquote(candidate_match[1]))
            except (ValueError, TypeError):
                pass  # Preserve the original log even if old telemetry cannot be decoded.
        decisions.append({
            "log_index": index, "turn_number": numbers[index], "seat": seat,
            "action": list(action), "before": before, "log": str(line),
            "decision_record": "recorded" if "[AI:" in line else "not_recorded",
            "decision_reason": reason_match[1] if reason_match else None,
            "candidate_record": "recorded" if candidates is not None else "not_recorded",
            "candidate_evaluations": candidates,
        })
        apply_action(state, seat, action)
        update_board(board, seat, action, hidden_receive=" (hidden)" in line)

    live = game.get("state")
    if live is not None and (state.hands != live.hands or state.turn != live.turn or state.phase != live.phase):
        raise ValueError("ログと現在の局面が一致しないため、再現用レポートを作成できません。")

    code_root = Path(__file__).resolve().parents[1] / "goita_ai2"
    digest = hashlib.sha256()
    for path in sorted(code_root.rglob("*.py")):
        digest.update(path.relative_to(code_root).as_posix().encode())
        digest.update(path.read_bytes())
    return {
        "format": "sorou-goita-ai-review", "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "round_id": game.get("member_kifu_round_id"),
        "round_number": game.get("round_count", 1),
        "dealer": game.get("dealer", "A"), "initial_hands": initial_hands,
        "kifu_moves": copy.deepcopy(game.get("kifu_moves", [])),
        "log": log, "decisions": decisions,
        "versions": {
            "ai_profile": game.get("ai_profile"),
            "deployment_revision": os.environ.get("RENDER_GIT_COMMIT") or None,
            "ai_source_sha256_at_capture": digest.hexdigest(),
        },
        "settings_at_capture": {
            seat: {key: getattr(agent, key, None) for key in SEARCH_FIELDS}
            for seat, agent in game.get("agents", {}).items()
        },
        "recording": {
            "decision_source": "original_log", "recalculated": False,
            "random_state_at_decision": "not_recorded",
            "search_settings_at_decision": "not_recorded",
            "hidden_hands_usage": "replay_only; evaluate using actor hand and public history",
        },
    }

"""Feature encoding and dependency-free inference for the experimental policy."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from goita_ai2.constants import ALL_SEATS, PIECE_TOTALS


Action = Tuple[str, Optional[str], Optional[str]]
PIECES = tuple(sorted(PIECE_TOTALS))
RELATIONS = ("self", "next", "partner", "previous")
PHASES = ("attack", "receive")
ACTION_TYPES = ("pass", "receive", "attack", "attack_after_block")
HISTORY_LENGTH = 8


def action_vocabulary() -> tuple[Action, ...]:
    actions: list[Action] = [("pass", None, None)]
    actions.extend(("receive", piece, None) for piece in PIECES)
    actions.extend(("attack", None, piece) for piece in PIECES)
    actions.extend(
        ("attack_after_block", block, attack)
        for block in PIECES
        for attack in PIECES
    )
    return tuple(actions)


ACTIONS = action_vocabulary()
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTIONS)}


def action_tuple(value: Sequence[Any] | Mapping[str, Any]) -> Action:
    if isinstance(value, Mapping):
        return (
            str(value.get("type", "")),
            None if value.get("block") is None else str(value.get("block")),
            None if value.get("attack") is None else str(value.get("attack")),
        )
    return (
        str(value[0]),
        None if value[1] is None else str(value[1]),
        None if value[2] is None else str(value[2]),
    )


def _one_hot(names: list[str], values: list[float], prefix: str, choices: Sequence[str], selected: Any) -> None:
    selected_text = str(selected)
    for choice in choices:
        names.append(f"{prefix}.{choice}")
        values.append(1.0 if selected_text == choice else 0.0)


def _piece_one_hot(names: list[str], values: list[float], prefix: str, selected: Any) -> None:
    _one_hot(names, values, prefix, ("none", *PIECES), "none" if selected is None else selected)


def _counts(names: list[str], values: list[float], prefix: str, payload: Mapping[str, Any], keys: Sequence[str]) -> None:
    for key in keys:
        names.append(f"{prefix}.{key}")
        values.append(float(payload.get(key, 0)))


def encode_state(state: Mapping[str, Any], legal_action_count: int) -> tuple[list[str], list[float]]:
    """Encode a public-information state into a stable numeric vector."""
    names: list[str] = []
    values: list[float] = []
    _one_hot(names, values, "phase", PHASES, state.get("phase"))
    _one_hot(names, values, "dealer", RELATIONS, state.get("dealer"))
    _one_hot(names, values, "attacker", ("none", *RELATIONS), state.get("attacker"))
    _piece_one_hot(names, values, "current_attack", state.get("current_attack"))
    names.extend(("own_score", "enemy_score", "score_difference"))
    own_score = float(state.get("own_score", 0))
    enemy_score = float(state.get("enemy_score", 0))
    values.extend((own_score, enemy_score, own_score - enemy_score))
    _counts(names, values, "own_initial_hand", dict(state.get("own_initial_hand", {}) or {}), PIECES)
    names.append("own_initial_had_both_royals")
    values.append(float(bool(state.get("own_initial_had_both_royals", False))))
    _counts(names, values, "own_hand", dict(state.get("own_hand", {}) or {}), PIECES)
    names.extend(("own_hand_size", "legal_action_count"))
    values.extend((float(state.get("own_hand_size", 0)), float(legal_action_count)))

    counters = dict(state.get("public_counters", {}) or {})
    _counts(names, values, "attack_piece_counts", dict(counters.get("attack_piece_counts", {}) or {}), PIECES)
    _counts(names, values, "receive_piece_counts", dict(counters.get("receive_piece_counts", {}) or {}), PIECES)
    _counts(names, values, "pass_counts", dict(counters.get("pass_counts", {}) or {}), RELATIONS)
    _counts(names, values, "face_down_counts", dict(counters.get("face_down_counts", {}) or {}), RELATIONS)
    _counts(names, values, "remaining_hand_sizes", dict(counters.get("remaining_hand_sizes", {}) or {}), RELATIONS)

    full_history = list(state.get("history", []) or [])
    relation_attacks = {relation: {piece: 0 for piece in PIECES} for relation in RELATIONS}
    relation_receives = {relation: {piece: 0 for piece in PIECES} for relation in RELATIONS}
    self_attack_receivers = {relation: {piece: 0 for piece in PIECES} for relation in RELATIONS}
    active_attacker = "none"
    active_piece: Optional[str] = None
    for raw_item in full_history:
        event = dict(raw_item or {})
        relation = str(event.get("actor", "none"))
        action_type = str(event.get("type", ""))
        block = None if event.get("block") is None else str(event.get("block"))
        attack = None if event.get("attack") is None else str(event.get("attack"))
        if action_type == "receive" and relation in relation_receives and block in PIECES:
            relation_receives[relation][block] += 1
            if active_attacker == "self" and active_piece in PIECES:
                self_attack_receivers[relation][active_piece] += 1
        if (
            action_type in {"attack", "attack_after_block"}
            and relation in relation_attacks
            and attack in PIECES
        ):
            relation_attacks[relation][attack] += 1
            active_attacker = relation
            active_piece = attack

    for relation in RELATIONS:
        _counts(
            names,
            values,
            f"relation_attack_piece_counts.{relation}",
            relation_attacks[relation],
            PIECES,
        )
        _counts(
            names,
            values,
            f"relation_receive_piece_counts.{relation}",
            relation_receives[relation],
            PIECES,
        )
    for relation in ("next", "partner", "previous"):
        _counts(
            names,
            values,
            f"self_attack_received_by.{relation}",
            self_attack_receivers[relation],
            PIECES,
        )

    history = full_history[-HISTORY_LENGTH:]
    history = ([None] * (HISTORY_LENGTH - len(history))) + history
    for slot, item in enumerate(history):
        event = dict(item or {})
        prefix = f"history.{slot}"
        names.append(f"{prefix}.present")
        values.append(float(item is not None))
        _one_hot(names, values, f"{prefix}.actor", RELATIONS, event.get("actor"))
        _one_hot(names, values, f"{prefix}.type", ACTION_TYPES, event.get("type"))
        names.append(f"{prefix}.block_known")
        values.append(float(bool(event.get("block_known", False))))
        _piece_one_hot(names, values, f"{prefix}.block", event.get("block"))
        _piece_one_hot(names, values, f"{prefix}.attack", event.get("attack"))
    return names, values


def record_features(record: Mapping[str, Any]) -> tuple[list[str], list[float]]:
    return encode_state(
        dict(record.get("state", {}) or {}),
        len(record.get("legal_actions", []) or []),
    )


def relative_seat(actor: str, other: Optional[str]) -> str:
    if other not in ALL_SEATS:
        return "none"
    offset = (ALL_SEATS.index(str(other)) - ALL_SEATS.index(actor)) % 4
    return RELATIONS[offset]


def piece_counts(pieces: Iterable[str]) -> Dict[str, int]:
    material = list(str(piece) for piece in pieces)
    return {piece: material.count(piece) for piece in PIECES}


def public_history(history: Sequence[Mapping[str, Any]], actor: str) -> list[Dict[str, Any]]:
    output: list[Dict[str, Any]] = []
    for item in history:
        action = action_tuple(list(item["action"]))
        action_type, block, attack = action
        event_actor = str(item["player"])
        hidden = action_type == "attack_after_block" and event_actor != actor
        output.append({
            "actor": relative_seat(actor, event_actor),
            "type": action_type,
            "block": None if hidden else block,
            "block_known": bool(block is not None and not hidden),
            "attack": attack,
        })
    return output


def public_counters(history: Sequence[Mapping[str, Any]], actor: str) -> Dict[str, Any]:
    attacks = {piece: 0 for piece in PIECES}
    receives = {piece: 0 for piece in PIECES}
    passes = {relation: 0 for relation in RELATIONS}
    hidden = {relation: 0 for relation in RELATIONS}
    played = {relation: 0 for relation in RELATIONS}
    for item in history:
        relation = relative_seat(actor, str(item["player"]))
        action_type, block, attack = action_tuple(list(item["action"]))
        if action_type == "pass":
            passes[relation] += 1
        elif action_type == "attack_after_block":
            hidden[relation] += 1
            played[relation] += 2
        elif action_type in {"receive", "attack"}:
            played[relation] += 1
        if action_type == "receive" and block in receives:
            receives[str(block)] += 1
        if attack in attacks:
            attacks[str(attack)] += 1
    return {
        "attack_piece_counts": attacks,
        "receive_piece_counts": receives,
        "pass_counts": passes,
        "face_down_counts": hidden,
        "remaining_hand_sizes": {
            relation: max(0, 8 - played[relation]) for relation in RELATIONS
        },
    }


def live_state_payload(
    state: Any,
    player: str,
    *,
    initial_hand: Sequence[str],
    history: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    own_team = "AC" if player in {"A", "C"} else "BD"
    enemy_team = "BD" if own_team == "AC" else "AC"
    return {
        "phase": str(state.phase),
        "dealer": relative_seat(player, state.dealer),
        "current_attack": state.current_attack,
        "attacker": relative_seat(player, state.attacker),
        "own_score": int(state.team_score.get(own_team, 0)),
        "enemy_score": int(state.team_score.get(enemy_team, 0)),
        "own_initial_hand": piece_counts(initial_hand),
        "own_initial_had_both_royals": "8" in initial_hand and "9" in initial_hand,
        "own_hand": piece_counts(state.hands[player]),
        "own_hand_size": len(state.hands[player]),
        "public_counters": public_counters(history, player),
        "history": public_history(history, player),
    }


class NeuralPolicyModel:
    """Load an exported MLP and rank legal actions without numpy/sklearn."""

    def __init__(self, payload: Mapping[str, Any]):
        self.payload = dict(payload)
        self.feature_names = tuple(str(item) for item in payload["feature_names"])
        self.mean = tuple(float(item) for item in payload["scaler"]["mean"])
        self.scale = tuple(float(item) for item in payload["scaler"]["scale"])
        self.class_ids = tuple(int(item) for item in payload["class_ids"])
        self.layers = tuple(payload["layers"])

    @classmethod
    def load(cls, path: Path) -> "NeuralPolicyModel":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def logits(self, names: Sequence[str], values: Sequence[float]) -> Dict[int, float]:
        if tuple(names) != self.feature_names:
            raise ValueError("neural policy feature schema mismatch")
        vector = [
            (float(value) - mean) / (scale if abs(scale) > 1e-12 else 1.0)
            for value, mean, scale in zip(values, self.mean, self.scale)
        ]
        for layer_index, layer in enumerate(self.layers):
            weights = layer["weights"]
            bias = layer["bias"]
            next_vector = [
                float(bias[column])
                + sum(vector[row] * float(weights[row][column]) for row in range(len(vector)))
                for column in range(len(bias))
            ]
            if layer_index + 1 < len(self.layers):
                next_vector = [max(0.0, value) for value in next_vector]
            vector = next_vector
        if len(vector) == 1 and len(self.class_ids) == 2:
            vector = [-vector[0], vector[0]]
        if len(vector) != len(self.class_ids):
            raise ValueError("neural policy output schema mismatch")
        return dict(zip(self.class_ids, vector))

    def rank_actions(
        self,
        state_payload: Mapping[str, Any],
        legal_actions: Sequence[Action],
    ) -> list[tuple[Action, float]]:
        names, values = encode_state(state_payload, len(legal_actions))
        scores = self.logits(names, values)
        ranked = [
            (tuple(action), float(scores.get(ACTION_TO_INDEX[tuple(action)], -math.inf)))
            for action in legal_actions
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked


__all__ = [
    "ACTIONS",
    "ACTION_TO_INDEX",
    "Action",
    "NeuralPolicyModel",
    "action_tuple",
    "encode_state",
    "live_state_payload",
    "record_features",
]

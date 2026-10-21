"""Bounded visual history for the VRChat bridge (not game logic).

Only VRChat-reserved seats have exact hands; website and AI seats stay masked,
even at a win. This history is never included in browser state responses.
"""

from backend.vrc_solo_players import seats as vrc_seats

SEATS = "ABCD"
MAX_STEPS = 128


def _digits(values, length):
    return "".join(str(value) if str(value) in "123456789" and len(str(value)) == 1 else "0"
                   for value in (list(values) + [None] * length)[:length])


def _frame(game, actor="-", kind="-"):
    state = game["state"]
    cells, hidden = [], []
    for seat in SEATS:
        row = game.get("board", {}).get(seat, {})
        receives = list(row.get("receive", []))
        flags = list(row.get("receive_hidden", []))
        # Deliberately mask history even when state.finished is already True.
        safe = ["1" if seat not in vrc_seats(game) and index < len(flags) and flags[index] and value is not None else value
                for index, value in enumerate(receives)]
        cells.append(_digits(safe, 4) + _digits(row.get("attack", []), 4))
        hidden.extend("1" if i < len(flags) and flags[i] else "0" for i in range(4))
    frame = ["0", actor, kind, "".join(cells), "".join(hidden),
            "".join(str(p) for p in state.hands["A"]),
            "".join(str(len(state.hands[s])) for s in SEATS),
            state.turn if state.turn in tuple(SEATS) else "-",
            "r" if state.phase == "receive" else "a",
            state.attacker if state.attacker in tuple(SEATS) else "-", str(state.current_attack or "0")]
    if game.get("vrc_web", {}).get("mixed"):
        frame.extend("".join(state.hands[seat]) if seat in vrc_seats(game) else "#" for seat in "BCD")
    return frame


def start(game):
    if game.get("vrc_solo_room") and game.get("is_started") and "vrc_playback" not in game:
        game["vrc_playback"] = {"base": _frame(game), "steps": [], "number": 0}


def _append(history, frame):
    history["number"] += 1
    frame[0] = str(history["number"])
    history["steps"].append(frame)
    del history["steps"][:-MAX_STEPS]


def record(game, player, action):
    history = game.get("vrc_playback")
    if not game.get("vrc_solo_room") or history is None:
        return
    action_type, _, attack = action
    kind = {"pass": "P", "receive": "R", "attack": "A", "attack_after_block": "A"}.get(action_type)
    if kind is None:
        return
    final = _frame(game, player, kind)
    previous = history["steps"][-1] if history["steps"] else history["base"]
    if action_type == "attack_after_block":
        # This rules action places two pieces atomically. Show its hidden/receive
        # piece first without changing the actual state or revealing its value.
        middle = list(final)
        middle[2] = "H"
        cells = list(middle[3])
        offset = SEATS.index(player) * 8 + 4
        added = [i for i in range(offset, offset + 4) if previous[3][i] == "0" and cells[i] != "0"]
        if len(added) == 1:
            cells[added[0]] = "0"
            middle[3] = "".join(cells)
            if player == "A":
                middle[5] += str(attack)
            elif len(middle) == 14 and player in vrc_seats(game):
                middle[10 + SEATS.index(player)] += str(attack)
            counts = list(middle[6])
            counts[SEATS.index(player)] = str(int(counts[SEATS.index(player)]) + 1)
            middle[6] = "".join(counts)
            middle[7:11] = [player, "a", previous[9], previous[10]]
            _append(history, middle)
    _append(history, final)


def fields(game):
    history = game.get("vrc_playback")
    if not history or not game.get("is_started"):
        return []
    frames = [history["base"], *history["steps"]]
    return [f"ply={history['number']}", "replay=" + ";".join(",".join(frame) for frame in frames)]

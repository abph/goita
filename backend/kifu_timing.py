"""Optional, rounded action intervals; never change game rules or save policy."""

import math
import time


def round_seconds(seconds):
    seconds = max(0.0, float(seconds))
    scale = 10 if seconds <= 4 else (1 if seconds < 10 else 0.2)
    return round(math.floor(seconds * scale + 0.5) / scale, 1)


def reset(game):
    game["kifu_move_times"] = [None] * len(game.get("kifu_moves", []))
    game["kifu_clock"] = None


def begin(game):
    if game.get("is_started") and game.get("kifu_clock") is None:
        game["kifu_clock"] = time.monotonic()


def record(game):
    """Called once per accepted action, immediately before its kifu row is added."""
    now = time.monotonic()
    previous = game.get("kifu_clock")
    size = len(game.get("kifu_moves", []))
    times = game.setdefault("kifu_move_times", [])
    times[:] = times[:size] + [None] * max(0, size - len(times))
    times.append(round_seconds(now - previous) if previous is not None else None)
    game["kifu_clock"] = now


def validate_times(values, size):
    if not isinstance(values, list) or len(values) != size:
        raise ValueError("時間情報と手順の件数が一致しません")
    for value in values:
        if value is not None and (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
            or value > 7 * 24 * 60 * 60
        ):
            raise ValueError("棋譜の時間情報が正しくありません")
    return list(values)


def saved_times(game):
    try:
        times = validate_times(game.get("kifu_move_times"), len(game.get("kifu_moves", [])))
    except ValueError:
        return {}
    return {"move_times": times} if any(value is not None for value in times) else {}

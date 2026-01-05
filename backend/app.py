from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from goita_ai2.state import GoitaState
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.simulate import _notify_public
from goita_ai2.utils import create_random_hands


ALL_SEATS = ["A", "B", "C", "D"]
MAIN_GID = "main"


def _validate_seat(s: str, *, name: str = "seat") -> str:
    s = (s or "").strip().upper()
    if s not in ALL_SEATS:
        raise HTTPException(status_code=400, detail=f"invalid {name}: {s} (must be A/B/C/D)")
    return s


def _normalize_hands(hands: Dict[str, List[Any]]) -> Dict[str, List[str]]:
    return {p: [str(x) for x in hands[p]] for p in ALL_SEATS}


def create_random_hands_no_five_shi(max_retry: int = 5000) -> Dict[str, List[str]]:
    last_hands: Dict[str, List[str]] = {p: [] for p in ALL_SEATS}
    for _ in range(max_retry):
        raw = create_random_hands()
        hands = _normalize_hands(raw)
        last_hands = hands
        if all(sum(1 for x in hands[p] if x == "1") <= 4 for p in ALL_SEATS):
            return hands
    return last_hands


app = FastAPI(title="Goita FastAPI (Render-ready)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== frontend 配信 ======
BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="frontend/index.html not found")
    return FileResponse(index_path)


# ====== 棋譜（kifu）用 ======
PIECE_KANJI: Dict[str, str] = {
    "9": "王", "8": "玉", "7": "飛", "6": "角",
    "5": "金", "4": "銀", "3": "馬", "2": "香", "1": "し",
}
PLAYER_IDX: Dict[str, str] = {"A": "0", "B": "1", "C": "2", "D": "3"}


def _hand_to_kifu_string(hand: List[Any]) -> str:
    return "".join(PIECE_KANJI.get(str(x), str(x)) for x in hand)


def _piece_to_kifu(v: Optional[str]) -> str:
    if v is None:
        return ""
    v = str(v)
    return PIECE_KANJI.get(v, v)


def _action_to_kifu_row(player: str, action: Tuple[str, Optional[str], Optional[str]]) -> List[str]:
    t, b, a = action
    pid = PLAYER_IDX[player]
    if t == "pass":
        return [pid, "パス", ""]
    if t == "receive":
        return [pid, _piece_to_kifu(b), ""]
    if t == "attack":
        return [pid, "", _piece_to_kifu(a)]
    if t == "attack_after_block":
        return [pid, _piece_to_kifu(b), _piece_to_kifu(a)]
    return [pid, t, ""]


def _compress_kifu_moves(moves: List[List[str]]) -> List[List[str]]:
    out: List[List[str]] = []
    for row in moves:
        if not row or len(row) < 3:
            continue
        pid, b, a = str(row[0]), str(row[1]), str(row[2])

        if b == "パス" or b.lower() == "pass":
            continue

        if out:
            lp, lb, la = out[-1]
            if lp == pid and la == "" and lb != "" and b == "" and a != "":
                out[-1] = [lp, lb, a]
                continue

        out.append([pid, b, a])
    return out


GAMES: Dict[str, Dict[str, Any]] = {}


class ActionModel(BaseModel):
    action_type: str = Field(..., description="pass / receive / attack / attack_after_block")
    block: Optional[str] = None
    attack: Optional[str] = None

    def to_tuple(self) -> Tuple[str, Optional[str], Optional[str]]:
        return (self.action_type, self.block, self.attack)


class StepRequest(BaseModel):
    player: str = Field(..., description="A/B/C/D")
    action: ActionModel


def _apply_action(state: GoitaState, player: str, action: Tuple[str, Optional[str], Optional[str]]) -> None:
    action_type, block, attack = action
    if action_type == "pass":
        state.apply_pass(player)
    elif action_type == "receive":
        if block is None:
            raise ValueError("receive には block が必要です")
        state.apply_receive(player, block)
    elif action_type == "attack":
        if attack is None:
            raise ValueError("attack には attack が必要です")
        state.apply_attack(player, attack)
    elif action_type == "attack_after_block":
        if block is None or attack is None:
            raise ValueError("attack_after_block には block と attack の両方が必要です")
        state.apply_attack_after_block(player, block, attack)
    else:
        raise ValueError(f"未知の action_type: {action_type}")


def _format_action(player: str, action: Tuple[str, Optional[str], Optional[str]]) -> str:
    t, b, a = action
    if t == "pass":
        return f"{player}: pass"
    if t == "receive":
        return f"{player}: receive {b}"
    if t == "attack":
        return f"{player}: attack {a}"
    if t == "attack_after_block":
        return f"{player}: block {b} -> attack {a}"
    return f"{player}: {t} (block={b}, attack={a})"


def _actions_to_json(actions: List[Tuple[str, Optional[str], Optional[str]]]) -> List[Dict[str, Any]]:
    return [{"action_type": t, "block": b, "attack": a} for (t, b, a) in actions]


def _build_scores(state: GoitaState) -> Dict[str, Any]:
    ts = getattr(state, "team_score", None)
    if isinstance(ts, dict):
        ac = ts.get("AC", 0)
        bd = ts.get("BD", 0)
        return {"A": ac, "C": ac, "B": bd, "D": bd}
    return {"A": 0, "B": 0, "C": 0, "D": 0}


def _new_board_snapshot() -> Dict[str, Dict[str, Any]]:
    return {p: {"receive": [None]*4, "attack": [None]*4, "receive_hidden": [False]*4} for p in ALL_SEATS}


def _push_first_empty(slots: List[Optional[str]], value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    for i in range(len(slots)):
        if slots[i] is None:
            slots[i] = value
            return i
    slots[-1] = value
    return len(slots) - 1


def _update_board_snapshot(board: Dict[str, Dict[str, Any]], player: str,
                          action: Tuple[str, Optional[str], Optional[str]], *,
                          hidden_receive: bool = False) -> None:
    t, b, a = action
    if player not in board:
        return
    if t == "receive":
        idx = _push_first_empty(board[player]["receive"], b)
        if idx is not None:
            board[player]["receive_hidden"][idx] = bool(hidden_receive)
    elif t == "attack":
        _push_first_empty(board[player]["attack"], a)
    elif t == "attack_after_block":
        idx = _push_first_empty(board[player]["receive"], b)
        if idx is not None:
            board[player]["receive_hidden"][idx] = bool(hidden_receive)
        _push_first_empty(board[player]["attack"], a)


def _is_hidden_receive_by_state_delta(state: GoitaState, player: str, action_type: str, before_len: int) -> bool:
    if action_type not in ("receive", "attack_after_block"):
        return False
    return len(state.face_down_hidden[player]) > before_len


def _state_public_view(
    state: GoitaState,
    *,
    viewer: str,
    log: List[str],
    board_public: Dict[str, Dict[str, Any]],
    reveal_hands: bool = False
) -> Dict[str, Any]:
    hands_view: Dict[str, Any] = {}
    for p in ALL_SEATS:
        if reveal_hands or p == viewer:
            hands_view[p] = list(state.hands[p])
        else:
            hands_view[p] = {"count": len(state.hands[p])}

    if reveal_hands:
        board_view = copy.deepcopy(board_public)
        for p in ALL_SEATS:
            rh = board_view.get(p, {}).get("receive_hidden")
            if isinstance(rh, list):
                board_view[p]["receive_hidden"] = [False for _ in rh]
    else:
        board_view = board_public

    return {
        "turn": state.turn,
        "phase": state.phase,
        "attacker": state.attacker,
        "current_attack": state.current_attack,
        "hands": hands_view,
        "team_score": getattr(state, "team_score", None),
        "scores": _build_scores(state),
        "board_public": board_view,
        "log": (log or [])[-200:],
        "finished": state.finished,
        "winner": state.winner,
    }


def _create_game_obj(dealer: str = "A") -> Dict[str, Any]:
    dealer = _validate_seat(dealer, name="dealer")
    hands = create_random_hands_no_five_shi()
    state = GoitaState(hands=hands, dealer=dealer)

    agents: Dict[str, RuleBasedAgent] = {s: RuleBasedAgent(name=f"CPU-{s}") for s in ALL_SEATS}
    for seat, ag in agents.items():
        ag.bind_player(seat)

    return {
        "state": state,
        "agents": agents,
        "log": [f"Game start. dealer={dealer}, table=main"],
        "board": _new_board_snapshot(),
        "init_hands": hands,
        "dealer": dealer,
        "kifu_moves": [],
        # 人間席（claimされた席）
        "human_seats": set(),  # type: Set[str]
    }


def _ensure_main_game(dealer: str = "A") -> None:
    if MAIN_GID not in GAMES:
        GAMES[MAIN_GID] = _create_game_obj(dealer=dealer)


@app.post("/games/main/reset")
def reset_main(dealer: str = "A"):
    GAMES[MAIN_GID] = _create_game_obj(dealer=dealer)
    return {"ok": True, "game_id": MAIN_GID, "dealer": dealer}


# 席のclaim（人間席として登録）
@app.post("/games/main/claim")
def claim_seat(seat: str):
    _ensure_main_game()
    seat = _validate_seat(seat, name="seat")
    game = GAMES[MAIN_GID]
    hs: Set[str] = game.setdefault("human_seats", set())
    hs.add(seat)
    return {"ok": True, "game_id": MAIN_GID, "human_seats": sorted(list(hs))}


@app.get("/games/{game_id}/state")
def get_state(game_id: str, viewer: str = "A", reveal_hands: int = 0):
    if game_id == MAIN_GID:
        _ensure_main_game()
    viewer = _validate_seat(viewer, name="viewer")
    game = GAMES.get(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="game not found")

    state: GoitaState = game["state"]

    payload = _state_public_view(
        state,
        viewer=viewer,
        log=game.get("log", []),
        board_public=game.get("board", _new_board_snapshot()),
        reveal_hands=bool(reveal_hands),
    )
    hs = game.get("human_seats", set())
    payload["human_seats"] = sorted(list(hs))
    return payload


@app.get("/games/{game_id}/legal_actions")
def get_legal_actions(game_id: str, player: str = "A"):
    if game_id == MAIN_GID:
        _ensure_main_game()
    player = _validate_seat(player, name="player")
    game = GAMES.get(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="game not found")

    state: GoitaState = game["state"]
    if state.finished or state.turn != player:
        return []
    return _actions_to_json(state.legal_actions(player))


@app.post("/games/{game_id}/step")
def step(game_id: str, req: StepRequest):
    if game_id == MAIN_GID:
        _ensure_main_game()

    player = _validate_seat(req.player, name="player")
    game = GAMES.get(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="game not found")

    state: GoitaState = game["state"]
    agents: Dict[str, RuleBasedAgent] = game["agents"]
    log: List[str] = game.setdefault("log", [])
    board = game.setdefault("board", _new_board_snapshot())
    human_seats: Set[str] = game.setdefault("human_seats", set())

    # stepを送ってきた席は人間席として登録（claim漏れ対策）
    human_seats.add(player)

    if state.finished:
        payload = _state_public_view(state, viewer=player, log=log, board_public=board)
        payload["human_seats"] = sorted(list(human_seats))
        return {"ok": True, "state": payload}

    if state.turn != player:
        raise HTTPException(status_code=400, detail=f"not your turn (turn={state.turn}, you={player})")

    action = req.action.to_tuple()

    before_fd = len(state.face_down_hidden[player])
    try:
        _apply_action(state, player, action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid action: {e}")

    hidden_receive = _is_hidden_receive_by_state_delta(state, player, action[0], before_fd)
    _update_board_snapshot(board, player, action, hidden_receive=hidden_receive)

    log.append(_format_action(player, action) + (" (hidden)" if hidden_receive else ""))
    game.setdefault("kifu_moves", []).append(_action_to_kifu_row(player, action))
    _notify_public(agents, state, player, action)

    # CPUを回す：次がhuman_seatsの誰かの番になったら止める
    safety = 0
    while (not state.finished) and (state.turn not in human_seats):
        safety += 1
        if safety > 2000:
            raise HTTPException(status_code=500, detail="safety stop: too many cpu steps")

        p = state.turn
        acts = state.legal_actions(p)
        if not acts:
            log.append(f"{p}: no legal actions (stop)")
            break

        cpu_action = agents[p].select_action(state, p, acts)

        before_fd = len(state.face_down_hidden[p])
        _apply_action(state, p, cpu_action)

        hidden_receive = _is_hidden_receive_by_state_delta(state, p, cpu_action[0], before_fd)
        _update_board_snapshot(board, p, cpu_action, hidden_receive=hidden_receive)

        log.append(_format_action(p, cpu_action) + (" (hidden)" if hidden_receive else ""))
        game.setdefault("kifu_moves", []).append(_action_to_kifu_row(p, cpu_action))
        _notify_public(agents, state, p, cpu_action)

    if state.finished:
        log.append(f"Game finished. winner={state.winner}, team_score={getattr(state, 'team_score', None)}")

    payload = _state_public_view(state, viewer=player, log=log, board_public=board)
    payload["human_seats"] = sorted(list(human_seats))
    return {"ok": True, "state": payload}


@app.get("/games/{game_id}/kifu", response_class=PlainTextResponse)
def get_kifu_yaml(game_id: str):
    if game_id == MAIN_GID:
        _ensure_main_game()

    game = GAMES.get(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="game not found")

    init_hands: Dict[str, List[Any]] = game.get("init_hands", {})
    dealer: str = game.get("dealer", "A")
    moves: List[List[str]] = _compress_kifu_moves(game.get("kifu_moves", []))

    state: GoitaState = game["state"]
    ts = getattr(state, "team_score", None)
    score = [0, 0]
    if isinstance(ts, dict):
        score = [int(ts.get("AC", 0)), int(ts.get("BD", 0))]

    h = {
        "p0": _hand_to_kifu_string(init_hands.get("A", [])),
        "p1": _hand_to_kifu_string(init_hands.get("B", [])),
        "p2": _hand_to_kifu_string(init_hands.get("C", [])),
        "p3": _hand_to_kifu_string(init_hands.get("D", [])),
    }
    uchidashi = int(PLAYER_IDX.get(dealer, "0"))

    lines: List[str] = []
    lines.append("version: 1.0")
    lines.append('p0: "プレイヤーA"')
    lines.append('p1: "プレイヤーB"')
    lines.append('p2: "プレイヤーC"')
    lines.append('p3: "プレイヤーD"')
    lines.append("log:")
    lines.append(" - hand:")
    lines.append(f'     p0: "{h["p0"]}"')
    lines.append(f'     p1: "{h["p1"]}"')
    lines.append(f'     p2: "{h["p2"]}"')
    lines.append(f'     p3: "{h["p3"]}"')
    lines.append(f"   uchidashi: {uchidashi}")
    lines.append(f"   score: [{score[0]},{score[1]}]")
    lines.append("   game:")
    for row in moves:
        a = str(row[0]).replace('"', '\\"')
        b = str(row[1]).replace('"', '\\"')
        c = str(row[2]).replace('"', '\\"')
        lines.append(f'    - ["{a}","{b}","{c}"]')
    return "\n".join(lines) + "\n"

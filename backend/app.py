from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# あなたのプロジェクト構成（goita_ai2/ が同階層にある前提）
from goita_ai2.state import GoitaState
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.simulate import _notify_public
from goita_ai2.utils import create_random_hands


app = FastAPI(title="Goita FastAPI (Render-ready)")

# Render上でGoogle Sites埋め込み等を考えるならCORSは広めでOK（本番で絞りたければ後で調整）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== frontend 配信（RenderでURL直打ちしたときに画面が出る） ======
BASE_DIR = Path(__file__).resolve().parents[1]  # プロジェクト直下（backend/ の1つ上）
FRONTEND_DIR = BASE_DIR / "frontend"

# frontend 配下のファイル（index.html含む）を /static で配信（任意だが便利）
# 例：/static/index.html でも開ける
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="frontend/index.html not found")
    return FileResponse(index_path)


# ====== ゲーム管理 ======
HUMAN_SEAT = "A"
CPU_SEATS = ["B", "C", "D"]
ALL_SEATS = ["A", "B", "C", "D"]

GAMES: Dict[str, Dict[str, Any]] = {}


class ActionModel(BaseModel):
    action_type: str = Field(..., description="pass / receive / attack / attack_after_block")
    block: Optional[str] = None
    attack: Optional[str] = None

    def to_tuple(self) -> Tuple[str, Optional[str], Optional[str]]:
        return (self.action_type, self.block, self.attack)


class StepRequest(BaseModel):
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
    return {
        p: {"receive": [None] * 4, "attack": [None] * 4, "receive_hidden": [False] * 4}
        for p in ALL_SEATS
    }


def _push_first_empty(slots: List[Optional[str]], value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    for i in range(len(slots)):
        if slots[i] is None:
            slots[i] = value
            return i
    slots[-1] = value
    return len(slots) - 1


def _update_board_snapshot(
    board: Dict[str, Dict[str, Any]],
    player: str,
    action: Tuple[str, Optional[str], Optional[str]],
    *,
    hidden_receive: bool = False,
) -> None:
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
    # face_down_hidden の増分で「伏せ」を検出
    if action_type not in ("receive", "attack_after_block"):
        return False
    return len(state.face_down_hidden[player]) > before_len


def _state_public_view(
    state: GoitaState,
    *,
    viewer: str,
    log: List[str],
    board_public: Dict[str, Dict[str, Any]],
    reveal_hands: bool = False,
) -> Dict[str, Any]:
    hands_view: Dict[str, Any] = {}
    for p in ALL_SEATS:
        if reveal_hands or p == viewer:
            hands_view[p] = list(state.hands[p])
        else:
            hands_view[p] = {"count": len(state.hands[p])}

    return {
        "turn": state.turn,
        "phase": state.phase,
        "attacker": state.attacker,
        "current_attack": state.current_attack,
        "hands": hands_view,
        "team_score": getattr(state, "team_score", None),
        "scores": _build_scores(state),
        "board_public": board_public,
        "log": (log or [])[-200:],
        "finished": state.finished,
        "winner": state.winner,
    }


@app.post("/games")
def create_game(dealer: str = "A"):
    hands = create_random_hands()
    state = GoitaState(hands=hands, dealer=dealer)

    agents: Dict[str, RuleBasedAgent] = {
        "B": RuleBasedAgent(name="CPU-B"),
        "C": RuleBasedAgent(name="CPU-C"),
        "D": RuleBasedAgent(name="CPU-D"),
    }
    for seat, ag in agents.items():
        ag.bind_player(seat)

    gid = str(uuid.uuid4())
    GAMES[gid] = {
        "state": state,
        "agents": agents,
        "log": [f"Game start. dealer={dealer}, human={HUMAN_SEAT}"],
        "board": _new_board_snapshot(),
    }
    return {"game_id": gid, "human_seat": HUMAN_SEAT, "dealer": dealer}


@app.get("/games/{game_id}/state")
def get_state(game_id: str, reveal_hands: int = 0):
    game = GAMES.get(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="game not found")

    state: GoitaState = game["state"]
    return _state_public_view(
        state,
        viewer=HUMAN_SEAT,
        log=game.get("log", []),
        board_public=game.get("board", _new_board_snapshot()),
        reveal_hands=bool(reveal_hands),
    )


@app.get("/games/{game_id}/legal_actions")
def get_legal_actions(game_id: str):
    game = GAMES.get(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="game not found")

    state: GoitaState = game["state"]
    if state.finished or state.turn != HUMAN_SEAT:
        return []
    return _actions_to_json(state.legal_actions(HUMAN_SEAT))


@app.post("/games/{game_id}/step")
def step(game_id: str, req: StepRequest):
    game = GAMES.get(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="game not found")

    state: GoitaState = game["state"]
    agents: Dict[str, RuleBasedAgent] = game["agents"]
    log: List[str] = game.setdefault("log", [])
    board = game.setdefault("board", _new_board_snapshot())

    if state.finished:
        return {"ok": True, "state": _state_public_view(state, viewer=HUMAN_SEAT, log=log, board_public=board)}

    if state.turn != HUMAN_SEAT:
        raise HTTPException(status_code=400, detail=f"not your turn (turn={state.turn})")

    action = req.action.to_tuple()

    before_fd = len(state.face_down_hidden[HUMAN_SEAT])
    try:
        _apply_action(state, HUMAN_SEAT, action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid action: {e}")

    hidden_receive = _is_hidden_receive_by_state_delta(state, HUMAN_SEAT, action[0], before_fd)
    _update_board_snapshot(board, HUMAN_SEAT, action, hidden_receive=hidden_receive)

    log.append(_format_action(HUMAN_SEAT, action) + (" (hidden)" if hidden_receive else ""))
    _notify_public(agents, state, HUMAN_SEAT, action)

    safety = 0
    while (not state.finished) and state.turn != HUMAN_SEAT:
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
        _notify_public(agents, state, p, cpu_action)

    if state.finished:
        log.append(f"Game finished. winner={state.winner}, team_score={getattr(state, 'team_score', None)}")

    return {"ok": True, "state": _state_public_view(state, viewer=HUMAN_SEAT, log=log, board_public=board)}


@app.delete("/games/{game_id}")
def delete_game(game_id: str):
    if game_id in GAMES:
        del GAMES[game_id]
    return {"ok": True}

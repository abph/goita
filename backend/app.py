from __future__ import annotations

import uuid
import copy
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# あなたのプロジェクト構成（goita_ai2/ が同階層にある前提）
from goita_ai2.state import GoitaState
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.simulate import _notify_public
from goita_ai2.utils import create_random_hands


def _normalize_hands(hands: Dict[str, List[Any]]) -> Dict[str, List[str]]:
    """handの要素型が int/str 混在でも state 側が扱いやすいように str に正規化する。"""
    return {p: [str(x) for x in hands[p]] for p in ALL_SEATS}


def create_random_hands_no_five_shi(max_retry: int = 5000) -> Dict[str, List[str]]:
    """create_random_hands() を使って配牌しつつ、
    どのプレイヤーにも '1'(し) が5枚以上配られないようにする。
    ※ hand要素が int の場合でも確実に判定できるように str に正規化してから数える。
    """
    last_hands: Dict[str, List[str]] = {p: [] for p in ALL_SEATS}
    for _ in range(max_retry):
        raw = create_random_hands()
        hands = _normalize_hands(raw)
        last_hands = hands
        if all(sum(1 for x in hands[p] if x == "1") <= 4 for p in ALL_SEATS):
            return hands
    return last_hands



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
ALL_SEATS = ["A", "B", "C", "D"]


# ====== 棋譜（kifu）出力用 ======
PIECE_KANJI: Dict[str, str] = {
    "9": "王",
    "8": "玉",
    "7": "飛",
    "6": "角",
    "5": "金",
    "4": "銀",
    "3": "馬",
    "2": "香",
    "1": "し",
}
PLAYER_IDX: Dict[str, str] = {"A": "0", "B": "1", "C": "2", "D": "3"}


def _hand_to_kifu_string(hand: List[Any]) -> str:
    # 手札の順は配牌時の並びをそのまま採用（必要ならここで並べ替えも可能）
    return "".join(PIECE_KANJI.get(str(x), str(x)) for x in hand)


def _piece_to_kifu(v: Optional[str]) -> str:
    if v is None:
        return ""
    v = str(v)
    return PIECE_KANJI.get(v, v)


def _action_to_kifu_row(player: str, action: Tuple[str, Optional[str], Optional[str]]) -> List[str]:
    """あなたの例に合わせて ["pid","block","attack"] 形式に変換する"""
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
    """棋譜の見やすさのために
    - pass 行を削除
    - 同一プレイヤーの連続した「受け→攻め」を 1 行に結合する
      例: ["1","玉",""] + ["1","","香"] => ["1","玉","香"]
    """
    out: List[List[str]] = []
    for row in moves:
        if not row or len(row) < 3:
            continue
        pid, b, a = str(row[0]), str(row[1]), str(row[2])

        # pass は出力しない
        if b == "パス" or b.lower() == "pass":
            continue

        if out:
            lp, lb, la = out[-1]
            # 同じプレイヤーで、前が受けのみ・今が攻めのみ → 結合
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
    action: ActionModel


class CreateGameRequest(BaseModel):
    dealer: str = "A"
    # 人間席（最大4人）。未指定なら全員人間。
    human_seats: List[str] = Field(default_factory=lambda: ["A","B","C","D"])
    # デバッグ用：全手札公開を管理者のみ許可（admin_token 必須）
    allow_reveal: bool = True


class JoinRequest(BaseModel):
    seat: str
    name: Optional[str] = None


class MultiStepRequest(BaseModel):
    seat: str
    token: str
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


    # みんなの手札を公開するときは、場の伏せ駒（receive_hidden）も公開する
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


@app.post("/games")
def create_game(req: CreateGameRequest):
    dealer = req.dealer.upper()
    if dealer not in ALL_SEATS:
        raise HTTPException(status_code=400, detail="dealer must be A/B/C/D")

    human_seats = [s.upper() for s in (req.human_seats or [])]
    hs: List[str] = []
    for s in human_seats:
        if s in ALL_SEATS and s not in hs:
            hs.append(s)
    if not hs:
        hs = ["A", "B", "C", "D"]

    hands = create_random_hands_no_five_shi()
    state = GoitaState(hands=hands, dealer=dealer)

    agents: Dict[str, Optional[RuleBasedAgent]] = {}
    for seat in ALL_SEATS:
        if seat in hs:
            agents[seat] = None
        else:
            ag = RuleBasedAgent(name=f"CPU-{seat}")
            ag.bind_player(seat)
            agents[seat] = ag

    gid = str(uuid.uuid4())
    admin_token = str(uuid.uuid4())
    GAMES[gid] = {
        "state": state,
        "agents": agents,
        "human_seats": hs,
        "seat_token": {},  # seat -> token
        "seat_name": {s: f"プレイヤー{s}" for s in ALL_SEATS},
        "admin_token": admin_token,
        "allow_reveal": bool(req.allow_reveal),
        "log": [f"Game start. dealer={dealer}, humans={hs}"],
        "board": _new_board_snapshot(),
        "init_hands": hands,
        "dealer": dealer,
        "kifu_moves": [],
        "lock": threading.Lock(),
    }
    return {"game_id": gid, "dealer": dealer, "human_seats": hs, "admin_token": admin_token}


@app.post("/games/{game_id}/join")
def join_game(game_id: str, req: JoinRequest):
    game = GAMES.get(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="game not found")

    seat = req.seat.upper()
    if seat not in ALL_SEATS:
        raise HTTPException(status_code=400, detail="seat must be A/B/C/D")

    if seat not in game.get("human_seats", []):
        raise HTTPException(status_code=400, detail="this seat is not a human seat")

    if seat in game.get("seat_token", {}):
        raise HTTPException(status_code=409, detail="seat already taken")

    token = str(uuid.uuid4())
    game["seat_token"][seat] = token
    if req.name:
        game["seat_name"][seat] = req.name

    game.setdefault("log", []).append(f"[JOIN] seat={seat}, name={game['seat_name'][seat]}")
    return {"game_id": game_id, "seat": seat, "token": token, "name": game["seat_name"][seat]}

@app.get("/games/{game_id}/state")
def get_state(
    game_id: str,
    seat: str,
    token: Optional[str] = None,
    reveal_hands: int = 0,
    admin: Optional[str] = None,
):
    game = GAMES.get(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="game not found")

    seat = seat.upper()
    if seat not in ALL_SEATS:
        raise HTTPException(status_code=400, detail="seat must be A/B/C/D")

    authed = (token is not None and game.get("seat_token", {}).get(seat) == token)

    allow_reveal = bool(game.get("allow_reveal", True))
    reveal = False
    if allow_reveal and reveal_hands:
        if admin and admin == game.get("admin_token"):
            reveal = True

    state: GoitaState = game["state"]
    view = _state_public_view(
        state,
        viewer=seat,
        log=game.get("log", []),
        board_public=game.get("board", _new_board_snapshot()),
        reveal_hands=reveal,
    )
    view["seat_name"] = game.get("seat_name", {})
    view["human_seats"] = game.get("human_seats", [])
    view["you_are"] = {"seat": seat, "authed": authed}
    view["admin_reveal"] = reveal
    return view


@app.get("/games/{game_id}/legal_actions")
def get_legal_actions(game_id: str, seat: str, token: str):
    game = GAMES.get(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="game not found")

    seat = seat.upper()
    if seat not in ALL_SEATS:
        raise HTTPException(status_code=400, detail="seat must be A/B/C/D")

    if game.get("seat_token", {}).get(seat) != token:
        raise HTTPException(status_code=403, detail="invalid token")

    state: GoitaState = game["state"]
    if state.finished or state.turn != seat:
        return []
    return _actions_to_json(state.legal_actions(seat))


@app.post("/games/{game_id}/step")
def step(game_id: str, req: MultiStepRequest):
    game = GAMES.get(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="game not found")

    seat = req.seat.upper()
    if seat not in ALL_SEATS:
        raise HTTPException(status_code=400, detail="seat must be A/B/C/D")

    if seat in game.get("human_seats", []):
        if game.get("seat_token", {}).get(seat) != req.token:
            raise HTTPException(status_code=403, detail="invalid token")

    lock = game.get("lock") or threading.Lock()
    with lock:
        state: GoitaState = game["state"]
        agents: Dict[str, Optional[RuleBasedAgent]] = game["agents"]
        log: List[str] = game.setdefault("log", [])
        board = game.setdefault("board", _new_board_snapshot())

        if state.finished:
            return {"ok": True, "state": _state_public_view(state, viewer=seat, log=log, board_public=board)}

        if state.turn != seat:
            raise HTTPException(status_code=400, detail=f"not your turn (turn={state.turn})")

        action = req.action.to_tuple()

        before_fd = len(state.face_down_hidden[seat])
        try:
            _apply_action(state, seat, action)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid action: {e}")

        hidden_receive = _is_hidden_receive_by_state_delta(state, seat, action[0], before_fd)
        _update_board_snapshot(board, seat, action, hidden_receive=hidden_receive)

        log.append(_format_action(seat, action) + (" (hidden)" if hidden_receive else ""))
        game.setdefault("kifu_moves", []).append(_action_to_kifu_row(seat, action))
        _notify_public({k: v for k, v in agents.items() if v is not None}, state, seat, action)

        # CPU席を自動で進める（次の手番が人間になるまで）
        safety = 0
        while not state.finished:
            p = state.turn
            if p in game.get("human_seats", []):
                break

            ag = agents.get(p)
            if ag is None:
                break

            safety += 1
            if safety > 2000:
                raise HTTPException(status_code=500, detail="safety stop: too many cpu steps")

            acts = state.legal_actions(p)
            if not acts:
                log.append(f"{p}: no legal actions (stop)")
                break

            cpu_action = ag.select_action(state, p, acts)
            before_fd = len(state.face_down_hidden[p])
            _apply_action(state, p, cpu_action)

            hidden_receive = _is_hidden_receive_by_state_delta(state, p, cpu_action[0], before_fd)
            _update_board_snapshot(board, p, cpu_action, hidden_receive=hidden_receive)

            log.append(_format_action(p, cpu_action) + (" (hidden)" if hidden_receive else ""))
            game.setdefault("kifu_moves", []).append(_action_to_kifu_row(p, cpu_action))
            _notify_public({k: v for k, v in agents.items() if v is not None}, state, p, cpu_action)

        if state.finished:
            log.append(f"Game finished. winner={state.winner}, team_score={getattr(state, 'team_score', None)}")

        return {"ok": True, "state": _state_public_view(state, viewer=seat, log=log, board_public=board)}

@app.get("/games/{game_id}/kifu", response_class=PlainTextResponse)
def get_kifu_yaml(game_id: str):
    """棋譜を YAML 形式で返す"""
    game = GAMES.get(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="game not found")

    init_hands: Dict[str, List[Any]] = game.get("init_hands", {})
    dealer: str = game.get("dealer", "A")
    moves: List[List[str]] = _compress_kifu_moves(game.get("kifu_moves", []))

    # プレイヤー名（必要なら後でUI化できます）
    p0, p1, p2, p3 = "プレイヤーA", "プレイヤーB", "プレイヤーC", "プレイヤーD"

    # スコア（team_score があれば使う）
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

    # YAML を手書き生成（依存ライブラリ不要）
    lines: List[str] = []
    lines.append("version: 1.0")
    lines.append(f'p0: "{p0}"')
    lines.append(f'p1: "{p1}"')
    lines.append(f'p2: "{p2}"')
    lines.append(f'p3: "{p3}"')
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
        a = str(row[0]).replace('"', '\"')
        b = str(row[1]).replace('"', '\"')
        c = str(row[2]).replace('"', '\"')
        lines.append(f'    - ["{a}","{b}","{c}"]')
    return "\n".join(lines) + "\n"


@app.get("/games/{game_id}/kifu.json")
def get_kifu_json(game_id: str):
    """棋譜を JSON 形式で返す（デバッグ/拡張用）"""
    game = GAMES.get(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="game not found")

    init_hands: Dict[str, List[Any]] = game.get("init_hands", {})
    dealer: str = game.get("dealer", "A")
    moves: List[List[str]] = _compress_kifu_moves(game.get("kifu_moves", []))

    state: GoitaState = game["state"]
    # 棋譜の score は「開始時点」を表すため常に [0,0]
    score = [0, 0]

    return {
        "version": "1.0",
        "p0": "プレイヤーA",
        "p1": "プレイヤーB",
        "p2": "プレイヤーC",
        "p3": "プレイヤーD",
        "log": [
            {
                "hand": {
                    "p0": _hand_to_kifu_string(init_hands.get("A", [])),
                    "p1": _hand_to_kifu_string(init_hands.get("B", [])),
                    "p2": _hand_to_kifu_string(init_hands.get("C", [])),
                    "p3": _hand_to_kifu_string(init_hands.get("D", [])),
                },
                "uchidashi": int(PLAYER_IDX.get(dealer, "0")),
                "score": score,
                "game": moves,
            }
        ],
    }


@app.delete("/games/{game_id}")
def delete_game(game_id: str):
    if game_id in GAMES:
        del GAMES[game_id]
    return {"ok": True}

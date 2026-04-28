from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

from fastapi import FastAPI, HTTPException, Body, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from goita_ai2.state import GoitaState
from goita_ai2.rule_based import RuleBasedAgent
from goita_ai2.simulate import _notify_public
from goita_ai2.utils import create_random_hands

# ★定数・マッピング辞書は constants.py からインポートするように統一
from goita_ai2.constants import ALL_SEATS, PIECE_TOTALS, PIECE_KANJI, PLAYER_IDX

MAIN_GID = "main"
NAME_MAX_LEN = 9

# =========================================================
# WebSocket 管理
# =========================================================
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, game_id: str):
        await websocket.accept()
        if game_id not in self.active_connections:
            self.active_connections[game_id] = []
        self.active_connections[game_id].append(websocket)

    def disconnect(self, websocket: WebSocket, game_id: str):
        if game_id in self.active_connections:
            if websocket in self.active_connections[game_id]:
                self.active_connections[game_id].remove(websocket)

    async def broadcast_update(self, game_id: str):
        if game_id in self.active_connections:
            for connection in self.active_connections[game_id]:
                try:
                    await connection.send_json({"type": "update"})
                except:
                    pass

manager = ConnectionManager()
# =========================================================

def _validate_seat(s: str, *, name: str = "seat") -> str:
    s = (s or "").strip().upper()
    if s not in ALL_SEATS:
        raise HTTPException(status_code=400, detail=f"invalid {name}: {s} (must be A/B/C/D)")
    return s

def _normalize_hands(hands: Dict[str, List[Any]]) -> Dict[str, List[str]]:
    return {p: [str(x) for x in hands[p]] for p in ALL_SEATS}

def create_random_hands_no_five_shi(max_retry: int = 5000) -> Dict[str, List[str]]:
    for _ in range(max_retry):
        raw = create_random_hands()
        hands = _normalize_hands(raw)
        if all(sum(1 for x in hands[p] if x == "1") <= 4 for p in ALL_SEATS):
            return hands
    raise RuntimeError(f"Failed to generate valid hands after {max_retry} retries.")

def build_hands_from_preset_counts(
    preset: Dict[str, Dict[str, int]],
    dealer: str,
    max_retry: int = 8000,
) -> Dict[str, List[str]]:
    p = {seat: {k: int(v) for k, v in (preset.get(seat) or {}).items()} for seat in ALL_SEATS}

    used_total = {k: 0 for k in PIECE_TOTALS}
    for seat in ALL_SEATS:
        seat_sum = 0
        for k, maxn in PIECE_TOTALS.items():
            n = int(p[seat].get(k, 0) or 0)
            if n < 0:
                raise ValueError("negative count")
            if n > 9:
                raise ValueError("count must be 0-9")
            seat_sum += n
            used_total[k] += n

        if seat_sum > 8:
            raise ValueError(f"{seat}: total pieces must be <= 8")
        if int(p[seat].get("1", 0) or 0) > 4:
            raise ValueError(f"{seat}: '1'(し) must be <= 4")

    for k, maxn in PIECE_TOTALS.items():
        if used_total[k] > maxn:
            raise ValueError(f"total of {k} exceeds max ({used_total[k]} > {maxn})")

    pool: List[str] = []
    for k, maxn in PIECE_TOTALS.items():
        pool.extend([k] * (maxn - used_total[k]))

    for _ in range(max_retry):
        pool2 = pool[:]
        random.shuffle(pool2)

        hands: Dict[str, List[str]] = {s: [] for s in ALL_SEATS}
        shi_cnt: Dict[str, int] = {s: 0 for s in ALL_SEATS}

        for seat in ALL_SEATS:
            for k in sorted(PIECE_TOTALS.keys()):
                n = int(p[seat].get(k, 0) or 0)
                if n:
                    hands[seat].extend([k] * n)
                    if k == "1":
                        shi_cnt[seat] += n

        ok = True
        for seat in ALL_SEATS:
            need = 8 - len(hands[seat])
            if need <= 0:
                continue

            fixed_kinds = {k for k, v in p[seat].items() if int(v) > 0}

            for _i in range(need):
                found = False
                for j in range(len(pool2)):
                    k = pool2[j]
                    if k in fixed_kinds:
                        continue
                    if k == "1" and shi_cnt[seat] >= 4:
                        continue
                    hands[seat].append(k)
                    if k == "1":
                        shi_cnt[seat] += 1
                    pool2.pop(j)
                    found = True
                    break
                if not found:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        if all(sum(1 for x in hands[s] if x == "1") <= 4 for s in ALL_SEATS):
            return {s: [str(x) for x in hands[s]] for s in ALL_SEATS}

    raise ValueError("failed to build hands from preset")


def _sanitize_player_name(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("\r", "").replace("\n", "")
    if len(s) > 9:
        s = s[:9]
    return s


app = FastAPI(title="Goita FastAPI (Render-ready)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="frontend/index.html not found")
    return FileResponse(index_path)


@app.websocket("/ws/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str, client_id: str = ""):
    await manager.connect(websocket, game_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, game_id)
        if client_id:
            game = GAMES.get(game_id)
            if game:
                hs = game.get("human_seats", {})
                if isinstance(hs, dict):
                    removed = False
                    for s, cid in list(hs.items()):
                        if cid == client_id:
                            del hs[s]
                            removed = True
                    if removed:
                        await manager.broadcast_update(game_id)
                        await manager.broadcast_update("lobby")


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

class NameRequest(BaseModel):
    seat: str
    name: str = ""

class ResetConfigBody(BaseModel):
    dealer: str = Field(default="A")
    preset_counts: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    requester: str = Field(default="W") 

class SettingsUpdateRequest(BaseModel):
    admin_password: str
    new_owner_name: str
    update_password: bool = False
    new_password: Optional[str] = None
    enable_effects: bool = True 


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
    return {p: {"receive": [None] * 4, "attack": [None] * 4, "receive_hidden": [False] * 4} for p in ALL_SEATS}


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
    if action_type not in ("receive", "attack_after_block"):
        return False
    return len(state.face_down_hidden[player]) > before_len


def _state_public_view(
    state: GoitaState,
    *,
    viewer: str,
    game_obj: Dict[str, Any]
) -> Dict[str, Any]:
    
    log = game_obj.get("log", [])
    board_public = game_obj.get("board", _new_board_snapshot())
    reveal_hands = game_obj.get("reveal_hands", False)
    human_seats = game_obj.get("human_seats", {})
    player_names = game_obj.get("player_names", {p: "" for p in ALL_SEATS})
    owner_name = game_obj.get("owner_name", "")
    is_started = game_obj.get("is_started", False)

    hands_view: Dict[str, Any] = {}
    
    if not is_started:
        for p in ALL_SEATS:
            hands_view[p] = {"count": 0}
        board_view = _new_board_snapshot()
        turn = None
        phase = ""
        attacker = ""
        current_attack = None
        scores = {"A": 0, "B": 0, "C": 0, "D": 0}
        team_score = {"AC": 0, "BD": 0}
        finished = False
        winner = None
    else:
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
            
        turn = state.turn
        phase = state.phase
        attacker = state.attacker
        current_attack = state.current_attack
        scores = _build_scores(state)
        team_score = getattr(state, "team_score", None)
        finished = state.finished
        winner = state.winner

    payload = {
        "is_started": is_started,
        "turn": turn,
        "phase": phase,
        "attacker": attacker,
        "current_attack": current_attack,
        "hands": hands_view,
        "team_score": team_score,
        "scores": scores,
        "board_public": board_view,
        "log": log[-200:],
        "finished": finished,
        "winner": winner,
        "player_names": player_names,
        "reveal_hands": reveal_hands,
        "owner_name": owner_name,
    }
    if isinstance(human_seats, dict):
        payload["human_seats"] = sorted(list(human_seats.keys()))
    else:
        payload["human_seats"] = sorted(list(human_seats))
    return payload


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
        "log": [],
        "board": _new_board_snapshot(),
        "init_hands": hands,
        "dealer": dealer,
        "kifu_moves": [],
        "human_seats": {}, 
        "player_names": {p: "" for p in ALL_SEATS},
        "password": None,
        "admin_password": None,
        "owner_name": "",
        "reveal_hands": False,
        "is_started": False,
        "enable_effects": True,
    }


def _ensure_main_game(dealer: Optional[str] = None) -> None:
    if MAIN_GID not in GAMES:
        d = dealer if dealer else random.choice(["A", "B", "C", "D"])
        game = _create_game_obj(dealer=d)
        game["owner_name"] = "メインルームA"
        GAMES[MAIN_GID] = game

def setup_supporter_rooms():
    supporter_data = [
        {"gid": "room-gold-01", "pass": None, "admin": "admin-a", "owner": "プライベートA"},
        {"gid": "room-silver-02", "pass": "goita-ai", "admin": "admin-b", "owner": "プライベートB"},
    ]
    for data in supporter_data:
        if data["gid"] not in GAMES:
            d = random.choice(["A", "B", "C", "D"])
            room = _create_game_obj(dealer=d)
            room["password"] = data["pass"]
            room["admin_password"] = data["admin"]
            room["owner_name"] = data["owner"]
            GAMES[data["gid"]] = room

setup_supporter_rooms()


def _check_effects(state: GoitaState, player: str, action: Tuple[str, Optional[str], Optional[str]], board_public: Dict[str, Dict[str, Any]]) -> List[str]:
    effects = []
    action_type, block, attack = action
    
    hand_len = len(state.hands[player])
    
    is_agari = False
    next_hand_len = hand_len
    if action_type == "attack":
        next_hand_len = hand_len - 1
        if hand_len == 1:
            is_agari = True
    elif action_type == "attack_after_block":
        next_hand_len = hand_len - 2
        if hand_len == 2:
            is_agari = True
    
    if action_type in ("attack", "attack_after_block") and attack is not None:
        
        attack_count = sum(1 for x in board_public.get(player, {}).get("attack", []) if x is not None)
        
        if attack_count == 2 and attack == "1":
            effects.append("uchidome")
            
        if attack_count == 2 and next_hand_len == 2:
            effects.append("reach")
            
        # ★ 修正：かかりごたえ（親の1番目の4枚駒の攻めに、相方が1番目の攻めで同じ駒を出す）
        partner_of_dealer = {"A":"C", "C":"A", "B":"D", "D":"B"}.get(state.dealer)
        if player == partner_of_dealer and attack_count == 0:
            if attack in ("2", "3", "4", "5"):
                dealer_attacks = [x for x in board_public.get(state.dealer, {}).get("attack", []) if x is not None]
                if len(dealer_attacks) > 0 and dealer_attacks[0] == attack:
                    effects.append("kakarigotae")
            
        if is_agari:
            if action_type == "attack_after_block":
                if (block == "8" and attack == "9") or (block == "9" and attack == "8"):
                    effects.append("damadama_agari")
                elif block == attack:
                    effects.append("baizuke")
                elif attack in ("8", "9"):
                    effects.append("ou_agari")
            elif action_type == "attack":
                if attack in ("8", "9"):
                    effects.append("ou_agari")
        else:
            if attack in ("8", "9"):
                other = "9" if attack == "8" else "8"
                if other in state.hands[player]:
                    effects.append("damadama")
            
    return effects


@app.get("/games/list")
def list_rooms():
    _ensure_main_game()
    def build_room_info(gid: str, data: dict):
        hs = data.get("human_seats", {})
        pn = data.get("player_names", {})
        seats_info = {}
        for s in ALL_SEATS:
            is_human = s in hs
            name = pn.get(s, "").strip()
            if is_human:
                seats_info[s] = name if name else "人間"
            else:
                seats_info[s] = "AI"

        owner_name = "メインルームA" if gid == MAIN_GID else data.get("owner_name", "サポーター")
        return {
            "game_id": gid,
            "is_private": data.get("password") is not None,
            "owner_name": owner_name,
            "player_count": len(hs),
            "seats": seats_info
        }

    rooms = [build_room_info(MAIN_GID, GAMES[MAIN_GID])]
    for gid, data in GAMES.items():
        if gid != MAIN_GID:
            rooms.append(build_room_info(gid, data))
            
    return {"rooms": rooms}


@app.post("/games/{game_id}/verify_password")
def verify_password(game_id: str, password: str = Body(..., embed=True)):
    if game_id not in GAMES:
        raise HTTPException(status_code=404, detail="部屋が存在しません")
    required_pass = GAMES[game_id].get("password")
    if not required_pass or required_pass == password:
        return {"ok": True}
    raise HTTPException(status_code=401, detail="合言葉が違います")


# =========================================================
# ルーム設定 API
# =========================================================
@app.post("/games/{game_id}/verify_admin")
def verify_admin(game_id: str, password: str = Body(..., embed=True)):
    game = GAMES.get(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="game not found")
    if game.get("admin_password") == password:
        return {
            "ok": True, 
            "owner_name": game.get("owner_name", ""),
            "is_private": game.get("password") is not None,
            "enable_effects": game.get("enable_effects", True)
        }
    raise HTTPException(status_code=401, detail="管理用パスワードが違います")


@app.post("/games/{game_id}/update_settings")
async def update_settings(game_id: str, req: SettingsUpdateRequest):
    game = GAMES.get(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="game not found")
    if game.get("admin_password") != req.admin_password:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    game["owner_name"] = _sanitize_player_name(req.new_owner_name)
    game["enable_effects"] = req.enable_effects
    if req.update_password:
        game["password"] = req.new_password if req.new_password else None
    
    await manager.broadcast_update(game_id)
    await manager.broadcast_update("lobby")
    return {"ok": True}


# =========================================================
# ゲーム操作 API
# =========================================================

@app.post("/games/{game_id}/start")
async def start_game(game_id: str, requester: str = "W"):
    if requester != "A":
        raise HTTPException(status_code=403, detail="Only player in seat A can start.")
    if game_id == MAIN_GID:
        _ensure_main_game()
    game = GAMES.get(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="game not found")
    if game.get("is_started"):
        return {"ok": False, "detail": "Already started"}
    
    game["is_started"] = True
    dealer = game.get("dealer", "A")
    game["log"].append(f"Game start. dealer={dealer}")
    
    await manager.broadcast_update(game_id)
    return {"ok": True}


@app.post("/games/{game_id}/toggle_reveal_hands")
async def toggle_reveal_hands(game_id: str, requester: str = "W"):
    if requester != "A":
        raise HTTPException(status_code=403, detail="Only player in seat A can toggle hands.")
    if game_id == MAIN_GID:
        _ensure_main_game()
    game = GAMES.get(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="game not found")
    
    game["reveal_hands"] = not game.get("reveal_hands", False)
    await manager.broadcast_update(game_id)
    return {"ok": True, "reveal_hands": game["reveal_hands"]}


@app.post("/games/{game_id}/reset")
async def reset_game(game_id: str, dealer: str = "A", requester: str = "W"):
    if requester != "A":
        raise HTTPException(status_code=403, detail="Only player in seat A can reset the game.")
    if game_id == MAIN_GID:
        _ensure_main_game(dealer=dealer)
    elif game_id not in GAMES:
        raise HTTPException(status_code=404, detail="game not found")

    old_game = GAMES.get(game_id, {})
    password = old_game.get("password")
    admin_password = old_game.get("admin_password")
    owner_name = old_game.get("owner_name", "")
    human_seats = old_game.get("human_seats", {})
    player_names = old_game.get("player_names", {p: "" for p in ALL_SEATS})
    enable_effects = old_game.get("enable_effects", True)
    
    new_game = _create_game_obj(dealer=dealer)
    new_game["password"] = password
    new_game["admin_password"] = admin_password
    new_game["owner_name"] = owner_name
    new_game["human_seats"] = human_seats
    new_game["player_names"] = player_names
    new_game["reveal_hands"] = False 
    new_game["is_started"] = False
    new_game["enable_effects"] = enable_effects 
    
    GAMES[game_id] = new_game
    
    await manager.broadcast_update(game_id)
    await manager.broadcast_update("lobby")
    return {"ok": True, "game_id": game_id, "dealer": dealer}


@app.post("/games/{game_id}/reset_config")
async def reset_game_config(game_id: str, body: ResetConfigBody):
    if body.requester != "A":
        raise HTTPException(status_code=403, detail="Only player in seat A can reset the game configuration.")
    if game_id == MAIN_GID:
        _ensure_main_game()
    elif game_id not in GAMES:
        raise HTTPException(status_code=404, detail="game not found")

    dealer = _validate_seat(body.dealer, name="dealer")
    preset = body.preset_counts or {}
    old_game = GAMES.get(game_id, {})
    password = old_game.get("password")
    admin_password = old_game.get("admin_password")
    owner_name = old_game.get("owner_name", "")
    human_seats = old_game.get("human_seats", {})
    player_names = old_game.get("player_names", {p: "" for p in ALL_SEATS})
    enable_effects = old_game.get("enable_effects", True)

    if preset:
        try:
            hands = build_hands_from_preset_counts(preset, dealer=dealer)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        new_game = _create_game_obj(dealer=dealer)
        new_game["state"] = GoitaState(hands=hands, dealer=dealer)
        new_game["board"] = _new_board_snapshot()
        new_game["log"] = []
        new_game["init_hands"] = hands
        new_game["dealer"] = dealer
        new_game["kifu_moves"] = []
        new_game["password"] = password
        new_game["admin_password"] = admin_password
        new_game["owner_name"] = owner_name
        new_game["human_seats"] = human_seats
        new_game["player_names"] = player_names
        new_game["reveal_hands"] = False
        new_game["is_started"] = False
        new_game["enable_effects"] = enable_effects
        GAMES[game_id] = new_game
    else:
        new_game = _create_game_obj(dealer=dealer)
        new_game["password"] = password
        new_game["admin_password"] = admin_password
        new_game["owner_name"] = owner_name
        new_game["human_seats"] = human_seats
        new_game["player_names"] = player_names
        new_game["reveal_hands"] = False
        new_game["is_started"] = False
        new_game["enable_effects"] = enable_effects
        GAMES[game_id] = new_game

    await manager.broadcast_update(game_id)
    await manager.broadcast_update("lobby")
    return {"ok": True, "game_id": game_id, "dealer": dealer, "preset": bool(preset)}


@app.post("/games/{game_id}/claim")
async def claim_seat(game_id: str, seat: str, client_id: str = ""):
    if game_id == MAIN_GID:
        _ensure_main_game()
    elif game_id not in GAMES:
        raise HTTPException(status_code=404, detail="game not found")
    seat = _validate_seat(seat, name="seat")
    game = GAMES[game_id]
    
    hs = game.setdefault("human_seats", {})
    if isinstance(hs, dict):
        for k, v in list(hs.items()):
            if v == client_id:
                del hs[k]
        hs[seat] = client_id
    else:
        game["human_seats"] = {seat: client_id}
        hs = game["human_seats"]
        
    await manager.broadcast_update(game_id)
    await manager.broadcast_update("lobby")
    return {"ok": True, "game_id": game_id, "human_seats": sorted(list(hs.keys()))}


@app.post("/games/{game_id}/release")
async def release_seat(game_id: str, seat: str, client_id: str = ""):
    if game_id == MAIN_GID:
        _ensure_main_game()
    elif game_id not in GAMES:
        raise HTTPException(status_code=404, detail="game not found")
    seat = _validate_seat(seat, name="seat")
    game = GAMES[game_id]
    hs = game.setdefault("human_seats", {})
    if isinstance(hs, dict):
        if seat in hs and hs[seat] == client_id:
            del hs[seat]
    
    await manager.broadcast_update(game_id)
    await manager.broadcast_update("lobby")
    return {"ok": True, "game_id": game_id}


@app.post("/games/{game_id}/set_name")
async def set_player_name(game_id: str, req: NameRequest):
    if game_id == MAIN_GID:
        _ensure_main_game()
    game = GAMES.get(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="game not found")
    seat = _validate_seat(req.seat, name="seat")
    name = _sanitize_player_name(req.name)
    pn: Dict[str, str] = game.setdefault("player_names", {p: "" for p in ALL_SEATS})
    pn[seat] = name

    await manager.broadcast_update(game_id)
    await manager.broadcast_update("lobby")
    return {"ok": True, "game_id": game_id, "player_names": pn}


@app.post("/games/{game_id}/step")
async def step(game_id: str, req: StepRequest):
    if game_id == MAIN_GID:
        _ensure_main_game()
    player = _validate_seat(req.player, name="player")
    game = GAMES.get(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="game not found")
    if not game.get("is_started"):
        raise HTTPException(status_code=400, detail="Game not started")

    state: GoitaState = game["state"]
    agents: Dict[str, RuleBasedAgent] = game["agents"]
    log: List[str] = game.setdefault("log", [])
    board = game.setdefault("board", _new_board_snapshot())
    
    if state.finished:
        return {"ok": True, "state": _state_public_view(state, viewer=player, game_obj=game)}

    if state.turn != player:
        raise HTTPException(status_code=400, detail=f"not your turn (turn={state.turn}, you={player})")
    
    action = req.action.to_tuple()
    
    effects = []
    if game.get("enable_effects", True):
        effects = _check_effects(state, player, action, board)

    before_fd = len(state.face_down_hidden[player])
    try:
        _apply_action(state, player, action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid action: {e}")
        
    hidden_receive = _is_hidden_receive_by_state_delta(state, player, action[0], before_fd)
    _update_board_snapshot(board, player, action, hidden_receive=hidden_receive)
    
    log_str = _format_action(player, action) + (" (hidden)" if hidden_receive else "")
    for ef in effects:
        log_str += f" [EFFECT:{ef}]"
    log.append(log_str)
    
    game.setdefault("kifu_moves", []).append(_action_to_kifu_row(player, action))
    _notify_public(agents, state, player, action)

    await manager.broadcast_update(game_id)
    return {"ok": True, "state": _state_public_view(state, viewer=player, game_obj=game)}


@app.post("/games/{game_id}/cpu_step")
async def cpu_step(game_id: str):
    if game_id == MAIN_GID:
        _ensure_main_game()
    game = GAMES.get(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="game not found")
    if not game.get("is_started"):
        return {"status": "ignored"}

    state: GoitaState = game["state"]
    human_seats = game.get("human_seats", {})
    if state.finished or (state.turn in human_seats):
        return {"status": "ignored"}
        
    agents: Dict[str, RuleBasedAgent] = game["agents"]
    log: List[str] = game.setdefault("log", [])
    board = game.setdefault("board", _new_board_snapshot())
    p = state.turn
    acts = state.legal_actions(p)
    if not acts:
        return {"status": "no_legal_actions"}
        
    cpu_action = agents[p].select_action(state, p, acts)
    
    effects = []
    if game.get("enable_effects", True):
        effects = _check_effects(state, p, cpu_action, board)

    before_fd_cpu = len(state.face_down_hidden[p])
    _apply_action(state, p, cpu_action)
    hidden_receive_cpu = _is_hidden_receive_by_state_delta(state, p, cpu_action[0], before_fd_cpu)
    _update_board_snapshot(board, p, cpu_action, hidden_receive=hidden_receive_cpu)
    
    log_str = _format_action(p, cpu_action) + (" (hidden)" if hidden_receive_cpu else "")
    for ef in effects:
        log_str += f" [EFFECT:{ef}]"
    log.append(log_str)
    
    game.setdefault("kifu_moves", []).append(_action_to_kifu_row(p, cpu_action))
    _notify_public(agents, state, p, cpu_action)
    if state.finished:
        msg = f"Game finished. winner={state.winner}, team_score={getattr(state, 'team_score', None)}"
        if not log or log[-1] != msg:
            log.append(msg)

    await manager.broadcast_update(game_id)
    return {"status": "ok"}

# =========================================================

@app.get("/games/{game_id}/state")
def get_state(game_id: str, viewer: str = "A", reveal_hands: int = 0):
    if game_id == MAIN_GID:
        _ensure_main_game()
    viewer = _validate_seat(viewer, name="viewer")
    game = GAMES.get(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="game not found")
    
    game_copy = copy.copy(game)
    if reveal_hands:
        game_copy["reveal_hands"] = True
        
    return _state_public_view(game["state"], viewer=viewer, game_obj=game_copy)


@app.get("/games/{game_id}/legal_actions")
def get_legal_actions(game_id: str, player: str = "A"):
    if game_id == MAIN_GID:
        _ensure_main_game()
    player = _validate_seat(player, name="player")
    game = GAMES.get(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="game not found")
    if not game.get("is_started"):
        return []
    
    state: GoitaState = game["state"]
    if state.finished or state.turn != player:
        return []
    return _actions_to_json(state.legal_actions(player))


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
    lines: List[str] = ["version: 1.0", 'p0: "プレイヤーA"', 'p1: "プレイヤーB"', 'p2: "プレイヤーC"', 'p3: "プレイヤーD"', "log:", " - hand:", f'     p0: "{h["p0"]}"', f'     p1: "{h["p1"]}"', f'     p2: "{h["p2"]}"', f'     p3: "{h["p3"]}"', f"   uchidashi: {uchidashi}", f"   score: [{score[0]},{score[1]}]", "   game:"]
    for row in moves:
        a, b, c = str(row[0]).replace('"', '\\"'), str(row[1]).replace('"', '\\"'), str(row[2]).replace('"', '\\"')
        lines.append(f'    - ["{a}","{b}","{c}"]')
    return "\n".join(lines) + "\n"
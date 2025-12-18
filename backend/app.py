# backend/app.py
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# パッケージ構成:
#   repo_root/
#     backend/app.py
#     frontend/index.html
#     goita_ai2/state.py
#     goita_ai2/rule_based.py
from goita_ai2.state import GoitaState
from goita_ai2.rule_based import RuleBasedAgent

PLAYERS = ["A", "B", "C", "D"]


def deal_hands_no_five_ones(max_tries: int = 5000) -> Dict[str, List[str]]:
    """
    1(し)が1人に5枚以上配られないように手札を配る。
    ＝各プレイヤーの「1」の枚数が 0〜4 に収まるまでシャッフルし直す。
    """
    base_deck = [str(i) for i in range(1, 10) for _ in range(4)]  # 1〜9 を各4枚（計36枚）

    last_hands: Dict[str, List[str]] = {p: [] for p in PLAYERS}
    for _ in range(max_tries):
        deck = base_deck[:]
        random.shuffle(deck)

        # 8枚×4人=32枚配る（残り4枚は未使用）
        hands = {p: deck[i * 8:(i + 1) * 8] for i, p in enumerate(PLAYERS)}
        last_hands = hands

        if all(h.count("1") <= 4 for h in hands.values()):
            return hands

    # ほぼ到達しない想定だが、念のため最後の配りを返す
    return last_hands


def _find_repo_root() -> Path:
    # backend/app.py から見て1つ上が repo_root 想定
    return Path(__file__).resolve().parents[1]


def _load_frontend_html() -> str:
    root = _find_repo_root()
    html_path = root / "frontend" / "index.html"
    if not html_path.exists():
        raise FileNotFoundError(f"frontend/index.html not found at {html_path}")
    return html_path.read_text(encoding="utf-8")


@dataclass
class PublicBoard:
    receive: List[Optional[str]] = field(default_factory=lambda: [None, None, None, None])
    attack: List[Optional[str]] = field(default_factory=lambda: [None, None, None, None])
    receive_hidden: List[bool] = field(default_factory=lambda: [False, False, False, False])

    def place_receive(self, piece: str, hidden: bool) -> None:
        for i in range(4):
            if self.receive[i] is None:
                self.receive[i] = piece
                self.receive_hidden[i] = bool(hidden)
                return

    def place_attack(self, piece: str) -> None:
        for i in range(4):
            if self.attack[i] is None:
                self.attack[i] = piece
                return


@dataclass
class GameSession:
    game_id: str
    state: GoitaState
    # 盤面表示用（state.pyに盤面保存が無い前提で、Web表示だけのために保持）
    board_public: Dict[str, PublicBoard] = field(default_factory=lambda: {p: PublicBoard() for p in PLAYERS})
    log: List[str] = field(default_factory=list)
    agents: Dict[str, Any] = field(default_factory=dict)


def _action_to_str(player: str, action: Dict[str, Any]) -> str:
    t = action.get("action_type")
    b = action.get("block")
    a = action.get("attack")
    return f"{player}: {t} block={b} attack={a}"


def _apply_action(session: GameSession, player: str, action: Dict[str, Any]) -> None:
    """
    action = {"action_type": "...", "block": "...", "attack": "..."}
    """
    t = action.get("action_type")
    block = action.get("block")
    attack = action.get("attack")

    if t == "pass":
        session.state.apply_pass(player)
        session.log.append(_action_to_str(player, action))
        return

    if t == "receive":
        if not block:
            raise HTTPException(status_code=400, detail="receive requires block")
        session.state.apply_receive(player, block)
        # 受けは表で置く（今回の仕様：通常の受けは隠さない）
        session.board_public[player].place_receive(block, hidden=False)
        session.log.append(_action_to_str(player, action))
        return

    if t == "attack":
        if not attack:
            raise HTTPException(status_code=400, detail="attack requires attack")
        session.state.apply_attack(player, attack)
        session.board_public[player].place_attack(attack)
        session.log.append(_action_to_str(player, action))
        return

    if t == "attack_after_block":
        if not block or not attack:
            raise HTTPException(status_code=400, detail="attack_after_block requires block and attack")
        session.state.apply_attack_after_block(player, block, attack)
        # この block は「伏せ」なので受けエリアに hidden=True で置く
        session.board_public[player].place_receive(block, hidden=True)
        session.board_public[player].place_attack(attack)
        session.log.append(_action_to_str(player, action))
        return

    raise HTTPException(status_code=400, detail=f"unknown action_type: {t}")


def _legal_actions_as_dicts(state: GoitaState, player: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for (t, b, a) in state.legal_actions(player):
        out.append({"action_type": t, "block": b, "attack": a})
    return out


def _state_response(session: GameSession, reveal_hands: bool) -> Dict[str, Any]:
    st = session.state
    hands: Dict[str, List[str]] = {}
    for p in PLAYERS:
        if reveal_hands or p == "A":
            hands[p] = list(st.hands[p])
        else:
            hands[p] = ["□"] * len(st.hands[p])

    board_pub = {
        p: {
            "receive": session.board_public[p].receive,
            "attack": session.board_public[p].attack,
            "receive_hidden": session.board_public[p].receive_hidden,
        }
        for p in PLAYERS
    }

    return {
        "game_id": session.game_id,
        "turn": st.turn,
        "phase": st.phase,
        "attacker": st.attacker or "",
        "current_attack": st.current_attack,
        "finished": st.finished,
        "winner": st.winner,
        "scores": st.team_score,
        "hands": hands,
        "board_public": board_pub,
        "log": session.log,
    }


def _run_cpu_until_human(session: GameSession, max_steps: int = 200) -> None:
    """
    Aの手番になるまでCPU(B,C,D)を進める。
    """
    steps = 0
    while (not session.state.finished) and session.state.turn != "A" and steps < max_steps:
        steps += 1
        p = session.state.turn
        ag = session.agents.get(p)
        actions = session.state.legal_actions(p)
        if not actions:
            session.log.append(f"[WARN] no legal actions for {p}")
            break

        if ag is not None and hasattr(ag, "select_action"):
            t, b, a = ag.select_action(session.state, p, actions)
            action = {"action_type": t, "block": b, "attack": a}
        else:
            t, b, a = random.choice(actions)
            action = {"action_type": t, "block": b, "attack": a}

        _apply_action(session, p, action)


app = FastAPI()

# Render / Sites での動作を楽にする（同一オリジン前提だが、必要なら許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# メモリ上にゲーム保持
GAMES: Dict[str, GameSession] = {}


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    return HTMLResponse(_load_frontend_html())


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/games")
def create_game() -> Dict[str, Any]:
    import uuid

    hands = deal_hands_no_five_ones()

    # 親はA固定（必要ならランダムにしてもOK）
    dealer = "A"
    st = GoitaState(hands=hands, dealer=dealer)

    # CPUエージェント
    agents: Dict[str, Any] = {
        "B": RuleBasedAgent(name="CPU-B"),
        "C": RuleBasedAgent(name="CPU-C"),
        "D": RuleBasedAgent(name="CPU-D"),
    }
    # 席固定（追跡用）
    for seat, ag in agents.items():
        if hasattr(ag, "bind_player"):
            ag.bind_player(seat)

    gid = str(uuid.uuid4())
    sess = GameSession(game_id=gid, state=st, agents=agents)

    # 初手がAでattackフェーズ（伏せ→攻め）なので、CPUはまだ回さない
    GAMES[gid] = sess
    return {"game_id": gid}


@app.get("/games/{game_id}/state")
def get_state(game_id: str, reveal_hands: int = Query(0)) -> Dict[str, Any]:
    sess = GAMES.get(game_id)
    if not sess:
        raise HTTPException(status_code=404, detail="game not found")
    return _state_response(sess, reveal_hands=bool(reveal_hands))


@app.get("/games/{game_id}/legal_actions")
def get_legal_actions(game_id: str) -> List[Dict[str, Any]]:
    sess = GAMES.get(game_id)
    if not sess:
        raise HTTPException(status_code=404, detail="game not found")

    player = sess.state.turn
    actions = _legal_actions_as_dicts(sess.state, player)
    return actions


@app.post("/games/{game_id}/step")
def step(game_id: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    sess = GAMES.get(game_id)
    if not sess:
        raise HTTPException(status_code=404, detail="game not found")

    if sess.state.finished:
        return _state_response(sess, reveal_hands=False)

    action = payload.get("action")
    if not isinstance(action, dict):
        raise HTTPException(status_code=400, detail="payload must contain action dict")

    # 人間はA固定
    if sess.state.turn != "A":
        raise HTTPException(status_code=400, detail=f"not A's turn (turn={sess.state.turn})")

    # 合法手チェック
    legal = _legal_actions_as_dicts(sess.state, "A")
    if action not in legal:
        raise HTTPException(status_code=400, detail={"error": "illegal action", "legal": legal, "got": action})

    # 適用（人間）
    _apply_action(sess, "A", action)

    # CPUを回す
    _run_cpu_until_human(sess)

    return _state_response(sess, reveal_hands=False)

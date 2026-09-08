"""Owner-only, ephemeral score attack rooms (no public room links)."""
import asyncio
import re
import time
from contextlib import asynccontextmanager, suppress

from fastapi import HTTPException, Response
from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse

from backend.member_api import require_member_origin
from backend.trace_results import trace_lifespan

PREFIX = "score-"
IDLE_SECONDS = 15 * 60


def is_score_room(game_id):
    return str(game_id).startswith(PREFIX)


class ScoreRoomGuard:
    """Protect every HTTP and WebSocket route before existing game handlers."""
    def __init__(self, app, games, identity, expire):
        self.app, self.games, self.identity, self.expire = app, games, identity, expire

    async def __call__(self, scope, receive, send):
        match = re.fullmatch(r"/(games|ws|voice)/(score-[^/]+)(?:/(.*))?", scope.get("path", ""))
        if scope["type"] not in {"http", "websocket"} or not match:
            return await self.app(scope, receive, send)
        channel, game_id, action = match.groups()
        request = HTTPConnection(scope)
        try:
            if scope["type"] == "websocket":
                scheme = "https" if request.url.scheme == "wss" else "http"
                if request.headers.get("origin") != f"{scheme}://{request.url.netloc}":
                    raise HTTPException(403, "同じサイトから操作してください。")
            else:
                require_member_origin(request)
            owner, _ = self.identity(request, Response())
            game = self.games.get(game_id)
            if not game:
                raise HTTPException(410, "スコアアタックのルームは終了しました。トップページから入り直してください。")
            if game.get("score_owner") != owner:
                raise HTTPException(403, "本人専用のルームです。")
            if time.monotonic() - game["score_last_active"] >= IDLE_SECONDS:
                await self.expire(game_id)
                raise HTTPException(410, "15分間操作がなかったため、ルームを終了しました。")
            if channel != "games":
                if channel != "ws" or request.query_params.get("client_id") != game["human_seats"]["A"]:
                    raise HTTPException(403, "このルームでは利用できません。")
            else:
                method = scope.get("method")
                allowed = ((method == "GET" and (action in {"state", "legal_actions", "kifu"}
                            or re.fullmatch(r"trace_results/(latest|history|[^/]+(?:/original)?)", action or "")))
                           or (method == "POST" and (action in {"step", "cpu_step", "set_name", "trace_random_start", "score_reset", "score_activity"}
                            or re.fullmatch(r"trace_results/[^/]+/retry", action or ""))))
                if not allowed:
                    raise HTTPException(403, "スコアアタックではこの操作は利用できません。")
                # Polling and AI turns must never keep an unattended room alive.
                if method == "POST" and action != "cpu_step":
                    game["score_last_active"] = time.monotonic()
            async def private_send(message):
                if message["type"] == "http.response.start":
                    message["headers"] = [(key, value) for key, value in message.get("headers", [])
                                          if key.lower() != b"cache-control"] + [(b"cache-control", b"no-store")]
                await send(message)
            return await self.app(scope, receive, private_send)
        except HTTPException as error:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 4403})
            else:
                await JSONResponse({"detail": error.detail}, status_code=error.status_code,
                                   headers={"Cache-Control": "no-store"})(scope, receive, send)


@asynccontextmanager
async def score_lifespan(app):
    async def sweep():
        while True:
            await asyncio.sleep(15)
            await app.state.sweep_score_rooms()
    async with trace_lifespan(app):
        task = asyncio.create_task(sweep())
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

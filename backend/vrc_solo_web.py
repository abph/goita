"""Browser partner lifecycle for a VRChat-owned Goita table.

The VRChat lease alone owns the match. Browser heartbeats reserve C but never
renew that lease. This module never exposes the lease key or other players' hands.
"""
import time

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.vrc_solo_lease import LeaseError


RECONNECT_SECONDS = 120
ONLINE_SECONDS = 30


class WebHeartbeat(BaseModel):
    client_id: str = Field(min_length=1, max_length=128)


class VrcSoloWeb:
    def __init__(self, core):
        self.core = core
        self.clock = time.monotonic

    def configure(self, game, room, session, mode, previous=None):
        old = (previous or {}).get("vrc_web", {})
        web = dict(old) if old else {
            "enabled": mode == "coop", "client_id": "", "last_seen": 0,
            "departed": False, "takeover": False,
        }
        web.update(session=session or old.get("session", ""), closed=False)
        game["vrc_web"] = web
        if not web["enabled"]:
            return
        game["ai_seats"] = ["B", "D"]
        if web["client_id"]:
            game["human_seats"]["C"] = web["client_id"]
            for key in ("player_names", "player_tags", "player_icons"):
                game[key]["C"] = (previous or {}).get(key, {}).get("C", "")
        elif web["takeover"]:
            game["ai_seats"].append("C")
        else:
            game["player_names"]["C"] = "サイト参加待ち"
        # A newly claimed cooperative table waits for C and an explicit VRC START.
        if previous is None:
            game["is_started"] = False

    def live(self, game):
        web = game.get("vrc_web")
        if not web or web.get("closed"):
            return False
        try:
            self.core["VRC_SOLO_LEASES"].require(game["vrc_solo_room"], web["session"])
        except LeaseError:
            self.close(game)
            return False
        return True

    def close(self, game):
        if game.get("vrc_web"):
            game["vrc_web"]["closed"] = True
        game["is_started"] = False
        game["vrc_pending"] = None
        for agent in game.get("agents", {}).values():
            cancel = getattr(agent, "cancel_background_search", None)
            if callable(cancel):
                cancel()

    def public(self, game, client_id=""):
        web = game.get("vrc_web", {})
        enabled = bool(web.get("enabled"))
        active = self.live(game) if web else False
        owner = web.get("client_id", "")
        elapsed = max(0, self.clock() - web.get("last_seen", 0))
        departed = web.get("departed", False)
        remaining = max(0, int(RECONNECT_SECONDS - elapsed + 0.999)) if owner and not departed else 0
        return {
            "enabled": enabled, "active": active, "ended": bool(web.get("closed")),
            "waiting": active and not game.get("is_started", False),
            "joined": bool(owner),
            "owned": active and bool(owner) and owner == client_id and not departed,
            "online": active and bool(owner) and not departed and elapsed < ONLINE_SECONDS,
            "reconnect_seconds": remaining,
            "can_join": enabled and active and (
                (bool(owner) and owner == client_id) or
                (not owner and not game.get("is_started") and not web.get("takeover"))
            ),
            "can_takeover": enabled and active and not web.get("takeover") and (
                not owner or departed or elapsed >= RECONNECT_SECONDS
            ),
        }

    def require_browser(self, game, client_id, seat="C"):
        web = game.get("vrc_web", {})
        if not web.get("enabled") or not self.live(game):
            raise HTTPException(409, "VRChat側の利用が終了しています。")
        if seat != "C" or not client_id or client_id.startswith("vrc-solo:"):
            raise HTTPException(403, "サイトから操作できるのはC席だけです。")
        return web

    async def join(self, game_id, seat, client_id):
        async with self.core["_game_turn_lock"](game_id):
            game = self.core["GAMES"][game_id]
            web = self.require_browser(game, client_id, seat)
            if len(client_id) > 128:
                raise HTTPException(400, "Invalid client identifier")
            if web["client_id"] and web["client_id"] != client_id:
                raise HTTPException(409, "C席は使用中です。観戦できます。")
            if not web["client_id"] and (game.get("is_started") or web.get("takeover")):
                raise HTTPException(409, "新しい参加者は対局開始前に参加してください。")
            new_participant = not web["client_id"]
            web.update(client_id=client_id, last_seen=self.clock(), departed=False)
            game["human_seats"]["C"] = client_id
            if new_participant:
                game["player_names"]["C"] = "サイトのプレイヤー"
            self.core["manager"].cancel_disconnect_release(game_id, client_id)
            await self.core["manager"].broadcast_update(game_id)
            return {"ok": True, "human_seats": ["A", "C"], "ai_seats": ["B", "D"]}

    async def leave(self, game_id, seat, client_id):
        async with self.core["_game_turn_lock"](game_id):
            game = self.core["GAMES"][game_id]
            web = self.require_browser(game, client_id, seat)
            if web["client_id"] != client_id:
                raise HTTPException(403, "C席の本人だけが退出できます。")
            # Keep the reservation until A explicitly confirms replacement by AI.
            web["departed"] = True
            await self.core["manager"].broadcast_update(game_id)
            return {"ok": True}

    def install(self, app):
        @app.middleware("http")
        async def restrict_browser_controls(request, call_next):
            parts = request.url.path.strip("/").split("/")
            if (request.method in {"POST", "PUT", "PATCH", "DELETE"}
                    and len(parts) >= 3 and parts[0] == "games"
                    and parts[1].startswith("vrc-solo-")):
                action = "/".join(parts[2:])
                if action not in {"claim", "release", "step", "set_name", "chat", "chat/ask_ai"}:
                    return JSONResponse({"detail": "連携卓の設定・開始・AI交代はVRChat側で操作してください。"}, status_code=403)
            return await call_next(request)

        @app.post("/vrc/solo/{room_id}/web/heartbeat")
        async def heartbeat(room_id: str, body: WebHeartbeat):
            try:
                game_id, _ = self.core["_vrc_solo_ids"](room_id)
            except ValueError:
                raise HTTPException(400, "Invalid room")
            async with self.core["_game_turn_lock"](game_id):
                game = self.core["GAMES"].get(game_id)
                if not game:
                    return {"enabled": False, "active": False}
                web = game.get("vrc_web", {})
                if self.live(game) and web.get("client_id") == body.client_id and not web.get("departed"):
                    web["last_seen"] = self.clock()
                return self.public(game, body.client_id)

        @app.get("/vrc/solo/{room_id}/takeover")
        async def takeover(room_id: str, session: str = ""):
            try:
                game_id, _ = self.core["_vrc_solo_ids"](room_id)
            except ValueError:
                return self.core["_vrc_solo_error"]("invalid_room")
            async with self.core["_game_turn_lock"](game_id):
                error = self.core["_vrc_solo_require"](room_id, session)
                if error is not None:
                    return error
                game = self.core["GAMES"].get(game_id)
                if not game or not game.get("vrc_web", {}).get("enabled"):
                    return self.core["_vrc_solo_error"]("not_coop")
                web = game["vrc_web"]
                if not web.get("takeover"):
                    if not self.public(game)["can_takeover"]:
                        return self.core["_vrc_solo_error"]("partner_online", "C席の再接続を待っています。")
                    web.update(client_id="", departed=False, takeover=True)
                    game["human_seats"].pop("C", None)
                    game["ai_seats"] = ["B", "C", "D"]
                    game["player_names"]["C"] = "そろうAI C"
                    game["is_started"] = True
                # Idempotent retry: never apply C's turn twice.
                try:
                    await self.core["_vrc_solo_run_ai"](game)
                except (ValueError, RuntimeError) as error:
                    await self.core["manager"].broadcast_update(game_id)
                    return self.core["_vrc_solo_error"]("ai_error", str(error))
                await self.core["manager"].broadcast_update(game_id)
                return self.core["_vrc_solo_response"](self.core["_vrc_solo_snapshot"](game))

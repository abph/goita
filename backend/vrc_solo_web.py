"""Browser partner lifecycle for a VRChat-owned Goita table.

The VRChat lease alone owns the match. Browser heartbeats reserve B/C/D but never
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
            "enabled": mode in {"coop", "multi", "mixed"}, "multi": mode in {"multi", "mixed"}, "mixed": mode == "mixed", "client_id": "", "last_seen": 0,
            "departed": False, "takeover": False,
        }
        web.update(session=session or old.get("session", ""), closed=False)
        # C's fields remain at the top level for compatibility with 0.3 clients.
        web["seats"] = {seat: dict(old.get("seats", {}).get(seat) or {
            "client_id": "", "last_seen": 0, "departed": False, "takeover": False,
        }) for seat in ("B", "D")}
        game["vrc_web"] = web
        if not web["enabled"]:
            return
        game["ai_seats"] = [seat for seat in ("B", "C", "D") if seat not in self.supported(web)]
        for seat in self.supported(web):
            slot = self.slot(web, seat)
            if slot["client_id"]:
                game["human_seats"][seat] = slot["client_id"]
                for key in ("player_names", "player_tags", "player_icons"):
                    game[key][seat] = (previous or {}).get(key, {}).get(seat, "")
            elif slot["takeover"] or previous is not None:
                game["ai_seats"].append(seat)
                game["player_names"][seat] = "そろうAI " + seat
            else:
                game["player_names"][seat] = "サイト参加待ち"
        game["ai_seats"].sort()
        # A second, explicit VRChat START fills remaining open seats with AI.
        if previous is None:
            game["is_started"] = False

    @staticmethod
    def supported(web):
        return ("B", "C", "D") if web.get("multi") else ("C",)

    @staticmethod
    def slot(web, seat):
        return web if seat == "C" else web["seats"][seat]

    def owned_seats(self, game, client_id):
        web = game.get("vrc_web", {})
        if not client_id or client_id.startswith("vrc-solo:") or not web.get("enabled") or web.get("closed"):
            return set()
        return {seat for seat in self.supported(web)
                if self.slot(web, seat).get("client_id") == client_id and not self.slot(web, seat).get("departed")}

    def touch(self, game, client_id, seat=None):
        web = game.get("vrc_web", {})
        for owned in self.owned_seats(game, client_id):
            if seat is None or owned == seat:
                self.slot(web, owned)["last_seen"] = self.clock()

    def start_waiting(self, game):
        web = game["vrc_web"]
        seats = self.public(game)["seats"]
        if not web.get("multi") and not seats["C"]["joined"]:
            return False
        if any(row["joined"] and not row["online"] for row in seats.values()):
            return False
        for seat in self.supported(web):
            if not self.slot(web, seat)["client_id"]:
                if seat not in game["ai_seats"]:
                    game["ai_seats"].append(seat)
                game["player_names"][seat] = "そろうAI " + seat
        game["ai_seats"].sort()
        game["is_started"] = True
        return True

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
        rows = {}
        occupied_by_client = {seat for seat in self.supported(web)
                              if web.get("enabled") and self.slot(web, seat).get("client_id") == client_id and client_id}
        for seat in self.supported(web) if enabled else ():
            slot = self.slot(web, seat)
            owner = slot.get("client_id", "")
            elapsed = max(0, self.clock() - slot.get("last_seen", 0))
            departed = slot.get("departed", False)
            is_ai = seat in game.get("ai_seats", [])
            remaining = max(0, int(RECONNECT_SECONDS - elapsed + 0.999)) if owner and not departed else 0
            rows[seat] = {
                "joined": bool(owner), "owned": active and bool(owner) and owner == client_id and not departed and slot.get("transport") != "vrc",
                "online": active and bool(owner) and not departed and elapsed < ONLINE_SECONDS,
                "reconnect_seconds": remaining, "is_ai": is_ai,
                "source": "vrchat" if slot.get("transport") == "vrc" else "website" if owner else "ai" if is_ai else "open",
                "can_join": active and slot.get("transport") != "vrc" and not (occupied_by_client - {seat}) and (
                    (bool(owner) and owner == client_id) or
                    (not owner and not game.get("is_started") and not slot.get("takeover"))
                ),
                "can_takeover": active and not slot.get("takeover") and not is_ai and (
                    not owner or departed or elapsed >= RECONNECT_SECONDS
                ),
            }
        result = {
            "enabled": enabled, "active": active, "ended": bool(web.get("closed")),
            "multi": bool(web.get("multi")), "mixed": bool(web.get("mixed")), "seats": rows,
            "waiting": active and not game.get("is_started", False),
        }
        result.update(rows.get("C") or {"joined": False, "owned": False, "online": False,
                                       "can_join": False, "can_takeover": False, "reconnect_seconds": 0})
        return result

    def require_browser(self, game, client_id, seat="C"):
        web = game.get("vrc_web", {})
        if not web.get("enabled") or not self.live(game):
            raise HTTPException(409, "VRChat側の利用が終了しています。")
        if seat not in self.supported(web) or not client_id or client_id.startswith("vrc-solo:"):
            raise HTTPException(403, "サイトから操作できるのは参加可能なB・C・D席だけです。")
        if self.slot(web, seat).get("transport") == "vrc":
            raise HTTPException(409, "この席はVRChatから参加中です。")
        return self.slot(web, seat)

    async def join(self, game_id, seat, client_id):
        async with self.core["_game_turn_lock"](game_id):
            game = self.core["GAMES"][game_id]
            slot = self.require_browser(game, client_id, seat)
            web = game["vrc_web"]
            if len(client_id) > 128:
                raise HTTPException(400, "Invalid client identifier")
            if any(self.slot(web, other)["client_id"] == client_id for other in self.supported(web) if other != seat):
                raise HTTPException(409, "参加できるのは1人1席です。席を変更する場合は先に退出してください。")
            if slot["client_id"] and slot["client_id"] != client_id:
                raise HTTPException(409, seat + "席は使用中です。観戦できます。")
            if not slot["client_id"] and (game.get("is_started") or slot.get("takeover")):
                raise HTTPException(409, "新しい参加者は対局開始前に参加してください。")
            new_participant = not slot["client_id"]
            slot.update(client_id=client_id, last_seen=self.clock(), departed=False)
            game["human_seats"][seat] = client_id
            if new_participant:
                game["player_names"][seat] = "サイトのプレイヤー " + seat
            self.core["manager"].cancel_disconnect_release(game_id, client_id)
            await self.core["manager"].broadcast_update(game_id)
            return {"ok": True, "human_seats": sorted(game["human_seats"]), "ai_seats": sorted(game["ai_seats"])}

    async def leave(self, game_id, seat, client_id):
        async with self.core["_game_turn_lock"](game_id):
            game = self.core["GAMES"][game_id]
            slot = self.require_browser(game, client_id, seat)
            if slot["client_id"] != client_id:
                raise HTTPException(403, "その席の本人だけが退出できます。")
            if game["vrc_web"].get("multi") and not game.get("is_started"):
                # Before the first deal, leaving makes the seat available again.
                slot.update(client_id="", last_seen=0, departed=False)
                game["human_seats"].pop(seat, None)
                game["player_names"][seat] = "サイト参加待ち"
            else:
                slot["departed"] = True
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
                if self.live(game):
                    self.touch(game, body.client_id)
                return self.public(game, body.client_id)

        @app.get("/vrc/solo/{room_id}/takeover")
        async def takeover(room_id: str, session: str = "", seat: str = "C"):
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
                if seat not in self.supported(web):
                    return self.core["_vrc_solo_error"]("invalid_seat")
                slot = self.slot(web, seat)
                if not slot.get("takeover"):
                    if not self.public(game)["seats"][seat]["can_takeover"]:
                        return self.core["_vrc_solo_error"]("partner_online", seat + "席の再接続を待っています。")
                    slot.update(client_id="", departed=False, takeover=True, transport="")
                    game["human_seats"].pop(seat, None)
                    game["ai_seats"] = sorted(set(game["ai_seats"]) | {seat})
                    game["player_names"][seat] = "そろうAI " + seat
                    if not web.get("multi"):
                        game["is_started"] = True
                # Idempotent retry: never apply the replaced player's turn twice.
                try:
                    if game.get("is_started"):
                        await self.core["_vrc_solo_run_ai"](game)
                except (ValueError, RuntimeError) as error:
                    await self.core["manager"].broadcast_update(game_id)
                    return self.core["_vrc_solo_error"]("ai_error", str(error))
                await self.core["manager"].broadcast_update(game_id)
                return self.core["_vrc_solo_response"](self.core["_vrc_solo_snapshot"](game))

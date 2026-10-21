"""Seat reservations made by the instance's single serialized VRChat bridge.

The bridge checks the actual network-event caller; fixed URLs identify a seat,
not an authenticated person. Never accept these identifiers as browser owners.
"""


def seats(game):
    web = game.get("vrc_web", {})
    return {"A"} | {seat for seat in "BCD" if web.get("enabled")
                    and web_slot(web, seat).get("transport") == "vrc"
                    and web_slot(web, seat).get("client_id")}


def web_slot(web, seat):
    return web if seat == "C" else web.get("seats", {}).get(seat, {})


def may_act(game, seat):
    return seat in seats(game) and (seat == "A" or not web_slot(game["vrc_web"], seat).get("departed"))


class VrcSoloPlayers:
    def __init__(self, core):
        self.core = core

    def touch(self, game):
        web = game.get("vrc_web", {})
        for seat in seats(game) - {"A"}:
            slot = web_slot(web, seat)
            if not slot.get("departed"):
                slot["last_seen"] = self.core["VRC_WEB"].clock()

    def install(self, app):
        @app.get("/vrc/solo/{room_id}/player/{seat}/{command}")
        async def reserve(room_id: str, seat: str, command: str, session: str = ""):
            error_response = self.core["_vrc_solo_error"]
            if seat not in "BCD" or len(seat) != 1 or command not in {"join", "leave"}:
                return error_response("invalid_seat")
            try:
                game_id, _ = self.core["_vrc_solo_ids"](room_id)
            except ValueError:
                return error_response("invalid_room")
            async with self.core["_game_turn_lock"](game_id):
                error = self.core["_vrc_solo_require"](room_id, session)
                if error is not None:
                    return error
                game = self.core["GAMES"].get(game_id)
                if not game or not game.get("vrc_web", {}).get("mixed"):
                    return error_response("not_mixed")
                web = game["vrc_web"]
                slot = web_slot(web, seat)
                if command == "join":
                    if slot.get("client_id") and slot.get("transport") != "vrc":
                        return error_response("seat_busy", "この席はサイトで使用中です。")
                    if slot.get("transport") != "vrc":
                        if game.get("is_started") or slot.get("takeover"):
                            return error_response("already_started", "参加は対局開始前に行ってください。")
                        if len(seats(game)) >= 3:
                            return error_response("vrc_full", "VRChatから参加できるのは3人までです。")
                    elif slot.get("departed") and game.get("is_started"):
                        return error_response("already_started")
                    owner = "vrc-solo:" + room_id + ":" + seat
                    slot.update(client_id=owner, transport="vrc", departed=False, last_seen=self.core["VRC_WEB"].clock())
                    game["human_seats"][seat] = owner
                    game["player_names"][seat] = "VRChat Player " + seat
                elif slot.get("transport") == "vrc":
                    if game.get("is_started"):
                        slot["departed"] = True
                    else:
                        slot.update(client_id="", transport="", departed=False, last_seen=0)
                        game["human_seats"].pop(seat, None)
                        game["player_names"][seat] = "参加待ち"
                    game["vrc_pending"] = None
                await self.core["manager"].broadcast_update(game_id)
                return self.core["_vrc_solo_response"](self.core["_vrc_solo_snapshot"](game))

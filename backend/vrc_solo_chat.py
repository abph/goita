"""Room-only chat display and six public stamps for a leased VRChat table."""
import time
import unicodedata

STAMPS = ("greeting", "thanks", "thinking", "nice", "sorry", "surprised")
STAMP_COOLDOWN = 10.0


def clean(value, limit):
    text = "".join(" " if unicodedata.category(c).startswith("C") else c for c in str(value or ""))
    return " ".join(text.split())[:limit]


def display_lines(sender, message):
    prefix = sender + ": "
    prefix_width = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in prefix)
    line, width = prefix, prefix_width
    result = []
    for char in message:
        size = 2 if unicodedata.east_asian_width(char) in "WF" else 1
        if width + size > 36:
            result.append(line)
            line, width = prefix, prefix_width
        line += char
        width += size
    if line != prefix:
        result.append(line)
    return result


class VrcSoloChat:
    def __init__(self, core):
        self.core = core
        self.clock = time.monotonic

    def visible(self, game):
        web = game.get("vrc_web", {})
        if not web.get("enabled") or not self.core["VRC_WEB"].live(game):
            return False
        # Keep the panel during reconnect reservations and between rounds, but
        # spectators and AI seats never enable it.
        return any(self.core["VRC_WEB"].slot(web, seat).get("client_id")
                   and seat not in game.get("ai_seats", [])
                   for seat in self.core["VRC_WEB"].supported(web))

    def fields(self, game):
        show = self.visible(game)
        lines = []
        if show:
            for item in game.get("chat_messages", [])[-8:]:
                seat = item.get("seat", "W")
                sender = seat if seat in ("A", "B", "C", "D") else "観戦"
                message = item.get("message", "")
                if item.get("message_type") == "stamp":
                    message = self.core["CHAT_STAMPS"].get(item.get("stamp_id"), message)
                message = clean(message, 90)
                if message:
                    lines.extend(display_lines(sender, message))
        # Hex UTF-8 keeps user text out of G1's delimiters and Udon's parser.
        text = "\n".join(lines[-8:]).encode("utf-8").hex()
        return [f"chatshow={int(show)}", "chat=" + text]

    @staticmethod
    def carry(game, previous):
        if previous:
            game["chat_messages"] = list(previous.get("chat_messages", []))[-100:]
            if previous.get("vrc_stamp_last"):
                game["vrc_stamp_last"] = dict(previous["vrc_stamp_last"])

    def install(self, app):
        @app.get("/vrc/solo/{room_id}/chat")
        async def chat(room_id: str, session: str = ""):
            try:
                game_id, _ = self.core["_vrc_solo_ids"](room_id)
            except ValueError:
                return self.core["_vrc_solo_error"]("invalid_room")
            async with self.core["_game_turn_lock"](game_id):
                error = self.core["_vrc_solo_require"](room_id, session)
                if error is not None:
                    return error
                game = self.core["GAMES"].get(game_id)
                if game is None:
                    return self.core["_vrc_solo_error"]("not_started")
                return self.core["_vrc_solo_response"]("C1|" + "|".join(self.fields(game)))

        @app.get("/vrc/solo/{room_id}/stamp/{stamp_id}")
        async def stamp(room_id: str, stamp_id: str, session: str = ""):
            if stamp_id not in STAMPS:
                return self.core["_vrc_solo_error"]("invalid_stamp")
            try:
                game_id, _ = self.core["_vrc_solo_ids"](room_id)
            except ValueError:
                return self.core["_vrc_solo_error"]("invalid_room")
            async with self.core["_game_turn_lock"](game_id):
                error = self.core["_vrc_solo_require"](room_id, session)
                if error is not None:
                    return error
                game = self.core["GAMES"].get(game_id)
                if game is None:
                    return self.core["_vrc_solo_error"]("not_started")
                if not self.visible(game):
                    return self.core["_vrc_solo_error"]("chat_unavailable")
                now = self.clock()
                last = game.get("vrc_stamp_last")
                if last and now - last["at"] < STAMP_COOLDOWN:
                    # Retrying the same fixed URL is an acknowledgement, not a
                    # second chat item. Other stamps must wait out the cooldown.
                    if last["stamp"] == stamp_id:
                        return self.core["_vrc_solo_response"](last["response"])
                    return self.core["_vrc_solo_error"]("stamp_cooldown")
                label = self.core["CHAT_STAMPS"][stamp_id]
                item = {
                    "seat": "A", "sender": self.core["_chat_sender_label"](game, "A"),
                    "sender_name": self.core["_chat_sender_name"](game, "A"),
                    "tag": "", "sender_icon": "", "message": label,
                    "ts": self.core["_next_chat_timestamp"](), "message_type": "stamp",
                    "stamp_id": stamp_id, "origin": "room", "room_id": game_id,
                    "room_name": game.get("owner_name") or game_id,
                }
                messages = game.setdefault("chat_messages", [])
                messages.append(item)
                del messages[:-100]
                response = f"S1|stamp={stamp_id}|id={item['ts']}"
                game["vrc_stamp_last"] = {"stamp": stamp_id, "at": now, "response": response}
                await self.core["manager"].broadcast_update(game_id)
                return self.core["_vrc_solo_response"](response)

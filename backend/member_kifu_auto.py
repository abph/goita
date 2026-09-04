"""Opted-in saves for seated members still connected when a round finishes."""

import logging
import secrets

from backend.member_api import MEMBER_COOKIE
from backend.member_store import MemberError

LOGGER = logging.getLogger(__name__)


def save_connected_round(store, game, connections):
    if not getattr(game.get("state"), "finished", False) or not game.get("last_completed_kifu"):
        return []
    round_id = game.setdefault("member_kifu_round_id", secrets.token_hex(24))
    events = []
    for seat, client_id in dict(game.get("human_seats", {})).items():
        if not client_id or seat in game.get("ai_seats", []):
            continue
        for connection in list(connections.get(client_id, ())):
            # An unrelated page must not use a member cookie to trigger a save.
            scheme = "https" if connection.url.scheme == "wss" else "http"
            if connection.headers.get("origin") != f"{scheme}://{connection.url.netloc}":
                continue
            token = connection.cookies.get(MEMBER_COOKIE, "")
            if not token:
                continue
            member = None
            try:
                member = store.members.authenticate(token)
                if not store.members.kifu_auto_save(token):
                    continue
                result = store.save_automatic(token, round_id=round_id, seat=seat,
                                              payload=game["last_completed_kifu"])
            except MemberError as error:
                if member is None:
                    continue
                result = {"status": "unavailable", "member_id": member["member_id"]}
            except Exception:
                # Never log names, tokens, hands or kifu contents on failure.
                LOGGER.error("Automatic member kifu save failed")
                result = {"status": "error", "member_id": member["member_id"]} if member else None
            if result:
                events.append((connection, dict(result, type="member_kifu_auto", round_id=round_id)))
    return events


async def send_save_result(connection, event):
    try:
        await connection.send_json(event)
    except Exception:
        # Saving has already committed; a disconnected notification cannot undo it.
        pass

"""Private member library endpoints, including a room-entry proof for locked rooms."""

import hashlib
import hmac
import secrets
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import Field

from backend.member_api import MEMBER_COOKIE, MemberInput, PrivateRoute
from backend.research_kifu_store import RESEARCH_KIFU_TAGS, normalize_research_kifu_tags


_ROOM_SECRET = secrets.token_bytes(32)


def _room_cookie(game_id):
    return "goita_kifu_room_" + hashlib.sha256(game_id.encode()).hexdigest()[:16]


def _room_proof(game_id, password):
    return hmac.new(_ROOM_SECRET, (game_id + "\0" + str(password)).encode(), hashlib.sha256).hexdigest()


def grant_kifu_room_access(response, request, game_id, password):
    response.set_cookie(_room_cookie(game_id), _room_proof(game_id, password),
                        httponly=True, samesite="strict", secure=request.url.scheme == "https",
                        path="/api/member/kifu", max_age=86400)


def require_kifu_room_access(request, game_id, password):
    if password and not hmac.compare_digest(request.cookies.get(_room_cookie(game_id), ""), _room_proof(game_id, password)):
        raise HTTPException(403, "この部屋に合言葉で入室してから保存してください。")


class Metadata(MemberInput):
    my_seat: Literal["", "A", "B", "C", "D", "spectator"] | None = None
    title: str = Field(default="", max_length=80)
    memo: str = Field(default="", max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=len(RESEARCH_KIFU_TAGS))


class SaveInput(Metadata):
    game_id: str = Field(min_length=1, max_length=100)
    anonymous: bool = False


class ImportInput(Metadata):
    kifu_text: str = Field(min_length=1, max_length=200_000)


def _tags(tags):
    if any(tag not in RESEARCH_KIFU_TAGS for tag in tags):
        raise HTTPException(400, "使用できないタグです")
    return normalize_research_kifu_tags(tags)


def create_member_kifu_router(store, snapshot, parse, *, persistent=False):
    router = APIRouter(prefix="/api/member/kifu", route_class=PrivateRoute)

    def token(request):
        return request.cookies.get(MEMBER_COOKIE, "")

    @router.post("/list")
    def list_records(request: Request):
        return {"records": store.list(token(request)), "persistent": persistent, "limit": store.LIMIT}

    @router.post("/save")
    def save(request: Request, data: SaveInput):
        store.members.authenticate(token(request), require_paid=True)
        payload = snapshot(request, data.game_id, data.anonymous)
        payload["my_seat"] = data.my_seat or ""
        record = store.save(token(request), title=data.title.strip() or f"第{payload.get('round_index', 1)}局",
                            memo=data.memo.strip(), tags=_tags(data.tags), payload=payload)
        return {"record": record, "persistent": persistent}

    @router.post("/import")
    def import_record(request: Request, data: ImportInput):
        store.members.authenticate(token(request), require_paid=True)
        try:
            payload = parse(data.kifu_text)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        payload["my_seat"] = data.my_seat or ""
        record = store.save(token(request), title=data.title.strip() or "読込棋譜", memo=data.memo.strip(),
                            tags=_tags(data.tags), payload=payload)
        return {"record": record, "persistent": persistent}

    @router.post("/statistics")
    def statistics(request: Request):
        return {"statistics": store.statistics(token(request))}

    @router.post("/{record_id}")
    def get(request: Request, record_id: str):
        return {"record": store.access(token(request), record_id)}

    @router.post("/{record_id}/edit")
    def edit(request: Request, record_id: str, data: Metadata):
        return {"record": store.access(token(request), record_id, action="edit", title=data.title,
                                      memo=data.memo, tags=_tags(data.tags), my_seat=data.my_seat)}

    @router.post("/{record_id}/delete")
    def delete(request: Request, record_id: str):
        store.access(token(request), record_id, action="delete")
        return {"ok": True}

    return router

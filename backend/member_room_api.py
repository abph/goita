"""Research-plan room management, authorized from the current member session."""

from fastapi import APIRouter, Request
from pydantic import Field, StrictBool

from backend.member_api import MEMBER_COOKIE, MemberInput, PrivateRoute
from backend.member_store import MemberError


class RoomSettingsInput(MemberInput):
    game_id: str = Field(min_length=1, max_length=64)
    new_owner_name: str = Field(max_length=12)
    update_password: StrictBool = False
    new_password: str = Field(default="", max_length=128)
    ai_profile: str = Field(max_length=64)
    show_legal_actions: StrictBool = False
    show_log: StrictBool = False


class RoomSeatInput(MemberInput):
    game_id: str = Field(min_length=1, max_length=64)
    seat: str = Field(pattern="^[ABCD]$")
    occupancy_token: str = Field(min_length=1, max_length=128)


def create_member_room_router(store, room_options, read_room, save_room, vacate_seat):
    router = APIRouter(prefix="/api/member/room", route_class=PrivateRoute)

    def authorized_room(request, expected=None):
        member = store.authenticate(request.cookies.get(MEMBER_COOKIE, ""), require_paid=True)
        room_id = member["managed_room_id"]
        if not member["research_enabled"] or not room_id:
            raise MemberError(403, "管理する部屋が割り当てられていません。")
        if room_id not in {room["game_id"] for room in room_options()} or (expected is not None and expected != room_id):
            raise MemberError(403, "この部屋の管理権限がありません。マイページを開き直してください。")
        return room_id

    @router.get("")
    async def read(request: Request):
        return {"room": read_room(authorized_room(request))}

    @router.post("/settings")
    async def save(request: Request, data: RoomSettingsInput):
        room_id = authorized_room(request, data.game_id)
        if data.ai_profile not in read_room(room_id)["ai_profiles"]:
            raise MemberError(400, "AIを選択してください。")
        await save_room(room_id, data)
        authorized_room(request, room_id)
        return {"room": read_room(room_id)}

    @router.post("/vacate")
    async def vacate(request: Request, data: RoomSeatInput):
        room_id = authorized_room(request, data.game_id)
        await vacate_seat(room_id, data)
        authorized_room(request, room_id)
        return {"room": read_room(room_id)}

    return router

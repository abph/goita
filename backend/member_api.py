"""Same-origin member APIs. Member sessions never grant room/site administration."""

from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from backend.member_store import MemberError, MemberStore


MEMBER_COOKIE = "goita_member_session"


def require_member_origin(request):
    if request is None or request.headers.get("X-Goita-Member") != "1" or request.headers.get("sec-fetch-site") == "cross-site":
        raise HTTPException(403, "同じサイトから操作してください。")
    origin = request.headers.get("origin")
    if origin:
        source = urlsplit(origin)
        if source.scheme != request.url.scheme or source.netloc != request.url.netloc:
            raise HTTPException(403, "同じサイトから操作してください。")


class PrivateRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def private_handler(request):
            try:
                # The custom header prevents simple cross-origin form submissions.
                # Origin validation also protects reads despite the legacy CORS policy.
                require_member_origin(request)
                response = await handler(request)
            except MemberError as error:
                response = JSONResponse({"detail": str(error)}, status_code=error.status)
            except HTTPException as error:
                response = JSONResponse({"detail": error.detail}, status_code=error.status_code)
            except RequestValidationError:
                # Do not echo rejected credential fields in validation responses.
                response = JSONResponse({"detail": "入力内容を確認してください。"}, status_code=422)
            response.headers["Cache-Control"] = "no-store"
            response.headers["Pragma"] = "no-cache"
            return response

        return private_handler


class MemberInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoginInput(MemberInput):
    member_id: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class PasswordInput(MemberInput):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


class CreateInput(MemberInput):
    member_id: str = Field(min_length=4, max_length=32)
    paid_enabled: StrictBool = True
    paid_until: str | None = Field(default=None, max_length=10)


class UpdateInput(MemberInput):
    enabled: StrictBool
    paid_enabled: StrictBool
    paid_until: str | None = Field(default=None, max_length=10)


def create_member_router(store: MemberStore, require_admin, *, persistent=False, force_secure=False):
    router = APIRouter(route_class=PrivateRoute)

    def token(request):
        return request.cookies.get(MEMBER_COOKIE, "")

    def set_session(request, response, value, seconds):
        response.set_cookie(MEMBER_COOKIE, value, max_age=seconds, path="/", httponly=True,
                            secure=force_secure or request.url.scheme == "https", samesite="strict")

    @router.get("/api/member/session")
    def session(request: Request):
        if not token(request):
            return {"authenticated": False, "member": None}
        try:
            member = store.authenticate(token(request), allow_temporary=True)
            return {"authenticated": True, "member": member}
        except MemberError as error:
            if error.status != 401:
                raise
            return {"authenticated": False, "member": None}

    @router.post("/api/member/login")
    def login(request: Request, response: Response, data: LoginInput):
        member, value, seconds = store.login(data.member_id, data.password)
        # Replace the previous session on this device, without touching other members.
        store.logout(token(request))
        set_session(request, response, value, seconds)
        return {"authenticated": True, "member": member}

    @router.get("/api/member/me")
    def me(request: Request):
        return {"member": store.authenticate(token(request))}

    @router.post("/api/member/password")
    def password(request: Request, response: Response, data: PasswordInput):
        member, value, seconds = store.change_password(token(request), data.current_password, data.new_password)
        set_session(request, response, value, seconds)
        return {"authenticated": True, "member": member}

    @router.post("/api/member/logout")
    def logout(request: Request, response: Response):
        store.logout(token(request))
        response.delete_cookie(MEMBER_COOKIE, path="/", httponly=True,
                               secure=force_secure or request.url.scheme == "https", samesite="strict")
        return {"ok": True}

    @router.get("/admin/api/members")
    def members(request: Request):
        require_admin(request)
        return {"members": store.list_members(), "persistent": persistent}

    @router.post("/admin/api/members")
    def create(request: Request, data: CreateInput):
        require_admin(request)
        return store.create(**data.model_dump())

    @router.put("/admin/api/members/{member_id}")
    def update(request: Request, member_id: str, data: UpdateInput):
        require_admin(request)
        return {"member": store.update(member_id, **data.model_dump())}

    @router.post("/admin/api/members/{member_id}/reset-password")
    def reset(request: Request, member_id: str):
        require_admin(request)
        return store.reset_password(member_id)

    @router.delete("/admin/api/members/{member_id}")
    def delete(request: Request, member_id: str):
        require_admin(request)
        store.delete(member_id)
        return {"ok": True}

    return router

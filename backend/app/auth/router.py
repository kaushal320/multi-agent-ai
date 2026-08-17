from fastapi import APIRouter, Request, Response

from app.auth.schemas import LoginRequest, UserOut
from app.auth.service import delete_session, login_with_firebase_token
from app.core.config import settings

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=UserOut)
async def login(body: LoginRequest, response: Response):
    user, session_id = await login_with_firebase_token(body.token)

    response.set_cookie(
        key=settings.session_cookie_name,
        value=session_id,
        max_age=settings.session_ttl_seconds,
        httponly=True,
        secure=settings.environment != "development",
        samesite="none" if settings.environment != "development" else "lax",
    )

    return UserOut(
        id=str(user.id),
        firebase_uid=user.firebase_uid,
        name=user.name,
        email=user.email,
        avatar=user.avatar,
    )


@router.get("/logout")
async def logout(request: Request, response: Response):
    session_id = request.cookies.get(settings.session_cookie_name)
    if session_id:
        await delete_session(session_id)
    response.delete_cookie(settings.session_cookie_name)
    return {"message": "Logged out successfully"}

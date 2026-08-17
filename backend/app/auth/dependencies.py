from fastapi import HTTPException, Request, status

from app.auth.service import get_session_user
from app.core.config import settings


async def get_current_user(request: Request) -> dict:
    """Reads the session cookie, looks it up in Redis, and returns the cached user
    payload. Use as a route dependency: `user: dict = Depends(get_current_user)`."""
    session_id = request.cookies.get(settings.session_cookie_name)
    if not session_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    user = await get_session_user(session_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired")

    return user

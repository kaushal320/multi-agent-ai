import json
import secrets

from app.auth.schemas import UserOut
from app.core.config import settings
from app.core.firebase import verify_id_token
from app.core.redis_client import redis_client
from app.models.user import User


def _session_key(session_id: str) -> str:
    return f"session:{session_id}"


async def login_with_firebase_token(token: str) -> tuple[User, str]:
    """Verifies the Firebase ID token, creates the user in MongoDB if this is their
    first login, then creates a session in Redis. Returns (user, session_id)."""
    decoded = verify_id_token(token)
    firebase_uid = decoded["uid"]

    user = await User.find_one(User.firebase_uid == firebase_uid)
    if not user:
        user = User(
            firebase_uid=firebase_uid,
            name=decoded.get("name"),
            email=decoded.get("email"),
            avatar=decoded.get("picture"),
        )
        await user.insert()

    session_id = secrets.token_urlsafe(32)
    payload = UserOut(
        id=str(user.id),
        firebase_uid=user.firebase_uid,
        name=user.name,
        email=user.email,
        avatar=user.avatar,
    ).model_dump()

    await redis_client.set(
        _session_key(session_id),
        json.dumps(payload),
        ex=settings.session_ttl_seconds,
    )
    return user, session_id


async def get_session_user(session_id: str) -> dict | None:
    raw = await redis_client.get(_session_key(session_id))
    if not raw:
        return None
    return json.loads(raw)


async def delete_session(session_id: str) -> None:
    await redis_client.delete(_session_key(session_id))

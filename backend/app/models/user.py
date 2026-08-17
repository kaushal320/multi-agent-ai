from datetime import datetime, timezone

from beanie import Document, Indexed
from pydantic import Field


class User(Document):
    firebase_uid: Indexed(str, unique=True)
    name: str | None = None
    email: str | None = None
    avatar: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "users"

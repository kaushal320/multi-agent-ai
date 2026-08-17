from datetime import datetime, timezone
from typing import Optional

from beanie import Document, Indexed
from pydantic import Field


class User(Document):
    firebase_uid: Indexed(str, unique=True)
    name: Optional[str] = None
    email: Optional[str] = None
    avatar: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "users"

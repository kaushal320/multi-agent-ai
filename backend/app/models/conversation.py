from datetime import datetime, timezone

from beanie import Document
from pydantic import Field


class Conversation(Document):
    user_id: str
    title: str = "New Chat"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "conversations"

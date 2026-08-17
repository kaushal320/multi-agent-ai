from datetime import datetime, timezone
from typing import Literal

from beanie import Document, PydanticObjectId
from pydantic import Field


class Message(Document):
    conversation_id: PydanticObjectId
    role: Literal["user", "assistant"]
    content: str
    images: list[str] = Field(default_factory=list)
    token_usage: dict = Field(default_factory=dict)
    agent: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "messages"

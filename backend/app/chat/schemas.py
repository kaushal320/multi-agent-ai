from typing import Literal

from beanie import PydanticObjectId
from pydantic import BaseModel, Field


class UpdateConversationRequest(BaseModel):
    conversation_id: PydanticObjectId
    title: str = Field(..., max_length=200)


class SaveMessageRequest(BaseModel):
    conversation_id: PydanticObjectId
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=50000)
    images: list[str] = []

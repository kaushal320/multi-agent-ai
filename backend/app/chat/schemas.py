from typing import Literal

from beanie import PydanticObjectId
from pydantic import BaseModel


class UpdateConversationRequest(BaseModel):
    conversation_id: PydanticObjectId
    title: str


class SaveMessageRequest(BaseModel):
    conversation_id: PydanticObjectId
    role: Literal["user", "assistant"]
    content: str
    images: list[str] = []

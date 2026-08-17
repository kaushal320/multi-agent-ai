from datetime import datetime, timezone
from typing import Literal, Optional

from beanie import PydanticObjectId

from app.models.conversation import Conversation
from app.models.message import Message


async def _get_owned_conversation(
    user_id: str, conversation_id: PydanticObjectId
) -> Optional[Conversation]:
    return await Conversation.find_one(
        Conversation.id == conversation_id, Conversation.user_id == user_id
    )


async def create_conversation(user_id: str, title: str = "New Chat") -> Conversation:
    conversation = Conversation(user_id=user_id, title=title)
    await conversation.insert()
    return conversation


async def get_conversations(user_id: str) -> list[Conversation]:
    return (
        await Conversation.find(Conversation.user_id == user_id)
        .sort(-Conversation.updated_at)
        .to_list()
    )


async def update_conversation(
    user_id: str, conversation_id: PydanticObjectId, title: str
) -> Optional[Conversation]:
    conversation = await _get_owned_conversation(user_id, conversation_id)
    if not conversation:
        return None
    conversation.title = title
    conversation.updated_at = datetime.now(timezone.utc)
    await conversation.save()
    return conversation


async def save_message(
    user_id: str,
    conversation_id: PydanticObjectId,
    role: Literal["user", "assistant"],
    content: str,
    images: list[str] = [],
) -> Optional[Message]:
    conversation = await _get_owned_conversation(user_id, conversation_id)
    if not conversation:
        return None
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        images=images,
    )
    await message.insert()
    conversation.updated_at = datetime.now(timezone.utc)
    await conversation.save()
    return message


async def get_messages(
    user_id: str, conversation_id: PydanticObjectId
) -> Optional[list[Message]]:
    conversation = await _get_owned_conversation(user_id, conversation_id)
    if not conversation:
        return None
    return (
        await Message.find(Message.conversation_id == conversation_id)
        .sort(Message.created_at)
        .to_list()
    )

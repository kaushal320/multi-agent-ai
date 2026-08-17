from datetime import datetime, timezone
from typing import Literal

from beanie import PydanticObjectId

from app.models.conversation import Conversation
from app.models.message import Message


async def _get_owned_conversation(
    user_id: str, conversation_id: PydanticObjectId
) -> Conversation | None:
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
) -> Conversation | None:
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
    images: list[str] | None = None,
    token_usage: dict | None = None,
    agent: str = "",
) -> Message | None:
    conversation = await _get_owned_conversation(user_id, conversation_id)
    if not conversation:
        return None
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        images=images or [],
        token_usage=token_usage or {},
        agent=agent,
    )
    await message.insert()
    conversation.updated_at = datetime.now(timezone.utc)
    await conversation.save()
    return message


async def get_messages(
    user_id: str, conversation_id: PydanticObjectId
) -> list[Message] | None:
    conversation = await _get_owned_conversation(user_id, conversation_id)
    if not conversation:
        return None
    return (
        await Message.find(Message.conversation_id == conversation_id)
        .sort(Message.created_at)
        .to_list()
    )


async def get_user_token_usage(user_id: str, days: int = 30) -> dict:
    """Aggregate token usage for a user over the last N days."""
    from datetime import timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    messages = (
        await Message.find(
            Message.role == "assistant",
            Message.created_at >= cutoff,
        )
        .to_list()
    )

    total_input = 0
    total_output = 0
    total_tokens = 0
    by_agent = {}

    for msg in messages:
        usage = msg.token_usage or {}
        inp = usage.get("input_tokens", 0)
        out = usage.get("output_tokens", 0)
        tot = usage.get("total_tokens", 0)
        total_input += inp
        total_output += out
        total_tokens += tot

        agent = msg.agent or "unknown"
        if agent not in by_agent:
            by_agent[agent] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "requests": 0}
        by_agent[agent]["input_tokens"] += inp
        by_agent[agent]["output_tokens"] += out
        by_agent[agent]["total_tokens"] += tot
        by_agent[agent]["requests"] += 1

    return {
        "period_days": days,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_tokens,
        "by_agent": by_agent,
    }

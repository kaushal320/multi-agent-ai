import json
import logging

from pymongo.errors import PyMongoError

from app.core.redis_client import redis_client
from app.models.message import Message

logger = logging.getLogger("cortex.agents.memory")

MEMORY_TTL_SECONDS = 60 * 60 * 24
MAX_MEMORY_MESSAGES = 20


def _memory_key(conversation_id: str) -> str:
    return f"messages:{conversation_id}"


async def get_memory(conversation_id: str) -> list[dict]:
    raw = await redis_client.get(_memory_key(conversation_id))
    if raw:
        return json.loads(raw)

    # Fallback: load last N messages from MongoDB when Redis is cold
    try:
        from beanie import PydanticObjectId

        messages = (
            await Message.find(
                Message.conversation_id == PydanticObjectId(conversation_id)
            )
            .sort(-Message.created_at)
            .limit(MAX_MEMORY_MESSAGES)
            .to_list()
        )
        messages.reverse()
        return [{"role": m.role, "content": m.content} for m in messages]
    except PyMongoError as e:
        logger.debug("MongoDB fallback failed: %s", e)
        return []


async def add_message(conversation_id: str, role: str, content: str) -> None:
    messages = await get_memory(conversation_id)
    messages.append({"role": role, "content": content})
    messages = messages[-MAX_MEMORY_MESSAGES:]
    await redis_client.set(
        _memory_key(conversation_id),
        json.dumps(messages),
        ex=MEMORY_TTL_SECONDS,
    )

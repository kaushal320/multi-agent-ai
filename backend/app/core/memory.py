import json

from app.core.redis_client import redis_client

MEMORY_TTL_SECONDS = 60 * 60 * 24
MAX_MEMORY_MESSAGES = 20


def _memory_key(conversation_id: str) -> str:
    return f"messages:{conversation_id}"


async def get_memory(conversation_id: str) -> list[dict]:
    raw = await redis_client.get(_memory_key(conversation_id))
    if not raw:
        return []
    return json.loads(raw)


async def add_message(conversation_id: str, role: str, content: str) -> None:
    messages = await get_memory(conversation_id)
    messages.append({"role": role, "content": content})
    messages = messages[-MAX_MEMORY_MESSAGES:]
    await redis_client.set(
        _memory_key(conversation_id),
        json.dumps(messages),
        ex=MEMORY_TTL_SECONDS,
    )

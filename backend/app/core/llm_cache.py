import hashlib
import json
import logging

import redis

from app.core.redis_client import redis_client

logger = logging.getLogger("cortex.agents.cache")

CACHE_TTL_SECONDS = 60 * 60  # 1 hour
CACHE_KEY_PREFIX = "llm_cache:"


def _cache_key(model_name: str, messages: list) -> str:
    """Generate a deterministic cache key from model name + messages."""
    payload = json.dumps(
        {"model": model_name, "messages": messages}, sort_keys=True, default=str
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"{CACHE_KEY_PREFIX}{digest}"


async def get_cached_response(model_name: str, messages: list) -> dict | None:
    """Look up a cached LLM response. Returns None on miss."""
    try:
        key = _cache_key(model_name, messages)
        raw = await redis_client.get(key)
        if raw:
            logger.debug("Cache HIT for model=%s", model_name)
            return json.loads(raw)
        logger.debug("Cache MISS for model=%s", model_name)
    except redis.RedisError as e:
        logger.debug("Cache read error (treating as miss): %s", e)
    return None


async def set_cached_response(
    model_name: str, messages: list, response: str, ttl: int = CACHE_TTL_SECONDS
) -> None:
    """Store an LLM response in cache."""
    try:
        key = _cache_key(model_name, messages)
        await redis_client.set(key, json.dumps({"response": response}), ex=ttl)
    except redis.RedisError as e:
        logger.debug("Cache write error (non-fatal): %s", e)


async def invalidate_cache(prefix: str = CACHE_KEY_PREFIX) -> int:
    """Clear all cached LLM responses. Returns number of keys deleted."""
    try:
        keys = []
        async for key in redis_client.scan_iter(f"{prefix}*"):
            keys.append(key)
        if keys:
            return await redis_client.delete(*keys)
    except redis.RedisError as e:
        logger.debug("Cache invalidation error: %s", e)
    return 0

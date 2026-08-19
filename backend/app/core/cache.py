"""
Unified Caching Layer for Embeddings and LLM Responses

Uses Redis with TTL-based expiration. Free-tier compatible.
"""

import hashlib
import json
import logging
import time
from typing import Any

import redis.asyncio as redis

from app.core.config import settings
from app.core.observability import obs

logger = logging.getLogger("cortex.cache")

# Cache configuration
EMBEDDING_CACHE_TTL = 60 * 60 * 24 * 7  # 7 days (embeddings are stable)
LLM_CACHE_TTL = 60 * 60  # 1 hour
CACHE_KEY_PREFIX = "cortex:cache:"

# Redis client (reuse existing)
redis_client = redis.from_url(settings.redis_url, decode_responses=True)


def _make_key(prefix: str, *parts: str) -> str:
    """Generate a deterministic cache key."""
    payload = ":".join(parts)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"{CACHE_KEY_PREFIX}{prefix}:{digest}"


# ============================================================
# Embeddings Cache
# ============================================================

async def get_cached_embedding(text: str, model: str) -> list[float] | None:
    """Retrieve cached embedding vector."""
    key = _make_key("emb", model, text)
    start = time.perf_counter()

    try:
        raw = await redis_client.get(key)
        if raw:
            obs.metric("cache.embedding.hit", 1, model=model)
            logger.debug("Embedding cache HIT: model=%s, len=%d", model, len(text))
            return json.loads(raw)
        obs.metric("cache.embedding.miss", 1, model=model)
        logger.debug("Embedding cache MISS: model=%s, len=%d", model, len(text))
    except redis.RedisError as e:
        obs.metric("cache.embedding.error", 1, model=model)
        logger.debug("Embedding cache error (treating as miss): %s", e)

    return None


async def set_cached_embedding(text: str, model: str, vector: list[float], ttl: int = EMBEDDING_CACHE_TTL) -> None:
    """Store embedding vector in cache."""
    key = _make_key("emb", model, text)
    try:
        await redis_client.set(key, json.dumps(vector), ex=ttl)
        logger.debug("Embedding cached: model=%s, dim=%d", model, len(vector))
    except redis.RedisError as e:
        logger.debug("Embedding cache write error: %s", e)


# ============================================================
# LLM Response Cache (Enhanced)
# ============================================================

async def get_cached_llm_response(model: str, messages: list, **kwargs) -> str | None:
    """Retrieve cached LLM response with parameter awareness."""
    # Include relevant kwargs in cache key (temperature, max_tokens, etc.)
    param_str = json.dumps({k: v for k, v in kwargs.items() if k in ("temperature", "max_tokens", "top_p")}, sort_keys=True)
    msg_str = json.dumps(messages, sort_keys=True, default=str)
    key = _make_key("llm", model, hashlib.md5(msg_str.encode()).hexdigest(), param_str)

    try:
        raw = await redis_client.get(key)
        if raw:
            obs.metric("cache.llm.hit", 1, model=model)
            logger.debug("LLM cache HIT: model=%s", model)
            return json.loads(raw).get("response")
        obs.metric("cache.llm.miss", 1, model=model)
        logger.debug("LLM cache MISS: model=%s", model)
    except redis.RedisError as e:
        obs.metric("cache.llm.error", 1, model=model)
        logger.debug("LLM cache error: %s", e)

    return None


async def set_cached_llm_response(
    model: str,
    messages: list,
    response: str,
    ttl: int = LLM_CACHE_TTL,
    **kwargs
) -> None:
    """Store LLM response in cache with parameter awareness."""
    param_str = json.dumps({k: v for k, v in kwargs.items() if k in ("temperature", "max_tokens", "top_p")}, sort_keys=True)
    msg_str = json.dumps(messages, sort_keys=True, default=str)
    key = _make_key("llm", model, hashlib.md5(msg_str.encode()).hexdigest(), param_str)

    try:
        await redis_client.set(key, json.dumps({"response": response, "model": model}), ex=ttl)
        logger.debug("LLM response cached: model=%s", model)
    except redis.RedisError as e:
        logger.debug("LLM cache write error: %s", e)


# ============================================================
# Search Results Cache
# ============================================================

async def get_cached_search(query: str, engine: str = "tavily") -> list | None:
    """Retrieve cached web search results."""
    key = _make_key("search", engine, query.lower().strip())

    try:
        raw = await redis_client.get(key)
        if raw:
            obs.metric("cache.search.hit", 1, engine=engine)
            logger.debug("Search cache HIT: engine=%s, query=%s", engine, query[:50])
            return json.loads(raw)
        obs.metric("cache.search.miss", 1, engine=engine)
        logger.debug("Search cache MISS: engine=%s, query=%s", engine, query[:50])
    except redis.RedisError as e:
        obs.metric("cache.search.error", 1, engine=engine)
        logger.debug("Search cache error: %s", e)

    return None


async def set_cached_search(query: str, results: list, engine: str = "tavily", ttl: int = 60 * 30) -> None:
    """Cache web search results (shorter TTL for freshness)."""
    key = _make_key("search", engine, query.lower().strip())

    try:
        await redis_client.set(key, json.dumps(results), ex=ttl)
        logger.debug("Search results cached: engine=%s, count=%d", engine, len(results))
    except redis.RedisError as e:
        logger.debug("Search cache write error: %s", e)


# ============================================================
# RAG Context Cache
# ============================================================

def _rag_key(collection: str, query: str) -> str:
    """RAG cache key with the collection name in the clear (not hashed).

    Including the collection literally lets us invalidate all cached RAG
    context for a conversation when a new document is uploaded, which the
    fully-hashed key format used to make impossible.
    """
    digest = hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]
    return f"{CACHE_KEY_PREFIX}rag:{collection}:{digest}"


async def get_cached_rag_context(collection: str, query: str) -> str | None:
    """Retrieve cached RAG context for a query."""
    key = _rag_key(collection, query)

    try:
        raw = await redis_client.get(key)
        if raw:
            obs.metric("cache.rag.hit", 1, collection=collection)
            logger.debug("RAG cache HIT: collection=%s", collection)
            return raw
        obs.metric("cache.rag.miss", 1, collection=collection)
        logger.debug("RAG cache MISS: collection=%s", collection)
    except redis.RedisError as e:
        obs.metric("cache.rag.error", 1, collection=collection)
        logger.debug("RAG cache error: %s", e)

    return None


async def set_cached_rag_context(collection: str, query: str, context: str, ttl: int = 60 * 60) -> None:
    """Cache RAG context."""
    key = _rag_key(collection, query)

    try:
        await redis_client.set(key, context, ex=ttl)
        logger.debug("RAG context cached: collection=%s", collection)
    except redis.RedisError as e:
        logger.debug("RAG cache write error: %s", e)


async def invalidate_rag_cache(collection: str) -> int:
    """Invalidate all cached RAG context for a conversation's collection.

    Called when a new document is uploaded so the next question reflects the
    freshly indexed PDF instead of a previously cached answer.
    """
    return await invalidate_cache(f"rag:{collection}:*")


# ============================================================
# Cache Management
# ============================================================

async def invalidate_cache(pattern: str = "*") -> int:
    """Invalidate cache keys matching pattern."""
    try:
        keys = []
        async for key in redis_client.scan_iter(f"{CACHE_KEY_PREFIX}{pattern}"):
            keys.append(key)
        if keys:
            deleted = await redis_client.delete(*keys)
            obs.metric("cache.invalidated", deleted)
            logger.info("Cache invalidated: %d keys", deleted)
            return deleted
    except redis.RedisError as e:
        logger.error("Cache invalidation error: %s", e)
    return 0


async def get_cache_stats() -> dict:
    """Get cache statistics."""
    stats = {"keys": 0, "memory": "unknown"}
    try:
        keys = 0
        async for _ in redis_client.scan_iter(f"{CACHE_KEY_PREFIX}*"):
            keys += 1
        stats["keys"] = keys
        # Try to get memory info
        info = await redis_client.info("memory")
        stats["memory"] = info.get("used_memory_human", "unknown")
    except Exception as e:
        logger.debug("Cache stats error: %s", e)
    return stats


# ============================================================
# Backward Compatibility (for existing llm_cache.py usage)
# ============================================================

CACHE_TTL_SECONDS = LLM_CACHE_TTL
CACHE_KEY_PREFIX_LEGACY = "llm_cache:"


def _legacy_cache_key(model_name: str, messages: list) -> str:
    """Legacy cache key format for backward compatibility."""
    payload = json.dumps({"model": model_name, "messages": messages}, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"{CACHE_KEY_PREFIX_LEGACY}{digest}"


async def get_cached_response(model_name: str, messages: list) -> dict | None:
    """Backward compatible cache lookup."""
    key = _legacy_cache_key(model_name, messages)
    try:
        raw = await redis_client.get(key)
        if raw:
            logger.debug("Legacy cache HIT for model=%s", model_name)
            return json.loads(raw)
        logger.debug("Legacy cache MISS for model=%s", model_name)
    except redis.RedisError as e:
        logger.debug("Legacy cache read error: %s", e)
    return None


async def set_cached_response(model_name: str, messages: list, response: str, ttl: int = CACHE_TTL_SECONDS) -> None:
    """Backward compatible cache store."""
    key = _legacy_cache_key(model_name, messages)
    try:
        await redis_client.set(key, json.dumps({"response": response}), ex=ttl)
    except redis.RedisError as e:
        logger.debug("Legacy cache write error: %s", e)


async def invalidate_cache_legacy(prefix: str = CACHE_KEY_PREFIX_LEGACY) -> int:
    """Clear legacy cached LLM responses."""
    try:
        keys = []
        async for key in redis_client.scan_iter(f"{prefix}*"):
            keys.append(key)
        if keys:
            return await redis_client.delete(*keys)
    except redis.RedisError as e:
        logger.debug("Legacy cache invalidation error: %s", e)
    return 0
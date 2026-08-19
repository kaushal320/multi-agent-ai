import asyncio
import logging
import os
import re

from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from app.core.config import settings
from app.core.llm_cache import get_cached_response, set_cached_response

logger = logging.getLogger("cortex.agents.models")

GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
MAX_PROMPT_TOKENS = int(os.getenv("MAX_PROMPT_TOKENS", "4000"))
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_CACHE_ENABLED = os.getenv("LLM_CACHE_ENABLED", "true").lower() == "true"


class ThinkingBlockFilter:
    """Remove streamed <think> blocks, including tags split across chunks."""

    _open_tag = "<think>"
    _close_tag = "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._inside_thinking = False

    def feed(self, text: str) -> str:
        self._buffer += text
        output = []

        while self._buffer:
            if self._inside_thinking:
                end = self._buffer.find(self._close_tag)
                if end == -1:
                    self._buffer = self._buffer[-(len(self._close_tag) - 1) :]
                    break
                self._buffer = self._buffer[end + len(self._close_tag) :]
                self._inside_thinking = False
                continue

            start = self._buffer.find(self._open_tag)
            if start == -1:
                keep = len(self._open_tag) - 1
                if len(self._buffer) > keep:
                    output.append(self._buffer[:-keep])
                    self._buffer = self._buffer[-keep:]
                break

            output.append(self._buffer[:start])
            self._buffer = self._buffer[start + len(self._open_tag) :]
            self._inside_thinking = True

        return "".join(output)

    def finish(self) -> str:
        if self._inside_thinking:
            return ""
        remaining = self._buffer
        self._buffer = ""
        return remaining


class TokenBudgetExceeded(Exception):
    def __init__(self, estimated_tokens: int, limit: int):
        self.estimated_tokens = estimated_tokens
        self.limit = limit
        super().__init__(
            f"Prompt too large: ~{estimated_tokens} tokens (limit: {limit})"
        )


class LLMRetryError(Exception):
    def __init__(self, agent_name: str, attempts: int, last_error: str):
        self.agent_name = agent_name
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"LLM call failed after {attempts} attempts for '{agent_name}': {last_error}"
        )


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return len(text) // 4


def check_token_budget(prompt: str, extra_context: str = "") -> None:
    """Raise TokenBudgetExceeded if prompt + context would exceed the limit."""
    total = _estimate_tokens(prompt) + _estimate_tokens(extra_context)
    if total > MAX_PROMPT_TOKENS:
        raise TokenBudgetExceeded(total, MAX_PROMPT_TOKENS)


def _messages_to_cache_key(messages) -> list:
    """Convert langchain messages to a serializable format for caching."""
    key = []
    for msg in messages:
        if hasattr(msg, "content"):
            key.append(
                {"role": getattr(msg, "type", "unknown"), "content": msg.content}
            )
        elif isinstance(msg, dict):
            key.append(msg)
        else:
            key.append({"role": "unknown", "content": str(msg)})
    return key


def _extract_usage(response) -> dict:
    """Extract token usage from LLM response if available."""
    usage = {}
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        usage = {
            "input_tokens": getattr(response.usage_metadata, "input_tokens", 0) or 0,
            "output_tokens": getattr(response.usage_metadata, "output_tokens", 0) or 0,
            "total_tokens": getattr(response.usage_metadata, "total_tokens", 0) or 0,
        }
    elif hasattr(response, "response_metadata") and response.response_metadata:
        meta = response.response_metadata
        if "token_usage" in meta:
            tu = meta["token_usage"]
            usage = {
                "input_tokens": tu.get("prompt_tokens", 0),
                "output_tokens": tu.get("completion_tokens", 0),
                "total_tokens": tu.get("total_tokens", 0),
            }
    return usage


def _get_model_raw(agent_name: str):
    """Return the underlying LLM instance (no retry wrapper)."""
    if agent_name in ("chat", "search", "router"):
        return ChatGroq(
            model=GROQ_MODEL,
            api_key=settings.groq_api_key,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=0,
        )
    if agent_name == "coding":
        return ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            api_key=settings.google_api_key,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=0,
        )
    return _get_model_raw("chat")


def get_model(agent_name: str):
    """Return a model with retry logic and optional caching wrapping the raw LLM."""

    class RetryWrapper:
        def __init__(self, raw_model, name: str):
            self._raw = raw_model
            self._name = name

        async def ainvoke(self, messages, **kwargs):
            # Check cache for non-streaming calls (skip for router/tool-bound models)
            use_cache = LLM_CACHE_ENABLED and not kwargs.get("tools")
            if use_cache:
                cache_key = _messages_to_cache_key(messages)
                cached = await get_cached_response(self._name, cache_key)
                if cached:
                    return AIMessage(content=cached["response"])

            last_err = None
            for attempt in range(1, LLM_MAX_RETRIES + 1):
                try:
                    result = await self._raw.ainvoke(messages, **kwargs)
                    # Cache successful responses
                    if use_cache and hasattr(result, "content"):
                        cache_key = _messages_to_cache_key(messages)
                        await set_cached_response(self._name, cache_key, result.content)
                    return result
                except (ValueError, ConnectionError, TimeoutError, OSError) as e:
                    last_err = e
                    logger.warning(
                        "[%s] LLM call failed (attempt %d/%d): %s",
                        self._name,
                        attempt,
                        LLM_MAX_RETRIES,
                        type(e).__name__,
                    )
                    if attempt < LLM_MAX_RETRIES:
                        await asyncio.sleep(min(2**attempt, 8))
            raise LLMRetryError(self._name, LLM_MAX_RETRIES, str(last_err))

        async def astream(self, messages, **kwargs):
            last_err = None
            for attempt in range(1, LLM_MAX_RETRIES + 1):
                try:
                    async for chunk in self._raw.astream(messages, **kwargs):
                        yield chunk
                    return
                except (ValueError, ConnectionError, TimeoutError, OSError) as e:
                    last_err = e
                    logger.warning(
                        "[%s] LLM stream failed (attempt %d/%d): %s",
                        self._name,
                        attempt,
                        LLM_MAX_RETRIES,
                        type(e).__name__,
                    )
                    if attempt < LLM_MAX_RETRIES:
                        await asyncio.sleep(min(2**attempt, 8))
            raise LLMRetryError(self._name, LLM_MAX_RETRIES, str(last_err))

        def bind_tools(self, tools, **kwargs):
            bound = self._raw.bind_tools(tools, **kwargs)
            wrapper = RetryWrapper(bound, self._name)
            return wrapper

    return RetryWrapper(_get_model_raw(agent_name), agent_name)


def content_text(content) -> str:
    """Extract plain text from a model's response content, which may be a string
    (Groq) or a list of content parts (Gemini returns
    [{'type': 'text', 'text': '...'}, ...])."""
    if isinstance(content, str):
        return content
    parts = []
    for item in content or []:
        if isinstance(item, dict):
            parts.append(item.get("text") or item.get("content") or "")
        elif hasattr(item, "text"):
            parts.append(item.text or "")
    return re.sub(r"<think>.*?</think>\\s*", "", "".join(parts), flags=re.DOTALL)


# Structured routing and agent handoff models
from typing import Literal

from pydantic import BaseModel, Field


class AgentHandoff(BaseModel):
    """Structured handoff between agents."""

    from_agent: str
    to_agent: str
    payload: dict
    reason: str


class RouterDecision(BaseModel):
    """Structured output from the router agent."""

    agent: Literal[
        "chat", "search", "coding", "pdf", "ppt", "image", "rag", "research_rag"
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    handoff: AgentHandoff | None = None

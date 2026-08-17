import logging
import time
from typing import Any

from app.core.pii import redact_pii

logger = logging.getLogger("cortex.agents")


def log_agent_start(agent_name: str, state: dict[str, Any]) -> float:
    logger.info(
        "[%s] Agent started | conversation=%s | prompt=%.80s",
        agent_name,
        state.get("conversation_id", "unknown"),
        redact_pii(state.get("prompt", "")),
    )
    return time.perf_counter()


def log_agent_success(
    agent_name: str,
    state: dict[str, Any],
    t0: float,
    **extra: Any,
) -> None:
    duration_ms = round((time.perf_counter() - t0) * 1000, 1)
    detail = " | ".join(f"{k}={v}" for k, v in extra.items()) if extra else ""
    logger.info(
        "[%s] Agent completed | conversation=%s | duration_ms=%.1f%s",
        agent_name,
        state.get("conversation_id", "unknown"),
        duration_ms,
        f" | {detail}" if detail else "",
    )


def log_agent_failure(
    agent_name: str,
    state: dict[str, Any],
    exc: Exception,
) -> None:
    logger.exception(
        "[%s] Agent FAILED | conversation=%s | error=%s",
        agent_name,
        state.get("conversation_id", "unknown"),
        type(exc).__name__,
    )

import time
from typing import Any

from app.core.observability import obs
from app.core.pii import redact_pii


def log_agent_start(agent_name: str, state: dict[str, Any]) -> float:
    """Log agent start, return timestamp for duration calculation."""
    return obs.agent_start(
        agent_name=agent_name,
        conversation_id=state.get("conversation_id", "unknown"),
        prompt=state.get("prompt", ""),
    )


def log_agent_success(
    agent_name: str,
    state: dict[str, Any],
    t0: float,
    **extra: Any,
) -> None:
    """Log agent completion with duration."""
    obs.agent_complete(
        agent_name=agent_name,
        conversation_id=state.get("conversation_id", "unknown"),
        start_time=t0,
        **extra,
    )


def log_agent_failure(
    agent_name: str,
    state: dict[str, Any],
    exc: Exception,
) -> None:
    """Log agent failure with error details."""
    obs.agent_failure(
        agent_name=agent_name,
        conversation_id=state.get("conversation_id", "unknown"),
        error=exc,
    )

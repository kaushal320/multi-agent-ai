"""
Unified Observability Module

Centralizes all logging, tracing, and metrics.
Primary backend: Logfire (when configured)
Fallback: Python stdlib logging

Usage:
    from app.core.observability import obs

    # Structured logging
    obs.info("Agent started", agent="search", conversation_id="abc123")

    # Spans for distributed tracing
    with obs.span("agent_execution", agent="rag", conversation_id="abc123"):
        result = await do_work()

    # Metrics
    obs.metric("tokens_used", 1500, agent="chat", model="gpt-4")

    # Errors
    obs.error("Agent failed", agent="coding", error=exc, conversation_id="abc123")
"""

import logging
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any

from app.core.config import settings
from app.core.pii import redact_pii

# Try to import logfire
try:
    import logfire

    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False

# Stdlib logger fallback
_logger = logging.getLogger("cortex.observability")


class _NoOpSpan:
    """No-op context manager for when Logfire is unavailable."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _NoOpMetric:
    """No-op metric recorder."""

    def __call__(self, *args, **kwargs):
        pass


class Observability:
    """
    Unified observability interface.

    Automatically uses Logfire when available and configured,
    otherwise falls back to structured stdlib logging.
    """

    def __init__(self):
        self._initialized = False

    def configure(self) -> None:
        """Configure Logfire if available and token provided."""
        if self._initialized:
            return

        if LOGFIRE_AVAILABLE and settings.logfire_token:
            logfire.configure(
                service_name="cortex-ai-backend",
                token=settings.logfire_token,
                advanced=logfire.AdvancedOptions(base_url=settings.logfire_base_url),
                send_to_logfire="if-token-present",
            )
            # Auto-instrument common libraries
            try:
                logfire.instrument_fastapi()
                logfire.instrument_httpx()
                logfire.instrument_pydantic()
            except Exception:
                pass
        self._initialized = True

    # ============================================================
    # Logging (structured, with PII redaction)
    # ============================================================

    def _log(
        self,
        level: str,
        msg: str,
        *,
        pii_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Internal: log with automatic PII redaction."""
        # Redact PII from kwargs
        safe_kwargs = {}
        for k, v in kwargs.items():
            if pii_fields and k in pii_fields and isinstance(v, str):
                safe_kwargs[k] = redact_pii(v)
            else:
                safe_kwargs[k] = v

        if LOGFIRE_AVAILABLE and self._initialized:
            getattr(logfire, level)(msg, **safe_kwargs)
        else:
            # Stdlib fallback with structured format
            extra_str = " | ".join(f"{k}={v}" for k, v in safe_kwargs.items())
            full_msg = f"{msg} | {extra_str}" if extra_str else msg
            getattr(_logger, level)(full_msg)

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._log("debug", msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        self._log("info", msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._log("warning", msg, **kwargs)

    def warn(self, msg: str, **kwargs: Any) -> None:
        self.warning(msg, **kwargs)

    def error(self, msg: str, *, error: Exception | None = None, **kwargs: Any) -> None:
        if error:
            kwargs["error_type"] = type(error).__name__
            kwargs["error_message"] = str(error)
        self._log("error", msg, **kwargs)

    def exception(self, msg: str, *, error: Exception | None = None, **kwargs: Any) -> None:
        if error:
            kwargs["error_type"] = type(error).__name__
            kwargs["error_message"] = str(error)
        if LOGFIRE_AVAILABLE and self._initialized:
            logfire.exception(msg, **kwargs)
        else:
            _logger.exception(msg, extra=kwargs)

    # ============================================================
    # Spans (distributed tracing)
    # ============================================================

    @contextmanager
    def span(self, name: str, **attributes: Any):
        """Create a tracing span. Use as context manager."""
        if LOGFIRE_AVAILABLE and self._initialized:
            with logfire.span(name, **attributes) as span:
                yield span
        else:
            # Stdlib fallback: log start/end
            start = time.perf_counter()
            self.debug(f"SPAN START: {name}", **attributes)
            try:
                yield _NoOpSpan()
            finally:
                duration_ms = round((time.perf_counter() - start) * 1000, 1)
                self.debug(f"SPAN END: {name}", duration_ms=duration_ms, **attributes)

    @asynccontextmanager
    async def async_span(self, name: str, **attributes: Any):
        """Async version of span (same implementation)."""
        with self.span(name, **attributes) as span:
            yield span

    # ============================================================
    # Metrics
    # ============================================================

    def metric(self, name: str, value: float, **attributes: Any) -> None:
        """Record a metric value."""
        if LOGFIRE_AVAILABLE and self._initialized:
            # Logfire uses info-level for metrics; can also use logfire.metric if available
            logfire.info(f"METRIC: {name}", metric_name=name, metric_value=value, **attributes)
        else:
            self.debug(f"METRIC: {name}", metric_value=value, **attributes)

    def counter(self, name: str, increment: float = 1, **attributes: Any) -> None:
        """Increment a counter metric."""
        self.metric(name, increment, metric_type="counter", **attributes)

    def histogram(self, name: str, value: float, **attributes: Any) -> None:
        """Record a histogram value (e.g., latency)."""
        self.metric(name, value, metric_type="histogram", **attributes)

    # ============================================================
    # High-level agent-specific helpers
    # ============================================================

    def agent_start(
        self,
        agent_name: str,
        conversation_id: str,
        prompt: str,
        **extra: Any,
    ) -> float:
        """Log agent start, return timestamp for duration calculation."""
        self.info(
            "Agent started",
            agent=agent_name,
            conversation_id=conversation_id,
            prompt=redact_pii(prompt),
            **extra,
        )
        return time.perf_counter()

    def agent_complete(
        self,
        agent_name: str,
        conversation_id: str,
        start_time: float,
        **extra: Any,
    ) -> None:
        """Log agent completion with duration."""
        duration_ms = round((time.perf_counter() - start_time) * 1000, 1)
        self.info(
            "Agent completed",
            agent=agent_name,
            conversation_id=conversation_id,
            duration_ms=duration_ms,
            **extra,
        )
        # Also record as histogram metric
        self.histogram(f"agent.{agent_name}.duration_ms", duration_ms)

    def agent_failure(
        self,
        agent_name: str,
        conversation_id: str,
        error: Exception,
        **extra: Any,
    ) -> None:
        """Log agent failure with error details."""
        self.error(
            "Agent failed",
            agent=agent_name,
            conversation_id=conversation_id,
            error=error,
            **extra,
        )
        self.counter(f"agent.{agent_name}.errors", 1)

    def router_decision(
        self,
        agent: str,
        confidence: float,
        reasoning: str,
        conversation_id: str,
        prompt: str,
    ) -> None:
        """Log router classification decision."""
        self.info(
            "Router decision",
            agent=agent,
            confidence=confidence,
            reasoning=reasoning,
            conversation_id=conversation_id,
            prompt=redact_pii(prompt),
        )
        self.metric("router.confidence", confidence, agent=agent)

    def guardrails_check(
        self,
        check_type: str,  # "input" or "output"
        passed: bool,
        conversation_id: str,
        prompt: str,
        violation_category: str | None = None,
    ) -> None:
        """Log guardrails check result."""
        if passed:
            self.info(
                f"Guardrails {check_type} check PASSED",
                conversation_id=conversation_id,
                prompt_length=len(prompt),
            )
        else:
            self.warning(
                f"Guardrails {check_type} VIOLATION",
                category=violation_category,
                conversation_id=conversation_id,
                prompt=redact_pii(prompt),
            )
        self.counter(f"guardrails.{check_type}.{'pass' if passed else 'fail'}", 1)

    def stream_finished(
        self,
        agent: str,
        conversation_id: str,
        response_length: int,
        token_usage: dict | None = None,
    ) -> None:
        """Log stream completion with token usage."""
        self.info(
            "Stream finished",
            agent=agent,
            conversation_id=conversation_id,
            response_length=response_length,
        )
        if token_usage:
            self.metric("tokens.input", token_usage.get("input_tokens", 0))
            self.metric("tokens.output", token_usage.get("output_tokens", 0))
            self.metric("tokens.total", token_usage.get("total_tokens", 0))

    def hybrid_search(
        self,
        conversation_id: str,
        vector_results: int,
        bm25_results: int,
        fused_results: int,
    ) -> None:
        """Log hybrid search (RAG) results."""
        self.info(
            "Hybrid search completed",
            conversation_id=conversation_id,
            vector_results=vector_results,
            bm25_results=bm25_results,
            fused_results=fused_results,
        )

    def parallel_execution(
        self,
        conversation_id: str,
        tasks: list[str],
    ) -> None:
        """Log parallel task execution."""
        self.info(
            "Parallel execution started",
            conversation_id=conversation_id,
            tasks=tasks,
        )


# Singleton instance
obs = Observability()

# Auto-configure on import
obs.configure()

# Convenience exports
__all__ = ["obs", "Observability"]
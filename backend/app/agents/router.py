import json
import logging
import uuid

from fastapi import Annotated, APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agents.chat_agent import chat_node_stream
from app.agents.coding_agent import coding_node_stream
from app.agents.graph import graph
from app.agents.image_agent import image_node
from app.agents.models import (
    ThinkingBlockFilter,
    TokenBudgetExceeded,
    check_token_budget,
)
from app.agents.pdf_agent import pdf_node
from app.agents.ppt_agent import ppt_node
from app.agents.rag_agent import rag_research_node
from app.agents.router_agent import router_node
from app.agents.search_agent import search_node
from app.auth.dependencies import get_current_user
from app.chat import service as chat_service
from app.core import guardrails, memory
from app.core.pii import redact_pii

logger = logging.getLogger("cortex.agents.router")

try:
    import logfire

    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])

STREAMED_AGENTS = {"chat", "search", "coding", "rag"}

NON_STREAMED_NODES = {
    "pdf": pdf_node,
    "ppt": ppt_node,
    "image": image_node,
}


class ChatRequest(BaseModel):
    prompt: str = Field(..., max_length=8000, description="User prompt text")
    conversation_id: str = Field(..., max_length=128)
    agent: str = Field(default="auto", max_length=32)


def _initial_state(body: ChatRequest) -> dict:
    return {
        "prompt": body.prompt,
        "agent": body.agent.lower(),
        "conversation_id": body.conversation_id,
        "request_id": uuid.uuid4().hex[:12],
        "ai_response": "",
        "search_results": [],
        "images": [],
        "rag_context": "",
        "rag_sources": [],
        "orchestration_plan": [],
        "token_usage": {},
    }


async def _visible_tokens(tokens):
    """Yield streamed content without model reasoning blocks."""
    thinking_filter = ThinkingBlockFilter()
    async for token in tokens:
        visible = thinking_filter.feed(token)
        if visible:
            yield visible
    visible = thinking_filter.finish()
    if visible:
        yield visible


def _get_limiter():
    """Import limiter at runtime to avoid circular imports."""
    from app.main import limiter

    return limiter


@router.post("/chat")
async def chat(
    request: Request,
    body: ChatRequest,
    user: Annotated[dict, Depends(get_current_user)],
):
    limiter = _get_limiter()
    limiter.limit("30/minute")(chat)
    try:
        check_token_budget(body.prompt)
    except TokenBudgetExceeded as e:
        return StreamingResponse(
            iter(
                [
                    f"data: {json.dumps({'token': f'Prompt too long: {e}'})}\n\n",
                    "data: [DONE]\n\n",
                ]
            ),
            media_type="text/event-stream",
        )

    await chat_service.save_message(
        user["id"], body.conversation_id, "user", body.prompt
    )
    await memory.add_message(body.conversation_id, "user", body.prompt)

    guardrails.validate_input_prompt(body.prompt)

    config = {"configurable": {"thread_id": body.conversation_id}}
    result = await graph.ainvoke(_initial_state(body), config=config)

    guardrails.validate_output_response(result["ai_response"])

    usage = result.get("token_usage", {})
    await chat_service.save_message(
        user["id"],
        body.conversation_id,
        "assistant",
        result["ai_response"],
        images=result.get("images", []),
        token_usage=usage,
        agent=result.get("agent", ""),
    )
    await memory.add_message(body.conversation_id, "assistant", result["ai_response"])

    return {
        "answer": result["ai_response"],
        "images": result.get("images", []),
        "orchestration_plan": result.get("orchestration_plan", []),
        "token_usage": usage,
    }


@router.post("/chat/stream")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    user: Annotated[dict, Depends(get_current_user)],
):
    try:
        check_token_budget(body.prompt)
    except TokenBudgetExceeded as exc:
        error_msg = f"Prompt too long: {exc}"

        async def token_error():
            yield f"data: {json.dumps({'token': error_msg})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(token_error(), media_type="text/event-stream")

    await chat_service.save_message(
        user["id"], body.conversation_id, "user", body.prompt
    )
    await memory.add_message(body.conversation_id, "user", body.prompt)

    state = _initial_state(body)

    async def event_stream():
        full_response = ""
        images = []
        resolved_agent = "chat"
        req_id = state["request_id"]

        logger.info(
            "[%s] Stream started | conversation=%s | prompt=%.80s",
            req_id,
            body.conversation_id,
            redact_pii(body.prompt),
        )

        try:
            # 1. Guardrails AI Inspection
            try:
                guardrails.validate_input_prompt(body.prompt)
            except guardrails.GuardrailViolationError as err:
                if LOGFIRE_AVAILABLE:
                    logfire.warn("Guardrails Policy Violation: {err}", err=str(err))
                yield f"data: {json.dumps({'token': f'Guardrails Policy Violation: {err!s}'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Fast-Path Greeting check (Zero LLM Token Cost)
            fast_greeting = guardrails.check_fast_path_greeting(body.prompt)
            if fast_greeting:
                if LOGFIRE_AVAILABLE:
                    logfire.info("FastPath Router Greeting Matched Zero Tokens Used")
                yield f"data: {json.dumps({'agent': 'chat'})}\n\n"
                yield f"data: {json.dumps({'token': fast_greeting})}\n\n"
                yield "data: [DONE]\n\n"
                full_response = fast_greeting
                return

            # 2. Router Agent Execution
            if LOGFIRE_AVAILABLE:
                with logfire.span("router_agent_execution", mode=body.agent):
                    state.update(await router_node(state))
            else:
                state.update(await router_node(state))

            resolved_agent = state["agent"]

            # Emit resolved agent to frontend
            yield f"data: {json.dumps({'agent': resolved_agent})}\n\n"
            yield f"data: {json.dumps({'plan': state.get('orchestration_plan', [])})}\n\n"

            # 3. Agent Execution Node with Logfire Tracing
            if LOGFIRE_AVAILABLE:
                logfire.info(
                    "Router Agent Orchestrated Request -> [{agent}]",
                    agent=resolved_agent,
                    conversation_id=body.conversation_id,
                )

            if resolved_agent == "search":
                if LOGFIRE_AVAILABLE:
                    with logfire.span("search_agent_execution"):
                        state.update(await search_node(state))
                else:
                    state.update(await search_node(state))

            if resolved_agent in ("rag", "research_rag"):
                state.update(await rag_research_node(state))
                if resolved_agent == "research_rag" and not state.get("ai_response"):
                    state.update(await search_node(state))

            if resolved_agent == "coding" and not state.get("ai_response"):
                if LOGFIRE_AVAILABLE:
                    with logfire.span("coding_agent_stream"):
                        async for token in _visible_tokens(coding_node_stream(state)):
                            full_response += token
                            yield f"data: {json.dumps({'token': token})}\n\n"
                else:
                    async for token in _visible_tokens(coding_node_stream(state)):
                        full_response += token
                        yield f"data: {json.dumps({'token': token})}\n\n"
            elif resolved_agent in ("rag", "research_rag") and not state.get(
                "ai_response"
            ):
                if LOGFIRE_AVAILABLE:
                    with logfire.span("answer_synthesis_stream", agent=resolved_agent):
                        async for token in _visible_tokens(chat_node_stream(state)):
                            full_response += token
                            yield f"data: {json.dumps({'token': token})}\n\n"
                else:
                    async for token in _visible_tokens(chat_node_stream(state)):
                        full_response += token
                        yield f"data: {json.dumps({'token': token})}\n\n"
            elif resolved_agent in ("chat", "search") and not state.get("ai_response"):
                if LOGFIRE_AVAILABLE:
                    with logfire.span("chat_agent_stream", agent=resolved_agent):
                        async for token in _visible_tokens(chat_node_stream(state)):
                            full_response += token
                            yield f"data: {json.dumps({'token': token})}\n\n"
                else:
                    async for token in _visible_tokens(chat_node_stream(state)):
                        full_response += token
                        yield f"data: {json.dumps({'token': token})}\n\n"
            else:
                if state.get("ai_response"):
                    full_response = state["ai_response"]
                    yield f"data: {json.dumps({'token': full_response})}\n\n"
                else:
                    node = NON_STREAMED_NODES.get(resolved_agent)
                    if node:
                        if LOGFIRE_AVAILABLE:
                            with logfire.span(f"{resolved_agent}_agent_execution"):
                                result = await node(state)
                        else:
                            result = await node(state)

                        if result is not None:
                            full_response = result.get("ai_response", "")
                            images = result.get("images", [])
                            yield f"data: {json.dumps({'token': full_response})}\n\n"
                    else:
                        async for token in _visible_tokens(chat_node_stream(state)):
                            full_response += token
                            yield f"data: {json.dumps({'token': token})}\n\n"

            # 4. Guardrails Output Inspection
            guardrails.validate_output_response(full_response)

            # 5. Emit token usage
            usage = state.get("token_usage", {})
            if usage:
                yield f"data: {json.dumps({'token_usage': usage})}\n\n"

            if images:
                yield f"data: {json.dumps({'images': images})}\n\n"

            yield "data: [DONE]\n\n"
        except Exception:
            logger.exception("[%s] Stream error | agent=%s", req_id, resolved_agent)
            error_msg = "An unexpected error occurred. Please try again."
            yield f"data: {json.dumps({'token': f'**Error**: {error_msg}'})}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            logger.info(
                "[%s] Stream finished | agent=%s | response_length=%d",
                req_id,
                resolved_agent,
                len(full_response),
            )
            if LOGFIRE_AVAILABLE:
                logfire.info(
                    "Finished stream for {agent}",
                    agent=resolved_agent,
                    conversation_id=body.conversation_id,
                    length=len(full_response),
                )
            if full_response:
                await chat_service.save_message(
                    user["id"],
                    body.conversation_id,
                    "assistant",
                    full_response,
                    images=images,
                    token_usage=state.get("token_usage", {}),
                    agent=resolved_agent,
                )
                await memory.add_message(
                    body.conversation_id, "assistant", full_response
                )

    return StreamingResponse(event_stream(), media_type="text/event-stream")

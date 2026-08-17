import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agents.chat_agent import chat_node_stream
from app.agents.coding_agent import coding_node, coding_node_stream
from app.agents.graph import graph
from app.agents.image_agent import image_node
from app.agents.pdf_agent import pdf_node
from app.agents.ppt_agent import ppt_node
from app.agents.rag_agent import rag_node_stream
from app.agents.router_agent import router_node
from app.agents.search_agent import search_node
from app.auth.dependencies import get_current_user
from app.chat import service as chat_service
from app.core import guardrails, memory

try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False

router = APIRouter(prefix="/api/agent", tags=["agent"])

STREAMED_AGENTS = {"chat", "search", "coding", "rag"}

NON_STREAMED_NODES = {
    "pdf": pdf_node,
    "ppt": ppt_node,
    "image": image_node,
}


class ChatRequest(BaseModel):
    prompt: str
    conversation_id: str
    agent: str = "auto"


def _initial_state(body: ChatRequest) -> dict:
    return {
        "prompt": body.prompt,
        "agent": body.agent.lower(),
        "conversation_id": body.conversation_id,
        "ai_response": "",
        "search_results": [],
        "images": [],
    }


@router.post("/chat")
async def chat(body: ChatRequest, user: dict = Depends(get_current_user)):
    await chat_service.save_message(user["id"], body.conversation_id, "user", body.prompt)
    await memory.add_message(body.conversation_id, "user", body.prompt)

    # Perform Guardrails AI validation
    guardrails.validate_input_prompt(body.prompt)

    result = await graph.ainvoke(_initial_state(body))

    guardrails.validate_output_response(result["ai_response"])

    await chat_service.save_message(
        user["id"],
        body.conversation_id,
        "assistant",
        result["ai_response"],
        images=result.get("images", []),
    )
    await memory.add_message(body.conversation_id, "assistant", result["ai_response"])

    return {"answer": result["ai_response"], "images": result.get("images", [])}


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest, user: dict = Depends(get_current_user)):
    await chat_service.save_message(user["id"], body.conversation_id, "user", body.prompt)
    await memory.add_message(body.conversation_id, "user", body.prompt)

    state = _initial_state(body)

    async def event_stream():
        full_response = ""
        images = []
        resolved_agent = "chat"

        try:
            # 1. Guardrails AI Inspection
            try:
                guardrails.validate_input_prompt(body.prompt)
            except guardrails.GuardrailViolationError as err:
                if LOGFIRE_AVAILABLE:
                    logfire.warn("⛔ [Guardrails AI] Policy Violation: {err}", err=str(err), prompt=body.prompt)
                yield f"data: {json.dumps({'token': f'⚠️ **Guardrails Policy Violation**: {str(err)}'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Fast-Path Greeting check (Zero LLM Token Cost)
            fast_greeting = guardrails.check_fast_path_greeting(body.prompt)
            if fast_greeting:
                if LOGFIRE_AVAILABLE:
                    logfire.info("⚡ [FastPath Router] Greeting Matched -> Zero Tokens Used", prompt=body.prompt)
                yield f"data: {json.dumps({'agent': 'chat'})}\n\n"
                yield f"data: {json.dumps({'token': fast_greeting})}\n\n"
                yield "data: [DONE]\n\n"
                full_response = fast_greeting
                return

            # 2. Router Agent Execution
            if LOGFIRE_AVAILABLE:
                with logfire.span("router_agent_execution", mode=body.agent):
                    resolved_agent = (await router_node(state))["agent"]
            else:
                resolved_agent = (await router_node(state))["agent"]

            state["agent"] = resolved_agent

            # Emit resolved agent to frontend
            yield f"data: {json.dumps({'agent': resolved_agent})}\n\n"

            # 3. Agent Execution Node with Logfire Tracing
            if LOGFIRE_AVAILABLE:
                logfire.info("🔀 [Router Agent] Orchestrated Request -> [{agent}]", agent=resolved_agent, conversation_id=body.conversation_id)


            if resolved_agent == "search":
                if LOGFIRE_AVAILABLE:
                    with logfire.span("search_agent_execution"):
                        state.update(await search_node(state))
                else:
                    state.update(await search_node(state))

            if resolved_agent == "coding" and not state.get("ai_response"):
                if LOGFIRE_AVAILABLE:
                    with logfire.span("coding_agent_stream"):
                        async for token in coding_node_stream(state):
                            full_response += token
                            yield f"data: {json.dumps({'token': token})}\n\n"
                else:
                    async for token in coding_node_stream(state):
                        full_response += token
                        yield f"data: {json.dumps({'token': token})}\n\n"
            elif resolved_agent == "rag" and not state.get("ai_response"):
                if LOGFIRE_AVAILABLE:
                    with logfire.span("rag_agent_stream"):
                        async for token in rag_node_stream(state):
                            full_response += token
                            yield f"data: {json.dumps({'token': token})}\n\n"
                else:
                    async for token in rag_node_stream(state):
                        full_response += token
                        yield f"data: {json.dumps({'token': token})}\n\n"
            elif resolved_agent in ("chat", "search") and not state.get("ai_response"):

                if LOGFIRE_AVAILABLE:
                    with logfire.span("chat_agent_stream", agent=resolved_agent):
                        async for token in chat_node_stream(state):
                            full_response += token
                            yield f"data: {json.dumps({'token': token})}\n\n"
                else:
                    async for token in chat_node_stream(state):
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
                        async for token in chat_node_stream(state):
                            full_response += token
                            yield f"data: {json.dumps({'token': token})}\n\n"

            # 4. Guardrails Output Inspection
            guardrails.validate_output_response(full_response)

            if images:
                yield f"data: {json.dumps({'images': images})}\n\n"

            yield "data: [DONE]\n\n"
        finally:
            if LOGFIRE_AVAILABLE:
                logfire.info(
                    "Finished stream for {agent}",
                    agent=resolved_agent,
                    conversation_id=body.conversation_id,
                    length=len(full_response),
                )
            await chat_service.save_message(
                user["id"],
                body.conversation_id,
                "assistant",
                full_response,
                images=images,
            )
            await memory.add_message(body.conversation_id, "assistant", full_response)

    return StreamingResponse(event_stream(), media_type="text/event-stream")



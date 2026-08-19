from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.logging import log_agent_failure, log_agent_start, log_agent_success
from app.agents.models import _extract_usage, content_text, get_model
from app.agents.state import AgentState
from app.core import memory

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Answer the user's question clearly and "
    "conversationally. If search results are provided, use them as context to "
    "give an accurate, up-to-date answer and mention your sources where relevant."
)


async def _build_messages(state: AgentState) -> list:
    history = await memory.get_memory(state["conversation_id"])

    system_prompt = SYSTEM_PROMPT
    if state.get("search_results"):
        context = "\n\n".join(str(item) for item in state["search_results"])
        system_prompt += (
            "\n\nUse the following search results as context to answer the "
            f"user's question:\n{context}"
        )

    if state.get("rag_context"):
        system_prompt += (
            "\n\nUse the following retrieved document context as evidence. "
            "Do not claim facts that are not supported by this context:\n"
            f"{state['rag_context']}"
        )

    messages = [SystemMessage(content=system_prompt)]
    for entry in history:
        role = entry.get("role")
        content = entry.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=state["prompt"]))
    return messages


async def chat_node(state: AgentState) -> dict:
    t0 = log_agent_start("chat", state)
    try:
        messages = await _build_messages(state)
        result = await get_model("chat").ainvoke(messages)
        response = content_text(result.content)
        usage = _extract_usage(result)
        log_agent_success(
            "chat",
            state,
            t0,
            response_length=len(response),
            tokens=usage.get("total_tokens", 0),
        )
        return {"ai_response": response, "token_usage": usage}
    except Exception as exc:
        log_agent_failure("chat", state, exc)
        raise


async def chat_node_stream(state: AgentState):
    """Streams the chat model's response tokens as they arrive."""
    t0 = log_agent_start("chat_stream", state)
    try:
        messages = await _build_messages(state)
        token_count = 0
        async for chunk in get_model("chat", streaming=True).astream(messages):
            content = content_text(chunk.content)
            if content:
                token_count += 1
                yield content
        log_agent_success("chat_stream", state, t0, tokens_yielded=token_count)
    except Exception as exc:
        log_agent_failure("chat_stream", state, exc)
        raise

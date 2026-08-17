from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.models import content_text, get_model
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
    messages = await _build_messages(state)
    result = await get_model("chat").ainvoke(messages)
    return {"ai_response": content_text(result.content)}


async def chat_node_stream(state: AgentState):
    """Streams the chat model's response tokens as they arrive."""
    messages = await _build_messages(state)
    async for chunk in get_model("chat").astream(messages):
        content = content_text(chunk.content)
        if content:
            yield content

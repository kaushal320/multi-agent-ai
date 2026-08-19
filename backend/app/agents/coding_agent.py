from app.agents.logging import log_agent_failure, log_agent_start, log_agent_success
from app.agents.models import _extract_usage, content_text, get_model
from app.agents.state import AgentState

CODING_SYSTEM_PROMPT = """You are an expert coding and software engineering assistant.
When answering coding questions, tutorials, or requests:
1. Provide clear, structured markdown explanations with conceptual key points.
2. Present complete, runnable, and well-commented code inside fenced markdown code blocks (e.g. ```python ... ``` or ```javascript ... ```).
3. Clearly separate explanations, bullet points, and code blocks so the user can easily read the tutorial and copy individual code snippets."""


async def coding_node(state: AgentState) -> dict:
    t0 = log_agent_start("coding", state)
    try:
        result = await get_model("coding").ainvoke(
            [
                ("system", CODING_SYSTEM_PROMPT),
                ("human", state["prompt"]),
            ]
        )
        response = content_text(result.content).strip()
        usage = _extract_usage(result)
        log_agent_success(
            "coding",
            state,
            t0,
            response_length=len(response),
            tokens=usage.get("total_tokens", 0),
        )
        return {"ai_response": response, "token_usage": usage}
    except Exception as exc:
        log_agent_failure("coding", state, exc)
        raise


async def coding_node_stream(state: AgentState):
    """Streams the coding agent's response tokens as they arrive."""
    t0 = log_agent_start("coding_stream", state)
    try:
        messages = [
            ("system", CODING_SYSTEM_PROMPT),
            ("human", state["prompt"]),
        ]
        token_count = 0
        async for chunk in get_model("coding", streaming=True).astream(messages):
            content = content_text(chunk.content)
            if content:
                token_count += 1
                yield content
        log_agent_success("coding_stream", state, t0, tokens_yielded=token_count)
    except Exception as exc:
        log_agent_failure("coding_stream", state, exc)
        raise

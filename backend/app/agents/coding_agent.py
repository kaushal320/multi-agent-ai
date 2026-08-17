import re

from app.agents.models import content_text, get_model
from app.agents.state import AgentState

CODING_SYSTEM_PROMPT = """You are an expert coding and software engineering assistant.
When answering coding questions, tutorials, or requests:
1. Provide clear, structured markdown explanations with conceptual key points.
2. Present complete, runnable, and well-commented code inside fenced markdown code blocks (e.g. ```python ... ``` or ```javascript ... ```).
3. Clearly separate explanations, bullet points, and code blocks so the user can easily read the tutorial and copy individual code snippets."""


try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False


async def coding_node(state: AgentState) -> dict:
    if LOGFIRE_AVAILABLE:
        with logfire.span("coding_node_ainvoke", prompt_length=len(state["prompt"])):
            result = await get_model("coding").ainvoke(
                [
                    ("system", CODING_SYSTEM_PROMPT),
                    ("human", state["prompt"]),
                ]
            )
    else:
        result = await get_model("coding").ainvoke(
            [
                ("system", CODING_SYSTEM_PROMPT),
                ("human", state["prompt"]),
            ]
        )
    return {"ai_response": content_text(result.content).strip()}


async def coding_node_stream(state: AgentState):
    """Streams the coding agent's response tokens as they arrive."""
    messages = [
        ("system", CODING_SYSTEM_PROMPT),
        ("human", state["prompt"]),
    ]
    if LOGFIRE_AVAILABLE:
        logfire.info("Started streaming tokens from Coding Agent")

    async for chunk in get_model("coding").astream(messages):
        content = content_text(chunk.content)
        if content:
            yield content



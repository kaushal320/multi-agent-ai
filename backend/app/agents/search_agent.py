import json

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from app.agents.logging import log_agent_failure, log_agent_start, log_agent_success
from app.agents.models import _extract_usage, content_text, get_model
from app.agents.state import AgentState
from app.core.config import settings


@tool
def search_tool(query: str) -> str:
    """Search the web for up-to-date information about the given query."""
    data = TavilySearch(
        tavily_api_key=settings.tavily_api_key,
        include_images=True,
    ).invoke({"query": query})
    return json.dumps(data)


async def search_node(state: AgentState) -> dict:
    """Let the model decide whether (and how) to call the search tool.

    If the model returns tool calls, execute them and hand the results to the
    chat node as context. If the model decides a search isn't needed, answer
    directly and skip the chat handoff."""
    t0 = log_agent_start("search", state)
    try:
        model = get_model("search").bind_tools([search_tool])
        response = await model.ainvoke([HumanMessage(state["prompt"])])

        usage = _extract_usage(response)

        if not response.tool_calls:
            response_text = content_text(response.content)
            log_agent_success(
                "search",
                state,
                t0,
                result_type="direct_answer",
                response_length=len(response_text),
            )
            return {"ai_response": response_text, "token_usage": usage}

        search_results: list[str] = []
        images: list[str] = []
        for call in response.tool_calls:
            data = json.loads(await search_tool.ainvoke(call["args"]))
            results = data.get("results", []) if isinstance(data, dict) else []
            search_results.extend(
                str(item.get("content") or item.get("title") or "")
                for item in results
                if isinstance(item, dict)
            )
            if isinstance(data, dict):
                images.extend(data.get("images", []))

        log_agent_success(
            "search",
            state,
            t0,
            result_type="tool_calls",
            tool_calls=len(response.tool_calls),
            results_count=len(search_results),
            images_count=len(images),
        )
        return {
            "search_results": search_results,
            "images": images,
            "token_usage": usage,
        }
    except Exception as exc:
        log_agent_failure("search", state, exc)
        raise

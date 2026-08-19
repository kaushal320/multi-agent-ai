import json
import logging
import asyncio

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from app.agents.logging import log_agent_failure, log_agent_start, log_agent_success
from app.agents.models import _extract_usage, content_text, get_model
from app.agents.state import AgentState
from app.core.cache import get_cached_search, set_cached_search
from app.core.config import settings
from app.core.observability import obs

logger = logging.getLogger("cortex.agents.search")


# Module-level search client (no cache in tool - cache at node level)
_tavily_search = TavilySearch(
    tavily_api_key=settings.tavily_api_key,
    include_images=True,
)


@tool
def search_tool(query: str) -> str:
    """Search the web for up-to-date information about the given query."""
    data = _tavily_search.invoke({"query": query})
    return json.dumps(data)


async def search_node(state: AgentState) -> dict:
    """Let the model decide whether (and how) to call the search tool.

    If the model returns tool calls, execute them and hand the results to the
    chat node as context. If the model decides a search isn't needed, answer
    directly and skip the chat handoff."""
    t0 = log_agent_start("search", state)
    prompt = state["prompt"]

    # Check search cache first
    cached_results = await get_cached_search(prompt)
    if cached_results:
        obs.metric("search.cache.hit", 1)
        logger.info("Search cache HIT for prompt: %s", prompt[:50])

        # Return cached results directly without calling model
        search_results = [
            str(item.get("content") or item.get("title") or "")
            for item in cached_results
            if isinstance(item, dict)
        ]
        images = []
        # Try to get images from cached results if available
        for item in cached_results:
            if isinstance(item, dict) and "images" in item:
                images.extend(item.get("images", []))

        log_agent_success(
            "search",
            state,
            t0,
            result_type="cached",
            results_count=len(search_results),
            images_count=len(images),
        )
        return {
            "search_results": search_results,
            "images": images,
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }

    try:
        model = get_model("search").bind_tools([search_tool])
        response = await model.ainvoke([HumanMessage(prompt)])

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
        raw_tool_results = []

        for call in response.tool_calls:
            data = json.loads(await search_tool.ainvoke(call["args"]))
            results = data.get("results", []) if isinstance(data, dict) else []
            search_results.extend(
                str(item.get("content") or item.get("title") or "")
                for item in results
                if isinstance(item, dict)
            )
            raw_tool_results.extend(results)
            if isinstance(data, dict):
                images.extend(data.get("images", []))

        # Cache the raw search results for future queries
        if raw_tool_results:
            await set_cached_search(prompt, raw_tool_results)
            obs.metric("search.cache.store", 1)

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

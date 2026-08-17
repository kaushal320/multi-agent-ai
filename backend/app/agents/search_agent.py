import json
import os

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from app.agents.models import content_text, get_model
from app.agents.state import AgentState


@tool
def search_tool(query: str) -> str:
    """Search the web for up-to-date information about the given query."""
    data = TavilySearch(
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        include_images=True,
    ).invoke({"query": query})
    return json.dumps(data)


async def search_node(state: AgentState) -> dict:
    """Let the model decide whether (and how) to call the search tool.

    If the model returns tool calls, execute them and hand the results to the
    chat node as context. If the model decides a search isn't needed, answer
    directly and skip the chat handoff."""
    model = get_model("search").bind_tools([search_tool])
    response = await model.ainvoke([HumanMessage(state["prompt"])])

    if not response.tool_calls:
        return {"ai_response": content_text(response.content)}

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
    return {"search_results": search_results, "images": images}

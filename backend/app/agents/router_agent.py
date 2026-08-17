import re

from app.agents.models import content_text, get_model
from app.agents.state import AgentState

ROUTER_AGENTS = ["chat", "search", "coding", "pdf", "ppt", "image", "rag"]

ROUTER_SYSTEM_PROMPT = """You are a router for an AI assistant. Based on the user's message, choose exactly ONE of the following agents and reply with a single word only:

- chat: general conversation, explanations, writing, or anything that does not require live web data
- search: questions that need current, factual, or real-time information from the web
- coding: requests to write, fix, or explain code, or to generate a code snippet in any language
- pdf: requests to create, generate, or draft a PDF document or report
- ppt: requests to create, generate, or draft a PowerPoint presentation or slide deck
- image: requests to generate, create, draw, or imagine an image or picture
- rag: questions about an uploaded document, or references to "the file", "the document", or "the PDF I uploaded"

Return only the agent name. Do not add any other text."""


try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False


async def router_node(state: AgentState) -> dict:
    if state.get("agent") and state["agent"] != "auto":
        if LOGFIRE_AVAILABLE:
            logfire.info("Manual agent selection: {agent}", agent=state["agent"])
        return {"agent": state["agent"]}

    if LOGFIRE_AVAILABLE:
        with logfire.span("router_node_classification", prompt=state["prompt"]):
            result = await get_model("router").ainvoke(
                [
                    ("system", ROUTER_SYSTEM_PROMPT),
                    ("human", state["prompt"]),
                ]
            )
            agent = re.sub(r"[^a-z]", "", content_text(result.content).strip().lower())
            if not agent or agent not in ROUTER_AGENTS:
                agent = "chat"
            logfire.info("Router classified prompt to agent: {agent}", agent=agent)
    else:
        result = await get_model("router").ainvoke(
            [
                ("system", ROUTER_SYSTEM_PROMPT),
                ("human", state["prompt"]),
            ]
        )
        agent = re.sub(r"[^a-z]", "", content_text(result.content).strip().lower())
        if not agent or agent not in ROUTER_AGENTS:
            agent = "chat"

    return {"agent": agent}


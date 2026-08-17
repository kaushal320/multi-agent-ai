import logging
import re

from app.agents.models import content_text, get_model
from app.agents.state import AgentState
from app.core.pii import redact_pii

logger = logging.getLogger("cortex.agents.router_agent")

ROUTER_AGENTS = ["chat", "search", "coding", "pdf", "ppt", "image", "rag", "research_rag"]

ROUTER_SYSTEM_PROMPT = """You are a router for an AI assistant. Based on the user's message, choose exactly ONE of the following agents and reply with a single word only:

- chat: general conversation, explanations, writing, or anything that does not require live web data
- search: questions that need current, factual, or real-time information from the web
- coding: requests to write, fix, or explain code, or to generate a code snippet in any language
- pdf: requests to create, generate, or draft a PDF document or report
- ppt: requests to create, generate, or draft a PowerPoint presentation or slide deck
- image: requests to generate, create, draw, or imagine an image or picture
- rag: questions about an uploaded document, or references to "the file", "the document", or "the PDF I uploaded"
- research_rag: questions that require both an uploaded document and current web information, comparison, or verification

Return only the agent name. Do not add any other text."""


try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False


async def router_node(state: AgentState) -> dict:
    if state.get("agent") and state["agent"] != "auto":
        logger.info("Manual agent selection: %s", state["agent"])
        if LOGFIRE_AVAILABLE:
            logfire.info("Manual agent selection: {agent}", agent=state["agent"])
        return {
            "agent": state["agent"],
            "orchestration_plan": ["manual agent selection", state["agent"]],
        }

    if LOGFIRE_AVAILABLE:
        with logfire.span("router_node_classification"):
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

    logger.info("Router classified to: %s | prompt=%.80s", agent, redact_pii(state["prompt"]))

    plans = {
        "chat": ["supervisor", "chat specialist"],
        "search": ["supervisor", "web research specialist", "answer synthesis specialist"],
        "coding": ["supervisor", "coding specialist"],
        "pdf": ["supervisor", "document generation specialist"],
        "ppt": ["supervisor", "presentation specialist"],
        "image": ["supervisor", "image generation specialist"],
        "rag": ["supervisor", "document retrieval specialist", "answer synthesis specialist"],
        "research_rag": ["supervisor", "document retrieval specialist", "web research specialist", "answer synthesis specialist"],
    }
    return {"agent": agent, "orchestration_plan": plans[agent]}

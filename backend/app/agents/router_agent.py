import logging

from app.agents.models import AgentHandoff, RouterDecision, get_model
from app.agents.state import AgentState
from app.core.observability import obs
from app.core.pii import redact_pii

logger = logging.getLogger("cortex.agents.router_agent")

ROUTER_AGENTS = [
    "chat",
    "search",
    "coding",
    "pdf",
    "ppt",
    "image",
    "rag",
    "research_rag",
]

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


async def router_node(state: AgentState) -> dict:
    if state.get("agent") and state["agent"] != "auto":
        obs.info("Manual agent selection", agent=state["agent"])
        return {
            "agent": state["agent"],
            "orchestration_plan": ["manual agent selection", state["agent"]],
        }

    model = get_model("router").with_structured_output(RouterDecision)
    decision = await model.ainvoke(
        [
            ("system", ROUTER_SYSTEM_PROMPT),
            ("human", state["prompt"]),
        ]
    )

    # Validate
    agent = decision.agent if decision.agent in ROUTER_AGENTS else "chat"

    obs.router_decision(
        agent=agent,
        confidence=decision.confidence,
        reasoning=decision.reasoning,
        conversation_id=state.get("conversation_id", "unknown"),
        prompt=state["prompt"],
    )

    plans = {
        "chat": ["supervisor", "chat specialist"],
        "search": [
            "supervisor",
            "web research specialist",
            "answer synthesis specialist",
        ],
        "coding": ["supervisor", "coding specialist"],
        "pdf": ["supervisor", "document generation specialist"],
        "ppt": ["supervisor", "presentation specialist"],
        "image": ["supervisor", "image generation specialist"],
        "rag": [
            "supervisor",
            "document retrieval specialist",
            "answer synthesis specialist",
        ],
        "research_rag": [
            "supervisor",
            "document retrieval specialist",
            "web research specialist",
            "answer synthesis specialist",
        ],
    }

    # Create structured handoff
    handoff = AgentHandoff(
        from_agent="router",
        to_agent=agent,
        payload={
            "original_prompt": state["prompt"],
            "plan": plans[agent],
            "conversation_id": state.get("conversation_id", "unknown"),
        },
        reason=decision.reasoning,
    )

    return {
        "agent": agent,
        "orchestration_plan": plans[agent],
        "reflection": decision.reasoning,
        "needs_more_info": decision.confidence < 0.5,
        "handoff": handoff.model_dump(),
    }

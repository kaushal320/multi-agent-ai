import logging

from app.agents.models import AgentHandoff, content_text, get_model
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


def _parse_agent(text: str) -> str:
    """Extract a known agent name from the model's (possibly noisy) reply.

    The model may wrap the word in quotes/punctuation or, on thinking models
    like Qwen3, embed it inside a <think> block. We fall back to 'chat' if no
    known agent token is found.
    """
    cleaned = content_text(text).strip().lower()
    tokens = cleaned.split()
    # Prefer an exact single-token match (handles "chat.", '"search"', etc.)
    for candidate in ROUTER_AGENTS:
        if candidate in tokens:
            return candidate
    # Otherwise, look for any known agent token anywhere in the text.
    for candidate in ROUTER_AGENTS:
        if candidate in cleaned:
            return candidate
    return "chat"

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

    # Use plain text generation instead of structured output.
    # This avoids the Qwen3 thinking-mode vs. tool-call conflict on Groq.
    model = get_model("router")
    resp = await model.ainvoke(
        [
            ("system", ROUTER_SYSTEM_PROMPT),
            ("human", state["prompt"]),
        ]
    )

    # Parse the agent name from the model's free-text reply.
    agent = _parse_agent(content_text(resp.content))

    # Simple confidence: 1.0 if exact match, 0.6 if found inside text, 0.3 otherwise.
    cleaned = content_text(resp.content).strip().lower()
    if agent in cleaned.split():
        confidence = 1.0
    elif agent in cleaned:
        confidence = 0.6
    else:
        confidence = 0.3

    reasoning = f"Router selected '{agent}' (confidence={confidence:.1f}) from model output: {resp.content[:200]!r}"

    obs.router_decision(
        agent=agent,
        confidence=confidence,
        reasoning=reasoning,
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
        reason=reasoning,
    )

    return {
        "agent": agent,
        "orchestration_plan": plans[agent],
        "reflection": reasoning,
        "needs_more_info": confidence < 0.5,
        "handoff": handoff.model_dump(),
    }

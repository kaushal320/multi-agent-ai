from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.chat_agent import chat_node
from app.agents.coding_agent import coding_node
from app.agents.image_agent import image_node
from app.agents.pdf_agent import pdf_node
from app.agents.ppt_agent import ppt_node
from app.agents.rag_agent import rag_research_node
from app.agents.router_agent import router_node
from app.agents.search_agent import search_node
from app.agents.state import AgentState


def _route(state: AgentState) -> str:
    return state.get("agent", "chat")


def _after_search(state: AgentState) -> str:
    return "end" if state.get("ai_response") else "chat"


def _after_rag(state: AgentState) -> str:
    if state.get("ai_response"):
        return "end"
    return "search" if state.get("agent") == "research_rag" else "chat"


builder = StateGraph(AgentState)
builder.add_node("router", router_node)
builder.add_node("search", search_node)
builder.add_node("chat", chat_node)
builder.add_node("coding", coding_node)
builder.add_node("pdf", pdf_node)
builder.add_node("ppt", ppt_node)
builder.add_node("image", image_node)
builder.add_node("rag_research", rag_research_node)
builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router",
    _route,
    {
        "chat": "chat",
        "search": "search",
        "coding": "coding",
        "pdf": "pdf",
        "ppt": "ppt",
        "image": "image",
        "rag": "rag_research",
        "research_rag": "rag_research",
    },
)
builder.add_conditional_edges("search", _after_search, {"chat": "chat", "end": END})
builder.add_conditional_edges(
    "rag_research",
    _after_rag,
    {"search": "search", "chat": "chat", "end": END},
)
builder.add_edge("chat", END)
builder.add_edge("coding", END)
builder.add_edge("pdf", END)
builder.add_edge("ppt", END)
builder.add_edge("image", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

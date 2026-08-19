from typing import TypedDict


class AgentState(TypedDict):
    prompt: str
    agent: str
    conversation_id: str
    request_id: str
    ai_response: str
    search_results: list
    images: list[str]
    rag_context: str
    rag_sources: list[dict]
    orchestration_plan: list[str]
    token_usage: dict
    # Iteration / reflection for reflexion loops
    iteration: int
    max_iterations: int
    reflection: str
    needs_more_info: bool
    # Structured handoff between agents
    handoff: dict | None
    # Dynamic fan-out
    fanout_agents: list[str]

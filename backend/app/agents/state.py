from typing import TypedDict


class AgentState(TypedDict):
    prompt: str
    agent: str
    conversation_id: str
    ai_response: str
    search_results: list
    images: list[str]

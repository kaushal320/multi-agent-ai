import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()


def get_model(agent_name: str):
    if agent_name in ("chat", "search", "router"):
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
        )
    if agent_name == "coding":
        return ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            api_key=os.getenv("GOOGLE_API_KEY"),
        )
    return get_model("chat")


def content_text(content) -> str:
    """Extract plain text from a model's response content, which may be a string
    (Groq) or a list of content parts (Gemini returns
    [{'type': 'text', 'text': '...'}, ...])."""
    if isinstance(content, str):
        return content
    parts = []
    for item in content or []:
        if isinstance(item, dict):
            parts.append(item.get("text") or item.get("content") or "")
        elif hasattr(item, "text"):
            parts.append(item.text or "")
    return "".join(parts)

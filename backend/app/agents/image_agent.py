from urllib.parse import quote

import httpx

from app.agents.state import AgentState
from app.core import storage

POLLINATIONS_URL = "https://image.pollinations.ai/prompt/{}"


async def image_node(state: AgentState) -> dict:
    prompt = state["prompt"].strip()
    url = POLLINATIONS_URL.format(quote(prompt))

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPError:
        return {
            "ai_response": "Sorry, I couldn't generate that image right now. Please try again with a more detailed description.",
            "images": [],
        }

    filename = "image.png"
    image_url = storage.save_file(filename, response.content)

    return {
        "ai_response": f"Here is your generated image:",
        "images": [image_url],
    }

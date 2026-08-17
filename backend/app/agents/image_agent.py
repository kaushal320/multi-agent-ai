from urllib.parse import quote

import httpx

from app.agents.logging import log_agent_failure, log_agent_start, log_agent_success
from app.agents.state import AgentState
from app.core import storage

POLLINATIONS_URL = "https://image.pollinations.ai/prompt/{}"


async def image_node(state: AgentState) -> dict:
    t0 = log_agent_start("image", state)
    try:
        prompt = state["prompt"].strip()
        url = POLLINATIONS_URL.format(quote(prompt))

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.get(url)
            response.raise_for_status()

        filename = "image.png"
        image_url = storage.save_file(filename, response.content)

        log_agent_success("image", state, t0, image_url=image_url)
        return {
            "ai_response": "Here is your generated image:",
            "images": [image_url],
        }
    except httpx.HTTPError as exc:
        log_agent_failure("image", state, exc)
        return {
            "ai_response": "Sorry, I couldn't generate that image right now. Please try again with a more detailed description.",
            "images": [],
        }
    except Exception as exc:
        log_agent_failure("image", state, exc)
        raise

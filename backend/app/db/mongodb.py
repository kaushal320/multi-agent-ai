from typing import Optional

from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

from app.core.config import settings
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.user import User

mongo_client: Optional[AsyncIOMotorClient] = None


async def init_db() -> None:
    """Connects to MongoDB Atlas and registers Beanie document models. Add new models
    to the document_models list below as you create them (Conversation, Message, ...)."""
    global mongo_client
    mongo_client = AsyncIOMotorClient(settings.mongo_uri)
    database = mongo_client[settings.mongo_db_name]
    await init_beanie(database=database, document_models=[User, Conversation, Message])

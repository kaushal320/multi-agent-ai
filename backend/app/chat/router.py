from beanie import PydanticObjectId
from fastapi import Annotated, APIRouter, Depends, HTTPException

from app.auth.dependencies import get_current_user
from app.chat import service
from app.chat.schemas import SaveMessageRequest, UpdateConversationRequest

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


def _dump(document) -> dict:
    """Serialize a Beanie document using field names (id, not the _id alias)."""
    return document.model_dump(mode="json", by_alias=False)


@router.post("/create_conversation")
async def create_conversation(user: Annotated[dict, Depends(get_current_user)]):
    conversation = await service.create_conversation(user["id"])
    return _dump(conversation)


@router.get("/get_conversations")
async def get_conversations(user: Annotated[dict, Depends(get_current_user)]):
    conversations = await service.get_conversations(user["id"])
    return [_dump(c) for c in conversations]


@router.post("/update_conversation")
async def update_conversation(
    body: UpdateConversationRequest,
    user: Annotated[dict, Depends(get_current_user)],
):
    conversation = await service.update_conversation(
        user["id"], body.conversation_id, body.title
    )
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return _dump(conversation)


@router.post("/save_message")
async def save_message(
    body: SaveMessageRequest,
    user: Annotated[dict, Depends(get_current_user)],
):
    message = await service.save_message(
        user["id"], body.conversation_id, body.role, body.content, body.images
    )
    if not message:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return _dump(message)


@router.get("/get_messages/{conversation_id}")
async def get_messages(
    conversation_id: PydanticObjectId,
    user: Annotated[dict, Depends(get_current_user)],
):
    messages = await service.get_messages(user["id"], conversation_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return [_dump(m) for m in messages]

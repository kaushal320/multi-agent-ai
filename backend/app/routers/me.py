from fastapi import Annotated, APIRouter, Depends

from app.auth.dependencies import get_current_user

router = APIRouter(prefix="/api/v1", tags=["user"])


@router.get("/me")
async def get_me(user: Annotated[dict, Depends(get_current_user)]):
    return user

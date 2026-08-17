from typing import Optional

from pydantic import BaseModel


class LoginRequest(BaseModel):
    token: str  # Firebase ID token from the frontend


class UserOut(BaseModel):
    id: str
    firebase_uid: str
    name: Optional[str] = None
    email: Optional[str] = None
    avatar: Optional[str] = None

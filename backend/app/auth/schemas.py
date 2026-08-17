from pydantic import BaseModel


class LoginRequest(BaseModel):
    token: str  # Firebase ID token from the frontend


class UserOut(BaseModel):
    id: str
    firebase_uid: str
    name: str | None = None
    email: str | None = None
    avatar: str | None = None

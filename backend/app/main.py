from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.agents.router import router as agent_router
from app.auth.router import router as auth_router
from app.chat.router import router as chat_router
from app.core import firebase  # noqa: F401  (import initializes the Firebase app)
from app.core.config import settings
from app.db.mongodb import init_db
from app.documents.router import router as documents_router
from app.routers.me import router as me_router


try:
    import logfire
    logfire.configure(
        service_name="cortex-ai-backend",
        token=settings.logfire_token if settings.logfire_token else None,
        advanced=logfire.AdvancedOptions(base_url=settings.logfire_base_url),
        send_to_logfire="if-token-present",
    )
    LOGFIRE_AVAILABLE = True
except Exception:
    LOGFIRE_AVAILABLE = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="Cortex AI", lifespan=lifespan)

if LOGFIRE_AVAILABLE:
    try:
        logfire.instrument_fastapi(app)
        logfire.instrument_httpx()
        logfire.instrument_pydantic()
    except Exception:
        pass


app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agent_router)
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(me_router)

app.mount(
    "/static",
    StaticFiles(directory="static", check_dir=False),
    name="static",
)


@app.get("/")
async def root():
    return {"message": "Cortex AI backend running"}

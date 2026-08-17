import logging
import sys
import time
from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI, Request

logger = logging.getLogger("cortex.main")
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.agents.router import router as agent_router
from app.auth.router import router as auth_router
from app.chat.router import router as chat_router
from app.core import firebase  # noqa: F401  (import initializes the Firebase app)
from app.core.config import settings
from app.db.mongodb import init_db
from app.documents.router import router as documents_router
from app.routers.me import router as me_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logging.getLogger("cortex.agents").setLevel(logging.INFO)


try:
    import logfire

    logfire.configure(
        service_name="cortex-ai-backend",
        token=settings.logfire_token if settings.logfire_token else None,
        advanced=logfire.AdvancedOptions(base_url=settings.logfire_base_url),
        send_to_logfire="if-token-present",
    )
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False


limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="Cortex AI",
    version="1.0.0",
    description="Multi-agent AI assistant with RAG, web search, coding, and document generation.",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

if LOGFIRE_AVAILABLE:
    try:
        logfire.instrument_fastapi(app)
        logfire.instrument_httpx()
        logfire.instrument_pydantic()
    except Exception as e:  # noqa: BLE001
        logger.debug("Logfire instrumentation error: %s", e)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if settings.environment != "development":
        response.headers["Strict-Transport-Security"] = (
            "max-age=63072000; includeSubDomains"
        )
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
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


@app.get("/api/v1/health")
@limiter.exempt
async def health_check():
    checks = {"status": "ok", "timestamp": time.time()}
    try:
        from app.core.redis_client import redis_client

        await redis_client.ping()
        checks["redis"] = "ok"
    except redis.RedisError as e:
        checks["redis"] = f"error: {type(e).__name__}"
        checks["status"] = "degraded"

    try:
        from app.core.vectorstore import qdrant_client

        qdrant_client.get_collections()
        checks["qdrant"] = "ok"
    except Exception as e:  # noqa: BLE001
        checks["qdrant"] = f"error: {type(e).__name__}"
        checks["status"] = "degraded"

    try:
        from app.db.mongodb import mongo_client

        await mongo_client.admin.command("ping")
        checks["mongodb"] = "ok"
    except Exception as e:  # noqa: BLE001
        checks["mongodb"] = f"error: {type(e).__name__}"
        checks["status"] = "degraded"

    status_code = 200 if checks["status"] == "ok" else 503
    return JSONResponse(content=checks, status_code=status_code)

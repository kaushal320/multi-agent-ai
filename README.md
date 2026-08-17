# Cortex AI — FastAPI Port

Single FastAPI app (routers instead of separate Node microservices), mirroring the
original Node/Express + Firebase + Redis + MongoDB auth flow.

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt  # requirements.txt lives at the repo root
cp .env.example .env             # fill in MONGO_URI, GROQ_API_KEY, etc.
# place your Firebase service-account JSON at the repo root and point
# FIREBASE_CREDENTIALS_PATH at it (e.g. ./multiagent-ai-....json)
```

Start Redis + Qdrant:
```bash
docker compose up -d
```

Run the API (from the repo root):
```bash
venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --reload --port 8000
```

Visit http://localhost:8000/docs for the interactive Swagger UI.

## Frontend wiring (no React changes needed beyond this)

In your Vite `.env`:
```
VITE_SERVER_URL=http://localhost:8000
```

Your existing `axios.create({ baseURL: import.meta.env.VITE_SERVER_URL, withCredentials: true })`
config works unchanged — FastAPI sets the same `httpOnly` session cookie, so
`/api/auth/login`, `/api/auth/logout`, `/api/me` behave exactly like the Node version.

## What's implemented so far

- Project skeleton (`app/core`, `app/db`, `app/models`, `app/auth`, `app/routers`)
- Config via `pydantic-settings` (`.env`)
- Async Redis client
- Firebase Admin token verification
- MongoDB (Motor + Beanie) with a `User` document
- `POST /api/auth/login` — verifies Firebase ID token, upserts user, creates a
  Redis-backed session, sets an `httpOnly` cookie
- `GET /api/auth/logout` — clears the Redis session + cookie
- `GET /api/me` — protected route, returns the cached session user (this is your
  `protect` middleware equivalent — reuse `Depends(get_current_user)` on any future
  protected route)

## Agent endpoints

- `POST /api/agent/chat` — sends a prompt through the LangGraph (router → chat /
  search / coding / pdf / ppt / image / rag). Body: `{prompt, conversation_id, agent: "auto"}`.
- `POST /api/agent/chat/stream` — SSE variant. Emits events:
  - `data: {"token": "..."}` per streamed chunk (chat/search) or as a single event
    (coding/pdf/ppt/image)
  - `data: {"images": [...]}` when the agent produced images (optional)
  - `data: [DONE]` at the end
- `POST /api/documents/upload` — multipart `file` (PDF only) + `conversation_id`;
  indexes the document into Qdrant for RAG.

**Frontend note (backend only for now):** `/api/agent/chat/stream` uses SSE — the
React frontend needs an `EventSource` or a `fetch` + `ReadableStream` reader on that
endpoint to consume the `token` events incrementally. The frontend has not been
touched yet.

## RAG (vector search)

Uploads a PDF per conversation, splits it into 1000-char chunks (150 overlap),
embeds with `GoogleGenerativeAIEmbeddings` (`gemini-embedding-001`, 3,072 dimensions), and stores it in
a Qdrant collection `conv_<conversation_id>`. The `rag` agent retrieves the top 4
chunks for a question and answers strictly from that context.

## Docker

`docker compose up --build` runs Redis, Qdrant, and the FastAPI app. The app build
context is the repo root (since `requirements.txt` and `.env` live there);
`backend/Dockerfile` copies them in. Qdrant data persists in the `qdrant_storage`
volume; generated files persist via `./static:/app/static`.

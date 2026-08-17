FROM python:3.11-slim

WORKDIR /app

# Install uv for ultra-fast package installation (10x-50x faster than standard pip)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

COPY backend/ ./backend/
COPY .env .
COPY multiagent-ai-1509a-firebase-adminsdk-fbsvc-48c09390a9.json .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]

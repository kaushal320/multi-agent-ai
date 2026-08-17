FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for ultra-fast package installation (10x-50x faster than standard pip)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

COPY backend/ ./backend/
# .env is supplied at runtime via env_file in docker-compose — never bake secrets into images.
# Firebase credentials are supplied at runtime as a Docker secret.

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--log-level", "info"]

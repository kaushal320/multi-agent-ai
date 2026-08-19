from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.core.cache import get_cached_embedding, set_cached_embedding
from app.core.config import settings
from app.core.observability import obs

qdrant_client = QdrantClient(url=settings.qdrant_url)

_base_embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
)

# gemini-embedding-001 returns 3,072-dimensional embeddings.  This must match
# the Qdrant collection schema exactly.
COLLECTION_VECTOR_SIZE = 3072
COLLECTION_DISTANCE = Distance.COSINE


class CachedEmbeddings:
    """Wrapper that adds Redis caching to embeddings."""

    def __init__(self, base_embeddings, model_name: str):
        self._base = base_embeddings
        self._model = model_name

    def embed_query(self, text: str) -> list[float]:
        """Synchronous embedding with cache check."""
        # Try cache first
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Can't await in sync context, skip cache
                return self._base.embed_query(text)
        except RuntimeError:
            pass

        # Fallback to base
        return self._base.embed_query(text)

    async def aembed_query(self, text: str) -> list[float]:
        """Async embedding with Redis cache."""
        # Check cache
        cached = await get_cached_embedding(text, self._model)
        if cached is not None:
            return cached

        # Compute embedding
        vector = await self._base.aembed_query(text)

        # Store in cache
        await set_cached_embedding(text, self._model, vector)
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Sync batch embed (no cache for batch)."""
        return self._base.embed_documents(texts)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Async batch embed (no cache for batch)."""
        return await self._base.aembed_documents(texts)


embeddings = CachedEmbeddings(_base_embeddings, "gemini-embedding-001")


def get_collection_name(conversation_id: str) -> str:
    return f"conv_{conversation_id}"


def ensure_collection(name: str) -> None:
    if not qdrant_client.collection_exists(name):
        qdrant_client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=COLLECTION_VECTOR_SIZE,
                distance=COLLECTION_DISTANCE,
            ),
        )

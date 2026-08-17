from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.core.config import settings

qdrant_client = QdrantClient(url=settings.qdrant_url)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
)

# gemini-embedding-001 returns 3,072-dimensional embeddings.  This must match
# the Qdrant collection schema exactly.
COLLECTION_VECTOR_SIZE = 3072
COLLECTION_DISTANCE = Distance.COSINE


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

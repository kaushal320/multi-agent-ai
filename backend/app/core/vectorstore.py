import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=768,
)

COLLECTION_VECTOR_SIZE = 768  # gemini-embedding-001 (dims capped via output_dimensionality)
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

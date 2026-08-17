from app.agents.models import content_text, get_model
from app.agents.state import AgentState
from app.core import vectorstore

RAG_SYSTEM_PROMPT = (
    "Answer only using the provided document context. If the answer isn't in "
    "the context, say so."
)


try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False


async def rag_node(state: AgentState) -> dict:
    collection = vectorstore.get_collection_name(state["conversation_id"])
    if not vectorstore.qdrant_client.collection_exists(collection):
        return {
            "ai_response": "No documents uploaded yet for this conversation. Upload a PDF first.",
            "images": [],
        }

    if LOGFIRE_AVAILABLE:
        with logfire.span("rag_vectorstore_search", collection=collection):
            query_vector = vectorstore.embeddings.embed_query(state["prompt"])
            hits = vectorstore.qdrant_client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=4,
            )
            logfire.info("📚 [RAG Agent] Retrieved {count} context chunks", count=len(hits))
    else:
        query_vector = vectorstore.embeddings.embed_query(state["prompt"])
        hits = vectorstore.qdrant_client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=4,
        )

    context = "\n\n".join(
        str(hit.payload.get("page_content", "")) for hit in hits
    )

    result = await get_model("chat").ainvoke(
        [
            ("system", RAG_SYSTEM_PROMPT),
            ("human", f"Document context:\n{context}\n\nQuestion: {state['prompt']}"),
        ]
    )
    return {"ai_response": content_text(result.content), "images": []}


async def rag_node_stream(state: AgentState):
    """Streams RAG agent tokens directly using Qdrant document vector context."""
    collection = vectorstore.get_collection_name(state["conversation_id"])
    if not vectorstore.qdrant_client.collection_exists(collection):
        yield "No documents uploaded yet for this conversation. Please click the attachment paperclip icon to upload a PDF first."
        return

    if LOGFIRE_AVAILABLE:
        with logfire.span("rag_vectorstore_search", collection=collection, prompt=state["prompt"]):
            query_vector = vectorstore.embeddings.embed_query(state["prompt"])
            hits = vectorstore.qdrant_client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=4,
            )
            logfire.info("📚 [RAG Agent] Vectorstore retrieved {count} document chunks", count=len(hits))
    else:
        query_vector = vectorstore.embeddings.embed_query(state["prompt"])
        hits = vectorstore.qdrant_client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=4,
        )

    context = "\n\n".join(
        str(hit.payload.get("page_content", "")) for hit in hits if hit.payload
    )

    messages = [
        ("system", RAG_SYSTEM_PROMPT),
        ("human", f"Document context:\n{context}\n\nQuestion: {state['prompt']}"),
    ]

    async for chunk in get_model("chat").astream(messages):
        content = content_text(chunk.content)
        if content:
            yield content


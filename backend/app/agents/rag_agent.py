from app.agents.logging import log_agent_failure, log_agent_start, log_agent_success
from app.agents.models import _extract_usage, content_text, get_model
from app.agents.state import AgentState
from app.core import vectorstore

RAG_SYSTEM_PROMPT = (
    "Answer only using the provided document context. If the answer isn't in "
    "the context, say so."
)


async def rag_node(state: AgentState) -> dict:
    """Backward-compatible RAG answer node used outside the orchestration graph."""
    t0 = log_agent_start("rag", state)
    try:
        research = await rag_research_node(state)
        if research.get("ai_response"):
            log_agent_success("rag", state, t0, result_type="no_context")
            return research

        result = await get_model("chat").ainvoke(
            [
                ("system", RAG_SYSTEM_PROMPT),
                (
                    "human",
                    f"Document context:\n{research['rag_context']}\n\nQuestion: {state['prompt']}",
                ),
            ]
        )
        response = content_text(result.content)
        usage = _extract_usage(result)
        log_agent_success(
            "rag", state, t0, result_type="synthesized", response_length=len(response)
        )
        return {"ai_response": response, "images": [], "token_usage": usage}
    except Exception as exc:
        log_agent_failure("rag", state, exc)
        raise


async def rag_research_node(state: AgentState) -> dict:
    """Retrieve document evidence for a downstream synthesis agent."""
    t0 = log_agent_start("rag_research", state)
    try:
        collection = vectorstore.get_collection_name(state["conversation_id"])
        if not vectorstore.qdrant_client.collection_exists(collection):
            log_agent_success("rag_research", state, t0, result_type="no_collection")
            return {
                "ai_response": "No documents uploaded yet for this conversation. Upload a PDF first.",
            }

        query_vector = vectorstore.embeddings.embed_query(state["prompt"])
        hits = vectorstore.qdrant_client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=4,
        ).points

        sources = [hit.payload or {} for hit in hits]
        context = "\n\n".join(str(source.get("page_content", "")) for source in sources)

        log_agent_success(
            "rag_research",
            state,
            t0,
            collection=collection,
            chunks_retrieved=len(hits),
        )
        return {"rag_context": context, "rag_sources": sources}
    except Exception as exc:
        log_agent_failure("rag_research", state, exc)
        raise


async def rag_node_stream(state: AgentState):
    """Streams RAG agent tokens directly using Qdrant document vector context."""
    t0 = log_agent_start("rag_stream", state)
    try:
        collection = vectorstore.get_collection_name(state["conversation_id"])
        if not vectorstore.qdrant_client.collection_exists(collection):
            log_agent_success("rag_stream", state, t0, result_type="no_collection")
            yield "No documents uploaded yet for this conversation. Please click the attachment paperclip icon to upload a PDF first."
            return

        query_vector = vectorstore.embeddings.embed_query(state["prompt"])
        hits = vectorstore.qdrant_client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=4,
        ).points

        context = "\n\n".join(
            str(hit.payload.get("page_content", "")) for hit in hits if hit.payload
        )

        messages = [
            ("system", RAG_SYSTEM_PROMPT),
            ("human", f"Document context:\n{context}\n\nQuestion: {state['prompt']}"),
        ]

        token_count = 0
        async for chunk in get_model("chat").astream(messages):
            content = content_text(chunk.content)
            if content:
                token_count += 1
                yield content

        log_agent_success(
            "rag_stream",
            state,
            t0,
            collection=collection,
            chunks_retrieved=len(hits),
            tokens_yielded=token_count,
        )
    except Exception as exc:
        log_agent_failure("rag_stream", state, exc)
        raise

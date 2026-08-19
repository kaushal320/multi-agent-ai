from app.agents.logging import log_agent_failure, log_agent_start, log_agent_success
from app.agents.models import _extract_usage, content_text, get_model
from app.agents.state import AgentState
from app.core import vectorstore
from app.core.cache import get_cached_rag_context, set_cached_rag_context
from app.core.observability import obs
import logging

# BM25 for hybrid search
try:
    from rank_bm25 import BM25Okapi

    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

logger = logging.getLogger("cortex.agents.rag")

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
    """Retrieve document evidence for a downstream synthesis agent using hybrid search."""
    t0 = log_agent_start("rag_research", state)
    prompt = state["prompt"]
    conversation_id = state["conversation_id"]

    try:
        # Check RAG cache first
        collection = vectorstore.get_collection_name(conversation_id)
        cached_context = await get_cached_rag_context(collection, prompt)
        if cached_context:
            obs.metric("rag.cache.hit", 1, collection=collection)
            logger.info("RAG cache HIT for conversation=%s, prompt=%s", conversation_id, prompt[:50])

            log_agent_success(
                "rag_research",
                state,
                t0,
                collection=collection,
                result_type="cached",
            )
            return {"rag_context": cached_context, "rag_sources": []}

        if not vectorstore.qdrant_client.collection_exists(collection):
            log_agent_success("rag_research", state, t0, result_type="no_collection")
            return {
                "ai_response": "No documents uploaded yet for this conversation. Upload a PDF first.",
            }

        # Vector search (semantic)
        query_vector = vectorstore.embeddings.embed_query(prompt)
        vector_hits = vectorstore.qdrant_client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=10,  # Get more for fusion
        ).points

        # BM25 keyword search (if available)
        bm25_hits = []
        if BM25_AVAILABLE:
            # Scroll all points to build BM25 index (cached per collection)
            all_points = vectorstore.qdrant_client.scroll(
                collection_name=collection,
                limit=1000,
                with_payload=True,
            )[0]
            if all_points:
                corpus = [
                    str(p.payload.get("page_content", ""))
                    for p in all_points
                    if p.payload
                ]
                tokenized_corpus = [doc.lower().split() for doc in corpus]
                bm25 = BM25Okapi(tokenized_corpus)
                tokenized_query = state["prompt"].lower().split()
                bm25_scores = bm25.get_scores(tokenized_query)
                # Get top 10 BM25 results
                bm25_top_indices = sorted(
                    range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
                )[:10]
                bm25_hits = [
                    all_points[i] for i in bm25_top_indices if bm25_scores[i] > 0
                ]

        # Reciprocal Rank Fusion (RRF)
        def rrf_fusion(vector_results, bm25_results, k=60):
            """Fuse results using Reciprocal Rank Fusion."""
            fused = {}
            for rank, hit in enumerate(vector_results):
                doc_id = hit.id
                fused[doc_id] = fused.get(doc_id, 0) + 1 / (k + rank + 1)
            for rank, hit in enumerate(bm25_results):
                doc_id = hit.id
                fused[doc_id] = fused.get(doc_id, 0) + 1 / (k + rank + 1)
            # Sort by fused score
            sorted_ids = sorted(fused.keys(), key=lambda x: fused[x], reverse=True)
            # Reconstruct hits in fused order
            all_hits = {hit.id: hit for hit in vector_results + bm25_results}
            return [all_hits[doc_id] for doc_id in sorted_ids[:4]]

        fused_hits = rrf_fusion(vector_hits, bm25_hits)

        sources = [hit.payload or {} for hit in fused_hits]
        context = "\n\n".join(str(source.get("page_content", "")) for source in sources)

        # Cache the RAG context
        await set_cached_rag_context(collection, prompt, context)
        obs.metric("rag.cache.store", 1, collection=collection)

        log_agent_success(
            "rag_research",
            state,
            t0,
            collection=collection,
            vector_results=len(vector_hits),
            bm25_results=len(bm25_hits),
            fused_results=len(fused_hits),
        )
        return {"rag_context": context, "rag_sources": sources}
    except Exception as exc:
        log_agent_failure("rag_research", state, exc)
        raise


async def rag_node_stream(state: AgentState):
    """Streams RAG agent tokens directly using hybrid search (vector + BM25)."""
    t0 = log_agent_start("rag_stream", state)
    try:
        collection = vectorstore.get_collection_name(state["conversation_id"])
        if not vectorstore.qdrant_client.collection_exists(collection):
            log_agent_success("rag_stream", state, t0, result_type="no_collection")
            yield "No documents uploaded yet for this conversation. Please click the attachment paperclip icon to upload a PDF first."
            return

        # Vector search
        query_vector = vectorstore.embeddings.embed_query(state["prompt"])
        vector_hits = vectorstore.qdrant_client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=10,
        ).points

        # BM25 keyword search
        bm25_hits = []
        if BM25_AVAILABLE:
            all_points = vectorstore.qdrant_client.scroll(
                collection_name=collection,
                limit=1000,
                with_payload=True,
            )[0]
            if all_points:
                corpus = [
                    str(p.payload.get("page_content", ""))
                    for p in all_points
                    if p.payload
                ]
                tokenized_corpus = [doc.lower().split() for doc in corpus]
                bm25 = BM25Okapi(tokenized_corpus)
                tokenized_query = state["prompt"].lower().split()
                bm25_scores = bm25.get_scores(tokenized_query)
                bm25_top_indices = sorted(
                    range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
                )[:10]
                bm25_hits = [
                    all_points[i] for i in bm25_top_indices if bm25_scores[i] > 0
                ]

        # RRF Fusion
        def rrf_fusion(vector_results, bm25_results, k=60):
            fused = {}
            for rank, hit in enumerate(vector_results):
                doc_id = hit.id
                fused[doc_id] = fused.get(doc_id, 0) + 1 / (k + rank + 1)
            for rank, hit in enumerate(bm25_results):
                doc_id = hit.id
                fused[doc_id] = fused.get(doc_id, 0) + 1 / (k + rank + 1)
            sorted_ids = sorted(fused.keys(), key=lambda x: fused[x], reverse=True)
            all_hits = {hit.id: hit for hit in vector_results + bm25_results}
            return [all_hits[doc_id] for doc_id in sorted_ids[:4]]

        fused_hits = rrf_fusion(vector_hits, bm25_hits)

        context = "\n\n".join(
            str(hit.payload.get("page_content", ""))
            for hit in fused_hits
            if hit.payload
        )

        messages = [
            ("system", RAG_SYSTEM_PROMPT),
            ("human", f"Document context:\n{context}\n\nQuestion: {state['prompt']}"),
        ]

        token_count = 0
        async for chunk in get_model("chat", streaming=True).astream(messages):
            content = content_text(chunk.content)
            if content:
                token_count += 1
                yield content

        log_agent_success(
            "rag_stream",
            state,
            t0,
            collection=collection,
            chunks_retrieved=len(fused_hits),
            tokens_yielded=token_count,
        )
    except Exception as exc:
        log_agent_failure("rag_stream", state, exc)
        raise

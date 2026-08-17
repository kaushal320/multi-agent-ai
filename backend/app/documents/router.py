import os
import tempfile
from uuid import uuid4

from fastapi import Annotated, APIRouter, Depends, File, Form, HTTPException, UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import PointStruct

from app.auth.dependencies import get_current_user
from app.core.vectorstore import (
    embeddings,
    ensure_collection,
    get_collection_name,
    qdrant_client,
)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post("/upload")
async def upload_document(
    file: Annotated[UploadFile, File()],
    conversation_id: Annotated[str, Form()],
    user: Annotated[dict, Depends(get_current_user)],
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB.",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(pages)
    finally:
        os.unlink(tmp_path)

    if not chunks:
        raise HTTPException(
            status_code=400, detail="No extractable text found in the PDF"
        )

    collection = get_collection_name(conversation_id)
    ensure_collection(collection)

    vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
    points = [
        PointStruct(
            id=uuid4().hex,
            vector=vector,
            payload={
                "conversation_id": conversation_id,
                "source_filename": file.filename,
                "chunk_index": i,
                "page_content": chunks[i].page_content,
            },
        )
        for i, vector in enumerate(vectors)
    ]
    qdrant_client.upsert(collection_name=collection, points=points)

    return {"message": "Document indexed", "chunks": len(chunks)}

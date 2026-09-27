"""Pinecone index management + LangChain vectorstore wrapper."""
from __future__ import annotations
from typing import Optional

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

from .config import (
    PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_CLOUD,
    PINECONE_REGION, EMBED_DIM,
)
from .embeddings import get_embeddings


def get_pinecone_client() -> Pinecone:
    if not PINECONE_API_KEY:
        raise RuntimeError(
            "PINECONE_API_KEY is missing. Add it to .env or export it."
        )
    return Pinecone(api_key=PINECONE_API_KEY)


def ensure_index(pc: Optional[Pinecone] = None) -> None:
    pc = pc or get_pinecone_client()
    existing = {i["name"] for i in pc.list_indexes()}
    if PINECONE_INDEX_NAME in existing:
        return
    print(f"[vectorstore] creating index '{PINECONE_INDEX_NAME}' "
          f"({EMBED_DIM}d, cosine, {PINECONE_CLOUD}/{PINECONE_REGION})")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )


def get_vectorstore() -> PineconeVectorStore:
    pc = get_pinecone_client()
    ensure_index(pc)
    index = pc.Index(PINECONE_INDEX_NAME)
    return PineconeVectorStore(index=index, embedding=get_embeddings())


def upsert_chunks(chunks, batch_size: int = 50) -> int:
    vs = get_vectorstore()
    docs = [
        Document(page_content=c.text, metadata=c.to_metadata())
        for c in chunks
    ]
    total = 0
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        vs.add_documents(batch, ids=[d.metadata["chunk_id"] for d in batch])
        total += len(batch)
        print(f"[vectorstore] upserted {total}/{len(docs)}")
    return total


def fetch_chunk(chunk_id: str) -> Optional[dict]:
    pc = get_pinecone_client()
    index = pc.Index(PINECONE_INDEX_NAME)
    res = index.fetch(ids=[chunk_id])
    vectors = res.get("vectors", {}) if hasattr(res, "get") else res.vectors
    if not vectors or chunk_id not in vectors:
        return None
    v = vectors[chunk_id]
    md = dict(getattr(v, "metadata", {}) or {})
    md["chunk_id"] = chunk_id
    return md

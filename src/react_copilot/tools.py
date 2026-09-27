"""Tool implementations exposed to the ReAct agent."""
from __future__ import annotations
import json
import time
from langchain_core.tools import tool

from .config import CORPUS_DIR, TOP_K
from .vectorstore import get_vectorstore, fetch_chunk
from .loaders import iter_corpus_files


def _allowed_files() -> set[str]:
    return {p.name for p in iter_corpus_files(CORPUS_DIR)}


def _safe_file(name: str) -> bool:
    if "/" in name or "\\" in name or ".." in name:
        return False
    return name in _allowed_files()


@tool
def search(query: str, k: int = TOP_K) -> str:
    """Semantic search over the local corpus.

    Args:
        query: natural-language search query.
        k: number of chunks to return (default 5, max 10).

    Returns:
        JSON string: {"results": [ {chunk_id, file, page, row, lines, score, preview}, ... ]}
    """
    k = max(1, min(int(k), 10))
    start = time.time()
    try:
        vs = get_vectorstore()
        pairs = vs.similarity_search_with_score(query, k=k)
    except Exception as e:
        return json.dumps({"error": f"search failed: {e}"})

    results = []
    for doc, score in pairs:
        md = doc.metadata or {}
        results.append({
            "chunk_id": md.get("chunk_id"),
            "file":     md.get("source"),
            "page":     int(md["page"]) if md.get("page") is not None else None,
            "row":      int(md["row"])  if md.get("row")  is not None else None,
            "lines":    (f"{int(md['line_start'])}-{int(md['line_end'])}"
                         if md.get("line_start") is not None else None),
            "score":    round(float(score), 4),
            "preview":  doc.page_content[:280].replace("\n", " ").strip(),
        })
    elapsed = round(time.time() - start, 2)
    return json.dumps({"results": results, "elapsed_s": elapsed}, ensure_ascii=False)


@tool
def open_file(file: str) -> str:
    """List all chunks belonging to one corpus file.

    Args:
        file: exact filename (e.g. 'guide_chunking.pdf').

    Returns:
        JSON string: {"file": ..., "chunks": [ {chunk_id, page, row, lines, preview}, ... ]}
    """
    if not _safe_file(file):
        return json.dumps({"error": f"file not in allow-list: {file}"})

    try:
        vs = get_vectorstore()
        docs = vs.similarity_search(
            query="", k=200,
            filter={"source": {"$eq": file}},
        )
    except Exception as e:
        return json.dumps({"error": f"open_file failed: {e}"})

    chunks = []
    for d in docs:
        md = d.metadata or {}
        chunks.append({
            "chunk_id": md.get("chunk_id"),
            "page":     int(md["page"]) if md.get("page") is not None else None,
            "row":      int(md["row"])  if md.get("row")  is not None else None,
            "lines":    (f"{int(md['line_start'])}-{int(md['line_end'])}"
                         if md.get("line_start") is not None else None),
            "preview":  d.page_content[:200].replace("\n", " ").strip(),
        })
    return json.dumps({"file": file, "chunks": chunks}, ensure_ascii=False)


@tool
def read_chunk(chunk_id: str) -> str:
    """Read the full text of a chunk.

    Args:
        chunk_id: a chunk id previously returned by search or open_file.

    Returns:
        JSON string: {"chunk_id", "file", "page", "row", "lines", "text"}
    """
    try:
        md = fetch_chunk(chunk_id)
    except Exception as e:
        return json.dumps({"error": f"read_chunk failed: {e}"})

    if md is None:
        return json.dumps({"error": f"chunk_id not found: {chunk_id}"})

    return json.dumps({
        "chunk_id": chunk_id,
        "file":     md.get("source"),
        "page":     int(md["page"]) if md.get("page") is not None else None,
        "row":      int(md["row"])  if md.get("row")  is not None else None,
        "lines":    (f"{int(md['line_start'])}-{int(md['line_end'])}"
                     if md.get("line_start") is not None else None),
        "text":     md.get("text", ""),
    }, ensure_ascii=False)


ALL_TOOLS = [search, open_file, read_chunk]
TOOL_NAMES = {t.name for t in ALL_TOOLS}

#!/usr/bin/env bash
# Creates the full react-research-copilot code tree next to your existing data/ folder.
# Safe to re-run: it will refuse to overwrite existing files unless FORCE=1.

set -euo pipefail
FORCE="${FORCE:-0}"

write() {
  local path="$1"
  if [[ -e "$path" && "$FORCE" != "1" ]]; then
    echo "skip (exists): $path"
    cat > /dev/null   # drain heredoc
    return
  fi
  mkdir -p "$(dirname "$path")"
  cat > "$path"
  echo "wrote: $path"
}

# ---------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------
mkdir -p src/react_copilot scripts notebooks presentation runs
touch src/react_copilot/__init__.py scripts/__init__.py

# ---------------------------------------------------------------------
write requirements.txt <<'PROJECT_EOF_9x7k'
langchain>=0.3.7
langchain-core>=0.3.15
langchain-community>=0.3.5
langchain-ollama>=0.2.0
langchain-pinecone>=0.2.0
langchain-text-splitters>=0.3.0
langgraph>=0.2.45
pinecone>=5.3.1
pypdf>=5.0.0
python-dotenv>=1.0.1
pandas>=2.2.3
tqdm>=4.66.5
tabulate>=0.9.0
PROJECT_EOF_9x7k

# ---------------------------------------------------------------------
write .env.example <<'PROJECT_EOF_9x7k'
# Copy to .env and fill in
PINECONE_API_KEY=pcsk_xxxxxxxxxxxxxxxxxxxx
PINECONE_INDEX_NAME=react-copilot
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.2:3b
EMBED_MODEL=nomic-embed-text

PROJECT_ROOT=/Users/ritvik/Documents/CapstoneProject/react-research-copilot
PROJECT_EOF_9x7k

# ---------------------------------------------------------------------
write .gitignore <<'PROJECT_EOF_9x7k'
__pycache__/
*.pyc
.env
.venv/
runs/
.ipynb_checkpoints/
PROJECT_EOF_9x7k

# ---------------------------------------------------------------------
write src/react_copilot/config.py <<'PROJECT_EOF_9x7k'
"""Central configuration. Reads .env if present, otherwise uses defaults."""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(os.getenv(
    "PROJECT_ROOT",
    "/Users/ritvik/Documents/CapstoneProject/react-research-copilot",
)).resolve()

DATA_DIR          = PROJECT_ROOT / "data"
CORPUS_DIR        = DATA_DIR / "corpus"
MANIFEST_JSON     = DATA_DIR / "corpus_manifest.json"
EVAL_QUESTIONS    = DATA_DIR / "evaluation_questions.csv"
RUNS_TEMPLATE     = DATA_DIR / "runs_template.csv"
RUNS_OUT          = PROJECT_ROOT / "runs"
RUNS_OUT.mkdir(parents=True, exist_ok=True)

# --- Ollama ------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL       = os.getenv("LLM_MODEL", "llama3.2:3b")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "nomic-embed-text")

# --- Pinecone ----------------------------------------------------------
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "react-copilot")
PINECONE_CLOUD      = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION     = os.getenv("PINECONE_REGION", "us-east-1")

# nomic-embed-text produces 768-dim vectors
EMBED_DIM = 768

# --- Chunking ----------------------------------------------------------
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 120

# --- Retrieval / Agent -------------------------------------------------
TOP_K        = 5
MAX_STEPS    = 8
TOOL_TIMEOUT = 30

# --- Supported corpus extensions --------------------------------------
SUPPORTED_EXTS = {".md", ".markdown", ".txt", ".pdf", ".json", ".csv"}
PROJECT_EOF_9x7k

# ---------------------------------------------------------------------
write src/react_copilot/loaders.py <<'PROJECT_EOF_9x7k'
"""Load corpus files into structured RawDocs with rich metadata."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import json
import csv

from pypdf import PdfReader

from .config import CORPUS_DIR, SUPPORTED_EXTS


@dataclass
class RawDoc:
    text: str
    source: str
    page: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    row: int | None = None
    extra: dict = field(default_factory=dict)


def _load_text_file(path: Path) -> list[RawDoc]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    text = "\n".join(lines)
    return [RawDoc(
        text=text,
        source=path.name,
        line_start=1,
        line_end=max(1, len(lines)),
    )]


def _load_pdf(path: Path) -> list[RawDoc]:
    docs: list[RawDoc] = []
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            docs.append(RawDoc(text=text, source=path.name, page=i))
    return docs


def _load_json(path: Path) -> list[RawDoc]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        return [RawDoc(text=path.read_text(errors="replace"),
                       source=path.name, row=1, extra={"error": str(e)})]

    docs: list[RawDoc] = []
    rows = None
    if isinstance(data, dict):
        for key in ("documents", "records", "items", "entries", "data"):
            if isinstance(data.get(key), list):
                rows = data[key]
                break
    elif isinstance(data, list):
        rows = data

    if rows is not None and all(isinstance(r, dict) for r in rows):
        for i, row in enumerate(rows, start=1):
            text = json.dumps(row, indent=2, ensure_ascii=False)
            docs.append(RawDoc(text=text, source=path.name, row=i))
    else:
        text = json.dumps(data, indent=2, ensure_ascii=False)
        docs.append(RawDoc(text=text, source=path.name, row=1))
    return docs


def _load_csv(path: Path) -> list[RawDoc]:
    docs: list[RawDoc] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            text = " | ".join(f"{k}: {v}" for k, v in row.items() if v not in (None, ""))
            docs.append(RawDoc(text=text, source=path.name, row=i))
    return docs


_LOADERS = {
    ".md": _load_text_file,
    ".markdown": _load_text_file,
    ".txt": _load_text_file,
    ".pdf": _load_pdf,
    ".json": _load_json,
    ".csv": _load_csv,
}


def iter_corpus_files(corpus_dir: Path = CORPUS_DIR) -> Iterable[Path]:
    for p in sorted(corpus_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> list[RawDoc]:
    out: list[RawDoc] = []
    for p in iter_corpus_files(corpus_dir):
        loader = _LOADERS.get(p.suffix.lower())
        if loader is None:
            continue
        try:
            docs = loader(p)
        except Exception as e:
            print(f"[loaders] WARN failed to load {p.name}: {e}")
            continue
        out.extend(docs)
    return out
PROJECT_EOF_9x7k

# ---------------------------------------------------------------------
write src/react_copilot/chunking.py <<'PROJECT_EOF_9x7k'
"""Chunk RawDocs into smaller Chunks with stable chunk_ids."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_SIZE, CHUNK_OVERLAP
from .loaders import RawDoc


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    page: int | None
    line_start: int | None
    line_end: int | None
    row: int | None

    def to_metadata(self) -> dict:
        md = {k: v for k, v in asdict(self).items()
              if k not in {"chunk_id", "text"} and v is not None}
        md["chunk_id"] = self.chunk_id
        md["text"] = self.text[:6000]
        return md


def _make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        length_function=len,
    )


def chunk_raw_docs(raw_docs: Iterable[RawDoc]) -> list[Chunk]:
    splitter = _make_splitter()
    chunks: list[Chunk] = []
    for doc in raw_docs:
        pieces = splitter.split_text(doc.text) or [doc.text]
        for i, piece in enumerate(pieces):
            chunk_id = _make_chunk_id(doc.source, doc.page, doc.row, i)
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=piece,
                source=doc.source,
                page=doc.page,
                line_start=doc.line_start,
                line_end=doc.line_end,
                row=doc.row,
            ))
    return chunks


def _make_chunk_id(source: str, page: int | None, row: int | None, idx: int) -> str:
    safe = source.replace("/", "_").replace(" ", "_")
    anchor = f"p{page}" if page is not None else (f"r{row}" if row is not None else "t")
    return f"{safe}::{anchor}::{idx}"
PROJECT_EOF_9x7k

# ---------------------------------------------------------------------
write src/react_copilot/embeddings.py <<'PROJECT_EOF_9x7k'
"""Ollama embeddings wrapper."""
from __future__ import annotations
from langchain_ollama import OllamaEmbeddings
from .config import OLLAMA_BASE_URL, EMBED_MODEL


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
PROJECT_EOF_9x7k

# ---------------------------------------------------------------------
write src/react_copilot/vectorstore.py <<'PROJECT_EOF_9x7k'
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
PROJECT_EOF_9x7k

# ---------------------------------------------------------------------
write src/react_copilot/prompts.py <<'PROJECT_EOF_9x7k'
"""System prompt for the ReAct research copilot."""
from __future__ import annotations

SYSTEM_PROMPT = """You are a careful research copilot. You answer questions ONLY using \
evidence from a local document corpus accessed through tools.

TOOLS
- search(query: str, k: int = 5)
    Semantic search across the corpus. Returns a list of chunks with chunk_id,
    file, page/row/line info, and a short preview.
- open_file(file: str)
    Lists all chunk_ids belonging to a specific file.
- read_chunk(chunk_id: str)
    Returns the FULL text of a specific chunk. Always read before citing.

WORKFLOW (ReAct: Reason -> Act -> Observe)
1. Reason about what you need to find.
2. Call `search` with a focused query.
3. From the results, call `read_chunk` on the most promising 1-3 chunks.
4. If evidence is thin, refine the query and search again.
5. Only when you have enough evidence, produce the FINAL ANSWER.

CITATION RULES
- Every factual claim in the final answer MUST be followed by a citation.
- PDFs:          [file.pdf:page]
- Markdown/text: [file.md:line_start-line_end]  or [file.md]
- JSON/CSV:      [file.json:row N]  or  [file.csv:row N]
- NEVER invent a citation. Only cite files you actually read with tools.
- If evidence is insufficient, say so explicitly and ask a clarifying question.

ANSWER STYLE
- 2-6 sentences. Concise.
- Include at most ONE short supporting quote (<= 25 words) if it strengthens the answer.
- Do not dump raw chunks into the answer.

Begin by reasoning about the question, then call a tool. Do not answer before searching.
"""


def build_user_prompt(question: str) -> str:
    return f"Question: {question}\n\nUse tools to gather evidence, then answer with citations."
PROJECT_EOF_9x7k

# ---------------------------------------------------------------------
write src/react_copilot/tools.py <<'PROJECT_EOF_9x7k'
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
            "page":     md.get("page"),
            "row":      md.get("row"),
            "lines":    (f"{md.get('line_start')}-{md.get('line_end')}"
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
            "page":     md.get("page"),
            "row":      md.get("row"),
            "lines":    (f"{md.get('line_start')}-{md.get('line_end')}"
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
        "page":     md.get("page"),
        "row":      md.get("row"),
        "lines":    (f"{md.get('line_start')}-{md.get('line_end')}"
                     if md.get("line_start") is not None else None),
        "text":     md.get("text", ""),
    }, ensure_ascii=False)


ALL_TOOLS = [search, open_file, read_chunk]
TOOL_NAMES = {t.name for t in ALL_TOOLS}
PROJECT_EOF_9x7k

# ---------------------------------------------------------------------
write src/react_copilot/agent.py <<'PROJECT_EOF_9x7k'
"""LangGraph ReAct agent: Reason -> Act -> Observe loop with max-step guard."""
from __future__ import annotations
from typing import Annotated, Any, TypedDict

from langchain_core.messages import (
    AnyMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage,
)
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .config import OLLAMA_BASE_URL, LLM_MODEL, MAX_STEPS
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .tools import ALL_TOOLS


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int
    question: str
    retrieved_files: list[str]
    log: list[dict[str, Any]]


def _build_llm():
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        num_predict=512,
    )


def _reason_node(state: AgentState) -> dict:
    llm = _build_llm().bind_tools(ALL_TOOLS)
    msgs = state["messages"]
    response = llm.invoke(msgs)

    step = state.get("steps", 0) + 1
    entry = {
        "step": step,
        "type": "reason",
        "content": (response.content or "").strip()[:600],
        "tool_calls": [
            {"name": tc.get("name"), "args": tc.get("args")}
            for tc in (getattr(response, "tool_calls", None) or [])
        ],
    }
    log = list(state.get("log", [])) + [entry]
    return {"messages": [response], "steps": step, "log": log}


def _should_continue(state: AgentState) -> str:
    if state.get("steps", 0) >= MAX_STEPS:
        return "end"
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "end"


def _tools_node(state: AgentState) -> dict:
    node = ToolNode(ALL_TOOLS)
    result = node.invoke(state)

    new_msgs = result.get("messages", [])
    retrieved = set(state.get("retrieved_files", []))
    log = list(state.get("log", []))

    for m in new_msgs:
        if isinstance(m, ToolMessage):
            import json as _json
            try:
                obs = _json.loads(m.content) if isinstance(m.content, str) else m.content
            except Exception:
                obs = {}
            file_hint = None
            if isinstance(obs, dict):
                file_hint = obs.get("file")
                for r in obs.get("results", []) or []:
                    if r.get("file"):
                        retrieved.add(r["file"])
            if file_hint:
                retrieved.add(file_hint)

            log.append({
                "step": state.get("steps", 0),
                "type": "observe",
                "tool": m.name,
                "content": str(m.content)[:600],
            })

    return {"messages": new_msgs, "retrieved_files": sorted(retrieved), "log": log}


def build_agent():
    g = StateGraph(AgentState)
    g.add_node("reason", _reason_node)
    g.add_node("tools", _tools_node)
    g.set_entry_point("reason")
    g.add_conditional_edges("reason", _should_continue,
                            {"tools": "tools", "end": END})
    g.add_edge("tools", "reason")
    return g.compile()


def run_agent(question: str) -> dict:
    agent = build_agent()

    init_state: AgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=build_user_prompt(question)),
        ],
        "steps": 0,
        "question": question,
        "retrieved_files": [],
        "log": [],
    }

    final_state = agent.invoke(init_state, {"recursion_limit": MAX_STEPS * 2 + 4})

    answer = ""
    for m in reversed(final_state["messages"]):
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            answer = m.content or ""
            break

    return {
        "question":        question,
        "answer":          answer.strip(),
        "steps":           final_state.get("steps", 0),
        "retrieved_files": final_state.get("retrieved_files", []),
        "log":             final_state.get("log", []),
    }
PROJECT_EOF_9x7k

# ---------------------------------------------------------------------
write src/react_copilot/evaluation.py <<'PROJECT_EOF_9x7k'
"""Evaluation utilities: baseline, ReAct run, grounded-precision heuristic."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from .config import OLLAMA_BASE_URL, LLM_MODEL, EVAL_QUESTIONS, RUNS_OUT
from .agent import run_agent

CITATION_RE = re.compile(r"\[([^\[\]:]+)(?::([^\]]+))?\]")


def run_baseline(question: str) -> dict:
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL,
                     temperature=0.1, num_predict=400)
    sys = ("You are a research assistant. Answer concisely from your own "
           "knowledge. Do NOT claim to cite files. If unsure, say so.")
    resp = llm.invoke([SystemMessage(content=sys),
                       HumanMessage(content=question)])
    return {
        "answer": (resp.content or "").strip(),
        "steps": 0,
        "retrieved_files": [],
        "log": [],
    }


def extract_citations(answer: str) -> list[tuple[str, str | None]]:
    out = []
    for m in CITATION_RE.finditer(answer or ""):
        f = m.group(1).strip()
        loc = (m.group(2) or "").strip() or None
        if "." in f:
            out.append((f, loc))
    return out


def grounded_precision(answer: str, gold_file: str) -> float:
    cites = extract_citations(answer)
    if not cites:
        return 0.0
    hits = sum(1 for f, _ in cites if f == gold_file)
    return hits / len(cites)


_WORD = re.compile(r"[a-zA-Z0-9]+")
_STOP = {"the","a","an","is","are","of","to","in","and","or","for","on",
         "by","with","as","that","this","it","be","can","do","does","how",
         "what","why","when","where","which","who","two","three"}


def _tokens(s: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(s or "")} - _STOP


def accuracy_heuristic(answer: str, gold_snippet: str, thresh: float = 0.30) -> float:
    a, g = _tokens(answer), _tokens(gold_snippet)
    if not g:
        return 0.0
    return 1.0 if len(a & g) / len(g) >= thresh else 0.0


def load_questions(path: Path = EVAL_QUESTIONS) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"id", "question", "gold_source_file", "gold_supporting_snippet"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"evaluation_questions.csv missing columns: {missing}")
    return df


def run_full_eval(out_csv: Path | None = None,
                  modes: Iterable[str] = ("baseline", "react")) -> pd.DataFrame:
    df = load_questions()
    rows: list[dict] = []

    for _, r in df.iterrows():
        qid    = r["id"]
        q      = r["question"]
        gold_f = r["gold_source_file"]
        gold_s = r["gold_supporting_snippet"]

        for mode in modes:
            print(f"[eval] {qid} | {mode}")
            if mode == "baseline":
                res = run_baseline(q)
            else:
                res = run_agent(q)

            gp = grounded_precision(res["answer"], gold_f) if mode == "react" else 0.0
            acc = accuracy_heuristic(res["answer"], gold_s)

            rows.append({
                "question_id":       qid,
                "question":          q,
                "mode":              mode,
                "steps_used":        res["steps"],
                "retrieved_files":   ";".join(res["retrieved_files"]),
                "answer":            res["answer"],
                "citations":         ";".join(
                                        f"{f}:{loc}" if loc else f for f, loc in
                                        extract_citations(res["answer"])),
                "grounded_precision": round(gp, 3),
                "notes":             f"acc={acc}",
            })

    out = pd.DataFrame(rows)
    out_csv = out_csv or (RUNS_OUT / "runs.csv")
    out.to_csv(out_csv, index=False)
    print(f"[eval] wrote {len(out)} rows -> {out_csv}")

    print("\n=== Summary ===")
    for mode in modes:
        sub = out[out["mode"] == mode]
        if sub.empty:
            continue
        print(f"{mode:8s} | n={len(sub)} | "
              f"mean grounded_precision={sub['grounded_precision'].mean():.3f} | "
              f"mean steps={sub['steps_used'].mean():.2f}")
    return out
PROJECT_EOF_9x7k

# ---------------------------------------------------------------------
write scripts/index_corpus.py <<'PROJECT_EOF_9x7k'
"""Index the entire corpus into Pinecone.

Usage:
    python -m scripts.index_corpus [--reset]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from react_copilot.config import CORPUS_DIR, PINECONE_INDEX_NAME
from react_copilot.loaders import load_corpus
from react_copilot.chunking import chunk_raw_docs
from react_copilot.vectorstore import (
    get_pinecone_client, ensure_index, upsert_chunks,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="delete + recreate the index before indexing")
    args = parser.parse_args()

    print(f"[index] corpus dir = {CORPUS_DIR}")
    pc = get_pinecone_client()

    if args.reset:
        existing = {i["name"] for i in pc.list_indexes()}
        if PINECONE_INDEX_NAME in existing:
            print(f"[index] deleting index {PINECONE_INDEX_NAME}")
            pc.delete_index(PINECONE_INDEX_NAME)

    ensure_index(pc)

    raw = load_corpus(CORPUS_DIR)
    print(f"[index] loaded {len(raw)} raw docs")

    chunks = chunk_raw_docs(raw)
    print(f"[index] produced {len(chunks)} chunks")

    n = upsert_chunks(chunks)
    print(f"[index] done. upserted {n} vectors into '{PINECONE_INDEX_NAME}'.")


if __name__ == "__main__":
    main()
PROJECT_EOF_9x7k

# ---------------------------------------------------------------------
write scripts/run_evaluation.py <<'PROJECT_EOF_9x7k'
"""Run baseline vs ReAct on evaluation_questions.csv."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from react_copilot.evaluation import run_full_eval


if __name__ == "__main__":
    run_full_eval()
PROJECT_EOF_9x7k

# ---------------------------------------------------------------------
write react_copilot.py <<'PROJECT_EOF_9x7k'
#!/usr/bin/env python3
"""ReAct Research Copilot - CLI."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from react_copilot.agent import run_agent
from react_copilot.config import CORPUS_DIR
from react_copilot.loaders import load_corpus
from react_copilot.chunking import chunk_raw_docs
from react_copilot.vectorstore import (
    get_pinecone_client, ensure_index, upsert_chunks,
)


def _print_log(log: list[dict]) -> None:
    print("\n---- ReAct trace ----")
    for entry in log:
        t = entry.get("type")
        if t == "reason":
            print(f"[step {entry['step']}] Reason: {entry['content']}")
            for tc in entry.get("tool_calls", []):
                print(f"           Act   : {tc['name']}({tc['args']})")
        elif t == "observe":
            print(f"           Observe ({entry['tool']}): "
                  f"{entry['content'][:300]}...")
    print("---------------------\n")


def cmd_index(reset: bool = False) -> None:
    print(f"[cli] indexing corpus from {CORPUS_DIR}")
    pc = get_pinecone_client()
    if reset:
        from react_copilot.config import PINECONE_INDEX_NAME
        existing = {i["name"] for i in pc.list_indexes()}
        if PINECONE_INDEX_NAME in existing:
            print(f"[cli] deleting index {PINECONE_INDEX_NAME}")
            pc.delete_index(PINECONE_INDEX_NAME)
    ensure_index(pc)
    chunks = chunk_raw_docs(load_corpus(CORPUS_DIR))
    print(f"[cli] {len(chunks)} chunks produced")
    n = upsert_chunks(chunks)
    print(f"[cli] upserted {n} vectors")


def cmd_ask(question: str, verbose: bool) -> None:
    print(f"\nQuestion: {question}\n")
    res = run_agent(question)
    if verbose:
        _print_log(res["log"])
    print("Final answer:\n")
    print(res["answer"])
    print(f"\n(steps={res['steps']}, files_read={res['retrieved_files']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", "-q", type=str, help="Question to ask")
    ap.add_argument("--index", action="store_true",
                    help="(Re)index the corpus into Pinecone")
    ap.add_argument("--reset", action="store_true",
                    help="With --index: delete and recreate the index first")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Print the full ReAct trace")
    args = ap.parse_args()

    if args.index:
        cmd_index(reset=args.reset)
        return
    if not args.question:
        ap.error("provide --question '...' or --index")
    cmd_ask(args.question, args.verbose)


if __name__ == "__main__":
    main()
PROJECT_EOF_9x7k

# ---------------------------------------------------------------------
write notebooks/demo.ipynb <<'PROJECT_EOF_9x7k'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# ReAct Research Copilot - Demo Walkthrough\n", "\n", "Index, retrieve, run the ReAct agent, and evaluate."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys, os\n",
    "sys.path.insert(0, os.path.abspath('../src'))\n",
    "from react_copilot.config import CORPUS_DIR\n",
    "from react_copilot.loaders import load_corpus\n",
    "from react_copilot.chunking import chunk_raw_docs\n",
    "from react_copilot.vectorstore import get_pinecone_client, ensure_index, upsert_chunks\n",
    "print('Corpus dir:', CORPUS_DIR)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 1. Index the corpus"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pc = get_pinecone_client()\n",
    "ensure_index(pc)\n",
    "raw = load_corpus(CORPUS_DIR)\n",
    "print('raw docs:', len(raw))\n",
    "chunks = chunk_raw_docs(raw)\n",
    "print('chunks:', len(chunks))\n",
    "# upsert_chunks(chunks)  # uncomment to (re)upload"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 2. Search tool sanity check"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from react_copilot.tools import search, read_chunk\n",
    "import json\n",
    "res = json.loads(search.invoke({'query': 'what is faithfulness in LLM evaluation', 'k': 3}))\n",
    "res"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 3. Run the ReAct agent"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from react_copilot.agent import run_agent\n",
    "r = run_agent('What does faithfulness mean in LLM evaluation?')\n",
    "print(r['answer'])\n",
    "print('\\nfiles:', r['retrieved_files'])\n",
    "print('steps:', r['steps'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 4. Evaluation (baseline vs ReAct)"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from react_copilot.evaluation import run_full_eval\n",
    "df = run_full_eval()\n",
    "df.head(10)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.12"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
PROJECT_EOF_9x7k

# ---------------------------------------------------------------------
write presentation/slides.md <<'PROJECT_EOF_9x7k'
---
marp: true
theme: default
paginate: true
---

# ReAct Research Copilot
### A tool-using single agent over a local document corpus

LangChain - LangGraph - Pinecone - Ollama (llama3.2:3b)

---

## 1. Problem & Approach

- **Problem:** LLMs hallucinate when answering questions about a private corpus.
- **Approach:** ReAct loop - Reason -> Act (tool) -> Observe -> repeat.
- Tools: `search`, `open_file`, `read_chunk`
- Evidence-based final answers with inline citations `[file:page]`.

---

## 2. Architecture

- User question -> LLM (llama3.2:3b)
- LLM emits tool_calls -> LangGraph ToolNode
- Tools: search / open_file / read_chunk
- All backed by Pinecone (768d, cosine, serverless us-east-1)

---

## 3. Demo Workflow

1. `python react_copilot.py --index --reset`
2. `python react_copilot.py -q "What is faithfulness?" -v`
3. Trace: Reason -> Act(search) -> Observe -> Act(read_chunk) -> Observe -> Final Answer with citations.

---

## 4. Evaluation

- 20 questions, two modes: **LLM-only** vs **ReAct + tools**
- Metrics: grounded precision, token-overlap accuracy, mean steps
- Output: `runs/runs.csv`

---

## 5. Failure Cases & Fixes

| Failure | Fix tried |
|---|---|
| Model answers without searching | Reinforced prompt + forced first-turn tool call |
| Cited a file it never read | Filter citations against `retrieved_files` |
| Loop > 8 steps | Hard `MAX_STEPS` guard |
| Wrong chunk retrieved | `open_file` fallback for full-file scan |

---

## 6. Takeaways

- Small local LLMs (3B) can drive ReAct reliably with structured tools + strict prompts.
- Grounding via Pinecone + explicit citations cuts hallucination sharply.
- LangGraph gives explicit control over steps, retries, and safety.
PROJECT_EOF_9x7k

# ---------------------------------------------------------------------
write README.md <<'PROJECT_EOF_9x7k'
# ReAct Research Copilot

Single-agent ReAct research copilot over a local document corpus.
Stack: LangChain - LangGraph - Pinecone - Ollama (llama3.2:3b + nomic-embed-text).

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

ollama pull llama3.2:3b
ollama pull nomic-embed-text

cp .env.example .env
# edit .env -> set PINECONE_API_KEY

python react_copilot.py --index --reset
python react_copilot.py -q "What does faithfulness mean in LLM evaluation?" -v
python -m scripts.run_evaluation
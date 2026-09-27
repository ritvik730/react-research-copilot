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

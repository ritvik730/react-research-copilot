"""Ollama embeddings wrapper."""
from __future__ import annotations
from langchain_ollama import OllamaEmbeddings
from .config import OLLAMA_BASE_URL, EMBED_MODEL


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

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

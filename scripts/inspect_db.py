"""Quick read-only inspection of the persistent Chroma vector database.

Prints:
    - Total chunk count.
    - Breakdown of chunks by `source` metadata field.
    - Breakdown of chunks by `(source, year)` (chunks per indexed report).
    - A few sample chunks with their full metadata and a content preview.

Usage:
    python scripts/inspect_db.py

This script does not modify the database in any way. Safe to run at any time.
"""

from collections import Counter

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from local_rag_sandbox.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
)

SAMPLE_QUERY = "hydrogen"
NUM_SAMPLES = 3
CONTENT_PREVIEW_CHARS = 80


def main() -> None:
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        print(f"No Chroma database found at {CHROMA_DIR}.")
        print("Run scripts/ingest.py first to build the index.")
        return

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    all_records = vectorstore.get()
    metadatas = all_records.get("metadatas") or []
    total = len(metadatas)

    print(f"Chroma collection: {COLLECTION_NAME!r} at {CHROMA_DIR}")
    print(f"Total chunks:      {total}")

    if total == 0:
        print("\nCollection is empty.")
        return

    by_source = Counter(m.get("source", "<missing>") for m in metadatas)
    print("\nChunks per source:")
    for source, count in sorted(by_source.items()):
        print(f"  {source:<12} {count:>6}")

    by_source_year = Counter(
        (m.get("source", "<missing>"), m.get("year", "<missing>")) for m in metadatas
    )
    print("\nChunks per (source, year):")
    for (source, year), count in sorted(by_source_year.items()):
        print(f"  {source:<12} {year!s:<6} {count:>6}")

    print(f"\nSample chunks (similarity search for {SAMPLE_QUERY!r}):")
    samples = vectorstore.similarity_search(SAMPLE_QUERY, k=NUM_SAMPLES)
    for i, doc in enumerate(samples, start=1):
        preview = doc.page_content[:CONTENT_PREVIEW_CHARS].replace("\n", " ")
        print(f"\n  --- Chunk {i} ---")
        print(f"  metadata: {doc.metadata}")
        print(f"  content : {preview}...")


if __name__ == "__main__":
    main()

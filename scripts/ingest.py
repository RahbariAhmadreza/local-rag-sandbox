"""Ingest source documents into a persistent Chroma vector database.

Scans DATA_DIR for any files matching SUPPORTED_EXTENSIONS, dispatches
each to the right LangChain document loader, splits the loaded text into
overlapping chunks, embeds the chunks with the local Ollama embedding
model, and persists everything to CHROMA_DIR.

Re-running this script ADDS documents to the existing collection rather
than replacing it. To start fresh, delete CHROMA_DIR first:

    rm -rf chroma_db/
"""

import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from local_rag_sandbox.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DATA_DIR,
    EMBED_MODEL,
    SUPPORTED_EXTENSIONS,
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def load_path(path: Path) -> list[Document]:
    """Load a single file using the loader appropriate for its extension."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(path)).load()
    if ext in (".txt", ".md"):
        return TextLoader(str(path), encoding="utf-8").load()
    if ext == ".csv":
        return CSVLoader(file_path=str(path)).load()
    raise ValueError(f"No loader registered for extension {ext!r} (file: {path})")


def discover_documents() -> list[Path]:
    """Return all files in DATA_DIR with a supported extension, sorted."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory does not exist: {DATA_DIR}")
    return [
        p
        for p in sorted(DATA_DIR.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def load_all(paths: list[Path]) -> list[Document]:
    """Load every file using the right loader; skip and report any that fail."""
    docs: list[Document] = []
    for path in paths:
        print(f"  Loading: {path.name}")
        try:
            loaded = load_path(path)
            docs.extend(loaded)
            print(f"    Loaded {len(loaded)} segment(s).")
        except Exception as exc:  # noqa: BLE001
            print(f"    SKIPPED ({type(exc).__name__}): {exc}")
    return docs


def main() -> None:
    print(f"Scanning {DATA_DIR} ...")
    files = discover_documents()

    if not files:
        print(f"No supported files found in {DATA_DIR}.")
        print(f"Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}")
        return

    print(f"Found {len(files)} file(s):")
    for path in files:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  - {path.name} ({size_mb:.1f} MB)")

    print("\nLoading documents:")
    documents = load_all(files)
    if not documents:
        print("No documents were successfully loaded. Aborting.")
        return
    print(f"\nTotal loaded segments: {len(documents)}")

    print(f"\nSplitting into chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks: {len(chunks)}")

    print(f"\nEmbedding with {EMBED_MODEL!r} and persisting to {CHROMA_DIR} ...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    start = time.perf_counter()
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )
    elapsed = time.perf_counter() - start

    print(
        f"\nDone in {elapsed:.1f}s. "
        f"Chroma collection '{COLLECTION_NAME}' saved to: {CHROMA_DIR}"
    )


if __name__ == "__main__":
    main()

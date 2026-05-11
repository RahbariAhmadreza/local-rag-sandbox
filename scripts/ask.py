"""Ask questions over the ingested document collection.

Loads the persistent Chroma vector database, embeds the user's question
with the same embedding model used during ingestion, retrieves the most
semantically similar chunks, and passes them as context to a local
ChatOllama LLM that returns a structured markdown answer.

Usage:
    python scripts/ask.py "Your question here"      # one-shot
    python scripts/ask.py                            # interactive prompt

Prerequisite: run scripts/ingest.py at least once so that CHROMA_DIR
contains an indexed collection.
"""

import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

from local_rag_sandbox.config import (
    CHAT_MODEL,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
)
from local_rag_sandbox.prompts import RAG_ANSWER_TEMPLATE

TOP_K = 4


def build_context(relevant_docs: list[Document]) -> str:
    """Format retrieved chunks into a single context block for the prompt."""
    parts: list[str] = []
    for i, doc in enumerate(relevant_docs, start=1):
        raw_source = doc.metadata.get("source", "unknown source")
        source = Path(raw_source).name if raw_source != "unknown source" else raw_source
        page = doc.metadata.get("page", "N/A")
        parts.append(
            f"Chunk {i}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def ask_question(query: str) -> None:
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
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    print(f"\nRetrieving the {TOP_K} most relevant chunks ...")
    relevant_docs = retriever.invoke(query)
    if not relevant_docs:
        print("No relevant chunks found. Either the index is empty or the question is way off-topic.")
        return
    print(f"Got {len(relevant_docs)} chunk(s). Sending to {CHAT_MODEL} ...\n")

    context = build_context(relevant_docs)
    prompt = ChatPromptTemplate.from_template(RAG_ANSWER_TEMPLATE)
    llm = ChatOllama(model=CHAT_MODEL, temperature=0)
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": query})

    print(response.content)


def main() -> None:
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:]).strip()
    else:
        query = input("Ask a question about the indexed documents: ").strip()

    if not query:
        print("Empty question. Exiting.")
        return

    ask_question(query)


if __name__ == "__main__":
    main()

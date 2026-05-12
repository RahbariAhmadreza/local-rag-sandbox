"""Ask questions over the ingested document collection.

Loads the persistent Chroma vector database, embeds the user's question
with the same embedding model used during ingestion, retrieves the most
semantically similar chunks, and passes them as context to a local
ChatOllama LLM that returns a structured markdown answer.

Usage:
    python scripts/ask.py "Your question here"                    # whole corpus
    python scripts/ask.py                                          # interactive prompt
    python scripts/ask.py --source dnv "..."                       # filter by publisher
    python scripts/ask.py --year 2024 "..."                        # filter by year
    python scripts/ask.py --topic global_hydrogen_review "..."     # filter by topic
    python scripts/ask.py --source iea --year 2024 "..."           # combined (AND)
    python scripts/ask.py --top-k 20 "..."                         # broader synthesis
    python scripts/ask.py --top-k 4 "..."                          # tighter context

Filters use AND semantics and exact-match. --source and --topic are
normalised to lowercase to match the metadata produced during ingestion.
Omitting all filters searches the full corpus (same as the bare command).
--top-k overrides the default retrieval depth (10). Lower values
(--top-k 4) give sharper context for narrow factual questions; higher
values (--top-k 20) help broad synthesis questions across many chunks.

Prerequisite: run scripts/ingest.py at least once so that CHROMA_DIR
contains an indexed collection.
"""

import argparse

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

TOP_K = 10


def build_where_filter(
    source: str | None,
    year: int | None,
    topic: str | None,
) -> dict | None:
    """Combine CLI filter flags into a Chroma `where` filter dict (AND semantics)."""
    conditions: list[dict] = []
    if source is not None:
        conditions.append({"source": source})
    if year is not None:
        conditions.append({"year": year})
    if topic is not None:
        conditions.append({"topic": topic})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def describe_filters(
    source: str | None,
    year: int | None,
    topic: str | None,
) -> str:
    """Return a one-line human-readable description of the active filter set."""
    parts: list[str] = []
    if source is not None:
        parts.append(f"source={source}")
    if year is not None:
        parts.append(f"year={year}")
    if topic is not None:
        parts.append(f"topic={topic}")
    if not parts:
        return "Full corpus: no metadata filters applied."
    return f"Filtered corpus: {', '.join(parts)}."


def build_context(relevant_docs: list[Document]) -> str:
    """Format retrieved chunks into a single context block for the prompt."""
    parts: list[str] = []
    for i, doc in enumerate(relevant_docs, start=1):
        source = doc.metadata.get("source", "unknown")
        year = doc.metadata.get("year", "N/A")
        filename = doc.metadata.get("filename", "unknown file")
        page = doc.metadata.get("page", "N/A")
        parts.append(
            f"Chunk {i}\n"
            f"Publisher: {source} ({year})\n"
            f"File:      {filename}\n"
            f"Page:      {page}\n"
            f"Content:\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def ask_question(
    query: str,
    source: str | None = None,
    year: int | None = None,
    topic: str | None = None,
    top_k: int = TOP_K,
) -> None:
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        print(f"No Chroma database found at {CHROMA_DIR}.")
        print("Run scripts/ingest.py first to build the index.")
        return

    filter_context = describe_filters(source, year, topic)
    print(f"\n{filter_context}")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    search_kwargs: dict = {"k": top_k}
    where_filter = build_where_filter(source, year, topic)
    if where_filter is not None:
        search_kwargs["filter"] = where_filter
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    print(f"Retrieving the {top_k} most relevant chunks ...")
    relevant_docs = retriever.invoke(query)
    if not relevant_docs:
        print(
            "No relevant chunks found. The filters may be too narrow, the index "
            "may be empty, or the question may be way off-topic."
        )
        return
    print(f"Got {len(relevant_docs)} chunk(s). Sending to {CHAT_MODEL} ...\n")

    context = build_context(relevant_docs)
    prompt = ChatPromptTemplate.from_template(RAG_ANSWER_TEMPLATE)
    llm = ChatOllama(model=CHAT_MODEL, temperature=0)
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": query,
        "filter_context": filter_context,
    })

    print(response.content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "question",
        nargs="*",
        help="Question to ask. If omitted, prompts interactively.",
    )
    parser.add_argument(
        "--source",
        help="Filter chunks to a specific publisher (e.g. dnv, iea). Case-insensitive.",
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Filter chunks to a specific year (e.g. 2024).",
    )
    parser.add_argument(
        "--topic",
        help="Filter chunks to a specific topic (e.g. global_hydrogen_review). Case-insensitive.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Number of chunks to retrieve from Chroma (default: {TOP_K}).",
    )
    args = parser.parse_args()

    if args.question:
        query = " ".join(args.question).strip()
    else:
        query = input("Ask a question about the indexed documents: ").strip()

    if not query:
        print("Empty question. Exiting.")
        return

    # Metadata in Chroma is stored lowercase (parent folder name + filename stem),
    # so normalise the user-facing flags to keep exact-match filters forgiving.
    source = args.source.lower() if args.source else None
    topic = args.topic.lower() if args.topic else None

    ask_question(query, source=source, year=args.year, topic=topic, top_k=args.top_k)


if __name__ == "__main__":
    main()

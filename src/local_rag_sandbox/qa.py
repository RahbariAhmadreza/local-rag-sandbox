"""Core RAG pipeline.

Pure logic. No printing, no CLI, no Streamlit imports. Callers pass an
optional `on_progress` callback to observe retrieval phases without
coupling the core to a particular front-end.

CLI lives in scripts/ask.py. UI lives in streamlit_app.py.
"""

from dataclasses import dataclass
from typing import Callable, Literal

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
from local_rag_sandbox.prompts import TEMPLATES_BY_DEPTH
from local_rag_sandbox.retrieval import (
    K_PER_SUBQUERY,
    build_grouped_context,
    cap_hits,
    decompose_question,
    dedupe_hits,
)

Depth = Literal["concise", "standard", "learning"]

DEPTH_TOP_K: dict[Depth, int] = {
    "concise":  6,
    "standard": 15,
    "learning": 20,
}


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk returned by retrieval, with display-ready metadata."""
    filename: str
    source: str
    year: int | str
    topic: str
    page_raw: int | str          # zero-based when int, else "N/A"
    page_display: int | str      # page_raw + 1 when int, else page_raw
    content: str
    retrieval_query: str | None = None  # focused subquery (Deep R&D multi-query only)


@dataclass(frozen=True)
class SubqueryResult:
    """Chunks retrieved for one focused subquery (before global dedupe/cap)."""
    query: str
    hits: list[RetrievedChunk]


@dataclass(frozen=True)
class RetrievalTrace:
    """Structured trace of Deep R&D multi-query retrieval."""
    original_query: str
    subquery_results: list[SubqueryResult]
    final_chunks: list[RetrievedChunk]


@dataclass(frozen=True)
class QAResult:
    """Structured result of one Q&A round."""
    question: str
    filter_context: str
    depth: Depth
    top_k_used: int
    chat_model: str
    answer: str
    retrieved_chunks: list[RetrievedChunk]


def build_where_filter(
    source: str | None,
    year: int | None,
    topic: str | None,
) -> dict | None:
    """Combine filter args into a Chroma `where` dict (AND semantics)."""
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
    """One-line human description of the active filter set."""
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


def normalise_filters(
    source: str | None,
    topic: str | None,
) -> tuple[str | None, str | None]:
    """Lowercase source and topic so exact-match filters are forgiving."""
    return (
        source.lower() if source else None,
        topic.lower() if topic else None,
    )


def _doc_to_chunk(
    doc: Document,
    *,
    retrieval_query: str | None = None,
) -> RetrievedChunk:
    """Convert a langchain Document into a RetrievedChunk with display page."""
    page_raw = doc.metadata.get("page", "N/A")
    # PyPDFLoader is zero-based; display +1 so citations match printed PDF
    # page labels. PDFs with Roman-numeral front matter will still be off —
    # accepted limitation for v1.
    page_display = page_raw + 1 if isinstance(page_raw, int) else page_raw
    return RetrievedChunk(
        filename=doc.metadata.get("filename", "unknown file"),
        source=doc.metadata.get("source", "unknown"),
        year=doc.metadata.get("year", "N/A"),
        topic=doc.metadata.get("topic", "unknown"),
        page_raw=page_raw,
        page_display=page_display,
        content=doc.page_content,
        retrieval_query=retrieval_query,
    )


def _build_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a single context block for the prompt."""
    parts: list[str] = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"Chunk {i}\n"
            f"Publisher: {c.source} ({c.year})\n"
            f"File:      {c.filename}\n"
            f"Page:      {c.page_display}\n"
            f"Content:\n{c.content}"
        )
    return "\n\n---\n\n".join(parts)


def _emit(on_progress: Callable[[str], None] | None, message: str) -> None:
    if on_progress is not None:
        on_progress(message)


def _retrieve_learning(
    vectorstore: Chroma,
    query: str,
    *,
    where_filter: dict | None,
    effective_top_k: int,
    llm: ChatOllama,
    on_progress: Callable[[str], None] | None,
) -> RetrievalTrace | None:
    """Multi-query retrieval for Deep R&D: decompose, search, dedupe, cap."""
    _emit(on_progress, "Generating focused retrieval queries ...")
    subqueries = decompose_question(query, llm)

    _emit(on_progress, "Focused retrieval queries:")
    for i, sq in enumerate(subqueries, start=1):
        _emit(on_progress, f"  {i}. {sq}")

    subquery_results: list[SubqueryResult] = []
    all_hits: list[tuple[Document, str]] = []
    n = len(subqueries)
    for i, subquery in enumerate(subqueries, start=1):
        _emit(on_progress, f"Retrieving query {i}/{n}: {subquery}")
        search_kwargs: dict = {"k": K_PER_SUBQUERY}
        if where_filter is not None:
            search_kwargs["filter"] = where_filter
        docs = vectorstore.similarity_search(subquery, **search_kwargs)
        per_query_hits = [
            _doc_to_chunk(doc, retrieval_query=subquery) for doc in docs
        ]
        subquery_results.append(SubqueryResult(query=subquery, hits=per_query_hits))
        for doc in docs:
            all_hits.append((doc, subquery))

    if not all_hits:
        return None

    merged = cap_hits(dedupe_hits(all_hits), effective_top_k)
    final_chunks = [
        _doc_to_chunk(doc, retrieval_query=retrieval_query)
        for doc, retrieval_query in merged
    ]
    _emit(
        on_progress,
        f"Merged {len(final_chunks)} unique chunk(s) from {n} focused quer{'y' if n == 1 else 'ies'}.",
    )
    return RetrievalTrace(
        original_query=query,
        subquery_results=subquery_results,
        final_chunks=final_chunks,
    )


def answer_question(
    query: str,
    *,
    source: str | None = None,
    year: int | None = None,
    topic: str | None = None,
    depth: Depth = "standard",
    top_k: int | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> QAResult | None:
    """Run the full RAG pipeline and return a structured result.

    Returns None when the index is missing or no chunks are retrieved.
    Status messages flow through the optional `on_progress` callback;
    callers without one stay silent.
    """
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        _emit(on_progress, f"No Chroma database found at {CHROMA_DIR}.")
        _emit(on_progress, "Run scripts/ingest.py first to build the index.")
        return None

    source, topic = normalise_filters(source, topic)
    effective_top_k = top_k if top_k is not None else DEPTH_TOP_K[depth]
    filter_context = describe_filters(source, year, topic)
    _emit(on_progress, f"\n{filter_context}")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    where_filter = build_where_filter(source, year, topic)
    llm = ChatOllama(model=CHAT_MODEL, temperature=0)

    if depth == "learning":
        trace = _retrieve_learning(
            vectorstore,
            query,
            where_filter=where_filter,
            effective_top_k=effective_top_k,
            llm=llm,
            on_progress=on_progress,
        )
        if trace is None:
            _emit(
                on_progress,
                "No relevant chunks found. The filters may be too narrow, the index "
                "may be empty, or the question may be way off-topic.",
            )
            return None
        chunks = trace.final_chunks
        context = build_grouped_context(chunks)
    else:
        search_kwargs: dict = {"k": effective_top_k}
        if where_filter is not None:
            search_kwargs["filter"] = where_filter
        retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

        _emit(on_progress, f"Retrieving the {effective_top_k} most relevant chunks ...")
        docs = retriever.invoke(query)
        if not docs:
            _emit(
                on_progress,
                "No relevant chunks found. The filters may be too narrow, the index "
                "may be empty, or the question may be way off-topic.",
            )
            return None

        chunks = [_doc_to_chunk(d) for d in docs]
        _emit(on_progress, f"Got {len(chunks)} chunk(s).")
        context = _build_context(chunks)

    _emit(on_progress, f"Sending to {CHAT_MODEL} ...\n")

    template = TEMPLATES_BY_DEPTH[depth]
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": query,
        "filter_context": filter_context,
    })

    return QAResult(
        question=query,
        filter_context=filter_context,
        depth=depth,
        top_k_used=effective_top_k,
        chat_model=CHAT_MODEL,
        answer=response.content,
        retrieved_chunks=chunks,
    )


def list_filter_options() -> tuple[list[str], list[int], list[str]]:
    """Read distinct source / year / topic values from Chroma metadata.

    Used by streamlit_app.py to populate the sidebar dropdowns. Returns
    empty lists when the index has not been built yet.
    """
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        return [], [], []
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    records = vectorstore.get()
    metadatas = records.get("metadatas") or []
    sources = sorted({m["source"] for m in metadatas if m.get("source")})
    years = sorted({m["year"] for m in metadatas if isinstance(m.get("year"), int)})
    topics = sorted({m["topic"] for m in metadatas if m.get("topic")})
    return sources, years, topics

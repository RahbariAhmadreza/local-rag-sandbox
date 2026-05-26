"""Deep R&D / learning-mode retrieval diagnostics (no answer generation).

Runs the same multi-query retrieval path as ``answer_question(depth="learning")``
and prints a human-readable report from ``RetrievalTrace``.

Usage:
    python scripts/diagnose_retrieval.py "What were the main hydrogen bottlenecks?"
    python scripts/diagnose_retrieval.py --source iea --year 2024 "..."
    python scripts/diagnose_retrieval.py --top-k 15 "..."

Requires a built Chroma index (``python scripts/ingest.py``) and a running Ollama
with the configured chat and embedding models.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

from local_rag_sandbox.config import (
    CHAT_MODEL,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
)
from local_rag_sandbox.qa import (
    DEPTH_TOP_K,
    RetrievedChunk,
    RetrievalTrace,
    _retrieve_learning,
    build_where_filter,
    describe_filters,
    normalise_filters,
)
from local_rag_sandbox.retrieval import content_hash

CONTENT_PREVIEW_CHARS = 120
_SEPARATOR = "=" * 72


@dataclass(frozen=True)
class CrossSubqueryDuplicate:
    """One chunk identity retrieved by more than one focused subquery."""

    filename: str
    page_display: int | str
    page_raw: int | str
    preview: str
    subquery_indices: tuple[int, ...]
    subquery_labels: tuple[str, ...]


def chunk_identity(chunk: RetrievedChunk) -> tuple[str, int | str, str]:
    """Dedup key aligned with retrieval.dedupe_hits (filename, page, content hash)."""
    return (chunk.filename, chunk.page_raw, content_hash(chunk.content))


def preview_content(text: str, max_chars: int = CONTENT_PREVIEW_CHARS) -> str:
    """Single-line preview for terminal output."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 1] + "…"


def find_cross_subquery_duplicates(trace: RetrievalTrace) -> list[CrossSubqueryDuplicate]:
    """Chunks whose identity appears under two or more distinct subqueries (raw hits)."""
    by_key: dict[tuple[str, int | str, str], dict[int, str]] = {}
    samples: dict[tuple[str, int | str, str], RetrievedChunk] = {}

    for sq_idx, sq in enumerate(trace.subquery_results, start=1):
        for hit in sq.hits:
            key = chunk_identity(hit)
            by_key.setdefault(key, {})[sq_idx] = sq.query
            samples.setdefault(key, hit)

    duplicates: list[CrossSubqueryDuplicate] = []
    for key, sq_map in by_key.items():
        if len(sq_map) < 2:
            continue
        sample = samples[key]
        indices = tuple(sorted(sq_map))
        labels = tuple(sq_map[i] for i in indices)
        duplicates.append(
            CrossSubqueryDuplicate(
                filename=sample.filename,
                page_display=sample.page_display,
                page_raw=sample.page_raw,
                preview=preview_content(sample.content),
                subquery_indices=indices,
                subquery_labels=labels,
            )
        )

    duplicates.sort(key=lambda d: (d.filename, str(d.page_display)))
    return duplicates


def _print_chunk_line(chunk: RetrievedChunk, *, indent: str = "    ") -> None:
    print(f"{indent}{chunk.filename}, p. {chunk.page_display}")
    print(f"{indent}  source={chunk.source}  year={chunk.year}  topic={chunk.topic}")
    if chunk.page_raw != chunk.page_display:
        print(f"{indent}  page (raw loader)={chunk.page_raw}")
    print(f"{indent}  preview: {preview_content(chunk.content)}")


def print_trace_report(
    trace: RetrievalTrace,
    *,
    filter_context: str,
    effective_top_k: int,
) -> None:
    """Print a structured retrieval diagnostic report."""
    print(_SEPARATOR)
    print("RETRIEVAL DIAGNOSTICS (learning / Deep R&D)")
    print(_SEPARATOR)
    print()
    print("Original question:")
    print(f"  {trace.original_query}")
    print()
    print("Filters:")
    print(f"  {filter_context}")
    print()
    print(f"Effective top-k (after dedupe/cap): {effective_top_k}")
    print()

    print("Generated subqueries:")
    if not trace.subquery_results:
        print("  (none)")
    else:
        for i, sq in enumerate(trace.subquery_results, start=1):
            print(f"  {i}. {sq.query}")
    print()

    print(_SEPARATOR)
    print("RAW HITS PER SUBQUERY")
    print(_SEPARATOR)
    for i, sq in enumerate(trace.subquery_results, start=1):
        print()
        print(f"Subquery {i}: {sq.query}")
        print(f"  Raw hits: {len(sq.hits)}")
        if not sq.hits:
            print("    (no hits)")
            continue
        for j, hit in enumerate(sq.hits, start=1):
            print(f"  Hit {j}:")
            _print_chunk_line(hit, indent="      ")

    print()
    print(_SEPARATOR)
    print("FINAL MERGED CHUNKS (deduped + capped)")
    print(_SEPARATOR)
    print(f"Count: {len(trace.final_chunks)}")
    print()
    if not trace.final_chunks:
        print("  (none)")
    else:
        for i, chunk in enumerate(trace.final_chunks, start=1):
            print(f"Chunk {i}:")
            _print_chunk_line(chunk, indent="  ")
            if chunk.retrieval_query:
                print(f"    kept from subquery: {chunk.retrieval_query}")
            print()

    cross = find_cross_subquery_duplicates(trace)
    print(_SEPARATOR)
    print("CROSS-SUBQUERY DUPLICATES (same chunk in multiple subquery raw hits)")
    print(_SEPARATOR)
    if not cross:
        print("  (none)")
    else:
        for i, dup in enumerate(cross, start=1):
            print()
            print(f"Duplicate {i}: {dup.filename}, p. {dup.page_display}")
            print(f"  page (raw loader)={dup.page_raw}")
            print(f"  preview: {dup.preview}")
            print(
                f"  seen in subqueries: "
                + ", ".join(
                    f"{idx} ({label!r})"
                    for idx, label in zip(dup.subquery_indices, dup.subquery_labels, strict=True)
                )
            )
    print()
    print(_SEPARATOR)


def run_diagnosis(
    query: str,
    *,
    source: str | None,
    year: int | None,
    topic: str | None,
    top_k: int | None,
) -> int:
    """Run learning-mode retrieval and print diagnostics. Returns process exit code."""
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        print(f"No Chroma database found at {CHROMA_DIR}.")
        print("Run scripts/ingest.py first to build the index.")
        return 1

    norm_source, norm_topic = normalise_filters(source, topic)
    effective_top_k = top_k if top_k is not None else DEPTH_TOP_K["learning"]
    filter_context = describe_filters(norm_source, year, norm_topic)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    where_filter = build_where_filter(norm_source, year, norm_topic)
    llm = ChatOllama(model=CHAT_MODEL, temperature=0)

    trace = _retrieve_learning(
        vectorstore,
        query,
        where_filter=where_filter,
        effective_top_k=effective_top_k,
        llm=llm,
        on_progress=None,
    )

    if trace is None:
        print(_SEPARATOR)
        print("RETRIEVAL DIAGNOSTICS (learning / Deep R&D)")
        print(_SEPARATOR)
        print()
        print("Original question:")
        print(f"  {query}")
        print()
        print("Filters:")
        print(f"  {filter_context}")
        print()
        print(f"Effective top-k (after dedupe/cap): {effective_top_k}")
        print()
        print("No chunks retrieved. Filters may be too narrow, the index may be empty,")
        print("or the question may be off-topic.")
        return 1

    print_trace_report(trace, filter_context=filter_context, effective_top_k=effective_top_k)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "question",
        nargs="+",
        help="Question to diagnose (learning-mode retrieval).",
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
        help="Filter chunks to a specific topic. Case-insensitive.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=f"Cap on unique chunks after merge (default: {DEPTH_TOP_K['learning']}).",
    )
    args = parser.parse_args()

    query = " ".join(args.question).strip()
    if not query:
        print("Empty question. Exiting.", file=sys.stderr)
        sys.exit(1)

    code = run_diagnosis(
        query,
        source=args.source,
        year=args.year,
        topic=args.topic,
        top_k=args.top_k,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()

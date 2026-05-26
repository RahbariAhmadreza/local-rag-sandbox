"""Batch retrieval eval for Deep R&D / learning mode (no answer generation).

Compares retrieval settings (e.g. subquery-k=5 vs k=10) using the real
``_retrieve_learning`` path and metrics from ``retrieval_eval``.

Usage:
    python scripts/eval_retrieval.py --compare
    python scripts/eval_retrieval.py --compare --case-id hydrogen_fid_offtake_investment
    python scripts/eval_retrieval.py --subquery-k 10
    python scripts/eval_retrieval.py --cases eval/retrieval_cases.json

Requires a built Chroma index and running Ollama (embed + chat models).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

from local_rag_sandbox.config import (
    CHAT_MODEL,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    PROJECT_ROOT,
)
from local_rag_sandbox.qa import (
    DEPTH_TOP_K,
    RetrievalTrace,
    _retrieve_learning,
    build_where_filter,
    normalise_filters,
)
from local_rag_sandbox.retrieval import K_PER_SUBQUERY
from local_rag_sandbox.retrieval_eval import (
    RetrievalProfile,
    RetrievalRunMetrics,
    empty_trace_metrics,
    filter_cases,
    format_metrics_table,
    format_profile_evidence_lines,
    format_top_filenames,
    load_cases,
    summarize_trace,
)

DEFAULT_CASES_PATH = PROJECT_ROOT / "eval" / "retrieval_cases.json"
COMPARE_PROFILES = (
    RetrievalProfile(name="k5", subquery_k=None),
    RetrievalProfile(name="k10", subquery_k=10),
)


def _resolve_profiles(
    *,
    compare: bool,
    subquery_k: int | None,
) -> list[RetrievalProfile]:
    if compare:
        if subquery_k is not None:
            print(
                "Warning: --subquery-k ignored when --compare is set.",
                file=sys.stderr,
            )
        return list(COMPARE_PROFILES)
    if subquery_k is not None:
        return [RetrievalProfile(name=f"k{subquery_k}", subquery_k=subquery_k)]
    return [RetrievalProfile(name="k5", subquery_k=None)]


def _run_profile(
    *,
    vectorstore: Chroma,
    llm: ChatOllama,
    case_question: str,
    source: str | None,
    year: int | None,
    topic: str | None,
    top_k: int,
    profile: RetrievalProfile,
) -> RetrievalTrace | None:
    norm_source, norm_topic = normalise_filters(source, topic)
    where_filter = build_where_filter(norm_source, year, topic)
    return _retrieve_learning(
        vectorstore,
        case_question,
        where_filter=where_filter,
        effective_top_k=top_k,
        llm=llm,
        on_progress=None,
        subquery_k=profile.subquery_k,
    )


def _metrics_for_run(
    trace: RetrievalTrace | None,
    *,
    profile: RetrievalProfile,
    top_k: int,
    expected_evidence: tuple,
) -> RetrievalRunMetrics:
    if trace is None:
        return empty_trace_metrics(
            profile_name=profile.name,
            subquery_k=profile.subquery_k,
            top_k=top_k,
            expected_evidence=expected_evidence,
        )
    return summarize_trace(
        trace,
        profile_name=profile.name,
        subquery_k=profile.subquery_k,
        top_k=top_k,
        expected_evidence=expected_evidence,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help=f"Path to retrieval cases JSON (default: {DEFAULT_CASES_PATH.relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        metavar="ID",
        help="Run only this case id (repeatable). Default: all cases.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run k5 (default subquery k) and k10 profiles side by side.",
    )
    parser.add_argument(
        "--subquery-k",
        type=int,
        default=None,
        help=f"Single profile with this per-subquery k (default: {K_PER_SUBQUERY} unless --compare).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=f"Final chunk cap after dedupe (default: {DEPTH_TOP_K['learning']}).",
    )
    args = parser.parse_args()

    if not args.cases.is_file():
        print(f"Case file not found: {args.cases}", file=sys.stderr)
        sys.exit(1)

    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        print(f"No Chroma database found at {CHROMA_DIR}.", file=sys.stderr)
        print("Run scripts/ingest.py first.", file=sys.stderr)
        sys.exit(1)

    try:
        cases = filter_cases(load_cases(args.cases), args.case_ids)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"Invalid cases file: {exc}", file=sys.stderr)
        sys.exit(1)

    top_k = args.top_k if args.top_k is not None else DEPTH_TOP_K["learning"]
    profiles = _resolve_profiles(compare=args.compare, subquery_k=args.subquery_k)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    llm = ChatOllama(model=CHAT_MODEL, temperature=0)

    print("RETRIEVAL EVAL (learning / Deep R&D)")
    print(f"Cases: {args.cases}  |  profiles: {', '.join(p.name for p in profiles)}  |  top_k: {top_k}")
    print()

    any_failed = False
    for case in cases:
        print("=" * 72)
        print(f"Case: {case.id}")
        print(f"Question: {case.question}")
        if case.source or case.year or case.topic:
            parts = []
            if case.source:
                parts.append(f"source={case.source}")
            if case.year is not None:
                parts.append(f"year={case.year}")
            if case.topic:
                parts.append(f"topic={case.topic}")
            print(f"Filters: {', '.join(parts)}")
        if case.notes:
            print(f"Notes: {case.notes}")
        print()

        rows: list[RetrievalRunMetrics] = []
        for profile in profiles:
            trace = _run_profile(
                vectorstore=vectorstore,
                llm=llm,
                case_question=case.question,
                source=case.source,
                year=case.year,
                topic=case.topic,
                top_k=top_k,
                profile=profile,
            )
            if trace is None:
                any_failed = True
                print(f"  [{profile.name}] no chunks retrieved", file=sys.stderr)
            rows.append(
                _metrics_for_run(
                    trace,
                    profile=profile,
                    top_k=top_k,
                    expected_evidence=case.expected_evidence,
                )
            )

        for line in format_metrics_table(rows):
            print(line)
        for row in rows:
            for detail_line in format_profile_evidence_lines(row):
                print(detail_line)
        if rows:
            top = rows[0]
            print(f"  top files (first profile): {format_top_filenames(top.filename_distribution)}")
        print()

    if any_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

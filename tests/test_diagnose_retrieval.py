"""Offline tests for diagnose_retrieval.py pure helpers."""

import contextlib
import importlib.util
import io
import sys
from pathlib import Path

from local_rag_sandbox.qa import RetrievedChunk, RetrievalTrace, SubqueryResult

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
_SPEC = importlib.util.spec_from_file_location(
    "diagnose_retrieval",
    _SCRIPTS_DIR / "diagnose_retrieval.py",
)
assert _SPEC is not None and _SPEC.loader is not None
dr = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = dr
_SPEC.loader.exec_module(dr)


def _chunk(
    filename: str = "a.pdf",
    page_raw: int = 0,
    content: str = "hello world",
    source: str = "pub",
    year: int = 2024,
    topic: str = "sample",
    retrieval_query: str | None = None,
) -> RetrievedChunk:
    page_display = page_raw + 1 if isinstance(page_raw, int) else page_raw
    return RetrievedChunk(
        filename=filename,
        source=source,
        year=year,
        topic=topic,
        page_raw=page_raw,
        page_display=page_display,
        content=content,
        retrieval_query=retrieval_query,
    )


def test_preview_content_collapses_whitespace() -> None:
    assert dr.preview_content("  hello   world  ") == "hello world"


def test_preview_content_truncates_long_text() -> None:
    text = "word " * 50
    out = dr.preview_content(text, max_chars=20)
    assert len(out) == 20
    assert out.endswith("…")


def test_preview_content_short_unchanged() -> None:
    assert dr.preview_content("short") == "short"


def test_chunk_identity_uses_content_hash() -> None:
    a = _chunk(content="hello   world")
    b = _chunk(content="hello world")
    assert dr.chunk_identity(a) == dr.chunk_identity(b)


def test_find_cross_subquery_duplicates_empty_trace() -> None:
    trace = RetrievalTrace(original_query="q", subquery_results=[], final_chunks=[])
    assert dr.find_cross_subquery_duplicates(trace) == []


def test_find_cross_subquery_duplicates_across_two_subqueries() -> None:
    shared = _chunk(filename="shared.pdf", page_raw=1, content="same body")
    trace = RetrievalTrace(
        original_query="q",
        subquery_results=[
            SubqueryResult(query="aspect A", hits=[shared]),
            SubqueryResult(query="aspect B", hits=[shared, _chunk(filename="other.pdf")]),
        ],
        final_chunks=[],
    )
    dups = dr.find_cross_subquery_duplicates(trace)
    assert len(dups) == 1
    assert dups[0].filename == "shared.pdf"
    assert dups[0].subquery_indices == (1, 2)
    assert dups[0].subquery_labels == ("aspect A", "aspect B")


def test_find_cross_subquery_duplicates_ignores_within_single_subquery() -> None:
    dup = _chunk(content="x")
    trace = RetrievalTrace(
        original_query="q",
        subquery_results=[SubqueryResult(query="only", hits=[dup, dup])],
        final_chunks=[],
    )
    assert dr.find_cross_subquery_duplicates(trace) == []


def test_print_trace_report_includes_key_sections() -> None:
    trace = RetrievalTrace(
        original_query="What are the risks?",
        subquery_results=[
            SubqueryResult(query="risk A", hits=[_chunk()]),
        ],
        final_chunks=[_chunk(retrieval_query="risk A")],
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dr.print_trace_report(
            trace,
            filter_context="Full corpus: no metadata filters applied.",
            effective_top_k=20,
        )
    out = buf.getvalue()
    assert "RETRIEVAL DIAGNOSTICS" in out
    assert "What are the risks?" in out
    assert "Generated subqueries:" in out
    assert "RAW HITS PER SUBQUERY" in out
    assert "FINAL MERGED CHUNKS" in out
    assert "CROSS-SUBQUERY DUPLICATES" in out

"""Unit tests for multi-query retrieval helpers (no Ollama / Chroma)."""

from dataclasses import dataclass

from langchain_core.documents import Document

from local_rag_sandbox.retrieval import (
    build_grouped_context,
    cap_hits,
    content_hash,
    dedupe_hits,
    document_dedup_key,
    normalize_content_for_hash,
    parse_subqueries,
)


@dataclass(frozen=True)
class _Chunk:
    filename: str
    page_display: int | str
    source: str
    year: int | str
    topic: str
    content: str
    retrieval_query: str | None = None


def _doc(
    filename: str = "report.pdf",
    page: int = 0,
    content: str = "hello world",
    source: str = "pub",
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "filename": filename,
            "page": page,
            "source": source,
            "year": 2024,
            "topic": "sample",
        },
    )


# --- parse_subqueries ---


def test_parse_subqueries_plain_lines() -> None:
    text = "cost barriers for the product\nslow adoption rates\nregulatory uncertainty"
    assert parse_subqueries(text) == [
        "cost barriers for the product",
        "slow adoption rates",
        "regulatory uncertainty",
    ]


def test_parse_subqueries_numbered_and_bulleted() -> None:
    text = """1. first aspect of the question
2) second aspect
- third aspect
* fourth aspect"""
    assert len(parse_subqueries(text)) == 4
    assert parse_subqueries(text)[0] == "first aspect of the question"
    assert parse_subqueries(text)[2] == "third aspect"


def test_parse_subqueries_strips_quotes() -> None:
    assert parse_subqueries('"quoted query"') == ["quoted query"]


def test_parse_subqueries_empty_returns_empty() -> None:
    assert parse_subqueries("") == []
    assert parse_subqueries("   \n\n  ") == []


def test_decompose_question_falls_back_when_parse_empty() -> None:
    class _EmptyLLM:
        def invoke(self, _messages: object) -> object:
            class _Resp:
                content = "   \n\n  "
            return _Resp()

    from local_rag_sandbox.retrieval import decompose_question

    question = "What are the main bottlenecks?"
    assert decompose_question(question, _EmptyLLM()) == [question]


def test_parse_subqueries_caps_at_five() -> None:
    text = "\n".join(f"query {i}" for i in range(10))
    assert len(parse_subqueries(text)) == 5


# --- content hash ---


def test_normalize_content_collapses_whitespace() -> None:
    assert normalize_content_for_hash("  hello   world\n\nfoo  ") == "hello world foo"


def test_content_hash_stable_for_same_normalized_text() -> None:
    a = content_hash("hello   world")
    b = content_hash("hello world")
    assert a == b
    assert len(a) == 16


def test_document_dedup_key_uses_filename_page_hash() -> None:
    doc = _doc(filename="a.pdf", page=3, content="same text")
    key = document_dedup_key(doc)
    assert key[0] == "a.pdf"
    assert key[1] == 3
    assert key[2] == content_hash("same text")


# --- dedupe_hits ---


def test_dedupe_hits_keeps_first_occurrence() -> None:
    doc_a = _doc(filename="a.pdf", page=1, content="shared")
    doc_b = _doc(filename="b.pdf", page=2, content="unique")
    hits = [
        (doc_a, "query one"),
        (doc_a, "query two"),  # duplicate content+page+file
        (doc_b, "query one"),
    ]
    result = dedupe_hits(hits)
    assert len(result) == 2
    assert result[0] == (doc_a, "query one")
    assert result[1] == (doc_b, "query one")


def test_dedupe_hits_keeps_different_pages() -> None:
    doc_p0 = _doc(page=0, content="text")
    doc_p1 = _doc(page=1, content="text")
    result = dedupe_hits([(doc_p0, "q1"), (doc_p1, "q1")])
    assert len(result) == 2


def test_cap_hits_limits_after_dedupe() -> None:
    hits = [(_doc(page=i, content=f"c{i}"), "q") for i in range(5)]
    assert len(cap_hits(dedupe_hits(hits), 3)) == 3


# --- build_grouped_context ---


def test_build_grouped_context_has_focused_query_headings() -> None:
    chunks = [
        _Chunk("a.pdf", 1, "pub", 2024, "t", "alpha", "aspect A"),
        _Chunk("b.pdf", 2, "pub", 2024, "t", "beta", "aspect B"),
    ]
    ctx = build_grouped_context(chunks)
    assert '=== Focused query 1: "aspect A" ===' in ctx
    assert '=== Focused query 2: "aspect B" ===' in ctx
    assert "File:    a.pdf" in ctx
    assert "Page:    1" in ctx


def test_build_grouped_context_no_duplicate_chunk_bodies() -> None:
    body = "unique passage text here"
    chunks = [_Chunk("x.pdf", 5, "s", 2020, "topic", body, "q1")]
    ctx = build_grouped_context(chunks)
    assert ctx.count(body) == 1

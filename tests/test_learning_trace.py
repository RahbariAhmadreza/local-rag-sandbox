"""Offline tests for Deep R&D RetrievalTrace (mocked Chroma / LLM)."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from local_rag_sandbox.qa import (
    RetrievalTrace,
    SubqueryResult,
    _retrieve_learning,
)
from local_rag_sandbox.retrieval import K_PER_SUBQUERY


def _doc(
    filename: str = "report.pdf",
    page: int = 0,
    content: str = "hello",
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "filename": filename,
            "page": page,
            "source": "pub",
            "year": 2024,
            "topic": "sample",
        },
    )


def test_retrieval_trace_dataclass_fields() -> None:
    trace = RetrievalTrace(
        original_query="What are the risks?",
        subquery_results=[SubqueryResult(query="risk A", hits=[])],
        final_chunks=[],
    )
    assert trace.original_query == "What are the risks?"
    assert trace.subquery_results[0].query == "risk A"
    assert trace.final_chunks == []


@patch("local_rag_sandbox.qa.decompose_question")
def test_retrieve_learning_returns_trace_with_per_subquery_hits(
    mock_decompose: MagicMock,
) -> None:
    mock_decompose.return_value = ["aspect A", "aspect B"]
    shared = _doc(filename="shared.pdf", page=1, content="duplicate body")
    unique_b = _doc(filename="b.pdf", page=2, content="only in B")

    def similarity_search(subquery: str, **kwargs: object) -> list[Document]:
        if subquery == "aspect A":
            return [shared, shared]
        return [shared, unique_b]

    vectorstore = MagicMock()
    vectorstore.similarity_search.side_effect = similarity_search

    trace = _retrieve_learning(
        vectorstore,
        "original question",
        where_filter=None,
        effective_top_k=10,
        llm=MagicMock(),
        on_progress=None,
    )

    assert trace is not None
    assert trace.original_query == "original question"
    assert len(trace.subquery_results) == 2
    assert trace.subquery_results[0].query == "aspect A"
    assert len(trace.subquery_results[0].hits) == 2
    assert trace.subquery_results[1].query == "aspect B"
    assert len(trace.subquery_results[1].hits) == 2

    # Global dedupe: shared.pdf p.1 appears once (first from aspect A)
    assert len(trace.final_chunks) == 2
    assert trace.final_chunks[0].filename == "shared.pdf"
    assert trace.final_chunks[0].retrieval_query == "aspect A"
    assert trace.final_chunks[1].filename == "b.pdf"
    assert trace.final_chunks[1].retrieval_query == "aspect B"


@patch("local_rag_sandbox.qa.decompose_question")
def test_retrieve_learning_returns_none_when_no_hits(
    mock_decompose: MagicMock,
) -> None:
    mock_decompose.return_value = ["aspect A"]
    vectorstore = MagicMock()
    vectorstore.similarity_search.return_value = []

    assert (
        _retrieve_learning(
            vectorstore,
            "original",
            where_filter=None,
            effective_top_k=5,
            llm=MagicMock(),
            on_progress=None,
        )
        is None
    )


@patch("local_rag_sandbox.qa.decompose_question")
def test_retrieve_learning_cap_limits_final_chunks(
    mock_decompose: MagicMock,
) -> None:
    mock_decompose.return_value = ["q"]
    docs = [_doc(filename=f"f{i}.pdf", page=i, content=f"c{i}") for i in range(5)]
    vectorstore = MagicMock()
    vectorstore.similarity_search.return_value = docs

    trace = _retrieve_learning(
        vectorstore,
        "original",
        where_filter=None,
        effective_top_k=3,
        llm=MagicMock(),
        on_progress=None,
    )

    assert trace is not None
    assert len(trace.subquery_results[0].hits) == 5
    assert len(trace.final_chunks) == 3


@patch("local_rag_sandbox.qa.decompose_question")
def test_retrieve_learning_default_subquery_k(
    mock_decompose: MagicMock,
) -> None:
    mock_decompose.return_value = ["q1", "q2"]
    vectorstore = MagicMock()
    vectorstore.similarity_search.return_value = []

    _retrieve_learning(
        vectorstore,
        "original",
        where_filter=None,
        effective_top_k=5,
        llm=MagicMock(),
        on_progress=None,
    )

    for call in vectorstore.similarity_search.call_args_list:
        assert call.kwargs["k"] == K_PER_SUBQUERY


@patch("local_rag_sandbox.qa.decompose_question")
def test_retrieve_learning_uses_subquery_k_override(
    mock_decompose: MagicMock,
) -> None:
    mock_decompose.return_value = ["q1", "q2"]
    vectorstore = MagicMock()
    vectorstore.similarity_search.return_value = [_doc()]

    _retrieve_learning(
        vectorstore,
        "original",
        where_filter=None,
        effective_top_k=5,
        llm=MagicMock(),
        on_progress=None,
        subquery_k=10,
    )

    for call in vectorstore.similarity_search.call_args_list:
        assert call.kwargs["k"] == 10

"""Offline tests: answer_question passes subquery_k to _retrieve_learning."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from local_rag_sandbox.qa import (
    QAResult,
    RetrievedChunk,
    RetrievalTrace,
    SubqueryResult,
    answer_question,
)


def _minimal_trace() -> RetrievalTrace:
    chunk = RetrievedChunk(
        filename="a.pdf",
        source="pub",
        year=2024,
        topic="t",
        page_raw=0,
        page_display=1,
        content="text",
        retrieval_query="q1",
    )
    return RetrievalTrace(
        original_query="question",
        subquery_results=[SubqueryResult(query="q1", hits=[chunk])],
        final_chunks=[chunk],
    )


@patch("local_rag_sandbox.qa._retrieve_learning")
@patch("local_rag_sandbox.qa.ChatPromptTemplate")
@patch("local_rag_sandbox.qa.ChatOllama")
@patch("local_rag_sandbox.qa.OllamaEmbeddings")
@patch("local_rag_sandbox.qa.Chroma")
@patch("local_rag_sandbox.qa.CHROMA_DIR")
def test_answer_question_learning_passes_subquery_k_to_retrieve_learning(
    mock_chroma_dir: MagicMock,
    mock_chroma_cls: MagicMock,
    mock_embeddings: MagicMock,
    mock_chat_ollama: MagicMock,
    mock_prompt_template: MagicMock,
    mock_retrieve_learning: MagicMock,
) -> None:
    mock_chroma_dir.exists.return_value = True
    mock_chroma_dir.iterdir.return_value = [Path("segment")]
    mock_retrieve_learning.return_value = _minimal_trace()

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = MagicMock(content="stub answer")
    mock_prompt_template.from_template.return_value.__or__.return_value = mock_chain

    result = answer_question(
        "question",
        depth="learning",
        subquery_k=10,
        on_progress=None,
    )

    assert isinstance(result, QAResult)
    mock_retrieve_learning.assert_called_once()
    assert mock_retrieve_learning.call_args.kwargs["subquery_k"] == 10

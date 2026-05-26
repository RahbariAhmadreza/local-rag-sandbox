"""Offline tests for retrieval_eval helpers."""

import json
from pathlib import Path

import pytest

from local_rag_sandbox.qa import RetrievedChunk, RetrievalTrace, SubqueryResult
from local_rag_sandbox.retrieval_eval import (
    EvidenceRecallResult,
    ExpectedEvidence,
    RetrievalCase,
    compute_evidence_recall,
    filter_cases,
    format_metrics_table,
    format_profile_evidence_lines,
    load_cases,
    summarize_trace,
)


def _chunk(
    filename: str = "a.pdf",
    page_display: int = 1,
    source: str = "iea",
) -> RetrievedChunk:
    return RetrievedChunk(
        filename=filename,
        source=source,
        year=2024,
        topic="t",
        page_raw=page_display - 1,
        page_display=page_display,
        content="body",
    )


def test_load_cases_valid(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "cases": [
                    {
                        "id": "one",
                        "question": "What is hydrogen?",
                        "expected_evidence": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cases = load_cases(path)
    assert len(cases) == 1
    assert cases[0].id == "one"


def test_load_cases_rejects_duplicate_id(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "cases": [
                    {"id": "dup", "question": "q1"},
                    {"id": "dup", "question": "q2"},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(path)


def test_filter_cases_unknown_id() -> None:
    cases = [RetrievalCase(id="a", question="q")]
    with pytest.raises(ValueError, match="unknown"):
        filter_cases(cases, ["missing"])


def test_evidence_recall_empty_expected_is_na() -> None:
    result = compute_evidence_recall([_chunk()], ())
    assert result.format_recall() == "n/a"
    assert result.recall_fraction is None


def test_evidence_recall_partial_hits() -> None:
    expected = (
        ExpectedEvidence("a.pdf", 1),
        ExpectedEvidence("b.pdf", 2),
    )
    result = compute_evidence_recall([_chunk("a.pdf", 1)], expected)
    assert result.format_recall() == "1/2"
    assert result.hit_items == (ExpectedEvidence("a.pdf", 1),)
    assert result.missed_items == (ExpectedEvidence("b.pdf", 2),)


def test_evidence_recall_full_match() -> None:
    expected = (ExpectedEvidence("a.pdf", 1), ExpectedEvidence("b.pdf", 2))
    chunks = [_chunk("a.pdf", 1), _chunk("b.pdf", 2)]
    result = compute_evidence_recall(chunks, expected)
    assert result.format_recall() == "2/2"
    assert result.missed_items == ()


def test_evidence_recall_no_match() -> None:
    expected = (ExpectedEvidence("x.pdf", 99),)
    result = compute_evidence_recall([_chunk("a.pdf", 1)], expected)
    assert result.format_recall() == "0/1"
    assert result.hit_items == ()
    assert result.missed_items == expected


def test_summarize_trace_matched_and_missed_evidence() -> None:
    expected = (
        ExpectedEvidence("a.pdf", 1),
        ExpectedEvidence("b.pdf", 2),
    )
    metrics = summarize_trace(
        RetrievalTrace(
            original_query="q",
            subquery_results=[],
            final_chunks=[_chunk("a.pdf", 1)],
        ),
        profile_name="k5",
        subquery_k=5,
        top_k=20,
        expected_evidence=expected,
    )
    assert metrics.matched_evidence == (ExpectedEvidence("a.pdf", 1),)
    assert metrics.missed_evidence == (ExpectedEvidence("b.pdf", 2),)
    assert metrics.recall_display == "1/2"


def test_summarize_trace_empty_expected_has_no_matched_missed() -> None:
    metrics = summarize_trace(
        RetrievalTrace(original_query="q", subquery_results=[], final_chunks=[_chunk()]),
        profile_name="k5",
        subquery_k=5,
        top_k=20,
        expected_evidence=(),
    )
    assert metrics.matched_evidence == ()
    assert metrics.missed_evidence == ()
    assert metrics.recall_display == "n/a"


def test_format_profile_evidence_lines() -> None:
    metrics = summarize_trace(
        RetrievalTrace(
            original_query="q",
            subquery_results=[],
            final_chunks=[_chunk("a.pdf", 10)],
        ),
        profile_name="k10",
        subquery_k=10,
        top_k=20,
        expected_evidence=(
            ExpectedEvidence("a.pdf", 10),
            ExpectedEvidence("c.pdf", 30),
        ),
    )
    lines = format_profile_evidence_lines(metrics)
    assert len(lines) == 2
    assert "[k10] matched: a.pdf p.10" in lines[0]
    assert "[k10] missed : c.pdf p.30" in lines[1]


def test_format_profile_evidence_lines_empty_when_no_expected() -> None:
    metrics = summarize_trace(
        RetrievalTrace(original_query="q", subquery_results=[], final_chunks=[]),
        profile_name="k5",
        subquery_k=None,
        top_k=20,
        expected_evidence=(),
    )
    assert format_profile_evidence_lines(metrics) == []


def test_summarize_trace_counts_and_distributions() -> None:
    trace = RetrievalTrace(
        original_query="q",
        subquery_results=[
            SubqueryResult(query="sq1", hits=[_chunk("a.pdf", 1, "iea"), _chunk("a.pdf", 2, "iea")]),
            SubqueryResult(query="sq2", hits=[_chunk("b.pdf", 3, "dnv")]),
        ],
        final_chunks=[
            _chunk("a.pdf", 1, "iea"),
            _chunk("a.pdf", 1, "iea"),
            _chunk("b.pdf", 3, "dnv"),
        ],
    )
    metrics = summarize_trace(
        trace,
        profile_name="k5",
        subquery_k=5,
        top_k=20,
        expected_evidence=(),
    )
    assert metrics.final_chunk_count == 3
    assert metrics.raw_hit_count == 3
    assert metrics.subquery_count == 2
    assert metrics.source_distribution == {"dnv": 1, "iea": 2}
    assert metrics.unique_filename_count == 2
    assert metrics.unique_page_count == 2
    assert metrics.dominant_filename_share == pytest.approx(2 / 3)
    assert metrics.recall_display == "n/a"


def test_summarize_trace_dominant_share_single_file() -> None:
    trace = RetrievalTrace(
        original_query="q",
        subquery_results=[],
        final_chunks=[_chunk("only.pdf", 1)] * 4,
    )
    metrics = summarize_trace(
        trace,
        profile_name="k10",
        subquery_k=10,
        top_k=20,
        expected_evidence=(),
    )
    assert metrics.dominant_filename_share == 1.0
    assert metrics.filename_distribution == {"only.pdf": 4}


def test_format_metrics_table_includes_recall_na() -> None:
    metrics = summarize_trace(
        RetrievalTrace(original_query="q", subquery_results=[], final_chunks=[]),
        profile_name="k5",
        subquery_k=None,
        top_k=20,
        expected_evidence=(),
    )
    table = "\n".join(format_metrics_table([metrics]))
    assert "n/a" in table
    assert "k5" in table


def test_evidence_recall_result_fraction() -> None:
    r = EvidenceRecallResult(total_expected=4, hits=2, hit_items=(), missed_items=())
    assert r.format_recall() == "2/4"
    assert r.recall_fraction == 0.5

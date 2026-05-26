"""Retrieval-only eval helpers (pure logic, no Ollama/Chroma).

Loads case definitions from JSON and computes metrics from RetrievalTrace.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_rag_sandbox.qa import RetrievedChunk, RetrievalTrace

SUPPORTED_VERSION = 1


@dataclass(frozen=True)
class ExpectedEvidence:
    filename: str
    page: int | str


@dataclass(frozen=True)
class RetrievalCase:
    id: str
    question: str
    source: str | None = None
    year: int | None = None
    topic: str | None = None
    notes: str | None = None
    expected_evidence: tuple[ExpectedEvidence, ...] = ()


@dataclass(frozen=True)
class EvidenceRecallResult:
    """Recall against expected (filename, page) on final chunks."""

    total_expected: int
    hits: int
    hit_items: tuple[ExpectedEvidence, ...]
    missed_items: tuple[ExpectedEvidence, ...]

    @property
    def recall_fraction(self) -> float | None:
        if self.total_expected == 0:
            return None
        return self.hits / self.total_expected

    def format_recall(self) -> str:
        if self.total_expected == 0:
            return "n/a"
        return f"{self.hits}/{self.total_expected}"


@dataclass(frozen=True)
class RetrievalRunMetrics:
    profile_name: str
    subquery_k: int | None
    top_k: int
    final_chunk_count: int
    raw_hit_count: int
    subquery_count: int
    source_distribution: dict[str, int]
    filename_distribution: dict[str, int]
    unique_filename_count: int
    unique_page_count: int
    evidence_recall: EvidenceRecallResult
    dominant_filename_share: float
    matched_evidence: tuple[ExpectedEvidence, ...]
    missed_evidence: tuple[ExpectedEvidence, ...]

    @property
    def recall_display(self) -> str:
        return self.evidence_recall.format_recall()


@dataclass(frozen=True)
class RetrievalProfile:
    name: str
    subquery_k: int | None  # None -> default K_PER_SUBQUERY at call site


def _parse_expected_item(raw: dict[str, Any]) -> ExpectedEvidence:
    if "filename" not in raw or "page" not in raw:
        raise ValueError("expected_evidence items require filename and page")
    page = raw["page"]
    if not isinstance(page, (int, str)):
        raise TypeError("expected_evidence page must be int or str")
    return ExpectedEvidence(filename=str(raw["filename"]), page=page)


def _parse_case(raw: dict[str, Any]) -> RetrievalCase:
    if "id" not in raw or "question" not in raw:
        raise ValueError("each case requires id and question")
    case_id = str(raw["id"]).strip()
    question = str(raw["question"]).strip()
    if not case_id or not question:
        raise ValueError("case id and question must be non-empty")

    expected_raw = raw.get("expected_evidence") or []
    if not isinstance(expected_raw, list):
        raise TypeError("expected_evidence must be a list")
    expected = tuple(_parse_expected_item(item) for item in expected_raw)

    year = raw.get("year")
    if year is not None and not isinstance(year, int):
        raise TypeError("year must be an integer when present")

    return RetrievalCase(
        id=case_id,
        question=question,
        source=raw.get("source"),
        year=year,
        topic=raw.get("topic"),
        notes=raw.get("notes"),
        expected_evidence=expected,
    )


def load_cases(path: Path) -> list[RetrievalCase]:
    """Load and validate retrieval eval cases from JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("case file must be a JSON object")

    version = data.get("version")
    if version != SUPPORTED_VERSION:
        raise ValueError(f"unsupported version {version!r}; expected {SUPPORTED_VERSION}")

    cases_raw = data.get("cases")
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ValueError("cases must be a non-empty list")

    cases = [_parse_case(item) for item in cases_raw]
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise ValueError(f"duplicate case id: {case.id!r}")
        seen.add(case.id)
    return cases


def filter_cases(cases: list[RetrievalCase], case_ids: list[str] | None) -> list[RetrievalCase]:
    if not case_ids:
        return cases
    wanted = set(case_ids)
    selected = [c for c in cases if c.id in wanted]
    missing = wanted - {c.id for c in selected}
    if missing:
        raise ValueError(f"unknown case id(s): {', '.join(sorted(missing))}")
    return selected


def _page_key(chunk: RetrievedChunk) -> tuple[str, int | str]:
    return (chunk.filename.lower(), chunk.page_display)


def _matches_expected(chunk: RetrievedChunk, expected: ExpectedEvidence) -> bool:
    if chunk.filename.lower() != expected.filename.lower():
        return False
    return chunk.page_display == expected.page


def compute_evidence_recall(
    chunks: list[RetrievedChunk],
    expected: tuple[ExpectedEvidence, ...],
) -> EvidenceRecallResult:
    if not expected:
        return EvidenceRecallResult(0, 0, (), ())

    hit_items: list[ExpectedEvidence] = []
    missed_items: list[ExpectedEvidence] = []
    for item in expected:
        if any(_matches_expected(c, item) for c in chunks):
            hit_items.append(item)
        else:
            missed_items.append(item)
    return EvidenceRecallResult(
        total_expected=len(expected),
        hits=len(hit_items),
        hit_items=tuple(hit_items),
        missed_items=tuple(missed_items),
    )


def summarize_trace(
    trace: RetrievalTrace,
    *,
    profile_name: str,
    subquery_k: int | None,
    top_k: int,
    expected_evidence: tuple[ExpectedEvidence, ...],
) -> RetrievalRunMetrics:
    """Compute v1 retrieval metrics from a trace."""
    final = trace.final_chunks
    raw_hit_count = sum(len(sq.hits) for sq in trace.subquery_results)

    source_distribution = dict(Counter(c.source for c in final))
    filename_distribution = dict(Counter(c.filename for c in final))
    unique_page_count = len({_page_key(c) for c in final})
    unique_filename_count = len({c.filename for c in final})

    if final:
        _top_name, top_count = Counter(c.filename for c in final).most_common(1)[0]
        dominant_filename_share = top_count / len(final)
    else:
        dominant_filename_share = 0.0

    evidence_recall = compute_evidence_recall(final, expected_evidence)

    return RetrievalRunMetrics(
        profile_name=profile_name,
        subquery_k=subquery_k,
        top_k=top_k,
        final_chunk_count=len(final),
        raw_hit_count=raw_hit_count,
        subquery_count=len(trace.subquery_results),
        source_distribution=source_distribution,
        filename_distribution=filename_distribution,
        unique_filename_count=unique_filename_count,
        unique_page_count=unique_page_count,
        evidence_recall=evidence_recall,
        dominant_filename_share=dominant_filename_share,
        matched_evidence=evidence_recall.hit_items,
        missed_evidence=evidence_recall.missed_items,
    )


def empty_trace_metrics(
    *,
    profile_name: str,
    subquery_k: int | None,
    top_k: int,
    expected_evidence: tuple[ExpectedEvidence, ...],
) -> RetrievalRunMetrics:
    """Metrics when retrieval returned no trace."""
    return summarize_trace(
        RetrievalTrace(original_query="", subquery_results=[], final_chunks=[]),
        profile_name=profile_name,
        subquery_k=subquery_k,
        top_k=top_k,
        expected_evidence=expected_evidence,
    )


def format_source_distribution(dist: dict[str, int]) -> str:
    if not dist:
        return "-"
    return ", ".join(f"{k}:{v}" for k, v in sorted(dist.items()))


def format_top_filenames(dist: dict[str, int], limit: int = 2) -> str:
    if not dist:
        return "-"
    items = sorted(dist.items(), key=lambda x: (-x[1], x[0]))[:limit]
    return ", ".join(f"{name}({count})" for name, count in items)


def format_evidence_item(item: ExpectedEvidence) -> str:
    return f"{item.filename} p.{item.page}"


def format_evidence_list(items: tuple[ExpectedEvidence, ...]) -> str:
    if not items:
        return "(none)"
    return "; ".join(format_evidence_item(item) for item in items)


def format_profile_evidence_lines(metrics: RetrievalRunMetrics) -> list[str]:
    """Matched/missed expected evidence for one profile (empty if no gold labels)."""
    if not metrics.evidence_recall.total_expected:
        return []
    prefix = f"  [{metrics.profile_name}]"
    return [
        f"{prefix} matched: {format_evidence_list(metrics.matched_evidence)}",
        f"{prefix} missed : {format_evidence_list(metrics.missed_evidence)}",
    ]


def format_metrics_table(rows: list[RetrievalRunMetrics]) -> list[str]:
    """Compact table lines for terminal output."""
    header = (
        f"{'profile':<8} {'final':>5} {'raw':>5} {'sqs':>4} "
        f"{'recall':>8} {'uniq_f':>6} {'uniq_p':>6} {'dom':>5} sources"
    )
    lines = [header, "-" * len(header)]
    for m in rows:
        dom_pct = f"{m.dominant_filename_share:.0%}" if m.final_chunk_count else "-"
        lines.append(
            f"{m.profile_name:<8} {m.final_chunk_count:>5} {m.raw_hit_count:>5} "
            f"{m.subquery_count:>4} {m.recall_display:>8} "
            f"{m.unique_filename_count:>6} {m.unique_page_count:>6} {dom_pct:>5} "
            f"{format_source_distribution(m.source_distribution)}"
        )
    return lines

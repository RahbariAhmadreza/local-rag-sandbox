"""Multi-query retrieval helpers (domain-agnostic).

Used by Deep R&D mode (depth="learning") to decompose a broad question into
focused subqueries, retrieve chunks per subquery, deduplicate, and format
grouped context for the answer prompt.
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from typing import Protocol

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage


class RetrievedChunkLike(Protocol):
    """Minimal chunk interface for grouped context formatting."""

    filename: str
    page_display: int | str
    source: str
    year: int | str
    topic: str
    content: str
    retrieval_query: str | None

K_PER_SUBQUERY: int = 5
SUBQUERY_MIN: int = 3
SUBQUERY_MAX: int = 5

_LINE_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"\d+[\.\)\]:\-]*\s*"  # numbered: 1. 1) 1:
    r"|[-*•]\s+"           # bullets
    r")*",
    re.IGNORECASE,
)

DECOMPOSE_PROMPT: str = """\
You help retrieve passages from a local document library.

Given the user question below, write {min_q} to {max_q} short focused search queries that would help find relevant passages.

Rules:
- Use only terms present in or strongly implied by the user's question.
- Do not add outside domain knowledge or assumptions.
- Each query should target a distinct aspect of the question.
- Write one search query per line.
- Output only the queries, nothing else.

User question:
{question}
"""


def normalize_content_for_hash(text: str) -> str:
    """Collapse whitespace for stable content hashing."""
    return " ".join(text.split())


def content_hash(text: str) -> str:
    """Return a short hex digest of normalized chunk content."""
    normalized = normalize_content_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def document_dedup_key(doc: Document) -> tuple[str, int | str, str]:
    """Identity key: filename + raw page + content hash."""
    filename = doc.metadata.get("filename", "unknown file")
    page_raw = doc.metadata.get("page", "N/A")
    return (filename, page_raw, content_hash(doc.page_content))


def parse_subqueries(text: str) -> list[str]:
    """Parse one-query-per-line LLM output; tolerate numbering and bullets."""
    queries: list[str] = []
    for raw_line in text.splitlines():
        line = _LINE_PREFIX_RE.sub("", raw_line).strip()
        if not line:
            continue
        # Drop common wrapper quotes
        if len(line) >= 2 and line[0] == line[-1] and line[0] in "\"'":
            line = line[1:-1].strip()
        if line:
            queries.append(line)
    return queries[:SUBQUERY_MAX]


def decompose_question(question: str, llm: BaseChatModel) -> list[str]:
    """Generate 3–5 focused subqueries; fall back to the original question."""
    prompt = DECOMPOSE_PROMPT.format(
        min_q=SUBQUERY_MIN,
        max_q=SUBQUERY_MAX,
        question=question,
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content if isinstance(response.content, str) else str(response.content)
        parsed = parse_subqueries(content)
        if parsed:
            return parsed
    except Exception:
        pass
    return [question]


def dedupe_hits(
    hits: list[tuple[Document, str]],
) -> list[tuple[Document, str]]:
    """Drop duplicate chunks; preserve first-seen retrieval order."""
    seen: set[tuple[str, int | str, str]] = set()
    unique: list[tuple[Document, str]] = []
    for doc, retrieval_query in hits:
        key = document_dedup_key(doc)
        if key in seen:
            continue
        seen.add(key)
        unique.append((doc, retrieval_query))
    return unique


def cap_hits(
    hits: list[tuple[Document, str]],
    max_chunks: int,
) -> list[tuple[Document, str]]:
    """Keep the first max_chunks hits after deduplication."""
    if max_chunks <= 0:
        return []
    return hits[:max_chunks]


def build_grouped_context(chunks: list[RetrievedChunkLike]) -> str:
    """Format chunks grouped by focused retrieval query for the answer prompt."""
    if not chunks:
        return ""

    groups: OrderedDict[str, list[RetrievedChunkLike]] = OrderedDict()
    for chunk in chunks:
        label = chunk.retrieval_query or "Original question"
        groups.setdefault(label, []).append(chunk)

    parts: list[str] = []
    parts.append(
        "Retrieval used several focused queries. Sections below are grouped by query. "
        "Cite using the File and Page lines in each chunk.\n"
    )

    chunk_num = 0
    for query_num, (query_text, group_chunks) in enumerate(groups.items(), start=1):
        parts.append(f'=== Focused query {query_num}: "{query_text}" ===\n')
        for chunk in group_chunks:
            chunk_num += 1
            parts.append(
                f"Chunk {chunk_num}\n"
                f"File:    {chunk.filename}\n"
                f"Page:    {chunk.page_display}\n"
                f"Source:  {chunk.source}\n"
                f"Year:    {chunk.year}\n"
                f"Topic:   {chunk.topic}\n"
                f"Content:\n{chunk.content}"
            )
        parts.append("")

    return "\n".join(parts).rstrip()

# GEMINI.md — local-rag-sandbox instructions

## Project purpose

This is a local/offline-first RAG sandbox for querying public energy-transition reports using Ollama, LangChain, Chroma, and Python.

The project is a learning and portfolio repository. Keep it simple, inspectable, and useful for practical report analysis.

## Current priorities

Focus on the retrieval engine, especially Deep R&D / learning mode.

Near-term priorities (in order):

1. Retrieval diagnostics
2. Retrieval eval harness
3. Evidence selection / ranking experiments
4. Only then answer synthesis or UI improvements

Meaning:
- Diagnostics first: inspect what retrieval is doing.
- Eval harness second: measure retrieval quality before changing algorithms.
- Ranking experiments third: RRF/reranking/source balancing only after baseline eval exists.
- Answer synthesis or UI later: do not tune prompts or Streamlit before retrieval is measured.

Do not prioritize Streamlit UI, ingestion dedupe, or large architectural rewrites unless explicitly requested.

## Development rules

- Prefer small, behavior-preserving changes.
- Keep defaults unchanged unless diagnostics/eval justify changing them.
- Add offline unit tests for helper logic.
- Do not add live Ollama/Chroma tests unless explicitly requested.
- Do not introduce heavy eval frameworks such as Ragas, TruLens, or MLflow.
- Do not implement RRF or prompt changes without a measurable baseline.
- Preserve local-first behavior and avoid cloud dependencies.

## Current retrieval notes

Deep R&D / learning mode:
- Decomposes the user question into focused subqueries.
- Retrieves chunks per subquery.
- Dedupes and caps final context.
- Uses RetrievalTrace for diagnostics.

Important finding:
Increasing `subquery-k` from 5 to 10 increased final chunks for one FID/offtake/investment question from 9 to 17, but the generated answer drifted. More context is not automatically better.

Therefore, before changing retrieval defaults, build/evaluate a small retrieval-only eval harness.

## Preferred next phase

Create a minimal retrieval eval harness:

- Store cases in `eval/retrieval_cases.json`.
- Add `scripts/eval_retrieval.py`.
- Compare retrieval settings such as default `subquery-k=5` and experimental `subquery-k=10`.
- Track source/page recall, final chunk count, source distribution, filename/page diversity, and duplicate overlap.
- Do not grade generated answers yet.

## Working style

Before editing files:
1. Inspect relevant code.
2. Propose file-level plan.
3. Keep scope narrow.
4. After edits, summarize changed files and tests.

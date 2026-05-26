# GEMINI.md — local-rag-sandbox instructions

## Project purpose

This is a local/offline-first RAG sandbox for querying public energy-transition reports using Ollama, LangChain, Chroma, and Python.

The project is a learning and portfolio repository. Keep it simple, inspectable, and useful for practical report analysis.

## Current checkpoint

Latest commit on `main`: **530235e** — *Show matched and missed retrieval eval evidence*.

## Current priorities

Focus on the retrieval engine, especially Deep R&D / learning mode.

Near-term priorities (in order):

1. Retrieval diagnostics — **shipped** (`scripts/diagnose_retrieval.py`, `RetrievalTrace`)
2. Retrieval eval harness — **shipped v1** (`eval/retrieval_cases.json`, `scripts/eval_retrieval.py`, matched/missed output)
3. **Phase 3.2 — eval hardening / dataset refinement** (current)
4. Evidence selection / ranking experiments (RRF/reranking/source balancing) — only after Phase 3.2 baseline
5. Only then answer synthesis or UI improvements

Meaning:
- Diagnostics first: inspect what retrieval is doing.
- Eval harness second: measure retrieval quality before changing algorithms.
- **Next:** refine partial `expected_evidence`, add small eval metrics (e.g. cross-subquery duplicate count, dominance warnings), run `--compare` across cases — not answer grading yet.
- Ranking experiments third: do not implement RRF blindly; DNV single-file dominance is a known risk.
- Answer synthesis or UI later: do not tune prompts or Streamlit before retrieval is measured.

Do not prioritize Streamlit UI, ingestion dedupe, or large architectural rewrites unless explicitly requested.

## Completed (retrieval tooling)

- `RetrievalTrace` / `SubqueryResult` on `_retrieve_learning` (`qa.py`)
- `scripts/diagnose_retrieval.py` — human retrieval report (no answer generation)
- `--subquery-k` on diagnose and expert `ask.py --depth learning` only
- `eval/retrieval_cases.json` — partial, corpus-specific reference labels (not full gold)
- `src/local_rag_sandbox/retrieval_eval.py` + `scripts/eval_retrieval.py --compare` (k5 vs k10)
- Matched/missed expected evidence lines per profile in eval output

Defaults unchanged: `K_PER_SUBQUERY = 5`, learning `top_k = 20`.

## Development rules

- Prefer small, behavior-preserving changes.
- Keep defaults unchanged unless diagnostics/eval justify changing them.
- Add offline unit tests for helper logic.
- Do not add live Ollama/Chroma tests unless explicitly requested.
- Do not introduce heavy eval frameworks such as Ragas, TruLens, or MLflow.
- Do not implement RRF or prompt changes without a measurable baseline from eval.
- Preserve local-first behavior and avoid cloud dependencies.

## Current retrieval notes

Deep R&D / learning mode:
- Decomposes the user question into focused subqueries.
- Retrieves chunks per subquery (`K_PER_SUBQUERY = 5` by default; override via `--subquery-k`).
- Dedupes and caps final context (`top_k` default 20 for learning).
- Uses `RetrievalTrace` for diagnostics and eval.

Important finding:
Increasing `subquery-k` from 5 to 10 increased final chunks for one FID/offtake/investment question from 9 to 17 and improved partial recall on reference pages, but the generated answer drifted away from investment/offtake/FID themes. **More retrieval volume is not automatically better.**

Therefore:
- Do not change default `K_PER_SUBQUERY` or learning `top_k` yet.
- Do not implement RRF without eval comparison across multiple cases.
- Do not tune prompts to “fix” retrieval selection problems.

## Preferred next phase (Phase 3.2)

Eval hardening and dataset refinement:

- Refine partial `expected_evidence` in `eval/retrieval_cases.json` (corpus-specific, indicative only).
- Add small eval metrics: cross-subquery duplicate count, dominance warnings (deferred in v1).
- Run `python scripts/eval_retrieval.py --compare` on all cases; record k5 vs k10 tables.
- Only after that, consider a minimal ranking experiment (e.g. per-file cap or simple fusion) gated by eval.

Still out of scope: answer generation grading, heavy eval frameworks, default changes without eval justification.

## Useful commands

```bash
pytest -v
python scripts/diagnose_retrieval.py --subquery-k 10 "Your question"
python scripts/eval_retrieval.py --compare
python scripts/eval_retrieval.py --compare --case-id hydrogen_fid_offtake_investment
```

## Working style

Before editing files:
1. Inspect relevant code.
2. Propose file-level plan.
3. Keep scope narrow.
4. After edits, summarize changed files and tests.

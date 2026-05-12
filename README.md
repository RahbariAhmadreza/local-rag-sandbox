# Local RAG Sandbox

> A local, offline-first Retrieval-Augmented Generation (RAG) sandbox for asking grounded questions over your own documents using Ollama, LangChain, and Chroma. Includes metadata-aware filters by source, year, and topic for multi-publisher / multi-year energy-transition report collections.

By **Ahmadreza Rahbari** ([@RahbariAhmadreza](https://github.com/RahbariAhmadreza)).

Everything runs on your machine. No cloud APIs. No data leaves your laptop. Apple Silicon-friendly.

---

## What it does

Drop your documents into `data/<publisher>/`, run two scripts, then ask questions about them. Every answer is structured, evidence-cited (filename + page), and grounded in chunks retrieved from your local Chroma database. Nothing is sent to the cloud.

A typical full-corpus query returns this shape:

```bash
$ python scripts/ask.py "What are the main types of electrolyzers covered in the corpus?"

Full corpus: no metadata filters applied.
Retrieving the 10 most relevant chunks ...
Got 10 chunk(s). Sending to qwen2.5:7b ...

## Question
What are the main types of electrolyzers covered in the corpus?

## Answer
A short, direct answer grounded in the retrieved chunks.

## Evidence
(2024_global_hydrogen_review.pdf, p. 42)
(2024_eto_main_report.pdf, p. 17)
...

## Interpretation
What the cited evidence implies for energy transition, policy,
investment, or technology strategy.

## Uncertainty
What is missing, unclear, or not supported by the retrieved context.
```

The headline Stage 2 feature is **metadata-aware filtered retrieval** — narrow the searched corpus before the question is asked:

```bash
# IEA reports only
python scripts/ask.py --source iea "How is electrolyser deployment tracked?"

# IEA 2024 reports only (AND semantics)
python scripts/ask.py --source iea --year 2024 "What were the main hydrogen bottlenecks?"

# Specific topic across all years and publishers
python scripts/ask.py --topic global_hydrogen_review "How has electrolyser scale-up evolved?"
```

Local. Offline. Apple Silicon-friendly.

## Why this exists

This project is both:

1. **A learning sandbox** for understanding practical LLM application workflows — local model calls, prompt templates, document loading, chunking, embeddings, vector stores, semantic retrieval, RAG, and evidence-grounded answers.
2. **A public portfolio project** demonstrating practical AI engineering at the intersection of LLM applications, RAG architecture, and energy-transition / hydrogen / electrification analysis.

The first example use case implemented in this repo is a RAG assistant over public energy-transition, hydrogen, and electrolyser reports. The architecture is general: swap the source documents, prompt, and models, and the same pipeline can serve legal contracts, internal policy documents, research archives, codebases, or any other text corpus.

## Architecture

```
Documents in data/<publisher>/YYYY_topic.pdf
    ↓
PyPDFLoader / TextLoader / CSVLoader
    ↓
Path-derived metadata: {source, year, topic, filename}
    ↓
RecursiveCharacterTextSplitter (1000-char chunks, 150-char overlap)
    ↓
Per-chunk metadata: {source, year, topic, filename, page}
    ↓
OllamaEmbeddings (nomic-embed-text → 768-dim vectors)
    ↓
Chroma (persisted to chroma_db/, HNSW index + metadata store)
    ↓
[query time]
    ↓
Optional metadata filter (--source / --year / --topic, AND semantics)
    ↓
Top-K retrieval (k=10 by default; tunable via --top-k)
    ↓
ChatPromptTemplate (Question / Answer / Evidence / Interpretation / Uncertainty)
    ↓
ChatOllama (qwen2.5:7b)
    ↓
Markdown answer with filename + page citations
```

The model **does not learn** from your documents — they are indexed in Chroma and relevant chunks are retrieved at question time and inserted into the prompt context. This is RAG, not fine-tuning.

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| Local model runtime | [Ollama](https://ollama.com) | Runs open-weight models fully offline on Apple Silicon. |
| Orchestration | [LangChain](https://www.langchain.com) | Standard glue for LLM applications. |
| Vector database | [Chroma](https://www.trychroma.com) | Local-first, persistent, embedded. No server. |
| Chat / reasoning | `qwen2.5:7b` (~4.7 GB) | Strong open-weight 7B model. |
| Fallback / fast | `llama3.2:3b` (~2 GB) | Quick smoke-test model. |
| Embedding | `nomic-embed-text` (~274 MB) | 768-dim embeddings, fast on CPU/GPU. |
| Packaging | `pyproject.toml` (src layout) | Modern Python project structure. |
| Dev tools | pytest, ruff | Tests and linting. |

## Project structure

```
local-rag-sandbox/
├── src/
│   └── local_rag_sandbox/
│       ├── __init__.py
│       ├── config.py          # paths, model names, supported extensions
│       └── prompts.py         # RAG prompt template (single, conditional)
├── scripts/
│   ├── test_ollama.py         # smoke test: can we reach Ollama?
│   ├── ingest.py              # recursively scan data/, chunk, embed, persist
│   ├── inspect_db.py          # read-only Chroma stats: chunk counts + samples
│   └── ask.py                 # query Chroma with optional metadata filters
├── tests/
│   ├── __init__.py
│   └── test_smoke.py          # imports & config sanity (8 tests)
├── data/                      # gitignored — see "Add source documents" for layout
│   ├── dnv/                   # one subfolder per publisher
│   │   ├── 2024_eto_main_report.pdf
│   │   └── ...
│   └── iea/
│       ├── 2024_global_hydrogen_review.pdf
│       └── ...
├── chroma_db/                 # persistent vector DB (gitignored, rebuildable)
├── eval/                      # placeholder for future evaluation questions
├── pyproject.toml             # package metadata + dev dependencies
├── requirements.txt           # pinned snapshot for full reproducibility
├── .gitignore
└── README.md
```

## Requirements

- macOS (developed on Apple M3 Pro, 18 GB RAM; works on any Apple Silicon).
- [Homebrew](https://brew.sh) (for installing Ollama).
- [Conda](https://www.anaconda.com/download) or Python 3.11+ with `venv`.
- ~10 GB free disk (models + database).

## Installation

### 1. Install Ollama

```bash
brew install --cask ollama
open -a Ollama   # launches the app once so the daemon starts
```

You should see a llama icon in your macOS menu bar after the launch.

### 2. Pull the local models

```bash
ollama pull nomic-embed-text   # embeddings, ~274 MB
ollama pull llama3.2:3b        # fast fallback chat, ~2 GB
ollama pull qwen2.5:7b         # main chat model, ~4.7 GB
```

Verify:

```bash
ollama list
```

### 3. Create the Python environment

Recommended — conda:

```bash
conda create -n local-rag-sandbox python=3.11 -y
conda activate local-rag-sandbox
```

Or with built-in `venv`:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install the package + dev tools

```bash
pip install -e ".[dev]"
```

This installs `local_rag_sandbox` in editable mode plus pytest and ruff.

To reproduce the exact pinned versions used during development:

```bash
pip install -r requirements.txt
```

### 5. Verify the install

```bash
pytest -v             # should report 8 passed
python scripts/test_ollama.py    # should print 3 bullets about grid congestion
```

If both succeed, the stack is working end-to-end.

## Usage

### 1. Add source documents

The ingest pipeline recursively scans `data/` and attaches per-chunk metadata derived from each file's path. The recommended layout is **one subfolder per publisher**, with filenames that start with the year:

```
data/
├── dnv/
│   ├── 2024_eto_main_report.pdf
│   └── 2025_eto_main_report.pdf
├── iea/
│   ├── 2024_global_hydrogen_review.pdf
│   └── 2025_global_hydrogen_review.pdf
└── ...
```

From this layout the ingestor derives:

| Metadata key | Source | Example |
|---|---|---|
| `source` | Parent folder name | `"dnv"`, `"iea"` |
| `year` | Filename prefix `YYYY_` (integer) | `2024` |
| `topic` | Filename after `YYYY_`, before extension | `"eto_main_report"` |
| `filename` | The bare filename, for citations | `"2024_eto_main_report.pdf"` |
| `page` | From `PyPDFLoader` | `42` |

Filenames must follow the `YYYY_<topic>.<ext>` convention for `year` and `topic` to be parsed. Files placed directly in `data/` (no publisher subfolder) get `source="unknown"` and still ingest — they just can't be filtered by source.

Supported file types:

| Extension | Loader |
|---|---|
| `.pdf` | `PyPDFLoader` |
| `.txt`, `.md` | `TextLoader` |
| `.csv` | `CSVLoader` |

Supported types are declared in `src/local_rag_sandbox/config.py` under `SUPPORTED_EXTENSIONS`. Adding a new extension requires (a) adding it to that tuple and (b) adding the matching loader to `scripts/ingest.py`. Commented-out future entries exist for `.docx`, `.xlsx`, and `.html`.

### 2. Build the vector database

```bash
python scripts/ingest.py
```

Output ends with something like:

```
Total chunks: 11388
Done in 190.8s. Chroma collection 'documents' saved to: .../chroma_db
```

Exact numbers depend on your corpus size. Re-running `ingest.py` currently re-adds duplicates; see [Known limitations](#known-limitations).

**Dry-run mode.** List discovered files and previewed metadata without loading, chunking, embedding, or writing anything:

```bash
python scripts/ingest.py --dry-run
```

Useful for verifying the path-derived metadata before a full ingest (e.g. confirming a filename matched the `YYYY_topic.pdf` pattern).

### 3. Inspect the database (optional)

Read-only sanity check of what's actually in Chroma:

```bash
python scripts/inspect_db.py
```

Prints total chunk count, breakdown by `source`, breakdown by `(source, year)`, and a few sample chunks with their full metadata. Useful after an ingest run to confirm metadata is attached as expected.

### 4. Ask questions

Whole-corpus question (no filters, same as v1):

```bash
python scripts/ask.py "What are the main types of electrolyzers covered in the corpus?"
```

Interactive mode (no positional question):

```bash
python scripts/ask.py
# > Ask a question about the indexed documents: <type here>
```

**Filtered queries.** Filters are optional, exact-match, and combined with AND semantics. `--source` and `--topic` are lowercased before matching, so case doesn't matter:

```bash
# Single publisher
python scripts/ask.py --source dnv "What does DNV say about hydrogen offtake?"

# Single year
python scripts/ask.py --year 2024 "What were the main hydrogen bottlenecks in 2024?"

# Single topic across all sources and years
python scripts/ask.py --topic global_hydrogen_review "How has electrolyser scale-up evolved?"

# Combined (AND): IEA 2024 reports only
python scripts/ask.py --source iea --year 2024 "..."
```

**Tune retrieval depth.** Default `TOP_K = 10`. Lower for sharper, narrow-question context; higher for broad synthesis:

```bash
python scripts/ask.py --top-k 4 "..."     # narrow factual question
python scripts/ask.py --top-k 20 "..."    # broad synthesis across many chunks
```

Before retrieval, the active filter set is printed. Examples:

```
Full corpus: no metadata filters applied.
Retrieving the 10 most relevant chunks ...
```

```
Filtered corpus: source=iea, year=2024.
Retrieving the 10 most relevant chunks ...
```

Answers follow a 5-section template:

```
## Question
Restated user question.

## Answer
Short, direct, evidence-grounded answer.

## Evidence
Filename + page citations from the retrieved chunks, e.g.
(2024_global_hydrogen_review.pdf, p. 42).

## Interpretation
What the cited evidence implies for energy transition, policy,
investment, or strategy.

## Uncertainty
What is missing, unclear, or not supported by the retrieved context.
```

## Configuration

All tunable constants live in [`src/local_rag_sandbox/config.py`](src/local_rag_sandbox/config.py):

| Constant | Default | What it does |
|---|---|---|
| `EMBED_MODEL` | `"nomic-embed-text"` | Ollama embedding model. |
| `CHAT_MODEL` | `"qwen2.5:7b"` | Main chat/reasoning model. |
| `FALLBACK_CHAT_MODEL` | `"llama3.2:3b"` | Faster, smaller chat model. |
| `COLLECTION_NAME` | `"documents"` | Chroma collection name. |
| `SUPPORTED_EXTENSIONS` | `(".pdf", ".txt", ".md", ".csv")` | File types `ingest.py` will pick up. |
| `CHUNK_SIZE` (in `ingest.py`) | `1000` | Characters per chunk. |
| `CHUNK_OVERLAP` (in `ingest.py`) | `150` | Overlap between adjacent chunks. |
| `TOP_K` (in `ask.py`) | `10` | Default number of chunks retrieved per question. Override per call with `--top-k`. |

Swap to the fallback model by changing `CHAT_MODEL` to `"llama3.2:3b"`.

## Known limitations

- **Re-ingestion adds duplicates.** Running `python scripts/ingest.py` twice without wiping `chroma_db/` re-adds every file's chunks. To rebuild cleanly:

  ```bash
  rm -rf chroma_db/
  python scripts/ingest.py
  ```

  Deterministic chunk IDs are on the roadmap to make re-ingest a no-op for already-indexed files.

- **No reranking.** Pure semantic top-K. Quality depends on how well chunks match the question wording.
- **No comparison mode yet.** Filters narrow the searched corpus but do not orchestrate per-year / per-source retrieval and merging. Asking "How did IEA's view change from 2021 to 2025?" still treats all retrieved chunks symmetrically. A dedicated comparison mode is on the roadmap.
- **Page numbers are zero-based.** `PyPDFLoader` reports pages starting from 0, so a citation like `(file.pdf, p. 17)` corresponds to page 18 as printed inside the PDF. A `(page + 1)` display fix is on the roadmap.
- **Tables and figures in PDFs extract poorly.** `pypdf` handles prose well; complex layouts less so. For graphics-heavy PDFs, try `pymupdf` or `unstructured` (not installed by default).
- **First-call latency.** Model loading into RAM costs 15–60 seconds on first run; subsequent runs reuse the warm model for ~5 minutes.

## Privacy

- All inference happens **locally** via Ollama on `localhost:11434`.
- Nothing is sent to OpenAI, Anthropic, Alibaba, or any cloud API.
- Your documents and prompts never leave your machine.
- `data/**/*.pdf` is gitignored by default (any subfolder depth) to prevent accidental commits of licensed material.

## Roadmap

Possible directions, in rough priority order:

- **Per-file deduplication via deterministic chunk IDs** — kills the duplicate-on-reingest footgun and makes re-ingest a no-op for unchanged files.
- **Comparison mode across multiple reports** — per-year / per-source retrieval with merging, for questions like *"How did IEA's view of electrolyser deployment change from 2021 to 2025?"*.
- **Reranking layer** — cross-encoder rerank of a wide top-N down to a sharp top-K to improve answer quality on broad questions.
- **Citation page display** — `(page + 1)` display fix so user-facing citations match printed PDF page labels.
- **Pydantic-typed structured answer outputs** for programmatic consumers downstream.
- **Simple eval harness** using `eval/test_questions.json` — score retrieval + answer quality empirically rather than by vibe.
- **Optional support for** `.docx`, `.xlsx`, `.html`.

## Acknowledgements

This project follows the practical patterns taught in DeepLearning.AI courses, the LangChain tutorials, and the wider open-weight model community. Built on the work of the Ollama, LangChain, Chroma, Qwen, Llama, and Nomic teams.

## License

[MIT](LICENSE)

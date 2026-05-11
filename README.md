# Local RAG Sandbox

> A local, offline-first Retrieval-Augmented Generation (RAG) sandbox for asking grounded questions over your own documents — PDFs, text files, document collections, or databases — using Ollama, LangChain, and Chroma.

By **Ahmadreza Rahbari** ([@RahbariAhmadreza](https://github.com/RahbariAhmadreza)).

Everything runs on your machine. No cloud APIs. No data leaves your laptop. Apple Silicon-friendly.

---

## What it does

Drop one or more documents into `data/`, run two scripts, then ask questions about them.

In the example below, the document indexed in `data/` is a public technical report on hydrogen-production electrolyzers — *Electrolyzers for Hydrogen Production: Technical and Economic Characteristics* — and that is what "the report" refers to in the answer:

```bash
$ python scripts/ask.py "What are the main types of electrolyzers covered in the report?"

Retrieving the 4 most relevant chunks ...
Got 4 chunk(s). Sending to qwen2.5:7b ...

## Answer
The main types of electrolyzers covered in the report are ALK (alkaline),
PEM (proton exchange membrane), SOEC (solid oxide electrolyzer cell), and
AEM (anion exchange membrane).

## Evidence from the report
Page 49, "Electrolyzer Technologies and Technology/Commercial Readiness
Levels" states: "This report focuses on the four leading types of
electrolyzer technologies now shaping global deployment: ALK, PEM, SOEC,
and AEM."

## Interpretation
The identification of these specific electrolyzer types is crucial for
understanding the current landscape of hydrogen production technology...

## Uncertainty
While the report identifies four main types of electrolyzers, it does not
provide detailed comparisons or rankings among these technologies...
```

Real example. Real output. Real local Mac.

## Why this exists

This project is both:

1. **A learning sandbox** for understanding practical LLM application workflows — local model calls, prompt templates, document loading, chunking, embeddings, vector stores, semantic retrieval, RAG, and evidence-grounded answers.
2. **A public portfolio project** demonstrating practical AI engineering at the intersection of LLM applications, RAG architecture, and energy-transition / hydrogen / electrification analysis.

The first example use case implemented in this repo is a RAG assistant over a public electrolyzer / hydrogen-production report, but the architecture is fully general: swap the source documents, prompt, and models, and the same pipeline serves legal contracts, internal policy documents, research archives, codebases, or any other text corpus.

## Architecture

```
Documents in data/
    ↓
PyPDFLoader / TextLoader / CSVLoader
    ↓
RecursiveCharacterTextSplitter (1000-char chunks, 150-char overlap)
    ↓
OllamaEmbeddings (nomic-embed-text → 768-dim vectors)
    ↓
Chroma (persisted to chroma_db/, HNSW index)
    ↓
[query time]
    ↓
Top-K retrieval (k=4 by default)
    ↓
ChatPromptTemplate (structured answer format)
    ↓
ChatOllama (qwen2.5:7b)
    ↓
Markdown answer with Evidence / Interpretation / Uncertainty sections
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
│       └── prompts.py         # RAG prompt template
├── scripts/
│   ├── test_ollama.py         # smoke test: can we reach Ollama?
│   ├── ingest.py              # build the vector database
│   └── ask.py                 # query the vector database
├── tests/
│   ├── __init__.py
│   └── test_smoke.py          # imports & config sanity (8 tests)
├── data/                      # put your source documents here (gitignored)
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

Drop any combination of supported files into `data/`:

| Extension | Loader |
|---|---|
| `.pdf` | `PyPDFLoader` |
| `.txt`, `.md` | `TextLoader` |
| `.csv` | `CSVLoader` |

Supported types are declared in `src/local_rag_sandbox/config.py` under `SUPPORTED_EXTENSIONS`. Adding a new extension requires (a) adding it to that tuple and (b) adding the matching loader to `scripts/ingest.py`. The config also has commented-out future entries for `.docx`, `.xlsx`, and `.html`.

### 2. Build the vector database

```bash
python scripts/ingest.py
```

Output ends with:

```
Total chunks: ~500
Done in 11.1s. Chroma collection 'documents' saved to: .../chroma_db
```

### 3. Ask questions

CLI argument mode:

```bash
python scripts/ask.py "What are the main types of electrolyzers covered in the report?"
```

Interactive mode:

```bash
python scripts/ask.py
# > Ask a question about the indexed documents: <type here>
```

Answers follow a 4-section template:

```
## Answer
Short direct answer.

## Evidence from the report
Page references and key details from the retrieved chunks.

## Interpretation
What it means for energy transition, policy, investment, or strategy.

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
| `TOP_K` (in `ask.py`) | `4` | Number of chunks retrieved per question. |

Swap to the fallback model by changing `CHAT_MODEL` to `"llama3.2:3b"`.

## Known limitations (v1)

- **Re-ingestion adds duplicates.** Running `python scripts/ingest.py` twice without wiping `chroma_db/` re-adds every file's chunks. To rebuild cleanly:

  ```bash
  rm -rf chroma_db/
  python scripts/ingest.py
  ```

- **No metadata filtering yet.** All chunks live in one collection. v2 will add per-document filtering.
- **No reranking.** Pure semantic top-K. Quality depends on how well chunks match the question wording.
- **Tables and figures in PDFs extract poorly.** `pypdf` handles prose well; complex layouts less so. For graphics-heavy PDFs, try `pymupdf` or `unstructured` (not installed by default).
- **First-call latency.** Model loading into RAM costs 15–60 seconds on first run; subsequent runs reuse the warm model for ~5 minutes.

## Privacy

- All inference happens **locally** via Ollama on `localhost:11434`.
- Nothing is sent to OpenAI, Anthropic, Alibaba, or any cloud API.
- Your documents and prompts never leave your machine.
- `data/*.pdf` is gitignored by default to prevent accidental commits of licensed material.

## Roadmap

Possible v2 directions, in rough priority order:

- Per-file deduplication via deterministic chunk IDs (kills the duplicate-on-reingest footgun).
- Source / publisher / year metadata for filtered questions ("only IEA reports", "only 2024 documents", etc.).
- Comparison mode across multiple reports.
- Pydantic-typed structured answer outputs.
- Reranking layer.
- Simple eval harness using `eval/test_questions.json`.
- Optional support for `.docx`, `.xlsx`, `.html`.

## Acknowledgements

This project follows the practical patterns taught in DeepLearning.AI courses, the LangChain tutorials, and the wider open-weight model community. Built on the work of the Ollama, LangChain, Chroma, Qwen, Llama, and Nomic teams.

## License

[MIT](LICENSE)

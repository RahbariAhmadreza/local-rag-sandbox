"""Project-wide configuration: paths, model names, and supported file types.

All scripts import constants from here so a model swap, path change, or
new file-type support is a one-line edit instead of a find-and-replace
across the codebase.

Source documents are discovered by scanning DATA_DIR for files matching
SUPPORTED_EXTENSIONS — not by pointing at a fixed filename. Drop any
combination of supported files into ``data/`` and they all get ingested.
"""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
CHROMA_DIR: Path = PROJECT_ROOT / "chroma_db"

EMBED_MODEL: str = "nomic-embed-text"
CHAT_MODEL: str = "qwen2.5:7b"
FALLBACK_CHAT_MODEL: str = "llama3.2:3b"

COLLECTION_NAME: str = "documents"

# File extensions that the ingest pipeline knows how to load. Each
# extension maps to a specific LangChain document loader in the ingest
# script. To add a new file type:
#   1. Add the extension here.
#   2. Add the matching loader to the dispatch in scripts/ingest_report.py.
#   3. Install any extra packages required (see commented entries below).
SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    ".pdf",     # PyPDFLoader               (pypdf — installed)
    ".txt",     # TextLoader                (langchain-community — installed)
    ".md",      # TextLoader                (langchain-community — installed)
    ".csv",     # CSVLoader                 (langchain-community — installed)
    # Future support — uncomment after installing the package(s) noted:
    # ".docx",  # Docx2txtLoader            (pip install docx2txt)
    # ".xlsx",  # UnstructuredExcelLoader   (pip install openpyxl pandas)
    # ".html",  # BSHTMLLoader              (pip install beautifulsoup4)
)

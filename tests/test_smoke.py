"""Smoke tests: verify package imports and basic configuration are sane.

These tests intentionally do NOT exercise Ollama or the network — they only
confirm that the package is installed correctly and constants are defined.
For the live Ollama check, run ``python scripts/test_ollama.py`` instead.
"""

from local_rag_sandbox import __version__
from local_rag_sandbox.config import (
    CHAT_MODEL,
    CHROMA_DIR,
    COLLECTION_NAME,
    DATA_DIR,
    EMBED_MODEL,
    FALLBACK_CHAT_MODEL,
    PROJECT_ROOT,
    SUPPORTED_EXTENSIONS,
)
from local_rag_sandbox.prompts import RAG_ANSWER_TEMPLATE


def test_version_exists() -> None:
    assert __version__


def test_project_root_is_a_directory() -> None:
    assert PROJECT_ROOT.is_dir()


def test_data_dir_under_project_root() -> None:
    assert DATA_DIR.parent == PROJECT_ROOT


def test_chroma_dir_under_project_root() -> None:
    assert CHROMA_DIR.parent == PROJECT_ROOT


def test_supported_extensions_non_empty_and_well_formed() -> None:
    assert SUPPORTED_EXTENSIONS
    assert all(ext.startswith(".") for ext in SUPPORTED_EXTENSIONS)
    assert all(ext == ext.lower() for ext in SUPPORTED_EXTENSIONS)


def test_pdf_is_supported() -> None:
    """The v1 example use case requires PDF support."""
    assert ".pdf" in SUPPORTED_EXTENSIONS


def test_model_names_are_set() -> None:
    assert CHAT_MODEL
    assert EMBED_MODEL
    assert FALLBACK_CHAT_MODEL
    assert COLLECTION_NAME


def test_rag_template_has_required_sections() -> None:
    for section in (
        "## Answer",
        "## Evidence from the report",
        "## Interpretation",
        "## Uncertainty",
    ):
        assert section in RAG_ANSWER_TEMPLATE

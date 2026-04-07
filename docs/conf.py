"""Sphinx configuration for researcher-ai documentation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT))

project = "researcher-ai"
author = "researcher-ai contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# Keep docs builds resilient on RTD where runtime-heavy packages may not be present.
autodoc_mock_imports = [
    "anthropic",
    "openai",
    "litellm",
    "langgraph",
    "chromadb",
    "sentence_transformers",
    "pysradb",
    "pdfplumber",
    "PIL",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "previous/**"]

html_theme = "furo"

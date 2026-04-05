"""Tier 4: Notebook structure validation.

Uses nbformat to verify that every completed phase notebook:
- Loads without error (valid JSON)
- Has at least one markdown cell and one code cell
- Has the expected minimum cell count
- Contains the required section headings
- Has the correct kernel metadata

No LLM calls, no network access — tests notebook *shape*, not execution output.
"""

from __future__ import annotations

import pytest
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(name: str):
    """Load and return a parsed notebook, skipping if nbformat is unavailable."""
    try:
        import nbformat
    except ImportError:
        pytest.skip("nbformat not installed")

    path = NOTEBOOKS_DIR / name
    if not path.exists():
        pytest.skip(f"{name} not yet created")

    nb = nbformat.read(str(path), as_version=4)
    # Assign missing cell IDs to suppress MissingIDFieldWarning, which will
    # become a hard error in future nbformat versions.
    import uuid
    for cell in nb.get("cells", []):
        if not cell.get("id"):
            cell["id"] = str(uuid.uuid4())[:8]
    return nb


def _cell_types(nb) -> list[str]:
    return [c["cell_type"] for c in nb["cells"]]


def _markdown_source(nb) -> str:
    return "\n".join(
        c["source"] for c in nb["cells"] if c["cell_type"] == "markdown"
    )


def _code_source(nb) -> str:
    return "\n".join(
        c["source"] for c in nb["cells"] if c["cell_type"] == "code"
    )


# ---------------------------------------------------------------------------
# Phase 1: Data Models
# ---------------------------------------------------------------------------

class TestNotebook01DataModels:
    """01_data_models.ipynb — Phase 1 data model demonstrations."""

    def test_loads_without_error(self):
        _load("01_data_models.ipynb")

    def test_has_markdown_and_code_cells(self):
        nb = _load("01_data_models.ipynb")
        types = _cell_types(nb)
        assert "markdown" in types
        assert "code" in types

    def test_minimum_cell_count(self):
        nb = _load("01_data_models.ipynb")
        assert len(nb["cells"]) >= 4

    def test_kernel_is_python(self):
        nb = _load("01_data_models.ipynb")
        lang = nb.get("metadata", {}).get("kernelspec", {}).get("language", "")
        assert lang.lower() in ("python", "python3", "")


# ---------------------------------------------------------------------------
# Phase 2: Paper Parser
# ---------------------------------------------------------------------------

class TestNotebook02PaperParser:
    """02_parse_paper.ipynb — Phase 2 PaperParser demonstrations."""

    def test_loads_without_error(self):
        _load("02_parse_paper.ipynb")

    def test_has_markdown_and_code_cells(self):
        nb = _load("02_parse_paper.ipynb")
        types = _cell_types(nb)
        assert "markdown" in types
        assert "code" in types

    def test_minimum_cell_count(self):
        nb = _load("02_parse_paper.ipynb")
        assert len(nb["cells"]) >= 8

    def test_imports_paper_parser(self):
        nb = _load("02_parse_paper.ipynb")
        code = _code_source(nb)
        assert "PaperParser" in code

    def test_covers_source_type_detection(self):
        nb = _load("02_parse_paper.ipynb")
        md = _markdown_source(nb)
        assert any(
            kw in md.lower() for kw in ("source type", "detect", "pmid", "doi", "pmcid")
        )

    def test_summary_cell_present(self):
        nb = _load("02_parse_paper.ipynb")
        md = _markdown_source(nb)
        assert "summary" in md.lower()


# ---------------------------------------------------------------------------
# Phase 3: Figure Parser
# ---------------------------------------------------------------------------

class TestNotebook03FigureParser:
    """03_parse_figures.ipynb — Phase 3 FigureParser demonstrations."""

    def test_loads_without_error(self):
        _load("03_parse_figures.ipynb")

    def test_has_markdown_and_code_cells(self):
        nb = _load("03_parse_figures.ipynb")
        types = _cell_types(nb)
        assert "markdown" in types
        assert "code" in types

    def test_minimum_cell_count(self):
        nb = _load("03_parse_figures.ipynb")
        assert len(nb["cells"]) >= 10

    def test_imports_figure_parser(self):
        nb = _load("03_parse_figures.ipynb")
        code = _code_source(nb)
        assert "FigureParser" in code

    def test_covers_caption_extraction(self):
        nb = _load("03_parse_figures.ipynb")
        md = _markdown_source(nb)
        assert any(kw in md.lower() for kw in ("caption", "extract"))

    def test_covers_subfigure_decomposition(self):
        nb = _load("03_parse_figures.ipynb")
        md = _markdown_source(nb)
        assert any(kw in md.lower() for kw in ("subfig", "panel", "decompos"))

    def test_covers_graceful_failure(self):
        nb = _load("03_parse_figures.ipynb")
        md = _markdown_source(nb)
        assert any(kw in md.lower() for kw in ("graceful", "stub", "fail"))

    def test_json_roundtrip_cell_present(self):
        nb = _load("03_parse_figures.ipynb")
        code = _code_source(nb)
        assert "model_dump_json" in code or "model_validate_json" in code

    def test_summary_cell_present(self):
        nb = _load("03_parse_figures.ipynb")
        md = _markdown_source(nb)
        assert "summary" in md.lower()


# ---------------------------------------------------------------------------
# Phase 4: Methods Parser (checked once created)
# ---------------------------------------------------------------------------

class TestNotebook04MethodsParser:
    """04_parse_methods.ipynb — Phase 4 MethodsParser demonstrations."""

    def test_loads_without_error(self):
        _load("04_parse_methods.ipynb")

    def test_has_markdown_and_code_cells(self):
        nb = _load("04_parse_methods.ipynb")
        types = _cell_types(nb)
        assert "markdown" in types
        assert "code" in types

    def test_minimum_cell_count(self):
        nb = _load("04_parse_methods.ipynb")
        assert len(nb["cells"]) >= 8

    def test_imports_methods_parser(self):
        nb = _load("04_parse_methods.ipynb")
        code = _code_source(nb)
        assert "MethodsParser" in code

    def test_covers_assay_identification(self):
        nb = _load("04_parse_methods.ipynb")
        md = _markdown_source(nb)
        assert any(kw in md.lower() for kw in ("assay", "method", "identif"))

    def test_covers_step_decomposition(self):
        nb = _load("04_parse_methods.ipynb")
        md = _markdown_source(nb)
        assert any(kw in md.lower() for kw in ("step", "pipeline", "decompos"))

    def test_summary_cell_present(self):
        nb = _load("04_parse_methods.ipynb")
        md = _markdown_source(nb)
        assert "summary" in md.lower()


# ---------------------------------------------------------------------------
# Phase 5: Data Parsers
# ---------------------------------------------------------------------------

class TestNotebook05DataParsers:
    """05_parse_data.ipynb — Phase 5 GEOParser and SRAParser demonstrations."""

    def test_loads_without_error(self):
        _load("05_parse_data.ipynb")

    def test_has_markdown_and_code_cells(self):
        nb = _load("05_parse_data.ipynb")
        types = _cell_types(nb)
        assert "markdown" in types
        assert "code" in types

    def test_minimum_cell_count(self):
        nb = _load("05_parse_data.ipynb")
        assert len(nb["cells"]) >= 8

    def test_imports_data_parsers(self):
        nb = _load("05_parse_data.ipynb")
        code = _code_source(nb)
        assert "GEOParser" in code
        assert "SRAParser" in code

    def test_covers_accession_validation(self):
        nb = _load("05_parse_data.ipynb")
        md = _markdown_source(nb)
        assert any(kw in md.lower() for kw in ("accession", "valid", "gse", "sra"))

    def test_covers_geo_and_sra(self):
        nb = _load("05_parse_data.ipynb")
        md = _markdown_source(nb)
        assert "geo" in md.lower()
        assert "sra" in md.lower()

    def test_summary_cell_present(self):
        nb = _load("05_parse_data.ipynb")
        md = _markdown_source(nb)
        assert "summary" in md.lower()


# ---------------------------------------------------------------------------
# Phase 7: Pipeline Builder
# ---------------------------------------------------------------------------

class TestNotebook06BuildPipeline:
    """06_build_pipeline.ipynb — Phase 7 PipelineBuilder demonstrations."""

    def test_loads_without_error(self):
        _load("06_build_pipeline.ipynb")

    def test_has_markdown_and_code_cells(self):
        nb = _load("06_build_pipeline.ipynb")
        types = _cell_types(nb)
        assert "markdown" in types
        assert "code" in types

    def test_minimum_cell_count(self):
        nb = _load("06_build_pipeline.ipynb")
        assert len(nb["cells"]) >= 10

    def test_kernel_is_python(self):
        nb = _load("06_build_pipeline.ipynb")
        lang = nb.get("metadata", {}).get("kernelspec", {}).get("language", "")
        assert lang.lower() in ("python", "python3", "")

    def test_imports_pipeline_builder(self):
        nb = _load("06_build_pipeline.ipynb")
        code = _code_source(nb)
        assert "PipelineBuilder" in code

    def test_imports_pipeline_generators(self):
        nb = _load("06_build_pipeline.ipynb")
        code = _code_source(nb)
        assert "SnakemakeGenerator" in code or "NextflowGenerator" in code or "JupyterGenerator" in code

    def test_covers_snakefile_output(self):
        nb = _load("06_build_pipeline.ipynb")
        md = _markdown_source(nb)
        assert any(kw in md.lower() for kw in ("snakefile", "snakemake"))

    def test_covers_nextflow_output(self):
        nb = _load("06_build_pipeline.ipynb")
        md = _markdown_source(nb)
        assert any(kw in md.lower() for kw in ("nextflow", "nf-core", "params"))

    def test_covers_jupyter_notebook_generation(self):
        nb = _load("06_build_pipeline.ipynb")
        md = _markdown_source(nb)
        assert any(kw in md.lower() for kw in ("jupyter", "notebook", "figure"))

    def test_covers_conda_environment(self):
        nb = _load("06_build_pipeline.ipynb")
        md = _markdown_source(nb)
        assert any(kw in md.lower() for kw in ("conda", "environment"))

    def test_uses_pipeline_backend_enum(self):
        nb = _load("06_build_pipeline.ipynb")
        code = _code_source(nb)
        assert "PipelineBackend" in code

    def test_saves_outputs_to_disk(self):
        nb = _load("06_build_pipeline.ipynb")
        code = _code_source(nb)
        assert "write_text" in code or "write(" in code or "Path(" in code


# ---------------------------------------------------------------------------
# Phase 8: End-to-End Integration
# ---------------------------------------------------------------------------

class TestNotebook07EndToEnd:
    """07_end_to_end.ipynb — Phase 8 crown jewel: full PMID-to-pipeline workflow."""

    def test_loads_without_error(self):
        _load("07_end_to_end.ipynb")

    def test_has_markdown_and_code_cells(self):
        nb = _load("07_end_to_end.ipynb")
        types = _cell_types(nb)
        assert "markdown" in types
        assert "code" in types

    def test_minimum_cell_count(self):
        nb = _load("07_end_to_end.ipynb")
        assert len(nb["cells"]) >= 18

    def test_kernel_is_python(self):
        nb = _load("07_end_to_end.ipynb")
        lang = nb.get("metadata", {}).get("kernelspec", {}).get("language", "")
        assert lang.lower() in ("python", "python3", "")

    def test_imports_all_parsers(self):
        nb = _load("07_end_to_end.ipynb")
        code = _code_source(nb)
        assert "PaperParser" in code
        assert "FigureParser" in code
        assert "MethodsParser" in code

    def test_imports_data_parsers(self):
        nb = _load("07_end_to_end.ipynb")
        code = _code_source(nb)
        assert "GEOParser" in code
        assert "SRAParser" in code

    def test_imports_software_parser(self):
        nb = _load("07_end_to_end.ipynb")
        code = _code_source(nb)
        assert "SoftwareParser" in code

    def test_imports_pipeline_builder(self):
        nb = _load("07_end_to_end.ipynb")
        code = _code_source(nb)
        assert "PipelineBuilder" in code

    def test_covers_all_nine_workflow_stages(self):
        nb = _load("07_end_to_end.ipynb")
        md = _markdown_source(nb).lower()
        stages = [
            "configuration",
            "parse paper",
            "parse figures",
            "parse methods",
            "fetch datasets",
            "parse software",
            "build pipeline",
            "inspect",
            "export",
        ]
        for stage in stages:
            assert stage in md, f"Missing workflow stage in markdown: '{stage}'"

    def test_has_paper_source_configuration(self):
        nb = _load("07_end_to_end.ipynb")
        code = _code_source(nb)
        assert "paper_source" in code

    def test_has_cache_dir_configuration(self):
        nb = _load("07_end_to_end.ipynb")
        code = _code_source(nb)
        assert "cache_dir" in code

    def test_has_backend_configuration(self):
        nb = _load("07_end_to_end.ipynb")
        code = _code_source(nb)
        assert "PipelineBackend" in code

    def test_exports_pipeline_to_disk(self):
        nb = _load("07_end_to_end.ipynb")
        code = _code_source(nb)
        assert "write_text" in code or "write(" in code

    def test_exports_snakefile(self):
        nb = _load("07_end_to_end.ipynb")
        code = _code_source(nb)
        assert "Snakefile" in code

    def test_exports_conda_env(self):
        nb = _load("07_end_to_end.ipynb")
        code = _code_source(nb)
        assert "environment.yml" in code or "conda_env_yaml" in code

    def test_exports_jupyter_notebook(self):
        nb = _load("07_end_to_end.ipynb")
        code = _code_source(nb)
        assert "figure_reproduction.ipynb" in code or "jupyter_content" in code

    def test_has_output_dir(self):
        nb = _load("07_end_to_end.ipynb")
        code = _code_source(nb)
        assert "output_dir" in code

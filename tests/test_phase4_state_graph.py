from __future__ import annotations

import subprocess
import types

from researcher_ai.models.method import AnalysisStep, Assay, AssayGraph, Method, MethodCategory
from researcher_ai.pipeline.builder import PipelineBuilder
from researcher_ai.pipeline.bash_tool import BashTool


def test_engineering_agent_validation_loop_passes(monkeypatch):
    builder = PipelineBuilder(validation_max_rounds=2)

    def _fake_run(cmd, cwd, capture_output, text, timeout, check):
        return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    snakefile, report = builder._validate_and_repair_snakefile(
        snakefile_content="rule all:\n    input:\n        []\n",
        profile={"partition": "hotel", "account": "csd786", "mem": "64G"},
    )

    assert "rule all:" in snakefile
    assert report["passed"] is True
    assert len(report["attempts"]) >= 2
    assert any("--lint" in str(a.get("cmd", "")) for a in report["attempts"])


def test_bash_tool_reports_missing_binary(monkeypatch):
    tool = BashTool(timeout_seconds=5)

    def _missing(*args, **kwargs):  # noqa: ARG001
        raise FileNotFoundError("snakemake")

    monkeypatch.setattr(subprocess, "run", _missing)
    result = tool.run(["snakemake", "--lint"], cwd="/tmp")
    assert result.status == "tool_unavailable"
    assert "not installed" in result.stderr


def test_engineering_agent_validation_skips_when_snakemake_missing(monkeypatch):
    builder = PipelineBuilder(validation_max_rounds=1)

    def _missing(*args, **kwargs):
        raise FileNotFoundError("snakemake")

    monkeypatch.setattr(subprocess, "run", _missing)
    _, report = builder._validate_and_repair_snakefile(
        snakefile_content="rule all:\n    input:\n        []\n",
        profile={"partition": "hotel", "account": "csd786", "mem": "64G"},
    )

    assert report["passed"] is True
    assert report["skipped"] == "snakemake_unavailable"


def test_engineering_agent_llm_repair_falls_back_to_deterministic(monkeypatch):
    builder = PipelineBuilder(validation_max_rounds=1)
    monkeypatch.setattr(
        "researcher_ai.pipeline.builder.extract_structured_data",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("llm unavailable")),
    )
    repaired = builder._repair_snakefile(
        "rule x:\n    output:\n        'x.txt'\n",
        "rule all missing in lint output",
    )
    assert "rule all:" in repaired


def test_pipeline_build_supports_local_profile(monkeypatch):
    builder = PipelineBuilder(validation_max_rounds=1, hpc_profile_name="local")
    monkeypatch.setattr(
        builder,
        "_validate_and_repair_snakefile",
        lambda *, snakefile_content, profile: (snakefile_content, {"passed": True, "attempts": []}),
    )
    assay = Assay(
        name="RNA-seq",
        description="RNA-seq computational analysis",
        data_type="sequencing",
        method_category=MethodCategory.computational,
        steps=[
            AnalysisStep(
                step_number=1,
                description="Align reads",
                input_data="reads.fastq.gz",
                output_data="aligned.bam",
                software="STAR",
                parameters={"threads": "8"},
            )
        ],
    )
    method = Method(assay_graph=AssayGraph(assays=[assay], dependencies=[]))
    pipeline = builder.build(method=method, datasets=[], software=[], figures=[])
    assert pipeline.tscc_slurm_profile is not None
    assert pipeline.tscc_slurm_profile["partition"] == "local"


def test_pipeline_build_injects_tscc_profile(monkeypatch):
    builder = PipelineBuilder(validation_max_rounds=1)
    monkeypatch.setattr(
        builder,
        "_validate_and_repair_snakefile",
        lambda *, snakefile_content, profile: (snakefile_content, {"passed": True, "attempts": []}),
    )

    assay = Assay(
        name="RNA-seq",
        description="RNA-seq computational analysis",
        data_type="sequencing",
        method_category=MethodCategory.computational,
        steps=[
            AnalysisStep(
                step_number=1,
                description="Align reads",
                input_data="reads.fastq.gz",
                output_data="aligned.bam",
                software="STAR",
                parameters={"threads": "8"},
            )
        ],
    )
    method = Method(assay_graph=AssayGraph(assays=[assay], dependencies=[]))
    pipeline = builder.build(method=method, datasets=[], software=[], figures=[])

    assert pipeline.tscc_slurm_profile is not None
    assert pipeline.tscc_slurm_profile["partition"] == "hotel"
    assert pipeline.tscc_slurm_profile["account"] == "csd786"

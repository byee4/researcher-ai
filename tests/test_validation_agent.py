from __future__ import annotations

from researcher_ai.models.method import (
    AnalysisStep,
    Assay,
    AssayGraph,
    EvidenceCategory,
    Method,
    MethodCategory,
)
from researcher_ai.parsers.validation_agent import ValidationAgent


class _FakeStore:
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def query(self, query: str, top_k: int = 1):  # noqa: ARG002
        q = query.lower()
        for key, text in self.mapping.items():
            if key.lower() in q:
                return [{"text": text, "source": "fake"}]
        return []


def _method_with_step(*, software: str = "STAR", version: str | None = None, params=None, description="align"):
    step = AnalysisStep(
        step_number=1,
        description=description,
        input_data="in.fastq.gz",
        output_data="out.bam",
        software=software,
        software_version=version,
        parameters=params or {},
    )
    assay = Assay(
        name="RNA-seq",
        description="RNA-seq processing",
        data_type="sequencing",
        method_category=MethodCategory.computational,
        steps=[step],
    )
    return Method(assay_graph=AssayGraph(assays=[assay], dependencies=[]))


def test_hallucinated_version_flagged_ungrounded():
    agent = ValidationAgent()
    method = _method_with_step(version="5.0.0")
    paper = _FakeStore({"star": "Reads were aligned using STAR."})

    report = agent.validate(method=method, paper_rag=paper, protocol_rag=None)
    version_verdict = next(v for v in report.verdicts if v.field.endswith("software_version"))
    assert version_verdict.evidence_category == EvidenceCategory.ungrounded
    assert version_verdict.action == "flag_ungrounded"


def test_grounded_parameter_is_kept():
    agent = ValidationAgent()
    method = _method_with_step(params={"--sjdbOverhang": "100"})
    paper = _FakeStore({"sjdboverhang": "STAR was run with --sjdbOverhang 100."})

    report = agent.validate(method=method, paper_rag=paper, protocol_rag=None)
    param_verdict = next(v for v in report.verdicts if ".parameters." in v.field)
    assert param_verdict.evidence_category == EvidenceCategory.stated_in_paper
    assert param_verdict.action == "keep"


def test_inferred_default_not_false_positive_ungrounded():
    agent = ValidationAgent()
    method = _method_with_step(params={"--outSAMtype": "BAM SortedByCoordinate"})
    paper = _FakeStore({"outsamtype": "Reads were aligned with STAR using default parameters."})

    report = agent.validate(method=method, paper_rag=paper, protocol_rag=None)
    param_verdict = next(v for v in report.verdicts if ".parameters." in v.field)
    assert param_verdict.evidence_category == EvidenceCategory.inferred_default
    assert param_verdict.action == "keep_as_default"


def test_template_gap_warning_emitted_for_missing_quantification_stage():
    agent = ValidationAgent()
    method = _method_with_step(description="align reads to genome")
    paper = _FakeStore({"rna": "RNA-seq methods"})

    report = agent.validate(method=method, paper_rag=paper, protocol_rag=None)
    assert any("template_missing_stages" in w for w in report.warnings)

#!/usr/bin/env python3
"""Run the researcher-ai parsing workflow for Django integration.

This script is intentionally executed in a subprocess to avoid module-name
collisions between:
- Django project module: researcher_ai (settings/urls)
- Package module: researcher_ai (parsers/models/pipeline)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

from researcher_ai.models.paper import PaperSource
from researcher_ai.models.dataset import Dataset
from researcher_ai.parsers.paper_parser import PaperParser
from researcher_ai.parsers.figure_parser import FigureParser
from researcher_ai.parsers.methods_parser import MethodsParser
from researcher_ai.parsers.software_parser import SoftwareParser
from researcher_ai.parsers.data.geo_parser import GEOParser
from researcher_ai.parsers.data.sra_parser import SRAParser
from researcher_ai.pipeline.builder import PipelineBuilder


_ACC_RE = re.compile(
    r"\b("
    r"GSE\d{4,8}|GSM\d{4,8}|GDS\d{3,7}|GPL\d{3,7}|"
    r"SRP\d{4,9}|SRX\d{4,9}|SRR\d{4,9}|ERP\d{4,9}|ERR\d{4,9}|"
    r"PRJNA\d{4,9}|PRJEB\d{4,9}"
    r")\b",
    re.IGNORECASE,
)


def _emit(progress: int, stage: str) -> None:
    print(f"PROGRESS|{progress}|{stage}", flush=True)


def _collect_accessions(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in _ACC_RE.finditer(text or ""):
        acc = m.group(1).upper()
        if acc not in seen:
            seen.add(acc)
            out.append(acc)
    return out


def _parse_dataset(accession: str) -> Optional[Dataset]:
    if accession.startswith(("GSE", "GSM", "GDS", "GPL")):
        return GEOParser().parse(accession)
    if accession.startswith(("SRP", "SRX", "SRR", "ERP", "ERR", "PRJNA", "PRJEB")):
        return SRAParser().parse(accession)
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="PMID or PDF path")
    parser.add_argument("--source-type", choices=["pmid", "pdf"], required=True)
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    source = args.source
    source_type = PaperSource.PMID if args.source_type == "pmid" else PaperSource.PDF
    output_path = Path(args.output)

    _emit(5, "Initializing workflow")
    paper_parser = PaperParser()
    figure_parser = FigureParser()
    methods_parser = MethodsParser()
    software_parser = SoftwareParser()
    pipeline_builder = PipelineBuilder()

    _emit(15, "Parsing paper")
    paper = paper_parser.parse(source, source_type=source_type)

    _emit(35, "Parsing figures")
    figures = figure_parser.parse_all_figures(paper)

    _emit(55, "Parsing methods")
    method = methods_parser.parse(paper, figures=figures, computational_only=True)

    _emit(70, "Parsing datasets")
    section_text = "\n".join((getattr(sec, "text", "") or "") for sec in (paper.sections or []))
    combined_text = "\n".join(
        [
            paper.raw_text or "",
            section_text,
            method.data_availability or "",
            method.code_availability or "",
            "\n".join(getattr(fig, "caption", "") or "" for fig in figures),
        ]
    )
    accessions = _collect_accessions(combined_text)
    datasets: list[Dataset] = []
    dataset_errors: list[str] = []
    for acc in accessions[:25]:
        try:
            ds = _parse_dataset(acc)
            if ds is not None:
                datasets.append(ds)
        except Exception as exc:  # pragma: no cover - best effort in workflow
            dataset_errors.append(f"{acc}: {type(exc).__name__}: {exc}")

    _emit(80, "Parsing software")
    software = software_parser.parse_from_method(method)

    _emit(92, "Building pipeline")
    pipeline = pipeline_builder.build(method, datasets, software, figures)

    _emit(98, "Serializing output")
    output = {
        "paper": paper.model_dump(mode="json"),
        "figures": [f.model_dump(mode="json") for f in figures],
        "method": method.model_dump(mode="json"),
        "datasets": [d.model_dump(mode="json") for d in datasets],
        "software": [s.model_dump(mode="json") for s in software],
        "pipeline": pipeline.model_dump(mode="json"),
        "dataset_parse_errors": dataset_errors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    _emit(100, "Completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

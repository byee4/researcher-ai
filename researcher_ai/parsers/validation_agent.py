"""Phase 3 validation agent for evidence-grounded method extraction."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import yaml

from researcher_ai.models.method import (
    Assay,
    AssayTemplate,
    EvidenceCategory,
    Method,
    ValidationReport,
    ValidationVerdict,
)

logger = logging.getLogger(__name__)


_DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_DEFAULT_PARAM_ALIASES: dict[str, set[str]] = {
    "star": {"outsamtype", "runcpu", "runthreadn", "twopassmode", "outfiltermultimapnmax"},
    "deseq2": {"fit_type", "beta_prior", "test", "cooks_cutoff"},
    "clippers": {"fdr", "bonferroni"},
}


class ValidationAgent:
    """Validate extracted method fields against paper/protocol evidence."""

    def __init__(self, *, template_dir: Optional[str | Path] = None):
        self.template_dir = Path(template_dir) if template_dir else _DEFAULT_TEMPLATE_DIR
        self.templates = self._load_templates(self.template_dir)

    def validate(
        self,
        *,
        method: Method,
        paper_rag,
        protocol_rag=None,
    ) -> ValidationReport:
        verdicts: list[ValidationVerdict] = []
        warnings: list[str] = []

        for assay in method.assays:
            template = self._match_template(assay.name)
            if template is not None:
                missing_stages = self._missing_required_stages(assay, template)
                if missing_stages:
                    warnings.append(
                        f"template_missing_stages:{assay.name}:{','.join(missing_stages)}"
                    )

            for step in assay.steps:
                if step.software:
                    verdicts.append(
                        self._validate_field(
                            field=f"{assay.name}.step_{step.step_number}.software",
                            claimed_value=step.software,
                            paper_rag=paper_rag,
                            protocol_rag=protocol_rag,
                            query=f"{step.software} {assay.name}",
                        )
                    )
                if step.software_version:
                    verdicts.append(
                        self._validate_field(
                            field=f"{assay.name}.step_{step.step_number}.software_version",
                            claimed_value=step.software_version,
                            paper_rag=paper_rag,
                            protocol_rag=protocol_rag,
                            query=f"{step.software or ''} {step.software_version} {assay.name}",
                        )
                    )
                for k, v in step.parameters.items():
                    verdicts.append(
                        self._validate_parameter(
                            assay=assay,
                            step_number=step.step_number,
                            software=(step.software or "").strip(),
                            key=str(k),
                            value=str(v),
                            paper_rag=paper_rag,
                            protocol_rag=protocol_rag,
                        )
                    )

        ungrounded_count = sum(1 for v in verdicts if v.evidence_category == EvidenceCategory.ungrounded)
        inferred_default_count = sum(
            1 for v in verdicts if v.evidence_category == EvidenceCategory.inferred_default
        )

        return ValidationReport(
            verdicts=verdicts,
            ungrounded_count=ungrounded_count,
            inferred_default_count=inferred_default_count,
            total_fields_checked=len(verdicts),
            warnings=warnings,
        )

    def _validate_parameter(
        self,
        *,
        assay: Assay,
        step_number: int,
        software: str,
        key: str,
        value: str,
        paper_rag,
        protocol_rag,
    ) -> ValidationVerdict:
        field = f"{assay.name}.step_{step_number}.parameters.{key}"
        query = f"{software} {key} {value} {assay.name}".strip()

        paper_hit = _best_hit_text(paper_rag, query)
        if _contains_claim(paper_hit, value) or _contains_claim(paper_hit, key):
            return ValidationVerdict(
                field=field,
                claimed_value=value,
                evidence_category=EvidenceCategory.stated_in_paper,
                evidence_source="paper",
                action="keep",
                rationale="Parameter evidence found directly in paper chunks.",
            )

        if self._is_default_parameter(software, key) and _mentions_default_usage(paper_hit):
            return ValidationVerdict(
                field=field,
                claimed_value=value,
                evidence_category=EvidenceCategory.inferred_default,
                evidence_source="paper",
                action="keep_as_default",
                rationale="Paper indicates default parameters for this software.",
            )

        protocol_hit = _best_hit_text(protocol_rag, query)
        if _contains_claim(protocol_hit, value) or _contains_claim(protocol_hit, key):
            return ValidationVerdict(
                field=field,
                claimed_value=value,
                evidence_category=EvidenceCategory.inferred_from_protocol,
                evidence_source="protocol",
                action="keep_as_default",
                rationale="Parameter grounded in protocol docs, not explicit paper text.",
            )

        return ValidationVerdict(
            field=field,
            claimed_value=value,
            evidence_category=EvidenceCategory.ungrounded,
            evidence_source=None,
            action="flag_ungrounded",
            rationale="No supporting evidence found in paper or protocol context.",
        )

    def _validate_field(
        self,
        *,
        field: str,
        claimed_value: str,
        paper_rag,
        protocol_rag,
        query: str,
    ) -> ValidationVerdict:
        paper_hit = _best_hit_text(paper_rag, query)
        if _contains_claim(paper_hit, claimed_value):
            return ValidationVerdict(
                field=field,
                claimed_value=claimed_value,
                evidence_category=EvidenceCategory.stated_in_paper,
                evidence_source="paper",
                action="keep",
                rationale="Field value appears in paper evidence chunks.",
            )

        protocol_hit = _best_hit_text(protocol_rag, query)
        if _contains_claim(protocol_hit, claimed_value):
            return ValidationVerdict(
                field=field,
                claimed_value=claimed_value,
                evidence_category=EvidenceCategory.inferred_from_protocol,
                evidence_source="protocol",
                action="keep_as_default",
                rationale="Field appears in protocol context only.",
            )

        return ValidationVerdict(
            field=field,
            claimed_value=claimed_value,
            evidence_category=EvidenceCategory.ungrounded,
            evidence_source=None,
            action="flag_ungrounded",
            rationale="No evidence found for claimed value.",
        )

    def _is_default_parameter(self, software: str, key: str) -> bool:
        sw = re.sub(r"\s+", "", software.lower())
        k = re.sub(r"[^a-z0-9]", "", key.lower())
        for prefix, params in _DEFAULT_PARAM_ALIASES.items():
            if sw.startswith(prefix) and k in params:
                return True
        return False

    def _missing_required_stages(self, assay: Assay, template: AssayTemplate) -> list[str]:
        step_text = " ".join(s.description.lower() for s in assay.steps)
        missing = [stage for stage in template.required_stages if stage.lower() not in step_text]
        return missing

    def _match_template(self, assay_name: str) -> Optional[AssayTemplate]:
        text = (assay_name or "").lower()
        if "rna" in text:
            return self.templates.get("rnaseq")
        if "chip" in text:
            return self.templates.get("chipseq")
        if "variant" in text or "wgs" in text:
            return self.templates.get("variant_calling")
        return None

    def _load_templates(self, template_dir: Path) -> dict[str, AssayTemplate]:
        out: dict[str, AssayTemplate] = {}
        if not template_dir.exists():
            return out
        for path in sorted(template_dir.glob("*_template.yaml")):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                template = AssayTemplate.model_validate(data)
                out[template.assay_type] = template
            except Exception as exc:
                logger.warning("Failed to load assay template %s: %s", path, exc)
        return out


def _best_hit_text(store, query: str) -> str:
    if store is None:
        return ""
    try:
        hits = store.query(query, top_k=1)
    except Exception:
        return ""
    if not hits:
        return ""
    top = hits[0] or {}
    return str(top.get("text", "") or "")


def _contains_claim(text: str, claim: str) -> bool:
    if not text or not claim:
        return False
    return claim.lower() in text.lower()


def _mentions_default_usage(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return "default" in lowered or "standard parameters" in lowered

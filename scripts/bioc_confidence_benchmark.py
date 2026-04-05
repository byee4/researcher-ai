#!/usr/bin/env python3
"""Generate BioC confidence benchmark artifacts from baseline/enhanced runs."""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _flatten_subfigures(figures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fig in figures:
        figure_id = str(fig.get("figure_id") or "")
        for sf in fig.get("subfigures", []) or []:
            label = str(sf.get("label") or "")
            rows.append(
                {
                    "subfigure_id": f"{figure_id}:{label}",
                    "figure_id": figure_id,
                    "label": label,
                    "composite_confidence": float(sf.get("composite_confidence") or 0.0),
                    "classification_confidence": float(sf.get("classification_confidence") or 0.0),
                    "bioc_contradiction": bool(sf.get("bioc_contradiction", False)),
                }
            )
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comp = [r["composite_confidence"] for r in rows]
    return {
        "subfigure_count": len(rows),
        "mean_composite_confidence": round(mean(comp), 3) if comp else 0.0,
        "median_composite_confidence": round(median(comp), 3) if comp else 0.0,
        "high_confidence_count": sum(1 for r in rows if r["classification_confidence"] >= 0.75),
        "ambiguous_count": sum(1 for r in rows if 40.0 <= r["composite_confidence"] <= 74.0),
        "contradiction_count": sum(1 for r in rows if r["bioc_contradiction"]),
    }


def _evaluate_gates(baseline: dict[str, Any], enhanced: dict[str, Any]) -> dict[str, Any]:
    mean_delta = enhanced["mean_composite_confidence"] - baseline["mean_composite_confidence"]
    amb_base = baseline["ambiguous_count"]
    amb_enh = enhanced["ambiguous_count"]
    amb_drop_abs = amb_base - amb_enh
    amb_drop_pct = (amb_drop_abs / amb_base) if amb_base else 0.0

    gates = {
        "mean_delta_ge_3": mean_delta >= 3.0,
        "ambiguous_drop_ge_10pct_or_abs2": (amb_drop_pct >= 0.10) or (amb_drop_abs >= 2),
        "contradictions_non_increasing": enhanced["contradiction_count"] <= baseline["contradiction_count"],
    }
    gates["all_pass"] = all(gates.values())
    gates["mean_delta"] = round(mean_delta, 3)
    gates["ambiguous_drop_abs"] = amb_drop_abs
    gates["ambiguous_drop_pct"] = round(amb_drop_pct, 4)
    return gates


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="BioC confidence benchmark report")
    parser.add_argument("--pmid", default="39303722")
    parser.add_argument("--baseline", required=True, help="Baseline workflow output JSON path.")
    parser.add_argument("--enhanced", required=True, help="Enhanced workflow output JSON path.")
    parser.add_argument("--outdir", default="parse_results/bioc")
    args = parser.parse_args()

    baseline_path = Path(args.baseline).resolve()
    enhanced_path = Path(args.enhanced).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    baseline_run = _load_json(baseline_path)
    enhanced_run = _load_json(enhanced_path)
    baseline_figures = baseline_run.get("figures", []) if isinstance(baseline_run.get("figures"), list) else []
    enhanced_figures = enhanced_run.get("figures", []) if isinstance(enhanced_run.get("figures"), list) else []

    baseline_rows = _flatten_subfigures(baseline_figures)
    enhanced_rows = _flatten_subfigures(enhanced_figures)
    baseline_summary = _summary(baseline_rows)
    enhanced_summary = _summary(enhanced_rows)
    gates = _evaluate_gates(baseline_summary, enhanced_summary)

    baseline_map = {r["subfigure_id"]: r for r in baseline_rows}
    enhanced_map = {r["subfigure_id"]: r for r in enhanced_rows}
    all_ids = sorted(set(baseline_map) | set(enhanced_map))
    per_subfigure: list[dict[str, Any]] = []
    for sid in all_ids:
        b = baseline_map.get(sid, {})
        e = enhanced_map.get(sid, {})
        per_subfigure.append(
            {
                "subfigure_id": sid,
                "composite_baseline": b.get("composite_confidence"),
                "composite_enhanced": e.get("composite_confidence"),
                "composite_delta": (
                    round(float(e.get("composite_confidence", 0.0)) - float(b.get("composite_confidence", 0.0)), 3)
                    if b or e
                    else None
                ),
                "classification_baseline": b.get("classification_confidence"),
                "classification_enhanced": e.get("classification_confidence"),
                "contradiction_baseline": bool(b.get("bioc_contradiction", False)),
                "contradiction_enhanced": bool(e.get("bioc_contradiction", False)),
            }
        )

    report = {
        "pmid": str(args.pmid),
        "baseline_path": str(baseline_path),
        "enhanced_path": str(enhanced_path),
        "baseline": baseline_summary,
        "enhanced": enhanced_summary,
        "mean_composite_delta": gates["mean_delta"],
        "contradiction_count_baseline": baseline_summary["contradiction_count"],
        "contradiction_count_enhanced": enhanced_summary["contradiction_count"],
        "gates": gates,
        "per_subfigure": per_subfigure,
    }

    baseline_figures_path = outdir / "pmid_39303722_figures_baseline.json"
    enhanced_figures_path = outdir / "pmid_39303722_figures_enhanced.json"
    _write_json(baseline_figures_path, {"figures": baseline_figures})
    _write_json(enhanced_figures_path, {"figures": enhanced_figures})

    baseline_text = json.dumps({"figures": baseline_figures}, indent=2, sort_keys=True)
    enhanced_text = json.dumps({"figures": enhanced_figures}, indent=2, sort_keys=True)
    diff = "\n".join(
        difflib.unified_diff(
            baseline_text.splitlines(),
            enhanced_text.splitlines(),
            fromfile=str(baseline_figures_path.name),
            tofile=str(enhanced_figures_path.name),
            lineterm="",
        )
    )
    (outdir / "pmid_39303722_figures_diff.patch").write_text(diff, encoding="utf-8")

    report_json = outdir / "bioc_confidence_39303722_report.json"
    _write_json(report_json, report)

    md_lines = [
        f"# BioC Confidence Benchmark ({args.pmid})",
        "",
        "## Summary",
        f"- Mean composite confidence: baseline `{baseline_summary['mean_composite_confidence']}` -> enhanced `{enhanced_summary['mean_composite_confidence']}` (delta `{gates['mean_delta']}`)",
        f"- Ambiguous panels (40-74): baseline `{baseline_summary['ambiguous_count']}` -> enhanced `{enhanced_summary['ambiguous_count']}`",
        f"- High-confidence panels (classification >= 0.75): baseline `{baseline_summary['high_confidence_count']}` -> enhanced `{enhanced_summary['high_confidence_count']}`",
        f"- Contradiction count: baseline `{baseline_summary['contradiction_count']}` -> enhanced `{enhanced_summary['contradiction_count']}`",
        "",
        "## Gates",
        f"- Mean delta >= +3.0: `{gates['mean_delta_ge_3']}`",
        f"- Ambiguous drop >=10% OR >=2 absolute: `{gates['ambiguous_drop_ge_10pct_or_abs2']}`",
        f"- Contradictions non-increasing: `{gates['contradictions_non_increasing']}`",
        f"- Overall pass: `{gates['all_pass']}`",
        "",
        "## Artifacts",
        "- `bioc_confidence_39303722_report.json`",
        "- `pmid_39303722_figures_baseline.json`",
        "- `pmid_39303722_figures_enhanced.json`",
        "- `pmid_39303722_figures_diff.patch`",
    ]
    (outdir / "bioc_confidence_39303722_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


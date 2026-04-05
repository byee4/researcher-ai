#!/usr/bin/env python3
"""Benchmark report for figure calibration (baseline vs calibrated)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from researcher_ai.models.figure import Figure
from researcher_ai.models.paper import Paper, PaperSource, PaperType
from researcher_ai.parsers.figure_calibration import FigureCalibrationEngine


def _load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import]

        data = yaml.safe_load(text) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}


def _parse_source(value: str) -> PaperSource:
    v = (value or "pmcid").strip().lower()
    for src in PaperSource:
        if src.value == v:
            return src
    return PaperSource.PMCID


def _parse_paper_type(value: str) -> PaperType:
    v = (value or "experimental").strip().lower()
    for pt in PaperType:
        if pt.value == v:
            return pt
    return PaperType.EXPERIMENTAL


def _norm_panel(label: str) -> str:
    raw = (label or "").strip().strip("()").lower()
    return raw[:1] if raw else ""


def _evaluate(expected: dict[str, Any], figures: list[Figure]) -> tuple[int, int]:
    total = 0
    correct = 0
    by_figure = {f.figure_id.strip().lower(): f for f in figures}

    for exp in expected.get("figures", []):
        figure_id = str(exp.get("figure_id") or "").strip().lower()
        if not figure_id:
            continue
        fig = by_figure.get(figure_id)
        if fig is None:
            continue
        sf_by = {_norm_panel(sf.label): sf for sf in fig.subfigures}
        panels = exp.get("panels") or {}
        for panel_label, fields in panels.items():
            if not isinstance(fields, dict):
                continue
            sf = sf_by.get(_norm_panel(str(panel_label)))
            if sf is None:
                for _ in fields.keys():
                    total += 1
                continue
            for key, value in fields.items():
                total += 1
                if key == "plot_type":
                    if sf.plot_type.value == str(value):
                        correct += 1
                elif key == "x_axis_label":
                    if sf.x_axis and (sf.x_axis.label or "") == str(value):
                        correct += 1
                elif key == "y_axis_label":
                    if sf.y_axis and (sf.y_axis.label or "") == str(value):
                        correct += 1
                elif key == "x_axis_scale":
                    if sf.x_axis and sf.x_axis.scale.value == str(value):
                        correct += 1
                elif key == "y_axis_scale":
                    if sf.y_axis and sf.y_axis.scale.value == str(value):
                        correct += 1
    return correct, total


def _collect_plot_type_pairs(expected: dict[str, Any], figures: list[Figure]) -> list[tuple[str, str]]:
    """Collect (expected_plot_type, predicted_plot_type) pairs for panel-level evaluation."""
    pairs: list[tuple[str, str]] = []
    by_figure = {f.figure_id.strip().lower(): f for f in figures}
    for exp in expected.get("figures", []):
        figure_id = str(exp.get("figure_id") or "").strip().lower()
        if not figure_id:
            continue
        fig = by_figure.get(figure_id)
        if fig is None:
            continue
        sf_by = {_norm_panel(sf.label): sf for sf in fig.subfigures}
        panels = exp.get("panels") or {}
        for panel_label, fields in panels.items():
            if not isinstance(fields, dict):
                continue
            expected_pt = str(fields.get("plot_type") or "").strip()
            if not expected_pt:
                continue
            sf = sf_by.get(_norm_panel(str(panel_label)))
            predicted_pt = sf.plot_type.value if sf is not None else "missing"
            pairs.append((expected_pt, predicted_pt))
    return pairs


def _per_plot_type_metrics(pairs: list[tuple[str, str]]) -> dict[str, dict[str, float | int]]:
    """Compute per-plot-type precision/recall from expected/predicted pairs."""
    expected_counts: dict[str, int] = {}
    tp: dict[str, int] = {}
    fp: dict[str, int] = {}
    fn: dict[str, int] = {}

    for expected_pt, predicted_pt in pairs:
        expected_counts[expected_pt] = expected_counts.get(expected_pt, 0) + 1
        if expected_pt == predicted_pt:
            tp[expected_pt] = tp.get(expected_pt, 0) + 1
        else:
            fn[expected_pt] = fn.get(expected_pt, 0) + 1
            fp[predicted_pt] = fp.get(predicted_pt, 0) + 1

    plot_types = sorted(set(expected_counts) | set(tp) | set(fp) | set(fn))
    metrics: dict[str, dict[str, float | int]] = {}
    for pt in plot_types:
        t = tp.get(pt, 0)
        p = fp.get(pt, 0)
        n = fn.get(pt, 0)
        precision = (t / (t + p)) if (t + p) else 0.0
        recall = (t / (t + n)) if (t + n) else 0.0
        metrics[pt] = {
            "tp": t,
            "fp": p,
            "fn": n,
            "support": expected_counts.get(pt, 0),
            "precision": precision,
            "recall": recall,
        }
    return metrics


def _build_expected(case: dict[str, Any]) -> dict[str, Any]:
    figures = []
    for item in case.get("expected", []) or []:
        if not isinstance(item, dict):
            continue
        figure_id = str(item.get("figure_id") or "").strip()
        panels = item.get("panels") or {}
        if figure_id and isinstance(panels, dict):
            figures.append({"figure_id": figure_id, "panels": panels})
    return {"figures": figures}


def run_report(fixtures_dir: Path, registry_path: Path | None = None) -> dict[str, Any]:
    engine = FigureCalibrationEngine(registry_path=registry_path)
    fixture_files = sorted([p for p in fixtures_dir.glob("*.yaml") if p.is_file()])

    baseline_correct = baseline_total = 0
    calibrated_correct = calibrated_total = 0
    baseline_plot_pairs: list[tuple[str, str]] = []
    calibrated_plot_pairs: list[tuple[str, str]] = []
    case_summaries: list[dict[str, Any]] = []

    for fixture in fixture_files:
        data = _load_yaml(fixture)
        for case in data.get("cases", []) or []:
            if not isinstance(case, dict):
                continue
            paper_raw = case.get("paper", {}) if isinstance(case.get("paper"), dict) else {}
            baseline_raw = case.get("baseline", {}) if isinstance(case.get("baseline"), dict) else {}
            baseline_figures_raw = baseline_raw.get("figures", []) if isinstance(baseline_raw.get("figures"), list) else []

            paper = Paper(
                title=str(paper_raw.get("title") or "Untitled"),
                source=_parse_source(str(paper_raw.get("source") or "pmcid")),
                source_path=str(paper_raw.get("source_path") or ""),
                pmcid=paper_raw.get("pmcid"),
                pmid=paper_raw.get("pmid"),
                paper_type=_parse_paper_type(str(paper_raw.get("paper_type") or "experimental")),
                figure_ids=[str(f.get("figure_id")) for f in baseline_figures_raw if isinstance(f, dict) and f.get("figure_id")],
            )
            baseline_figures = [Figure.model_validate(f) for f in baseline_figures_raw if isinstance(f, dict)]
            calibrated_figures = [engine.apply(paper, f) for f in baseline_figures]

            expected = _build_expected(case)
            b_corr, b_tot = _evaluate(expected, baseline_figures)
            c_corr, c_tot = _evaluate(expected, calibrated_figures)
            baseline_plot_pairs.extend(_collect_plot_type_pairs(expected, baseline_figures))
            calibrated_plot_pairs.extend(_collect_plot_type_pairs(expected, calibrated_figures))
            baseline_correct += b_corr
            baseline_total += b_tot
            calibrated_correct += c_corr
            calibrated_total += c_tot

            case_summaries.append(
                {
                    "fixture": fixture.name,
                    "case_id": str(case.get("case_id") or ""),
                    "baseline": {"correct": b_corr, "total": b_tot},
                    "calibrated": {"correct": c_corr, "total": c_tot},
                }
            )

    baseline_acc = (baseline_correct / baseline_total) if baseline_total else 0.0
    calibrated_acc = (calibrated_correct / calibrated_total) if calibrated_total else 0.0
    delta = calibrated_acc - baseline_acc
    rel_improvement = (delta / baseline_acc) if baseline_acc else 0.0
    baseline_plot_metrics = _per_plot_type_metrics(baseline_plot_pairs)
    calibrated_plot_metrics = _per_plot_type_metrics(calibrated_plot_pairs)

    return {
        "fixtures_dir": str(fixtures_dir),
        "cases": case_summaries,
        "baseline": {
            "correct": baseline_correct,
            "total": baseline_total,
            "accuracy": baseline_acc,
        },
        "calibrated": {
            "correct": calibrated_correct,
            "total": calibrated_total,
            "accuracy": calibrated_acc,
        },
        "baseline_per_plot_type": baseline_plot_metrics,
        "calibrated_per_plot_type": calibrated_plot_metrics,
        "delta_accuracy": delta,
        "relative_improvement": rel_improvement,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Figure calibration benchmark report")
    parser.add_argument(
        "--fixtures-dir",
        default="tests/fixtures/figure_calibration",
        help="Directory with calibration fixture YAML files.",
    )
    parser.add_argument(
        "--registry",
        default="",
        help="Optional path to calibration registry YAML.",
    )
    args = parser.parse_args()

    fixtures_dir = Path(args.fixtures_dir).resolve()
    registry_path = Path(args.registry).resolve() if args.registry else None
    report = run_report(fixtures_dir=fixtures_dir, registry_path=registry_path)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

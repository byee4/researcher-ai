"""Run-artifact reporting helpers for workflow outputs."""

from __future__ import annotations

from collections import Counter
from typing import Any


def summarize_figure_parsing(figures: list[Any]) -> dict[str, Any]:
    """Summarize figure parse warnings and decomposition modes."""
    warning_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    per_figure: list[dict[str, Any]] = []

    for fig in figures or []:
        if hasattr(fig, "model_dump"):
            payload = fig.model_dump(mode="json")
        elif isinstance(fig, dict):
            payload = fig
        else:
            payload = {}
        figure_id = str(payload.get("figure_id", ""))
        warnings = [str(w) for w in (payload.get("parse_warnings") or [])]
        for warning in warnings:
            warning_counts[warning] += 1
        mode = _classify_figure_decomposition_mode(warnings)
        mode_counts[mode] += 1
        per_figure.append(
            {
                "figure_id": figure_id,
                "decomposition_mode": mode,
                "parse_warnings": warnings,
            }
        )

    return {
        "figure_count": len(figures or []),
        "figures_with_any_parse_warnings": sum(1 for item in per_figure if item["parse_warnings"]),
        "decomposition_mode_counts": dict(mode_counts),
        "warning_counts": dict(warning_counts),
        "per_figure": per_figure,
    }


def summarize_method_parsing(method: Any) -> dict[str, Any]:
    """Summarize method parse warnings and explicitly list excluded assays."""
    if hasattr(method, "model_dump"):
        payload = method.model_dump(mode="json")
    elif isinstance(method, dict):
        payload = method
    else:
        payload = {}

    warnings = [str(w) for w in (payload.get("parse_warnings") or [])]
    warning_counts: Counter[str] = Counter()
    excluded_assays: list[dict[str, str]] = []
    for warning in warnings:
        key = warning.split(":", 1)[0]
        warning_counts[key] += 1
        parsed = _parse_filtered_assay_warning(warning)
        if parsed is not None:
            excluded_assays.append(parsed)

    return {
        "warning_counts": dict(warning_counts),
        "excluded_assays": excluded_assays,
        "excluded_assay_count": len(excluded_assays),
    }


def _classify_figure_decomposition_mode(warnings: list[str]) -> str:
    warning_set = {str(w) for w in warnings}
    if "subfigure_decomposition_caption_split_fallback" in warning_set:
        return "caption_split_fallback"
    if "subfigure_decomposition_timeout" in warning_set:
        return "timeout_fallback"
    if "subfigure_decomposition_empty_response" in warning_set:
        return "empty_response_no_split"
    if warning_set:
        return "llm_with_warnings"
    return "llm"


def _parse_filtered_assay_warning(warning: str) -> dict[str, str] | None:
    prefix = "assay_filtered_non_computational:"
    if not warning.startswith(prefix):
        return None
    raw = warning[len(prefix) :].strip()
    if "excluded (category=" not in raw:
        return {"name": raw.strip("' "), "category": "unknown"}
    name_part, rest = raw.split("excluded (category=", 1)
    name = name_part.strip().strip("'").strip()
    category = rest.split(",", 1)[0].strip().strip(")")
    return {"name": name, "category": category}

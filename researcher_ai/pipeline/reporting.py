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

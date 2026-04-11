#!/usr/bin/env python3
"""Investigate empty structured responses for Figure 2 subfigure decomposition.

This script runs repeated figure parsing experiments for PMID 39303722 (Figure 2)
using the production parser path and emits artifacts + a summary report.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from researcher_ai.parsers.figure_parser import FigureParser
from researcher_ai.parsers.paper_parser import PaperParser


@dataclass
class Variant:
    name: str
    runs: int
    response_mode: str
    max_tokens: int
    trim_in_text_to: int | None
    cache_enabled: bool


def _truthy(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@contextmanager
def _temporary_env(overrides: dict[str, str | None]) -> Iterator[None]:
    previous: dict[str, str | None] = {}
    for key, value in overrides.items():
        previous[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                events.append(payload)
        except json.JSONDecodeError:
            continue
    return events


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return float(statistics.quantiles(values, n=100, method="inclusive")[94])


def _run_once(
    *,
    paper,
    figure_id: str,
    variant: Variant,
    run_idx: int,
    runs_dir: Path,
    disable_fallbacks: bool,
) -> dict[str, Any]:
    run_dir = runs_dir / f"{variant.name}_run_{run_idx:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    llm_debug_path = run_dir / "llm_debug.jsonl"
    trace_path = run_dir / "figure_trace.json"

    env_overrides = {
        "RESEARCHER_AI_MODEL": "gpt-5.4",
        "RESEARCHER_AI_SUBFIGURE_DECOMPOSE_MAX_TOKENS": str(variant.max_tokens),
        "RESEARCHER_AI_SUBFIGURE_TIMEOUT_SECONDS": "0",
        "RESEARCHER_AI_FIGURE_TRACE_PATH": str(trace_path),
        "RESEARCHER_AI_LLM_DEBUG_EMPTY_RESPONSES": "1",
        "RESEARCHER_AI_LLM_DEBUG_EMPTY_RESPONSES_PATH": str(llm_debug_path),
        "RESEARCHER_AI_STRUCTURED_RESPONSE_FORMAT_MODE": variant.response_mode,
        "RESEARCHER_AI_DISABLE_MODEL_FALLBACKS": "1" if disable_fallbacks else "0",
    }

    paper_one = paper.model_copy(update={"figure_ids": [figure_id]})

    with _temporary_env(env_overrides):
        parser = FigureParser(cache_dir=str(run_dir / "cache") if variant.cache_enabled else None)
        if variant.trim_in_text_to is not None:
            original = parser._find_in_text_references

            def _trimmed_refs(paper_obj, fig_id):
                refs = original(paper_obj, fig_id)
                return refs[: variant.trim_in_text_to]

            parser._find_in_text_references = _trimmed_refs  # type: ignore[assignment]

        t0 = time.perf_counter()
        figures = parser.parse_all_figures(paper_one)
        total_duration = time.perf_counter() - t0

    figure = next((f for f in figures if f.figure_id == figure_id), None)
    trace_events = _read_json(trace_path) or []
    llm_events = _read_jsonl(llm_debug_path)

    decompose_event = None
    for event in trace_events:
        if event.get("figure_id") == figure_id and event.get("step") == "decompose_subfigures":
            decompose_event = event
            break

    parse_warnings = list(getattr(figure, "parse_warnings", [])) if figure is not None else []
    subfig_count = len(getattr(figure, "subfigures", [])) if figure is not None else 0

    empty_by_warning = "subfigure_decomposition_empty_response" in parse_warnings
    empty_by_trace = bool(decompose_event and decompose_event.get("status") == "empty_response")
    fallback_used = any(e.get("event") == "failover_structured" for e in llm_events)

    run_result = {
        "variant": variant.name,
        "run_index": run_idx,
        "figure_id": figure_id,
        "total_duration_s": round(float(total_duration), 4),
        "subfigure_count": subfig_count,
        "success_exact_8": subfig_count == 8,
        "parse_warnings": parse_warnings,
        "empty_response": bool(empty_by_warning or empty_by_trace),
        "fallback_used": fallback_used,
        "trace_decompose_status": (decompose_event or {}).get("status", "missing"),
        "trace_decompose_duration_s": float((decompose_event or {}).get("duration_s", 0.0) or 0.0),
        "llm_prompt_hash": next(
            (
                str(e.get("prompt_hash"))
                for e in llm_events
                if e.get("event") == "structured_start"
            ),
            "",
        ),
        "llm_prompt_len": next(
            (
                int(e.get("prompt_len", 0))
                for e in llm_events
                if e.get("event") == "structured_start"
            ),
            0,
        ),
        "llm_modes_seen": [
            str(e.get("mode"))
            for e in llm_events
            if e.get("event") == "api_call" and e.get("mode")
        ],
        "llm_content_lengths": [
            int(e.get("content_len", 0))
            for e in llm_events
            if e.get("event") == "content_observed"
        ],
        "llm_errors": [
            {
                "mode": e.get("mode"),
                "error_class": e.get("error_class"),
                "error_message": e.get("error_message"),
            }
            for e in llm_events
            if e.get("event") == "api_call_error"
        ],
        "run_dir": str(run_dir),
    }

    (run_dir / "run_result.json").write_text(json.dumps(run_result, indent=2), encoding="utf-8")
    return run_result


def _summarize_variant(name: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [float(r.get("trace_decompose_duration_s", 0.0)) for r in runs if r.get("trace_decompose_duration_s", 0.0) > 0]
    return {
        "variant": name,
        "runs": len(runs),
        "empty_response_rate": round(sum(1 for r in runs if r.get("empty_response")) / max(len(runs), 1), 4),
        "fallback_rate": round(sum(1 for r in runs if r.get("fallback_used")) / max(len(runs), 1), 4),
        "success_exact_8_rate": round(sum(1 for r in runs if r.get("success_exact_8")) / max(len(runs), 1), 4),
        "decompose_latency_p50_s": round(float(statistics.median(durations)) if durations else 0.0, 4),
        "decompose_latency_p95_s": round(_p95(durations), 4),
        "common_decompose_statuses": {
            status: sum(1 for r in runs if r.get("trace_decompose_status") == status)
            for status in sorted({str(r.get("trace_decompose_status", "missing")) for r in runs})
        },
    }


def _rank_hypotheses(summary_by_variant: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    hypotheses: list[dict[str, Any]] = []
    baseline = summary_by_variant.get("baseline", {})
    baseline_empty = float(baseline.get("empty_response_rate", 0.0) or 0.0)

    json_schema_only = summary_by_variant.get("json_schema_only", {})
    json_object_first = summary_by_variant.get("json_object_first", {})
    trimmed_context = summary_by_variant.get("trimmed_context", {})

    schema_empty = float(json_schema_only.get("empty_response_rate", 0.0) or 0.0)
    obj_first_empty = float(json_object_first.get("empty_response_rate", 0.0) or 0.0)
    trimmed_empty = float(trimmed_context.get("empty_response_rate", 0.0) or 0.0)

    if schema_empty > baseline_empty and obj_first_empty < baseline_empty:
        hypotheses.append(
            {
                "hypothesis": "Response-format sensitivity (json_schema path is less robust than json_object-first).",
                "confidence": "high",
                "evidence": {
                    "baseline_empty_rate": baseline_empty,
                    "json_schema_only_empty_rate": schema_empty,
                    "json_object_first_empty_rate": obj_first_empty,
                },
            }
        )

    if trimmed_empty + 0.05 < baseline_empty:
        hypotheses.append(
            {
                "hypothesis": "Prompt/context size contributes to empty responses.",
                "confidence": "medium",
                "evidence": {
                    "baseline_empty_rate": baseline_empty,
                    "trimmed_context_empty_rate": trimmed_empty,
                },
            }
        )

    if not hypotheses:
        hypotheses.append(
            {
                "hypothesis": "Provider/runtime intermittency is the dominant source of empty responses.",
                "confidence": "medium",
                "evidence": {
                    "baseline_empty_rate": baseline_empty,
                    "json_schema_only_empty_rate": schema_empty,
                    "json_object_first_empty_rate": obj_first_empty,
                    "trimmed_context_empty_rate": trimmed_empty,
                },
            }
        )

    return hypotheses


def _recommendation(summary_by_variant: dict[str, dict[str, Any]]) -> dict[str, str]:
    baseline = summary_by_variant.get("baseline", {})
    baseline_empty = float(baseline.get("empty_response_rate", 0.0) or 0.0)
    json_obj = summary_by_variant.get("json_object_first", {})
    obj_empty = float(json_obj.get("empty_response_rate", 0.0) or 0.0)

    if obj_empty + 0.05 < baseline_empty:
        return {
            "default": "go",
            "recommendation": "Adopt json_object-first for subfigure decomposition or make it adaptive after first empty content response.",
        }
    if baseline_empty > 0.15:
        return {
            "default": "go",
            "recommendation": "Keep current strategy but add stronger retry/fallback policy and context trimming guard for Figure 2-style long captions.",
        }
    return {
        "default": "no-go",
        "recommendation": "Keep current behavior; empty-response incidence is low under observed conditions.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pmid", default="39303722")
    parser.add_argument("--figure-id", default="Figure 2")
    parser.add_argument("--baseline-runs", type=int, default=20)
    parser.add_argument("--variant-runs", type=int, default=10)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--include-cache-variant", default="1", help="1/0")
    parser.add_argument("--disable-fallbacks", default="1", help="1/0")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path("parse_results") / f"figure2_empty_response_investigation_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[investigation] output_dir={out_dir}")
    print(f"[investigation] loading paper pmid={args.pmid}")
    paper = PaperParser().parse(args.pmid)

    variants: list[Variant] = [
        Variant("baseline", max(1, args.baseline_runs), "auto", 1200, None, False),
        Variant("json_schema_only", max(1, args.variant_runs), "json_schema_only", 1200, None, False),
        Variant("json_object_first", max(1, args.variant_runs), "json_object_first", 1200, None, False),
        Variant("max_tokens_1600", max(1, args.variant_runs), "auto", 1600, None, False),
        Variant("trimmed_context", max(1, args.variant_runs), "auto", 1200, 1, False),
    ]
    if _truthy(str(args.include_cache_variant)):
        variants.append(Variant("cache_on", max(1, args.variant_runs), "auto", 1200, None, True))

    all_runs: list[dict[str, Any]] = []
    for variant in variants:
        print(f"[investigation] variant={variant.name} runs={variant.runs}")
        for i in range(1, variant.runs + 1):
            print(f"  - run {i}/{variant.runs}")
            try:
                run_result = _run_once(
                    paper=paper,
                    figure_id=args.figure_id,
                    variant=variant,
                    run_idx=i,
                    runs_dir=runs_dir,
                    disable_fallbacks=_truthy(str(args.disable_fallbacks)),
                )
            except Exception as exc:
                run_result = {
                    "variant": variant.name,
                    "run_index": i,
                    "figure_id": args.figure_id,
                    "error_class": exc.__class__.__name__,
                    "error_message": str(exc),
                    "empty_response": False,
                    "fallback_used": False,
                    "success_exact_8": False,
                    "trace_decompose_status": "run_error",
                    "trace_decompose_duration_s": 0.0,
                }
            all_runs.append(run_result)

    summary_by_variant: dict[str, dict[str, Any]] = {}
    for variant in variants:
        rows = [r for r in all_runs if r.get("variant") == variant.name]
        summary_by_variant[variant.name] = _summarize_variant(variant.name, rows)

    hypotheses = _rank_hypotheses(summary_by_variant)
    recommendation = _recommendation(summary_by_variant)

    report = {
        "pmid": args.pmid,
        "figure_id": args.figure_id,
        "disable_fallbacks": _truthy(str(args.disable_fallbacks)),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary_by_variant": summary_by_variant,
        "hypotheses_ranked": hypotheses,
        "recommendation": recommendation,
        "run_count": len(all_runs),
        "runs": all_runs,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Figure 2 Empty Structured Response Investigation",
        "",
        f"- PMID: `{args.pmid}`",
        f"- Figure: `{args.figure_id}`",
        f"- Total runs: `{len(all_runs)}`",
        "",
        "## Variant Summary",
        "",
        "| Variant | Runs | Empty Rate | Fallback Rate | Success@8 | p50 (s) | p95 (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in [v.name for v in variants]:
        s = summary_by_variant[name]
        lines.append(
            f"| {name} | {s['runs']} | {s['empty_response_rate']:.2%} | {s['fallback_rate']:.2%} | {s['success_exact_8_rate']:.2%} | {s['decompose_latency_p50_s']:.2f} | {s['decompose_latency_p95_s']:.2f} |"
        )

    lines.extend([
        "",
        "## Ranked Hypotheses",
        "",
    ])
    for idx, h in enumerate(hypotheses, start=1):
        lines.append(f"{idx}. **{h['hypothesis']}** (confidence: `{h['confidence']}`)")
        lines.append(f"   - Evidence: `{json.dumps(h['evidence'], ensure_ascii=False)}`")

    lines.extend([
        "",
        "## Recommendation",
        "",
        f"- Decision: `{recommendation['default']}`",
        f"- Action: {recommendation['recommendation']}",
        "",
        "## Artifacts",
        "",
        "- `runs/*/llm_debug.jsonl`: per-attempt structured-call telemetry",
        "- `runs/*/figure_trace.json`: figure parser step telemetry",
        "- `runs/*/run_result.json`: normalized per-run result",
        "- `report.json`: machine-readable summary",
    ])

    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[investigation] wrote {out_dir / 'report.json'}")
    print(f"[investigation] wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()

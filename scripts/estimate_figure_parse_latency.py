#!/usr/bin/env python3
"""Estimate per-figure parsing latency for a paper with fail-fast controls.

Example:
  python scripts/estimate_figure_parse_latency.py \
    --source 40456907 \
    --source-type pmid \
    --max-figures 10 \
    --max-total-seconds 240 \
    --output /tmp/figure_latency_40456907.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
from pathlib import Path
from typing import Any

from researcher_ai.models.paper import PaperSource
from researcher_ai.parsers.figure_parser import FigureParser
from researcher_ai.parsers.paper_parser import PaperParser


def _pctl(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def _build_payload(
    *,
    source: str,
    source_type: str,
    requested: int,
    parse_elapsed_s: float,
    per_figure: list[dict[str, Any]],
    timed_out: bool,
    max_total_seconds: float,
) -> dict[str, Any]:
    ok_times = [row["seconds"] for row in per_figure if row["status"] == "ok"]
    return {
        "source": source,
        "source_type": source_type,
        "paper_parse_seconds": round(parse_elapsed_s, 4),
        "figure_count_requested": requested,
        "figure_count_timed": len(per_figure),
        "figure_count_ok": len(ok_times),
        "figure_count_error": len(per_figure) - len(ok_times),
        "timed_out": timed_out,
        "max_total_seconds": max_total_seconds,
        "latency_seconds": {
            "mean": round(statistics.mean(ok_times), 4) if ok_times else 0.0,
            "median_p50": round(_pctl(ok_times, 50), 4) if ok_times else 0.0,
            "p95": round(_pctl(ok_times, 95), 4) if ok_times else 0.0,
            "max": round(max(ok_times), 4) if ok_times else 0.0,
        },
        "per_figure": per_figure,
    }


def _write_payload(payload: dict[str, Any], output: str) -> None:
    if not output:
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="PMID or PDF path")
    parser.add_argument("--source-type", choices=["pmid", "pdf"], required=True)
    parser.add_argument("--max-figures", type=int, default=10, help="Max figures to time")
    parser.add_argument(
        "--max-total-seconds",
        type=float,
        default=0.0,
        help="Optional hard timeout for the full figure-timing run (0 disables).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON output path (prints JSON to stdout if omitted).",
    )
    args = parser.parse_args()

    source_type = PaperSource.PMID if args.source_type == "pmid" else PaperSource.PDF

    parse_start = time.perf_counter()
    paper = PaperParser().parse(args.source, source_type=source_type)
    parse_elapsed_s = time.perf_counter() - parse_start

    figure_ids = [
        fid for fid in (paper.figure_ids or [])
        if "supplementary figure" not in (fid or "").strip().lower()
    ][: max(0, args.max_figures)]
    paper = paper.model_copy(update={"figure_ids": figure_ids})

    fp = FigureParser()
    per_figure: list[dict[str, Any]] = []

    original_parse_from_context = fp._parse_figure_from_context

    def _timed_parse_from_context(
        figure_id: str,
        caption: str,
        in_text: list[str],
        *,
        bioc_fig_passages=None,
        bioc_results_passages=None,
    ):
        t0 = time.perf_counter()
        status = "ok"
        error = ""
        warning_count = 0
        subfigure_count = 0
        try:
            figure = original_parse_from_context(
                figure_id,
                caption,
                in_text,
                bioc_fig_passages=bioc_fig_passages,
                bioc_results_passages=bioc_results_passages,
            )
            warning_count = len(figure.parse_warnings)
            subfigure_count = len(figure.subfigures)
            return figure
        except Exception as exc:  # pragma: no cover - runtime diagnostic path
            status = "error"
            error = f"{exc.__class__.__name__}: {exc}"
            raise
        finally:
            elapsed = time.perf_counter() - t0
            per_figure.append(
                {
                    "figure_id": figure_id,
                    "status": status,
                    "seconds": round(elapsed, 4),
                    "subfigure_count": subfigure_count,
                    "warning_count": warning_count,
                    "error": error,
                }
            )
            payload = _build_payload(
                source=args.source,
                source_type=args.source_type,
                requested=args.max_figures,
                parse_elapsed_s=parse_elapsed_s,
                per_figure=per_figure,
                timed_out=False,
                max_total_seconds=args.max_total_seconds,
            )
            _write_payload(payload, args.output)

    fp._parse_figure_from_context = _timed_parse_from_context  # type: ignore[method-assign]

    timed_out = False
    max_total = float(args.max_total_seconds or 0.0)
    if max_total > 0:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            fut = executor.submit(fp.parse_all_figures, paper)
            try:
                fut.result(timeout=max_total)
            except concurrent.futures.TimeoutError:
                timed_out = True
                fut.cancel()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
    else:
        fp.parse_all_figures(paper)

    payload = _build_payload(
        source=args.source,
        source_type=args.source_type,
        requested=args.max_figures,
        parse_elapsed_s=parse_elapsed_s,
        per_figure=per_figure,
        timed_out=timed_out,
        max_total_seconds=max_total,
    )

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {path}")
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

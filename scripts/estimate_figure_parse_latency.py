#!/usr/bin/env python3
"""Estimate per-figure parsing latency for a paper.

Example:
  python scripts/estimate_figure_parse_latency.py \
    --source 40456907 \
    --source-type pmid \
    --max-figures 10 \
    --output /tmp/figure_latency_40456907.json
"""

from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="PMID or PDF path")
    parser.add_argument("--source-type", choices=["pmid", "pdf"], required=True)
    parser.add_argument("--max-figures", type=int, default=10, help="Max figures to time")
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

    fp = FigureParser()
    per_figure: list[dict[str, Any]] = []

    for figure_id in figure_ids:
        t0 = time.perf_counter()
        status = "ok"
        subfigure_count = 0
        warning_count = 0
        error = ""
        try:
            figure = fp.parse_figure(paper, figure_id)
            subfigure_count = len(figure.subfigures)
            warning_count = len(figure.parse_warnings)
        except Exception as exc:  # pragma: no cover - runtime diagnostic path
            status = "error"
            error = f"{exc.__class__.__name__}: {exc}"
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

    ok_times = [row["seconds"] for row in per_figure if row["status"] == "ok"]
    payload: dict[str, Any] = {
        "source": args.source,
        "source_type": args.source_type,
        "paper_parse_seconds": round(parse_elapsed_s, 4),
        "figure_count_requested": args.max_figures,
        "figure_count_timed": len(per_figure),
        "figure_count_ok": len(ok_times),
        "figure_count_error": len(per_figure) - len(ok_times),
        "latency_seconds": {
            "mean": round(statistics.mean(ok_times), 4) if ok_times else 0.0,
            "median_p50": round(_pctl(ok_times, 50), 4) if ok_times else 0.0,
            "p95": round(_pctl(ok_times, 95), 4) if ok_times else 0.0,
            "max": round(max(ok_times), 4) if ok_times else 0.0,
        },
        "per_figure": per_figure,
    }

    out = json.dumps(payload, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(out)
        print(f"Wrote {path}")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

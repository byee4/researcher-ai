from __future__ import annotations

import importlib.util
from pathlib import Path


def test_calibration_report_shows_improvement_on_fixture():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "figure_calibration_report.py"
    spec = importlib.util.spec_from_file_location("figure_calibration_report", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    fixtures_dir = repo_root / "tests" / "fixtures" / "figure_calibration"
    report = mod.run_report(fixtures_dir=fixtures_dir)

    assert len(report["cases"]) >= 5
    assert report["baseline"]["total"] > 0
    assert report["calibrated"]["accuracy"] >= report["baseline"]["accuracy"]
    assert report["delta_accuracy"] > 0
    assert "bar" in report["baseline_per_plot_type"]
    assert "bar" in report["calibrated_per_plot_type"]

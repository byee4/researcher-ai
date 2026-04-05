"""Tests for registry-driven figure calibration engine."""

from __future__ import annotations

import json

from researcher_ai.models.figure import Figure, PlotCategory, PlotType, SubFigure
from researcher_ai.models.paper import Paper, PaperSource, PaperType
from researcher_ai.parsers.figure_calibration import FigureCalibrationEngine


def _paper(pmcid: str = "PMC11633308") -> Paper:
    return Paper(
        title="test",
        source=PaperSource.PMCID,
        source_path=pmcid,
        paper_type=PaperType.EXPERIMENTAL,
        pmcid=pmcid,
        figure_ids=["Figure 1", "Figure 2"],
    )


def _figure(fig_id: str, labels: list[str]) -> Figure:
    return Figure(
        figure_id=fig_id,
        title=fig_id,
        caption="caption",
        purpose="purpose",
        subfigures=[
            SubFigure(
                label=lb,
                description=f"Panel {lb}",
                plot_type=PlotType.OTHER,
                plot_category=PlotCategory.COMPOSITE,
            )
            for lb in labels
        ],
    )


def test_registry_applies_pmc11633308_figure1_rules():
    engine = FigureCalibrationEngine()
    fig = _figure("Figure 1", ["A", "B", "C"])
    calibrated = engine.apply(_paper(), fig)
    by = {sf.label.lower(): sf for sf in calibrated.subfigures}

    assert by["a"].plot_type == PlotType.BAR
    assert by["a"].plot_category == PlotCategory.COMPOSITE
    assert len(by["a"].layers) == 2
    assert by["a"].layers[1].plot_type == PlotType.STACKED_BAR
    assert by["a"].x_axis is not None
    assert by["a"].y_axis is not None


def test_registry_applies_pmc11633308_figure2_rules():
    engine = FigureCalibrationEngine()
    fig = _figure("Figure 2", ["A", "B", "C", "D", "E", "F", "G", "H"])
    calibrated = engine.apply(_paper(), fig)
    by = {sf.label.lower(): sf for sf in calibrated.subfigures}

    assert by["a"].plot_type == PlotType.VENN
    assert by["b"].plot_type == PlotType.STACKED_BAR
    assert by["c"].plot_type == PlotType.TSNE
    assert by["d"].plot_type == PlotType.STACKED_BAR
    assert by["e"].plot_type == PlotType.STACKED_BAR
    assert by["f"].plot_type == PlotType.BUBBLE
    assert by["g"].plot_type == PlotType.UPSET
    assert by["h"].plot_type == PlotType.BAR


def test_registry_does_not_apply_to_other_papers():
    engine = FigureCalibrationEngine()
    fig = _figure("Figure 1", ["A"])
    calibrated = engine.apply(_paper("PMC0000000"), fig)
    assert calibrated.subfigures[0].plot_type == PlotType.OTHER


def test_global_scope_rule_applies_without_paper_identity(tmp_path):
    registry = {
        "version": 1,
        "rules": [
            {
                "rule_id": "global_scatter_to_bar",
                "scope": "global",
                "enabled": True,
                "priority": 1,
                "match": {"figure_id_pattern": "^figure\\s*9$"},
                "actions": {"plot_type": "bar"},
            }
        ],
    }
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    engine = FigureCalibrationEngine(registry_path=registry_path)
    fig = _figure("Figure 9", ["A"])
    calibrated = engine.apply(_paper("PMC0000000"), fig)
    assert calibrated.subfigures[0].plot_type == PlotType.BAR


def test_family_scope_rule_matches_title_pattern(tmp_path):
    registry = {
        "version": 1,
        "rules": [
            {
                "rule_id": "family_nature_bar",
                "scope": "family",
                "enabled": True,
                "priority": 1,
                "match": {
                    "title_pattern": "nature",
                    "figure_id_pattern": "^figure\\s*3$",
                },
                "actions": {"plot_type": "stacked_bar"},
            }
        ],
    }
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    engine = FigureCalibrationEngine(registry_path=registry_path)

    paper_match = Paper(
        title="Nature benchmark study",
        source=PaperSource.PMCID,
        source_path="PMC1",
        paper_type=PaperType.EXPERIMENTAL,
        pmcid="PMC1",
    )
    paper_no_match = Paper(
        title="Cell benchmark study",
        source=PaperSource.PMCID,
        source_path="PMC2",
        paper_type=PaperType.EXPERIMENTAL,
        pmcid="PMC2",
    )
    fig = _figure("Figure 3", ["A"])
    calibrated_yes = engine.apply(paper_match, fig)
    calibrated_no = engine.apply(paper_no_match, fig)
    assert calibrated_yes.subfigures[0].plot_type == PlotType.STACKED_BAR
    assert calibrated_no.subfigures[0].plot_type == PlotType.OTHER


def test_feature_flag_off_disables_calibration(tmp_path, monkeypatch):
    registry = {
        "version": 1,
        "rules": [
            {
                "rule_id": "paper_force",
                "scope": "paper",
                "enabled": True,
                "priority": 1,
                "match": {"pmcid": "PMC11633308", "figure_id_pattern": "^figure\\s*1$"},
                "actions": {"plot_type": "bar"},
            }
        ],
    }
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    monkeypatch.setenv("RESEARCHER_AI_FIGURE_CALIBRATION", "off")
    engine = FigureCalibrationEngine(registry_path=registry_path)
    fig = _figure("Figure 1", ["A"])
    calibrated = engine.apply(_paper(), fig)
    assert calibrated.subfigures[0].plot_type == PlotType.OTHER


def test_global_scope_guardrail_skips_high_confidence_override(tmp_path):
    registry = {
        "version": 1,
        "rules": [
            {
                "rule_id": "global_override",
                "scope": "global",
                "enabled": True,
                "priority": 1,
                "match": {"figure_id_pattern": "^figure\\s*1$"},
                "actions": {"plot_type": "bar"},
            }
        ],
    }
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    engine = FigureCalibrationEngine(registry_path=registry_path)
    high_conf = SubFigure(
        label="A",
        description="Panel A",
        plot_type=PlotType.OTHER,
        plot_category=PlotCategory.COMPOSITE,
        classification_confidence=0.95,
    )
    fig = Figure(figure_id="Figure 1", title="Figure 1", caption="c", purpose="p", subfigures=[high_conf])
    calibrated = engine.apply(_paper("PMC0000000"), fig)
    assert calibrated.subfigures[0].plot_type == PlotType.OTHER


def test_paper_scope_can_override_high_confidence(tmp_path):
    registry = {
        "version": 1,
        "rules": [
            {
                "rule_id": "paper_override",
                "scope": "paper",
                "enabled": True,
                "priority": 1,
                "match": {"pmcid": "PMC11633308", "figure_id_pattern": "^figure\\s*1$"},
                "actions": {"plot_type": "bar"},
            }
        ],
    }
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    engine = FigureCalibrationEngine(registry_path=registry_path)
    high_conf = SubFigure(
        label="A",
        description="Panel A",
        plot_type=PlotType.OTHER,
        plot_category=PlotCategory.COMPOSITE,
        classification_confidence=0.95,
    )
    fig = Figure(figure_id="Figure 1", title="Figure 1", caption="c", purpose="p", subfigures=[high_conf])
    calibrated = engine.apply(_paper(), fig)
    assert calibrated.subfigures[0].plot_type == PlotType.BAR

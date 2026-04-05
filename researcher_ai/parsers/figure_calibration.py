"""Registry-driven figure calibration engine.

Applies deterministic correction rules after base figure parsing. This provides
stable, auditable overrides for benchmark papers without hardcoding rules into
core parser logic.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from researcher_ai.models.figure import Axis, AxisScale, Figure, PlotCategory, PlotLayer, PlotType, SubFigure
from researcher_ai.models.paper import Paper

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML via optional dependency, falling back to JSON parser."""
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import]

        data = yaml.safe_load(text) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        # Support JSON-formatted registry files if YAML dependency is absent.
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except Exception:
            logger.warning("Could not parse registry file %s as YAML or JSON", path)
            return {}


def _norm_panel(label: str) -> str:
    raw = (label or "").strip().strip("()").lower()
    return raw[:1] if raw else ""


def _norm_figure_id(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _parse_plot_type(value: str, fallback: PlotType) -> PlotType:
    try:
        return PlotType((value or "").strip().lower())
    except Exception:
        return fallback


def _parse_plot_category(value: str, fallback: PlotCategory) -> PlotCategory:
    try:
        return PlotCategory((value or "").strip().lower())
    except Exception:
        return fallback


def _category_for_plot(plot_type: PlotType, fallback: PlotCategory) -> PlotCategory:
    mapping = {
        PlotType.BAR: PlotCategory.CATEGORICAL,
        PlotType.GROUPED_BAR: PlotCategory.CATEGORICAL,
        PlotType.STACKED_BAR: PlotCategory.CATEGORICAL,
        PlotType.BOX: PlotCategory.CATEGORICAL,
        PlotType.VIOLIN: PlotCategory.CATEGORICAL,
        PlotType.SCATTER: PlotCategory.RELATIONAL,
        PlotType.LINE: PlotCategory.RELATIONAL,
        PlotType.BUBBLE: PlotCategory.RELATIONAL,
        PlotType.HEATMAP: PlotCategory.MATRIX,
        PlotType.CLUSTERMAP: PlotCategory.MATRIX,
        PlotType.VENN: PlotCategory.FLOW,
        PlotType.UPSET: PlotCategory.FLOW,
        PlotType.SANKEY: PlotCategory.FLOW,
        PlotType.TSNE: PlotCategory.DIMENSIONALITY,
        PlotType.UMAP: PlotCategory.DIMENSIONALITY,
        PlotType.PCA: PlotCategory.DIMENSIONALITY,
        PlotType.IMAGE: PlotCategory.IMAGE,
    }
    return mapping.get(plot_type, fallback)


def _parse_axis_scale(value: str, fallback: AxisScale) -> AxisScale:
    try:
        return AxisScale((value or "").strip().lower())
    except Exception:
        return fallback


class FigureCalibrationEngine:
    """Apply registry rules to parsed Figure objects."""

    def __init__(self, registry_path: Optional[Path] = None):
        if registry_path is None:
            registry_path = Path(__file__).resolve().parents[1] / "calibration" / "figure_registry.yaml"
        self.registry_path = registry_path
        self.registry: dict[str, Any] = _load_yaml(registry_path) if registry_path.exists() else {"rules": []}
        rules = self.registry.get("rules", [])
        self.rules: list[dict[str, Any]] = [r for r in rules if isinstance(r, dict) and r.get("enabled", True)]
        self.rules.sort(key=lambda r: int(r.get("priority", 0)))
        self.enabled = self._env_flag_enabled()
        self.high_confidence_threshold = self._env_confidence_threshold()

    @staticmethod
    def _env_flag_enabled() -> bool:
        value = os.environ.get("RESEARCHER_AI_FIGURE_CALIBRATION", "on").strip().lower()
        return value not in {"0", "false", "off", "no", "disabled"}

    @staticmethod
    def _env_confidence_threshold() -> float:
        raw = os.environ.get("RESEARCHER_AI_FIGURE_CALIBRATION_CONFIDENCE_THRESHOLD", "0.75").strip()
        try:
            parsed = float(raw)
        except Exception:
            parsed = 0.75
        return max(0.0, min(1.0, parsed))

    def apply(self, paper: Paper, figure: Figure) -> Figure:
        """Apply matching rules to a figure and return a calibrated copy."""
        if not self.enabled:
            return figure
        fig = figure
        for rule in self.rules:
            if not self._rule_matches(rule, paper, fig):
                continue
            fig = self._apply_rule(fig, rule)
        return fig

    def _rule_matches(self, rule: dict[str, Any], paper: Paper, figure: Figure) -> bool:
        scope = str(rule.get("scope") or "paper").strip().lower()
        match = rule.get("match", {}) if isinstance(rule.get("match"), dict) else {}

        pmcid = (paper.pmcid or "").strip().upper()
        pmid = (paper.pmid or "").strip()
        title = (paper.title or "").strip()
        source = (paper.source.value if hasattr(paper.source, "value") else str(paper.source or "")).strip().lower()
        source_path = (paper.source_path or "").strip()
        paper_type = (paper.paper_type.value if hasattr(paper.paper_type, "value") else str(paper.paper_type or "")).strip().lower()
        figure_norm = _norm_figure_id(figure.figure_id)

        if match.get("pmcid") and str(match.get("pmcid")).strip().upper() != pmcid:
            return False
        if match.get("pmid") and str(match.get("pmid")).strip() != pmid:
            return False
        figure_pattern = str(match.get("figure_id_pattern") or "").strip()
        if figure_pattern and not re.search(figure_pattern, figure_norm, flags=re.IGNORECASE):
            return False

        if scope == "global":
            return True

        if scope == "family":
            family_checks: list[bool] = []
            source_types = match.get("source_types") or []
            if isinstance(source_types, list) and source_types:
                family_checks.append(source in {str(x).strip().lower() for x in source_types})
            source_path_pattern = str(match.get("source_path_pattern") or "").strip()
            if source_path_pattern:
                family_checks.append(bool(re.search(source_path_pattern, source_path, flags=re.IGNORECASE)))
            title_pattern = str(match.get("title_pattern") or "").strip()
            if title_pattern:
                family_checks.append(bool(re.search(title_pattern, title, flags=re.IGNORECASE)))
            paper_type_match = str(match.get("paper_type") or "").strip().lower()
            if paper_type_match:
                family_checks.append(paper_type == paper_type_match)
            # If no family-specific key is provided, treat as general pass-through.
            return all(family_checks) if family_checks else True

        # "paper" scope defaults to identifier-focused matching. If no explicit
        # identifiers are provided, allow (acts similarly to a constrained rule).
        return True

    def _apply_rule(self, figure: Figure, rule: dict[str, Any]) -> Figure:
        actions = rule.get("actions", {}) if isinstance(rule.get("actions"), dict) else {}
        match = rule.get("match", {}) if isinstance(rule.get("match"), dict) else {}
        scope = str(rule.get("scope") or "paper").strip().lower()
        panel_labels = [_norm_panel(str(x)) for x in (match.get("panel_labels") or [])]
        panel_labels = [x for x in panel_labels if x]

        updated_figure = figure
        if actions.get("title_override"):
            updated_figure = updated_figure.model_copy(update={"title": str(actions["title_override"])})
        if actions.get("caption_override"):
            updated_figure = updated_figure.model_copy(update={"caption": str(actions["caption_override"])})

        updated_subfigs: list[SubFigure] = []
        for sf in updated_figure.subfigures:
            label = _norm_panel(sf.label)
            if panel_labels and label not in panel_labels:
                updated_subfigs.append(sf)
                continue
            updated_subfigs.append(self._apply_actions_to_subfigure(sf, actions, scope=scope))
        return updated_figure.model_copy(update={"subfigures": updated_subfigs})

    def _apply_actions_to_subfigure(self, subfigure: SubFigure, actions: dict[str, Any], *, scope: str) -> SubFigure:
        sf = subfigure
        updates: dict[str, Any] = {}
        allow_high_conf_override = bool(actions.get("allow_high_confidence_override", False))
        high_conf_guard = (
            scope in {"global", "family"}
            and not allow_high_conf_override
            and sf.classification_confidence >= self.high_confidence_threshold
        )

        if actions.get("plot_type") and not high_conf_guard:
            plot_type = _parse_plot_type(str(actions["plot_type"]), sf.plot_type)
            updates["plot_type"] = plot_type
            if actions.get("plot_category"):
                updates["plot_category"] = _parse_plot_category(str(actions["plot_category"]), sf.plot_category)
            else:
                updates["plot_category"] = _category_for_plot(plot_type, sf.plot_category)

        if isinstance(actions.get("layers"), list) and not high_conf_guard:
            layers: list[PlotLayer] = []
            for idx, layer_pt in enumerate(actions["layers"]):
                plot_type = _parse_plot_type(str(layer_pt), sf.plot_type)
                layers.append(PlotLayer(plot_type=plot_type, is_primary=(idx == 0)))
            if layers:
                updates["layers"] = layers

        if "n_facets" in actions and not high_conf_guard:
            try:
                updates["n_facets"] = int(actions.get("n_facets"))
            except Exception:
                pass
        if "facet_variable" in actions and not high_conf_guard:
            updates["facet_variable"] = str(actions.get("facet_variable") or "")
        if "confidence" in actions and not high_conf_guard:
            try:
                updates["classification_confidence"] = float(actions.get("confidence"))
            except Exception:
                pass

        x_axis = sf.x_axis
        y_axis = sf.y_axis
        if actions.get("x_axis_label") or actions.get("x_axis_scale"):
            if x_axis is None:
                x_axis = Axis(label=str(actions.get("x_axis_label") or "x"))
            if actions.get("x_axis_label"):
                x_axis = x_axis.model_copy(update={"label": str(actions["x_axis_label"])})
            if actions.get("x_axis_scale"):
                x_axis = x_axis.model_copy(update={"scale": _parse_axis_scale(str(actions["x_axis_scale"]), x_axis.scale)})
        if actions.get("y_axis_label") or actions.get("y_axis_scale"):
            if y_axis is None:
                y_axis = Axis(label=str(actions.get("y_axis_label") or "y"))
            if actions.get("y_axis_label"):
                y_axis = y_axis.model_copy(update={"label": str(actions["y_axis_label"])})
            if actions.get("y_axis_scale"):
                y_axis = y_axis.model_copy(update={"scale": _parse_axis_scale(str(actions["y_axis_scale"]), y_axis.scale)})
        if x_axis is not None:
            updates["x_axis"] = x_axis
        if y_axis is not None:
            updates["y_axis"] = y_axis

        evidence_tag = str(actions.get("evidence_tag") or "").strip()
        if evidence_tag:
            evidence = [e for e in sf.evidence_spans if isinstance(e, str) and e.strip()]
            evidence.append(evidence_tag)
            updates["evidence_spans"] = list(dict.fromkeys(evidence))

        if not updates:
            return sf
        return sf.model_copy(update=updates)

from researcher_ai.pipeline.reporting import summarize_figure_parsing


def test_summarize_figure_parsing_classifies_modes_and_counts():
    figures = [
        {"figure_id": "Figure 1", "parse_warnings": []},
        {
            "figure_id": "Figure 2",
            "parse_warnings": [
                "subfigure_decomposition_empty_response",
                "subfigure_decomposition_caption_split_fallback",
            ],
        },
        {
            "figure_id": "Figure 3",
            "parse_warnings": ["subfigure_decomposition_timeout"],
        },
        {
            "figure_id": "Figure 4",
            "parse_warnings": ["multimodal_pdf_panel:low_confidence"],
        },
    ]

    summary = summarize_figure_parsing(figures)

    assert summary["figure_count"] == 4
    assert summary["figures_with_any_parse_warnings"] == 3
    assert summary["decomposition_mode_counts"]["llm"] == 1
    assert summary["decomposition_mode_counts"]["caption_split_fallback"] == 1
    assert summary["decomposition_mode_counts"]["timeout_fallback"] == 1
    assert summary["decomposition_mode_counts"]["llm_with_warnings"] == 1
    assert summary["warning_counts"]["subfigure_decomposition_empty_response"] == 1
    assert summary["warning_counts"]["subfigure_decomposition_caption_split_fallback"] == 1
    assert summary["warning_counts"]["subfigure_decomposition_timeout"] == 1

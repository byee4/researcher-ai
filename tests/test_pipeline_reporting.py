from researcher_ai.pipeline.reporting import summarize_figure_parsing, summarize_method_parsing


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


def test_summarize_method_parsing_extracts_excluded_assays():
    method = {
        "parse_warnings": [
            "assay_filtered_non_computational: 'Cell culture' excluded (category=experimental, computational_only=True)",
            "assay_filtered_non_computational: 'Immunofluorescence' excluded (category=experimental, computational_only=True)",
            "retrieval_parameter_gap: assay='RNA-seq' stage='align' rounds=2 unresolved=parameters",
        ]
    }
    summary = summarize_method_parsing(method)
    assert summary["excluded_assay_count"] == 2
    assert summary["warning_counts"]["assay_filtered_non_computational"] == 2
    assert summary["warning_counts"]["retrieval_parameter_gap"] == 1
    assert summary["excluded_assays"][0]["name"] == "Cell culture"
    assert summary["excluded_assays"][0]["category"] == "experimental"

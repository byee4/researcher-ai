# Agentic Refactor Plan: `researcher_ai` Architecture

This document serves as an architectural blueprint for transitioning the `researcher_ai` codebase from a linear ETL pipeline to a state-graph agentic workflow. It is written with system prompts and directives designed to be ingested by an AI coding assistant or autonomous engineering agent.

---

## Phase 1: Establish the Universal LLM Interface
**Target:** `researcher_ai/utils/llm.py`
**Objective:** Decouple parsing logic from specific provider APIs to allow dynamic model routing based on task complexity (e.g., using lightweight models for text routing and heavy reasoning models for method extraction).

> **Agent Implementation Prompt:**
> "Refactor `researcher_ai/utils/llm.py`. Replace all provider-specific functions (e.g., `ask_claude_structured`) with a single generic function: `extract_structured_data(model_router: str, prompt: str, schema: BaseModel)`. Implement `litellm` to handle API standardization across OpenAI, Anthropic, and Google Gemini. Ensure the function dynamically pulls the correct API key from the local Mac environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`) based on the requested `model_router` string."

---

## Phase 2: Multimodal Ingestion Pipeline
**Targets:** `researcher_ai/parsers/paper_parser.py`, `researcher_ai/parsers/figure_parser.py`
**Objective:** Prevent data loss during PDF extraction by preserving spatial formatting and passing raw image panels directly to multimodal models.

> **Agent Implementation Prompt:**
> "Update `PaperParser.parse()`. When the `PaperSource` is `PDF`, route the document through the `marker-pdf` library to generate spatially accurate Markdown, retaining table structures. Subsequently, refactor `FigureParser`. Disable the current regex and BioC heuristic pipeline. Implement a new flow that crops identified figure panels from the PDF and passes the raw image bytes alongside the caption text directly to a multimodal vision model (e.g., `gemini-3.1-pro`) to instantiate the `Figure` and `SubFigure` Pydantic models."

---

## Phase 3: RAG-Augmented Method Inference
**Target:** `researcher_ai/parsers/methods_parser.py`
**Objective:** Resolve omitted methodological details in academic papers by querying a localized vector database of standard bioinformatics protocols and software documentation.

> **Agent Implementation Prompt:**
> "Modify `MethodsParser`. Before finalizing the `AssayGraph`, the parser must evaluate the graph for missing computational parameters. Equip the reasoning LLM with a `search_protocol_docs` tool connected to a local vector store containing standard bioinformatics documentation (e.g., STAR aligner manuals, Seurat vignettes, and internal Yeo Lab standard operating procedures for eCLIP and RNA-seq). Instruct the LLM to query this database to infer and populate missing `AnalysisStep` parameters before passing Pydantic validation."

---

## Phase 4: State-Graph Orchestration & Agentic Execution
**Targets:** `scripts/run_workflow.py` (Replace), `researcher_ai/pipeline/builder.py`
**Objective:** Transition from a one-shot pipeline compiler to an iterative, stateful execution loop that validates its own code before final output.

> **Agent Implementation Prompt:**
> "Deprecate the linear `run_workflow.py` execution script. Implement a new orchestrator using LangGraph. Define the global state using the existing Pydantic models: `Paper`, `Method`, and `PipelineConfig`. 
> 
> Refactor `PipelineBuilder` into an autonomous Engineering Agent. This agent must read the populated `AssayGraph` state and generate a complete `Snakefile`. The agent must automatically inject SLURM execution profiles (`--partition`, `--account`, `--mem`) optimized specifically for the Triton Shared Computing Cluster (TSCC). 
> 
> Finally, provide the Engineering Agent with a local bash execution tool. Force the agent to run `snakemake --lint` or `snakemake -n` in a temporary sandbox. If the terminal returns an error, the agent must parse the traceback, rewrite the `Snakefile`, and re-test until the pipeline passes validation before returning the final state."

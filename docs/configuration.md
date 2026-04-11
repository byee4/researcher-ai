# Configuration

`researcher-ai` behavior is controlled by model routing config and environment variables.

## Model/provider config file

Default model and provider behavior lives in:
`researcher_ai/config/models.yaml`

Use this file to:
- map stable aliases to vendor model IDs,
- define provider API key precedence,
- enforce provider-specific constraints (temperature, min tokens, safety settings),
- adjust default timeout and token settings.

## Environment variables

## Provider credentials

- `OPENAI_API_KEY`
- `LLM_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `GOOGLE_API_KEY`

## Core runtime

- `RESEARCHER_AI_MODEL`: default model router (default `gpt-5.4`)
- `RESEARCHER_AI_LLM_TIMEOUT_SECONDS`: LLM timeout override (recommended default `90`)
- `RESEARCHER_AI_PROVIDER_MAX_RETRIES`: provider SDK retry cap (default `0`; prevents hidden timeout multiplication)
- `RESEARCHER_AI_LITELLM_VERBOSE`: enable LiteLLM debug/verbose logging (`1/true`, recommended default `0`)
- `RESEARCHER_AI_MAX_IMAGE_BYTES`: image-size cap for multimodal calls
- `RESEARCHER_AI_PARSE_FIGURES_TIMEOUT_SECONDS`: optional hard timeout for orchestrator figure parsing; degrades to empty figures on timeout (recommended default `0`)
- `RESEARCHER_AI_PARSE_FIGURES_TIMEOUT_PER_FIGURE_SECONDS`: per-figure floor used to auto-scale orchestrator figure timeout (default `60`)
- `RESEARCHER_AI_SKIP_FIGURES`: `1/true` skips figure parsing (recovery mode, recommended default `0`)
- `RESEARCHER_AI_SUBFIGURE_TIMEOUT_SECONDS`: optional timeout for panel decomposition requests (recommended default `0`)
- `RESEARCHER_AI_MAX_FIGURE_LLM_TIMEOUTS`: per-paper timeout budget before figure LLM circuit breaker opens (default `3`)
- `RESEARCHER_AI_SUBFIGURE_DECOMPOSE_MAX_TOKENS`: max output tokens for panel decomposition (default `1200`)
- `RESEARCHER_AI_FIGURE_PURPOSE_MAX_TOKENS`: max output tokens for figure purpose/title extraction (default `600`)
- `RESEARCHER_AI_FIGURE_METHODS_DATASETS_MAX_TOKENS`: max output tokens for figure methods/dataset extraction (default `350`)
- `RESEARCHER_AI_FIGURE_TRACE_PATH`: optional path to write per-step figure telemetry JSON trace
- `RESEARCHER_AI_BIOWORKFLOW_MODE`: BioWorkflow rollout mode (`off`, `warn`, `on`)
  - `off`: skip validation stage in orchestrator
  - `warn` (default): run validation and continue with warnings
  - `on`: strict; if ungrounded fields remain, return `needs_human_review`
- `RESEARCHER_AI_MAX_RETRIEVAL_REFINEMENT_ROUNDS`: hard cap for iterative retrieval rounds per stage (recommended default `2`)

Methods parser safeguard behavior:
- On quota/rate-limit assay parse failures, the parser opens an internal assay circuit breaker for that paper and parses remaining assays from local text fallback instead of continuing assay-level LLM calls.
- This behavior is surfaced via `assay_parse_circuit_opened` and `assay_fallback_no_llm_after_circuit` warnings in `Method.parse_warnings`.
- Retrieval refinement warnings are severity-typed:
  - `retrieval_circuit_breaker`: unresolved critical fields (for example missing software evidence).
  - `retrieval_parameter_gap`: only parameters remain unresolved after refinement rounds.
  - `retrieval_refinement_stalled`: refinement produced no novel evidence chunks for the stage.

Figure parser safeguard behavior:
- When panel decomposition returns an empty structured response, parser warnings include `subfigure_decomposition_empty_response`.
- If deterministic caption panel splitting is then used as fallback, parser warnings include `subfigure_decomposition_caption_split_fallback`.
- Workflow run artifacts now include `figure_parse_summary` with:
  - `decomposition_mode_counts` (`llm`, `caption_split_fallback`, `timeout_fallback`, `llm_with_warnings`, `empty_response_no_split`)
  - per-warning counts and per-figure decomposition mode/warning details.

## BioC / PubMed

- `RESEARCHER_AI_BIOC_ENABLED`: enable/disable BioC enrichment (recommended default `1`)
- `RESEARCHER_AI_BIOC_CACHE_DIR`: custom BioC cache directory
- `RESEARCHER_AI_BIOC_CACHE_TTL_SEC`: BioC cache TTL (seconds)
- `NCBI_API_KEY`: API key for higher NCBI throughput

## Figure calibration

- `RESEARCHER_AI_FIGURE_CALIBRATION`: `on`/`off` (recommended default `on`)
- `RESEARCHER_AI_FIGURE_CALIBRATION_CONFIDENCE_THRESHOLD`: threshold used by calibration logic

## Pipeline/HPC

- `RESEARCHER_AI_HPC_PROFILE`: `tscc` or `local` (recommended default `tscc`)
- `TSCC_SLURM_PARTITION`, `TSCC_SLURM_ACCOUNT`, `TSCC_SLURM_MEM`
- `LOCAL_SLURM_PARTITION`, `LOCAL_SLURM_ACCOUNT`, `LOCAL_SLURM_MEM`

## Example shell setup

```bash
export OPENAI_API_KEY="..."
export RESEARCHER_AI_MODEL="gpt-5.4"
export RESEARCHER_AI_LLM_TIMEOUT_SECONDS="90"
export RESEARCHER_AI_LITELLM_VERBOSE="0"
export RESEARCHER_AI_PARSE_FIGURES_TIMEOUT_SECONDS="0"
export RESEARCHER_AI_PARSE_FIGURES_TIMEOUT_PER_FIGURE_SECONDS="60"
export RESEARCHER_AI_SUBFIGURE_TIMEOUT_SECONDS="0"
export RESEARCHER_AI_MAX_FIGURE_LLM_TIMEOUTS="3"
export RESEARCHER_AI_SUBFIGURE_DECOMPOSE_MAX_TOKENS="1200"
export RESEARCHER_AI_FIGURE_PURPOSE_MAX_TOKENS="600"
export RESEARCHER_AI_FIGURE_METHODS_DATASETS_MAX_TOKENS="350"
export RESEARCHER_AI_BIOWORKFLOW_MODE="warn"
export RESEARCHER_AI_MAX_RETRIEVAL_REFINEMENT_ROUNDS="2"
export NCBI_API_KEY="..."
export RESEARCHER_AI_BIOC_ENABLED="1"
export RESEARCHER_AI_FIGURE_CALIBRATION="on"
export RESEARCHER_AI_HPC_PROFILE="tscc"
```

## PMID 39303722 benchmark profile (2026-04-10)

Use this profile when running PMID `39303722` with OpenAI-only credentials:

```bash
export RESEARCHER_AI_MODEL="gpt-5.4"
export RESEARCHER_AI_DISABLE_MODEL_FALLBACKS="1"
export RESEARCHER_AI_LLM_TIMEOUT_SECONDS="180"
export RESEARCHER_AI_PROVIDER_MAX_RETRIES="0"
export RESEARCHER_AI_PARSE_FIGURES_TIMEOUT_SECONDS="1800"
export RESEARCHER_AI_PARSE_FIGURES_TIMEOUT_PER_FIGURE_SECONDS="180"
export RESEARCHER_AI_SUBFIGURE_TIMEOUT_SECONDS="180"
export RESEARCHER_AI_MAX_FIGURE_LLM_TIMEOUTS="6"
export RESEARCHER_AI_MAX_RETRIEVAL_REFINEMENT_ROUNDS="2"
export RESEARCHER_AI_BIOWORKFLOW_MODE="warn"
```

Observed benchmark outcome (`one_more_recommended_net`):
- runtime: `535.42s`
- sections parsed: `8`
- figures parsed: `7`
- assays in assay graph: `22`
- assay stub parse warnings: `22`
- leading warning class: OpenAI `RateLimitError` (quota exceeded)

Interpretation: this profile prevents premature timeout/fallback degradation, but it cannot override upstream provider quota exhaustion. If warnings show quota exceeded, increase/restore provider quota before rerunning.

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
- `RESEARCHER_AI_LLM_TIMEOUT_SECONDS`: LLM timeout override
- `RESEARCHER_AI_PROVIDER_MAX_RETRIES`: provider SDK retry cap (default `0`; prevents hidden timeout multiplication)
- `RESEARCHER_AI_LITELLM_VERBOSE`: enable LiteLLM debug/verbose logging (`1/true`)
- `RESEARCHER_AI_MAX_IMAGE_BYTES`: image-size cap for multimodal calls
- `RESEARCHER_AI_PARSE_FIGURES_TIMEOUT_SECONDS`: optional hard timeout for orchestrator figure parsing; degrades to empty figures on timeout
- `RESEARCHER_AI_PARSE_FIGURES_TIMEOUT_PER_FIGURE_SECONDS`: per-figure floor used to auto-scale orchestrator figure timeout (default `60`)
- `RESEARCHER_AI_SKIP_FIGURES`: `1/true` skips figure parsing (recovery mode)
- `RESEARCHER_AI_SUBFIGURE_TIMEOUT_SECONDS`: optional timeout for panel decomposition requests
- `RESEARCHER_AI_MAX_FIGURE_LLM_TIMEOUTS`: per-paper timeout budget before figure LLM circuit breaker opens (default `3`)
- `RESEARCHER_AI_SUBFIGURE_DECOMPOSE_MAX_TOKENS`: max output tokens for panel decomposition (default `1200`)
- `RESEARCHER_AI_FIGURE_PURPOSE_MAX_TOKENS`: max output tokens for figure purpose/title extraction (default `600`)
- `RESEARCHER_AI_FIGURE_METHODS_DATASETS_MAX_TOKENS`: max output tokens for figure methods/dataset extraction (default `350`)
- `RESEARCHER_AI_FIGURE_TRACE_PATH`: optional path to write per-step figure telemetry JSON trace
- `RESEARCHER_AI_BIOWORKFLOW_MODE`: BioWorkflow rollout mode (`off`, `warn`, `on`)
  - `off`: skip validation stage in orchestrator
  - `warn` (default): run validation and continue with warnings
  - `on`: strict; if ungrounded fields remain, return `needs_human_review`
- `RESEARCHER_AI_MAX_RETRIEVAL_REFINEMENT_ROUNDS`: hard cap for iterative retrieval rounds per stage

## BioC / PubMed

- `RESEARCHER_AI_BIOC_ENABLED`: enable/disable BioC enrichment
- `RESEARCHER_AI_BIOC_CACHE_DIR`: custom BioC cache directory
- `RESEARCHER_AI_BIOC_CACHE_TTL_SEC`: BioC cache TTL (seconds)
- `NCBI_API_KEY`: API key for higher NCBI throughput

## Figure calibration

- `RESEARCHER_AI_FIGURE_CALIBRATION`: `on`/`off`
- `RESEARCHER_AI_FIGURE_CALIBRATION_CONFIDENCE_THRESHOLD`: threshold used by calibration logic

## Pipeline/HPC

- `RESEARCHER_AI_HPC_PROFILE`: `tscc` or `local`
- `TSCC_SLURM_PARTITION`, `TSCC_SLURM_ACCOUNT`, `TSCC_SLURM_MEM`
- `LOCAL_SLURM_PARTITION`, `LOCAL_SLURM_ACCOUNT`, `LOCAL_SLURM_MEM`

## Example shell setup

```bash
export OPENAI_API_KEY="..."
export RESEARCHER_AI_MODEL="gpt-5.4"
export RESEARCHER_AI_PARSE_FIGURES_TIMEOUT_SECONDS="90"
export RESEARCHER_AI_PARSE_FIGURES_TIMEOUT_PER_FIGURE_SECONDS="60"
export RESEARCHER_AI_MAX_FIGURE_LLM_TIMEOUTS="3"
export RESEARCHER_AI_SUBFIGURE_DECOMPOSE_MAX_TOKENS="1200"
export RESEARCHER_AI_BIOWORKFLOW_MODE="warn"
export RESEARCHER_AI_MAX_RETRIEVAL_REFINEMENT_ROUNDS="2"
export NCBI_API_KEY="..."
export RESEARCHER_AI_BIOC_ENABLED="1"
export RESEARCHER_AI_FIGURE_CALIBRATION="on"
export RESEARCHER_AI_HPC_PROFILE="tscc"
```

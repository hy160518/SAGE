# Multi-Modal Forensic Data Processing Service

Code repository accompanying the paper submission - implements a Master-Agent framework for processing images, voice, and text data with entity fusion.

## Setup

**Requirements:** Python 3.10

```bash
pip install -r requirements.txt
```

Configure API keys:
```bash
cp configs/api_keys.yaml.example configs/api_keys.yaml
# Fill in API keys for DashScope, OpenAI, ZhipuAI...
```

## Project Structure

- **Model Factory** (`src/models/`): Unified LLM interface (Qwen-VL, ChatGLM, Whisper)
- **Processors** (`src/processors/`): Image/Voice/Text workers with parallel processing
- **UIDN Fusion** (`src/fusion/`): Entity matching & relationship graph construction
- **Data Processing** (`src/data/`): Input validation & anonymization
- **Evaluation** (`eval/`): Performance metrics (ASR, extraction accuracy, downstream tasks)
- **Baselines** (`src/baselines/`): Adapters for AutoGen-MAC and LlamaIndex-Workflows

## Data & Privacy

See `configs/schemas/chat_messages.schema.json` for data format.

## Baselines

Two external baselines are implemented as adapters to match our workflow:

- AutoGen-MAC (`src/baselines/autogen_mac.py`): Multi-agent chat orchestration with Text/Image/Voice agents. Uses our processors as tools. If `autogen` is not installed, a fallback orchestrator is used.
- LlamaIndex-Workflows (`src/baselines/llamaindex_workflows.py`): Event-driven DAG with OCR/ASR/Entity steps and optional vector-store aggregation. If `llama-index-core` is unavailable, a fallback DAG simulation runs.

Quick run (optional):

```bash
python scripts/run_baselines.py
python scripts/run_e2e_baseline_pipeline.py
```

Install optional deps if needed:

```bash
pip install autogen>=0.3.0 llama-index-core>=0.10.0
```

### Realistic data acquisition via baselines

- AutoGen path: `collect_with_autogen()` creates a minimal agent setup (User + DataLoader-Agent) to orchestrate data loading via `ForensicDataLoader`. If `autogen` is missing, it falls back to direct loader.
- LlamaIndex path: `collect_with_llamaindex()` wraps loading in a Workflow `LoadStep` driven by `StartEvent`. If `llama-index-core` is missing, it falls back to direct loader.

End-to-end: collectors → baseline processors (OCR/ASR/Entities) → `UIDN` fusion → graph/timeline export.

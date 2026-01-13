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

- **AutoGen-MAC** ([https://github.com/microsoft/autogen](https://github.com/microsoft/autogen)): Multi-agent chat orchestration with Text/Image/Voice agents. Uses our processors as tools. **Requires `pyautogen` to be installed.**
- **LlamaIndex-Workflows** ([https://github.com/run-llama/llama_index](https://github.com/run-llama/llama_index)): Event-driven DAG with OCR/ASR/Entity steps and optional vector-store aggregation. **Requires `llama-index-core` to be installed.**

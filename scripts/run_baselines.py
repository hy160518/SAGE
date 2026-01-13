import json
from pathlib import Path

from src.pipeline.runner import PipelineRunner


def main():
    project_root = Path(__file__).parent.parent
    input_path = project_root / 'data' / 'annotations' / 'Case_A_chat_messages.json'

    runner = PipelineRunner()
    result_autogen = runner.run(input_path=str(input_path), baseline_mode='autogen')
    print('[AutoGen-MAC] result keys:', list(result_autogen.keys()))

    runner2 = PipelineRunner()
    result_llama = runner2.run(input_path=str(input_path), baseline_mode='llamaindex')
    print('[LlamaIndex-Workflows] result keys:', list(result_llama.keys()))

    out_dir = project_root / 'output' / 'baselines'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'autogen_results.json', 'w', encoding='utf-8') as f:
        json.dump(result_autogen.get('baseline_results', {}), f, ensure_ascii=False, indent=2)
    with open(out_dir / 'llamaindex_results.json', 'w', encoding='utf-8') as f:
        json.dump(result_llama.get('baseline_results', {}), f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()

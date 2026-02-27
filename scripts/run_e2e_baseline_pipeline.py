import json
from pathlib import Path

from typing import Optional
from src.baselines.data_collectors import collect_with_autogen, collect_with_llamaindex
from src.baselines.autogen_mac import AutoGenMACBaseline
from src.baselines.llamaindex_workflows import LlamaIndexWorkflowBaseline
from src.fusion.uidn_builder import UIDN


def run_pipeline(mode: str, csv_path: Optional[str] = None, media_dir: Optional[str] = None):
    project_root = Path(__file__).parent.parent
    config = {}

    if mode == 'autogen':
        data = collect_with_autogen(config, csv_path=csv_path, media_dir=media_dir)
        baseline = AutoGenMACBaseline(config)
    elif mode == 'llamaindex':
        data = collect_with_llamaindex(config, csv_path=csv_path, media_dir=media_dir)
        baseline = LlamaIndexWorkflowBaseline(config)
    else:
        raise ValueError('mode must be autogen or llamaindex')

    results = baseline.run(data)

    uidn = UIDN()
    uidn.process_worker_results(results)
    uidn.build_relationship_graph()
    uidn.detect_conflicts()
    uidn.generate_timeline()
    export = uidn.export_results()

    out_dir = project_root / 'output' / 'baselines'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'e2e_{mode}_uidn.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    print(f'E2E {mode} pipeline results saved to: {out_file}')


def main():
    project_root = Path(__file__).parent.parent
    csv_example = str(project_root / 'data' / 'we_chat_forensic.csv') 
    media_example = str(project_root / 'data' / 'media')  
    try:
        run_pipeline('autogen', csv_path=csv_example, media_dir=media_example)
    except Exception as e:
        print('[autogen] pipeline failed:', e)

    try:
        run_pipeline('llamaindex', csv_path=csv_example, media_dir=media_example)
    except Exception as e:
        print('[llamaindex] pipeline failed:', e)


if __name__ == '__main__':
    main()

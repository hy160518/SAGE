import sys
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from config import EvaluationConfig


class EvaluationRunner:
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results = {}
        self.start_time = datetime.now()
    
    def run(self, run_downstream: bool = True, 
            run_asr: bool = True, 
            run_drug_extraction: bool = True) -> Dict[str, Any]:
        print("=" * 80)
        print("Starting Evaluation")
        print("=" * 80)
        print()
        
        if run_downstream:
            print("[1/3] Running Downstream Task Metrics...")
            try:
                self._run_downstream_evaluation()
                self.results['downstream'] = 'SUCCESS'
                print("  ✓ Downstream Task Evaluation completed\n")
            except Exception as e:
                self.results['downstream'] = f'FAILED: {str(e)}'
                print(f"  ✗ Failed: {str(e)}\n")
        
        if run_asr:
            print("[2/3] Running ASR Performance Evaluation...")
            try:
                self._run_asr_evaluation()
                self.results['asr'] = 'SUCCESS'
                print("  ✓ ASR Performance Evaluation completed\n")
            except Exception as e:
                self.results['asr'] = f'FAILED: {str(e)}'
                print(f"  ✗ Failed: {str(e)}\n")
        
        if run_drug_extraction:
            print("[3/3] Running Drug Extraction Evaluation...")
            try:
                self._run_drug_extraction_evaluation()
                self.results['drug_extraction'] = 'SUCCESS'
                print("  ✓ Drug Extraction Evaluation completed\n")
            except Exception as e:
                self.results['drug_extraction'] = f'FAILED: {str(e)}'
                print(f"  ✗ Failed: {str(e)}\n")
        
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        summary = {
            'timestamp': self.start_time.isoformat(),
            'duration_seconds': duration,
            'results': self.results,
            'status': 'SUCCESS' if all(v == 'SUCCESS' for v in self.results.values()) else 'PARTIAL_SUCCESS'
        }
        
        self._save_summary(summary)
        self._print_summary(summary)
        
        return summary
    
    def _run_downstream_evaluation(self) -> None:
        sys.path.insert(0, str(self.project_root / 'eval'))
        from downstream_task_metrics import DownstreamEvaluator
        
        evaluator = DownstreamEvaluator(
            annotation_dir=str(self.project_root / 'data' / 'annotations'),
            results_dir=str(self.project_root / 'results')
        )
        evaluator.evaluate_all()
    
    def _run_asr_evaluation(self) -> None:
        sys.path.insert(0, str(self.project_root / 'eval'))
        from asr_performance import ASREvaluator
        
        evaluator = ASREvaluator(
            annotation_dir=str(self.project_root / 'data' / 'annotations'),
            results_dir=str(self.project_root / 'results')
        )
        evaluator.evaluate_all()
    
    def _run_drug_extraction_evaluation(self) -> None:
        sys.path.insert(0, str(self.project_root / 'eval'))
        from drug_extraction_accuracy import DrugExtractionEvaluator
        
        evaluator = DrugExtractionEvaluator(
            annotation_dir=str(self.project_root / 'data' / 'annotations'),
            results_dir=str(self.project_root / 'results')
        )
        evaluator.evaluate_all()
    
    def _save_summary(self, summary: Dict[str, Any]) -> None:
        summary_file = self.project_root / 'results' / 'evaluation_summary.json'
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def _print_summary(self, summary: Dict[str, Any]) -> None:
        print("=" * 80)
        print("Evaluation Summary")
        print("=" * 80)
        print(f"Status: {summary['status']}")
        print(f"Duration: {summary['duration_seconds']:.2f} seconds")
        print()
        print("Results:")
        for task, result in summary['results'].items():
            status_icon = "✓" if result == "SUCCESS" else "✗"
            print(f"  {status_icon} {task}: {result}")
        print()
        print(f"Summary saved to: results/evaluation_summary.json")
        print("=" * 80)


def main():
    runner = EvaluationRunner()
    runner.run(run_downstream=True, run_asr=True, run_drug_extraction=True)


if __name__ == '__main__':
    main()

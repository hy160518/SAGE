import json
from pathlib import Path
from typing import Dict, List, Any
from statistics import mean, stdev

from config import EvaluationConfig, ASR_METRICS


class ASRMetrics:
    
    @staticmethod
    def calculate_cer(reference: str, hypothesis: str) -> float:
        if not reference:
            return 1.0 if hypothesis else 0.0
        
        distance = ASRMetrics._levenshtein_distance(reference, hypothesis)
        return distance / len(reference)
    
    @staticmethod
    def calculate_wer(reference: List[str], hypothesis: List[str]) -> float:
        if not reference:
            return 1.0 if hypothesis else 0.0
        
        distance = ASRMetrics._levenshtein_distance(reference, hypothesis)
        return distance / len(reference)
    
    @staticmethod
    def _levenshtein_distance(s1, s2) -> int:
        if isinstance(s1, list):
            s1 = s1
        if isinstance(s2, list):
            s2 = s2
            
        if len(s1) < len(s2):
            return ASRMetrics._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]


class ASREvaluator:
    
    def __init__(self, annotation_dir: str = 'data/annotations', results_dir: str = 'results'):
        self.annotation_dir = Path(annotation_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = ASRMetrics()
    
    def evaluate(self, case_name: str) -> Dict[str, Any]:
        asr_file = self.annotation_dir / f'{case_name}_asr_ground_truth.json'
        if not asr_file.exists():
            print(f"⚠ ASR file not found: {asr_file}")
            return None
        
        with open(asr_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        
        if not samples:
            return None
        
        cer_scores = []
        wer_scores = []
        
        for sample in samples:
            reference = sample.get('reference_text', '')
            hypothesis = sample.get('hypothesis_text', '')
            
            cer = self.metrics.calculate_cer(reference, hypothesis)
            wer = self.metrics.calculate_wer(
                reference.split(), 
                hypothesis.split()
            )
            
            cer_scores.append(cer)
            wer_scores.append(wer)
        
        result = {
            'case': case_name,
            'num_samples': len(samples),
            'CER': {
                'avg': round(mean(cer_scores), 4),
                'min': round(min(cer_scores), 4),
                'max': round(max(cer_scores), 4),
                'std': round(stdev(cer_scores), 4) if len(cer_scores) > 1 else 0.0
            },
            'WER': {
                'avg': round(mean(wer_scores), 4),
                'min': round(min(wer_scores), 4),
                'max': round(max(wer_scores), 4),
                'std': round(stdev(wer_scores), 4) if len(wer_scores) > 1 else 0.0
            }
        }
        
        return result
    
    def evaluate_all(self) -> Dict[str, Any]:
        results = {
            'system': 'ASR_Evaluation',
            'metrics': ASR_METRICS['metrics'],
            'metric_definitions': ASR_METRICS['description'],
            'cases': {}
        }
        
        for case in EvaluationConfig.CASE_NAMES:
            case_result = self.evaluate(case)
            if case_result:
                results['cases'][case] = case_result
        
        output_file = self.results_dir / 'asr_performance.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ ASR evaluation saved to {output_file}")
        return results


def main():
    evaluator = ASREvaluator()
    evaluator.evaluate_all()


if __name__ == '__main__':
    main()

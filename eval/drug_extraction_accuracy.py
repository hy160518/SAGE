import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

from config import EvaluationConfig, DRUG_METRICS


class DrugExtractionMetrics:
    
    @staticmethod
    def calculate_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1-Score': round(f1, 4),
            'Support': tp + fn
        }


class DrugExtractionEvaluator:
    
    def __init__(self, annotation_dir: str = 'data/annotations', results_dir: str = 'results'):
        self.annotation_dir = Path(annotation_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = DrugExtractionMetrics()
    
    def evaluate(self, case_name: str) -> Dict[str, Any]:
        image_file = self.annotation_dir / f'{case_name}_image_annotations.json'
        if not image_file.exists():
            print(f"⚠ Image annotation file not found: {image_file}")
            return None
        
        with open(image_file, 'r', encoding='utf-8') as f:
            images = json.load(f)
        
        if not images:
            return None
        
        tp, fp, fn = 0, 0, 0
        by_type = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        for img in images:
            true_label = img.get('contains_drugs', False)
            pred_label = img.get('predicted_contains_drugs', False)
            img_type = img.get('image_type', 'unknown')
            
            if true_label and pred_label:
                tp_i, fp_i, fn_i = 1, 0, 0
            elif true_label and not pred_label:
                tp_i, fp_i, fn_i = 0, 0, 1
            elif not true_label and pred_label:
                tp_i, fp_i, fn_i = 0, 1, 0
            else:
                tp_i, fp_i, fn_i = 0, 0, 0
            
            tp += tp_i
            fp += fp_i
            fn += fn_i
            
            by_type[img_type]['tp'] += tp_i
            by_type[img_type]['fp'] += fp_i
            by_type[img_type]['fn'] += fn_i
        
        result = {
            'case': case_name,
            'total_images': len(images),
            'overall': self.metrics.calculate_metrics(tp, fp, fn),
            'by_type': {}
        }
        
        for img_type, stats in by_type.items():
            result['by_type'][img_type] = {
                'count': stats['tp'] + stats['fn'],
                'metrics': self.metrics.calculate_metrics(stats['tp'], stats['fp'], stats['fn'])
            }
        
        return result
    
    def evaluate_all(self) -> Dict[str, Any]:
        results = {
            'task': 'Drug_Extraction_from_Images',
            'metrics': DRUG_METRICS['metrics'],
            'metric_definitions': DRUG_METRICS['description'],
            'image_types': DRUG_METRICS['image_types'],
            'cases': {}
        }
        
        for case in EvaluationConfig.CASE_NAMES:
            case_result = self.evaluate(case)
            if case_result:
                results['cases'][case] = case_result
        
        output_file = self.results_dir / 'drug_extraction_accuracy.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Drug extraction evaluation saved to {output_file}")
        return results


def main():
    evaluator = DrugExtractionEvaluator()
    evaluator.evaluate_all()


if __name__ == '__main__':
    main()

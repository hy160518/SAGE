import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

from config import EvaluationConfig, DOWNSTREAM_METRICS



class DownstreamMetrics:
    
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


class RelationshipMetrics:
    
    @staticmethod
    def calculate_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'support': tp + fn
        }


class DownstreamEvaluator:

    def __init__(self, annotation_dir: str = 'data/annotations', results_dir: str = 'results'):
        self.annotation_dir = Path(annotation_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = RelationshipMetrics()
    
    def evaluate(self, case_name: str) -> Dict[str, Any]:
        annotations = self._load_annotations(case_name)
        if not annotations:
            return None
        
        task1_results = self._task_user_role_identification(annotations)
        
        task2_results = self._task_relationship_extraction(annotations)
        
        return {
            'case': case_name,
            'task1_user_role': task1_results,
            'task2_relationship': task2_results
        }
    
    def _load_annotations(self, case_name: str) -> Dict[str, Any]:
        files = {
            'user_roles': f'{case_name}_user_roles.json',
            'relationships': f'{case_name}_user_drug_relations.json',
            'messages': f'{case_name}_chat_messages.json'
        }
        
        annotations = {}
        for key, filename in files.items():
            filepath = self.annotation_dir / filename
            if not filepath.exists():
                print(f"⚠ {key} file not found: {filepath}")
                return None
            
            with open(filepath, 'r', encoding='utf-8') as f:
                annotations[key] = json.load(f)
        
        return annotations
    
    def _task_user_role_identification(self, annotations: Dict[str, Any]) -> Dict[str, Any]:
        user_roles_gt = annotations['user_roles']
        metrics = DownstreamMetrics()
        
        results = {}
        for role in DOWNSTREAM_METRICS['task1_roles']:
            tp = sum(1 for u in user_roles_gt if u.get('predicted_role') == role and u.get('true_role') == role)
            fp = sum(1 for u in user_roles_gt if u.get('predicted_role') == role and u.get('true_role') != role)
            fn = sum(1 for u in user_roles_gt if u.get('predicted_role') != role and u.get('true_role') == role)
            
            results[role] = metrics.calculate_metrics(tp, fp, fn)
        
        f1_scores = [results[role]['F1-Score'] for role in DOWNSTREAM_METRICS['task1_roles']]
        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        
        return {
            'by_role': results,
            'macro_f1': round(macro_f1, 4)
        }
    
    def _task_relationship_extraction(self, annotations: Dict[str, Any]) -> Dict[str, Any]:
        relationships_gt = annotations['relationships']
        metrics = DownstreamMetrics()
        
        tp = sum(1 for rel in relationships_gt if rel.get('is_related') and rel.get('predicted_related'))
        fp = sum(1 for rel in relationships_gt if not rel.get('is_related') and rel.get('predicted_related'))
        fn = sum(1 for rel in relationships_gt if rel.get('is_related') and not rel.get('predicted_related'))
        
        return metrics.calculate_metrics(tp, fp, fn)
    
    def evaluate_all(self) -> Dict[str, Any]:
        all_results = {
            'task1_name': DOWNSTREAM_METRICS['task1_name'],
            'task1_description': DOWNSTREAM_METRICS['description']['task1'],
            'task1_metrics': DOWNSTREAM_METRICS['task1_metrics'],
            'task2_name': DOWNSTREAM_METRICS['task2_name'],
            'task2_description': DOWNSTREAM_METRICS['description']['task2'],
            'task2_metrics': DOWNSTREAM_METRICS['task2_metrics'],
            'cases': {}
        }
        
        for case in EvaluationConfig.CASE_NAMES:
            result = self.evaluate(case)
            if result:
                all_results['cases'][case] = result
        
        output_file = self.results_dir / 'downstream_task_metrics.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Downstream evaluation saved to {output_file}")
        return all_results


def main():
    evaluator = DownstreamEvaluator()
    evaluator.evaluate_all()


if __name__ == '__main__':
    main()
from typing import Dict, List, Any


class EvaluationConfig:
    
    CASE_NAMES: List[str] = ['Case_A', 'Case_B']
    
    DEFAULT_ANNOTATION_DIR = 'data/annotations'
    
    DEFAULT_RESULTS_DIR = 'results'
    
    FILE_PATTERNS: Dict[str, str] = {
        'asr': '{case_name}_asr_ground_truth.json',
        'image': '{case_name}_image_annotations.json',
        'user_roles': '{case_name}_user_roles.json',
        'user_relations': '{case_name}_user_drug_relations.json',
        'chat_messages': '{case_name}_chat_messages.json',
    }
    
    ASR_METRICS: Dict[str, Any] = {
        'metrics': ['CER', 'WER'],
        'description': {
            'CER': 'Character Error Rate: (S+D+I)/N where S=substitutions, D=deletions, I=insertions',
            'WER': 'Word Error Rate: applies edit distance at word level'
        }
    }
    
    DRUG_EXTRACTION_METRICS: Dict[str, Any] = {
        'metrics': ['Precision', 'Recall', 'F1-Score'],
        'description': {
            'Precision': 'TP / (TP + FP)',
            'Recall': 'TP / (TP + FN)',
            'F1-Score': '2 * (Precision * Recall) / (Precision + Recall)'
        },
        'image_types': ['prescription', 'label', 'packaging', 'other']
    }
    
    DOWNSTREAM_METRICS: Dict[str, Any] = {
        'task1_name': 'User-Role Identification',
        'task1_roles': ['prescriber', 'recipient', 'benign_user'],
        'task1_metrics': ['Precision', 'Recall', 'F1-Score', 'Macro-F1'],
        'task2_name': 'User-Drug Relationship Extraction',
        'task2_metrics': ['Precision', 'Recall', 'F1-Score'],
        'description': {
            'task1': 'Identify user roles based on linguistic and behavioral patterns in forensic data',
            'task2': 'Extract user pairs with drug-related interactions from multimodal evidence'
        }
    }
    
    OUTPUT_FILES: Dict[str, str] = {
        'downstream': 'downstream_task_metrics.json',
        'asr': 'asr_performance.json',
        'drug_extraction': 'drug_extraction_accuracy.json',
        'summary': 'evaluation_summary.json'
    }
    
    @classmethod
    def get_file_pattern(cls, pattern_key: str) -> str:
        return cls.FILE_PATTERNS.get(pattern_key, '')
    
    @classmethod
    def get_output_file(cls, output_key: str) -> str:
        return cls.OUTPUT_FILES.get(output_key, '')
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        return {
            'case_names': cls.CASE_NAMES,
            'annotation_dir': cls.DEFAULT_ANNOTATION_DIR,
            'results_dir': cls.DEFAULT_RESULTS_DIR,
            'file_patterns': cls.FILE_PATTERNS,
            'asr_metrics': cls.ASR_METRICS,
            'drug_extraction_metrics': cls.DRUG_EXTRACTION_METRICS,
            'downstream_metrics': cls.DOWNSTREAM_METRICS,
            'output_files': cls.OUTPUT_FILES
        }


CASES = EvaluationConfig.CASE_NAMES
ASR_METRICS = EvaluationConfig.ASR_METRICS
DRUG_METRICS = EvaluationConfig.DRUG_EXTRACTION_METRICS
DOWNSTREAM_METRICS = EvaluationConfig.DOWNSTREAM_METRICS

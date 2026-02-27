import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

class ContextManager:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_dir = os.path.join(
            os.path.dirname(__file__),
            '..',
            'data',
            'context'
        )
        
        os.makedirs(self.base_dir, exist_ok=True)
        
        self.examples = {
            'drug_extraction': [],
            'entity_extraction': [],
            'voice_quality': [],
            'image_classification': []
        }
        
        self.history = {
            'successful_cases': [],
            'failed_cases': []
        }
        

        self._load_examples()
        

        self._load_history()
    
    def _load_examples(self):

        examples_file = os.path.join(self.base_dir, 'examples.json')
        
        if os.path.exists(examples_file):
            try:
                with open(examples_file, 'r', encoding='utf-8') as f:
                    self.examples = json.load(f)
            except Exception as e:
                print(f"failed to load the sample library: {str(e)}")
        else:

            self.examples = self._get_default_examples()
            self._save_examples()
    
    def _load_history(self):
        history_file = os.path.join(self.base_dir, 'history.json')
        
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
                print(f"✓ loaded history: {len(self.history['successful_cases'])} successful cases, "
                      f"{len(self.history['failed_cases'])} failed cases")
            except Exception as e:
                print(f"failed to load history: {str(e)}")
    
    def _save_examples(self):
        examples_file = os.path.join(self.base_dir, 'examples.json')
        
        try:
            with open(examples_file, 'w', encoding='utf-8') as f:
                json.dump(self.examples, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"failed to save examples: {str(e)}")
    
    def _save_history(self):
        history_file = os.path.join(self.base_dir, 'history.json')
        
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"failed to save history: {str(e)}")
    
    def get_few_shot_examples(self, task_type: str, num_examples: int = 3) -> List[Dict]:

        examples = self.examples.get(task_type, [])
        if len(examples) < num_examples:
            historical_examples = self._extract_examples_from_history(task_type)
            examples.extend(historical_examples)
        
        return examples[:num_examples]
    
    def add_successful_case(self, case: Dict[str, Any]):
        case['timestamp'] = datetime.now().isoformat()
        case['success'] = True
        
        self.history['successful_cases'].append(case)
        
        if len(self.history['successful_cases']) % 10 == 0:
            self._extract_examples_from_successful_cases()
        if len(self.history['successful_cases']) % 5 == 0:
            self._save_history()
    
    def add_failed_case(self, case: Dict[str, Any]):

        case['timestamp'] = datetime.now().isoformat()
        case['success'] = False
        
        self.history['failed_cases'].append(case)

        self._analyze_failure(case)

        if len(self.history['failed_cases']) % 5 == 0:
            self._save_history()
    
    def build_context(self, task_type: str, current_input: str) -> str:

        examples = self.get_few_shot_examples(task_type)
        
        if not examples:
            return current_input

        few_shot_text = "\n examples:\n"
        
        for i, example in enumerate(examples, 1):
            input_text = example.get('input', '')
            output_text = example.get('output', '')
            
            few_shot_text += f"\nExample {i}:\n"
            few_shot_text += f"input: {input_text}\n"
            few_shot_text += f"output: {output_text}\n"
        
        full_context = few_shot_text + f"\nNow please process the following input:\n{current_input}"
        
        return full_context
    
    def _extract_examples_from_history(self, task_type: str, max_examples: int = 5) -> List[Dict]:

        relevant_cases = [
            case for case in self.history['successful_cases']
            if case.get('task_type') == task_type and case.get('confidence', 0) >= 0.8
        ]
        
        relevant_cases.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        

        examples = []
        for case in relevant_cases[:max_examples]:
            example = {
                'input': case.get('input', ''),
                'output': case.get('output', ''),
                'confidence': case.get('confidence', 0)
            }
            examples.append(example)
        
        return examples
    
    def _extract_examples_from_successful_cases(self):
        
        for task_type in self.examples.keys():
            new_examples = self._extract_examples_from_history(task_type, max_examples=3)
            
            if new_examples:
                existing = self.examples[task_type]
                
                for new_ex in new_examples:
                    if not any(ex.get('input') == new_ex.get('input') for ex in existing):
                        existing.append(new_ex)
                
                existing.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                self.examples[task_type] = existing[:10]
        
        self._save_examples()
        print("Example library update completed")
    
    def _analyze_failure(self, case: Dict[str, Any]):

        failure_type = case.get('error_type', 'unknown')
        print(f"Record failed case: {failure_type}")

    def _get_default_examples(self) -> Dict[str, List[Dict]]:

        return {
            'drug_extraction': [
                {
                    'input': 'Chat records:\n[uuid001] Zhang San: Do you still have stock?\n[uuid001] Zhang San: The last batch of Magu was of good quality',
                    'output': '{"uuid001": "Drug1", "uuid001": "Drug1"}',
                    'confidence': 1.0
                },
                {
                    'input': 'Chat records:\n[uuid003] Wang Wu: Do you still have Drug2?\n[uuid003] Wang Wu: Give me some',
                    'output': '{"uuid003": "Drug2", "uuid003": "Drug2"}',
                    'confidence': 1.0
                }
            ],
            'entity_extraction': [
                {
                    'input': 'Zhang San sold 50 grams of Drug1 to Li Si at a hotel parking lot in XX District, XX City on the evening of March 15, 2023, for 5000 yuan.',
                    'output': json.dumps({
                        'persons': [
                            {'name': 'Zhangsan', 'role': 'Recipient'},
                            {'name': 'lisi', 'role': 'Prescriber'}
                        ],
                        'drugs': [{'name': 'Dru1', 'alias': ['Drug1'], 'quantity': '50 grams'}],
                        'times': [{'type': 'Time', 'time': 'March 15, 2023, evening'}],
                        'amounts': [{'value': '5000 yuan', 'type': 'Transaction Amount'}]
                    }, ensure_ascii=False),
                    'confidence': 1.0
                }
            ],
            'voice_quality': [],
            'image_classification': []
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'total_examples': sum(len(examples) for examples in self.examples.values()),
            'examples_by_type': {k: len(v) for k, v in self.examples.items()},
            'successful_cases': len(self.history['successful_cases']),
            'failed_cases': len(self.history['failed_cases']),
            'success_rate': self._calculate_success_rate()
        }
    
    def _calculate_success_rate(self) -> float:
        total = len(self.history['successful_cases']) + len(self.history['failed_cases'])
        if total == 0:
            return 0.0
        return len(self.history['successful_cases']) / total

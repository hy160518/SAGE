import json
import os
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from dashscope import Generation
import dashscope

from ..utils.retry_handler import smart_retry

class EnsembleConfig:
    def __init__(self):
        self.models = ['qwen-max', 'qwen-turbo']  
        self.voting_strategy = 'weighted' 
        self.weights = {'qwen-max': 0.7, 'qwen-turbo': 0.3}
        self.min_agreement = 0.6  

class AdaptiveParams:
    def __init__(self):
        self.base_temperature = 0.1
        self.base_top_p = 0.95
        self.adjustment_history = []
        self.success_rate = 1.0
    
    def adjust_for_retry(self, retry_count: int) -> Tuple[float, float]:

        if retry_count == 1:
            return self.base_temperature * 0.5, self.base_top_p * 0.9
        elif retry_count == 2:
            return self.base_temperature * 1.5, self.base_top_p * 1.05
        else:
            return self.base_temperature * 2.0, min(self.base_top_p * 1.1, 0.99)
    
    def adjust_for_quality(self, quality_score: float) -> Tuple[float, float]:
        if quality_score < 0.5:
            return self.base_temperature * 0.7, self.base_top_p * 0.9
        elif quality_score > 0.9:
            return self.base_temperature, self.base_top_p
        else:
            return self.base_temperature, self.base_top_p

class TextHandler:
    BATCH_SIZE = 30
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get('dashscope', {}).get('api_key')
        dashscope.api_key = self.api_key
        self.model = config.get('models', {}).get('text_model', 'qwen-max')
        self.temperature = config.get('parameters', {}).get('temperature', 0.1)
        self.top_p = config.get('parameters', {}).get('top_p', 0.95)
        self.prompts = self._load_prompts()
        self.few_shot_examples = self._load_few_shot_examples()
        
        self.enable_ensemble = config.get('enable_ensemble', False)
        self.ensemble_config = EnsembleConfig()
        self.adaptive_params = AdaptiveParams()
        self.quality_threshold = config.get('quality_threshold', 0.6)
        
        self.stats = {
            'total_processed': 0,
            'ensemble_used': 0,
            'adaptive_adjustments': 0,
            'quality_scores': []
        }
        
    def _load_prompts(self) -> Dict[str, str]:
        prompts = {}
        prompt_dir = os.path.join(os.path.dirname(__file__), '..', 'prompts')
        drug_prompt_path = os.path.join(prompt_dir, 'drug_extraction.txt')
        if os.path.exists(drug_prompt_path):
            with open(drug_prompt_path, 'r', encoding='utf-8') as f:
                prompts['drug_extraction'] = f.read()
        entity_prompt_path = os.path.join(prompt_dir, 'entity_extraction.txt')
        if os.path.exists(entity_prompt_path):
            with open(entity_prompt_path, 'r', encoding='utf-8') as f:
                prompts['entity_extraction'] = f.read()
        return prompts
    
    def _load_few_shot_examples(self) -> Dict[str, List[Dict]]:
        return {}
    
    def process_batch(self, text_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for i in range(0, len(text_entries), self.BATCH_SIZE):
            batch = text_entries[i:i + self.BATCH_SIZE]
            batch_results = self._process_single_batch(batch)
            results.extend(batch_results)
        return results
    
    def _process_single_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for entry in batch:
            try:
                drugs = self.extract_drugs(entry)
                entities = self.extract_entities(entry)
                validated = self._validate_result(drugs, entities)
                result = {
                    'uuid': entry.get('uuid'),
                    'content': entry.get('content'),
                    'drugs': drugs,
                    'entities': entities,
                    'validated': validated,
                    'confidence': self._calculate_confidence(drugs, entities)
                }
                results.append(result)
            except Exception as e:
                results.append({
                    'uuid': entry.get('uuid'),
                    'error': str(e),
                    'confidence': 0.0
                })
        return results
    
    def extract_drugs(self, text_entry: Dict[str, Any], model_override: Optional[str] = None, params_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        content = text_entry.get('content', '')
        uuid = text_entry.get('uuid', '')
        chat_text = f"[{uuid}] {content}"
        system_prompt = self.prompts.get('drug_extraction', '')
        few_shot_text = ""
        examples = self.few_shot_examples.get('drug_extraction') if isinstance(self.few_shot_examples, dict) else None
        if examples:
            few_shot_text = "\n\n" + "\n".join([f"{ex.get('input','')}\n{ex.get('output','')}" for ex in examples])
        full_prompt = system_prompt + few_shot_text + f"\n\n{chat_text}"
        
        self.stats['total_processed'] += 1
        
        try:
            if self.enable_ensemble and not model_override:
                result = self._extract_with_ensemble(full_prompt, 'drug')
                self.stats['ensemble_used'] += 1
                return result
            
            ov_model = model_override or self.model
            ov_temp = (params_override or {}).get('temperature', self.temperature)
            ov_top_p = (params_override or {}).get('top_p', self.top_p)
            
            result_text = self._call_api_with_adaptive_retry(
                full_prompt, model=ov_model, 
                temperature=ov_temp, top_p=ov_top_p
            )
            
            if result_text:
                drugs = self._parse_drug_json(result_text)
                quality = self._assess_extraction_quality(drugs, content)
                self.stats['quality_scores'].append(quality)
                
                return {
                    'drugs': drugs,
                    'quality_score': quality,
                    'model': ov_model
                }
            else:
                return {'drugs': {}, 'quality_score': 0.0}
        except Exception as e:
            return {'drugs': {}, 'error': str(e), 'quality_score': 0.0}
    
    def extract_entities(self, text_entry: Dict[str, Any], model_override: Optional[str] = None, params_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        content = text_entry.get('content', '')
        system_prompt = self.prompts.get('entity_extraction', '')
        full_prompt = system_prompt + f"\n\n{content}"
        try:
            ov_model = model_override or self.model
            ov_temp = (params_override or {}).get('temperature', self.temperature)
            ov_top_p = (params_override or {}).get('top_p', self.top_p)
            result_text = self._call_api_with_retry(full_prompt, model=ov_model, temperature=ov_temp, top_p=ov_top_p)
            if result_text:
                entities = self._parse_entity_json(result_text)
                return entities
            else:
                return {}
        except Exception:
            return {}
    
    def _parse_drug_json(self, text: str) -> Dict[str, str]:
        try:
            if '{' in text and '}' in text:
                json_start = text.index('{')
                json_end = text.rindex('}') + 1
                json_text = text[json_start:json_end]
                drugs = json.loads(json_text)
                if isinstance(drugs, dict):
                    return drugs
            return {}
        except json.JSONDecodeError:
            return {}
    
    def _parse_entity_json(self, text: str) -> Dict[str, Any]:
        try:
            if '{' in text and '}' in text:
                json_start = text.index('{')
                json_end = text.rindex('}') + 1
                json_text = text[json_start:json_end]
                entities = json.loads(json_text)
                return entities
            else:
                return {}
        except json.JSONDecodeError:
            return {}
    
    def _validate_result(self, drugs: Dict, entities: Dict) -> bool:
        if not isinstance(drugs, dict) or not isinstance(entities, dict):
            return False
        if drugs and 'drugs' in entities:
            entity_drugs = entities.get('drugs', [])
            if not entity_drugs:
                return False
        return True
    
    def _calculate_confidence(self, drugs: Dict, entities: Dict) -> float:
        confidence = 0.5
        if drugs:
            confidence += 0.2
        if entities:
            confidence += 0.1
        if self._validate_result(drugs, entities):
            confidence += 0.2
        return min(confidence, 1.0)

    def normalize_behavior(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        import re
        content_str = str(content)
        sender = metadata.get('sender', 'A')
        receiver = metadata.get('receiver', 'B')

        if 'transfer' in content_str.lower() or 'zhuanzhang' in content_str.lower():
            amount_match = re.search(r'(\d+(?:\.\d+)?)\s*元?', content_str)
            amount = amount_match.group(1) if amount_match else '0'
            status = 'received' if 'received' in content_str.lower() or 'yishoukuank' in content_str.lower() else ('returned' if 'returned' in content_str.lower() else 'pending')
            return {
                'type': 'transfer',
                'normalized': f"transfer: {amount} CNY from {sender} to {receiver}, {status}",
                'entities': {'amount': amount, 'recipient': receiver, 'status': status}
            }

        if 'red envelope' in content_str.lower() or 'hongbao' in content_str.lower():
            amount_match = re.search(r'(\d+(?:\.\d+)?)\s*元?', content_str)
            amount = amount_match.group(1) if amount_match else '0'
            status = 'expired' if 'expired' in content_str.lower() else 'opened'
            return {
                'type': 'red_envelope',
                'normalized': f"red_envelope: {amount} CNY from {sender} to {receiver}, {status}",
                'entities': {'amount': amount, 'status': status}
            }

        if 'voice call' in content_str.lower() or 'yuyin' in content_str.lower():
            duration_match = re.search(r'(\d+)\s*秒', content_str)
            duration = duration_match.group(1) if duration_match else '0'
            return {
                'type': 'call_voice',
                'normalized': f"call_voice: {duration} s between {sender} and {receiver}",
                'entities': {'duration': duration}
            }

        if 'video call' in content_str.lower() or 'shipin' in content_str.lower():
            duration_match = re.search(r'(\d+)\s*秒', content_str)
            duration = duration_match.group(1) if duration_match else '0'
            return {
                'type': 'call_video',
                'normalized': f"call_video: {duration} s between {sender} and {receiver}",
                'entities': {'duration': duration}
            }

        if 'add friend' in content_str.lower() or 'tianjiahaoyou' in content_str.lower():
            return {
                'type': 'add_contact',
                'normalized': f"add_contact: @{receiver}",
                'entities': {'contact': receiver}
            }

        if 'location' in content_str.lower() or 'weizhi' in content_str.lower():
            location_match = re.search(r'([^\s]{2,})', content_str)
            location = location_match.group(1) if location_match else 'unknown'
            return {
                'type': 'location',
                'normalized': f"location: {location}",
                'entities': {'location': location}
            }

        return {
            'type': 'message',
            'normalized': content_str[:200],
            'entities': {}
        }

    def _extract_with_ensemble(self, prompt: str, task_type: str) -> Dict[str, Any]:
        results = []
        
        for model in self.ensemble_config.models:
            try:
                result_text = self._call_api_with_retry(
                    prompt, model=model,
                    temperature=self.temperature,
                    top_p=self.top_p
                )
                
                if task_type == 'drug':
                    parsed = self._parse_drug_json(result_text)
                else:
                    parsed = self._parse_entity_json(result_text)
                
                results.append({
                    'model': model,
                    'result': parsed,
                    'weight': self.ensemble_config.weights.get(model, 1.0)
                })
            except Exception as e:
                print(f"Model {model} failed: {str(e)}")
        
        merged = self._merge_ensemble_results(results, task_type)
        return merged
    
    def _merge_ensemble_results(self, results: List[Dict], task_type: str) -> Dict[str, Any]:
        if not results:
            return {'drugs': {}} if task_type == 'drug' else {'entities': {}}
        
        if len(results) == 1:
            return {'drugs': results[0]['result']} if task_type == 'drug' else {'entities': results[0]['result']}
        
        if self.ensemble_config.voting_strategy == 'weighted':
            best = max(results, key=lambda x: x['weight'])
            return {
                'drugs' if task_type == 'drug' else 'entities': best['result'],
                'ensemble_info': {
                    'models_used': [r['model'] for r in results],
                    'selected_model': best['model']
                }
            }
        
        return {'drugs': {}} if task_type == 'drug' else {'entities': {}}
    
    def _call_api_with_adaptive_retry(self, prompt: str, model: Optional[str] = None, 
                                     temperature: Optional[float] = None, 
                                     top_p: Optional[float] = None,
                                     max_retries: int = 3) -> str:
        last_error = None
        
        for retry_count in range(max_retries):
            try:
                if retry_count > 0:
                    adj_temp, adj_top_p = self.adaptive_params.adjust_for_retry(retry_count)
                    self.stats['adaptive_adjustments'] += 1
                else:
                    adj_temp = temperature if temperature is not None else self.temperature
                    adj_top_p = top_p if top_p is not None else self.top_p
                
                result = self._call_api_with_retry(
                    prompt, model=model,
                    temperature=adj_temp,
                    top_p=adj_top_p
                )
                return result
                
            except Exception as e:
                last_error = e
                if retry_count < max_retries - 1:
                    print(f"Retry {retry_count + 1}/{max_retries}: {str(e)}")
                continue
        
        raise last_error if last_error else Exception("All retries failed")
    
    def _assess_extraction_quality(self, extracted: Dict, original_text: str) -> float:
        if not extracted:
            return 0.0
        
        score = 0.5  
        
        if len(extracted) > 0:
            score += 0.2
        if len(extracted) >= 3:
            score += 0.1
        
        for key, value in extracted.items():
            if isinstance(value, str) and len(value) > 0:
                score += 0.05
        
        return min(score, 1.0)
    
    @smart_retry
    def _call_api_with_retry(self, prompt: str, model: Optional[str] = None, temperature: Optional[float] = None, top_p: Optional[float] = None) -> str:
        response = Generation.call(
            model=model or self.model,
            prompt=prompt,
            temperature=self.temperature if temperature is None else temperature,
            top_p=self.top_p if top_p is None else top_p
        )
        if response.status_code == 200:
            return response.output.text.strip()
        else:
            raise Exception(f"API failed: status={response.status_code}, msg={response.message}")
    
    def get_statistics(self) -> Dict:
        avg_quality = sum(self.stats['quality_scores']) / len(self.stats['quality_scores']) if self.stats['quality_scores'] else 0.0
        
        return {
            'total_processed': self.stats['total_processed'],
            'ensemble_used': self.stats['ensemble_used'],
            'adaptive_adjustments': self.stats['adaptive_adjustments'],
            'average_quality': avg_quality,
            'quality_distribution': {
                'high (>0.8)': sum(1 for q in self.stats['quality_scores'] if q > 0.8),
                'medium (0.5-0.8)': sum(1 for q in self.stats['quality_scores'] if 0.5 <= q <= 0.8),
                'low (<0.5)': sum(1 for q in self.stats['quality_scores'] if q < 0.5)
            }
        }

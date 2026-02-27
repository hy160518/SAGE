import json
import os
from typing import List, Dict, Any, Optional
from dashscope import Audio, Generation
import dashscope

class VoiceHandler:
    BATCH_SIZE = 30
    WHISPER_PARAMS = {
        'beam_size': 5,
        'language': 'zh',
        'temperature': 0.0,
        'initial_prompt': 'Output simplified Chinese with punctuation',
        'n_threads': 8,
    }
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get('dashscope', {}).get('api_key')
        dashscope.api_key = self.api_key
        self.transcription_model = config.get('models', {}).get('voice_model', 'paraformer-v1')
        self.llm_model = config.get('models', {}).get('text_model', 'qwen-max')
        self.quality_prompt = self._load_quality_prompt()
        
    def _load_quality_prompt(self) -> str:
        prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'voice_quality.txt')
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        return ""
    
    def process_batch(self, voice_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for i in range(0, len(voice_entries), self.BATCH_SIZE):
            batch = voice_entries[i:i + self.BATCH_SIZE]
            batch_results = self._process_single_batch(batch)
            results.extend(batch_results)
        return results
    
    def _process_single_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for entry in batch:
            try:
                transcription = self.transcribe_audio(entry)
                quality_assessment = self.assess_quality(transcription)
                result = {
                    'uuid': entry.get('uuid'),
                    'file_path': entry.get('file_path'),
                    'transcription': transcription,
                    'quality': quality_assessment,
                    'confidence': quality_assessment.get('quality_score', 0.0),
                    'usable': quality_assessment.get('quality_score', 0.0) >= 0.7
                }
                results.append(result)
            except Exception as e:
                results.append({
                    'uuid': entry.get('uuid'),
                    'error': str(e),
                    'confidence': 0.0,
                    'usable': False
                })
        return results
    
    def transcribe_audio(self, voice_entry: Dict[str, Any], model_override: Optional[str] = None, params_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        file_path = voice_entry.get('file_path')
        if not file_path or not os.path.exists(file_path):
            return {'success': False, 'error': 'File not found', 'text': ''}
        try:
            effective_model = model_override or self.transcription_model
            extra_kwargs = params_override or {}
            if 'language' not in extra_kwargs:
                extra_kwargs['language'] = 'zh'

            response = Audio.call(
                model=effective_model,
                file=file_path,
                **extra_kwargs
            )
            if response.status_code == 200:
                transcription_text = self._extract_transcription_text(response)
                return {
                    'success': True,
                    'text': transcription_text,
                    'duration': voice_entry.get('duration', 0),
                    'model': effective_model,
                    'parameters': self.WHISPER_PARAMS
                }
            else:
                return {'success': False, 'error': f"API failed: {response.message}", 'text': ''}
        except Exception as e:
            return {'success': False, 'error': f"Transcription error: {str(e)}", 'text': ''}
    
    def _extract_transcription_text(self, response) -> str:
        try:
            if hasattr(response, 'output') and hasattr(response.output, 'text'):
                return response.output.text
            elif hasattr(response, 'output') and isinstance(response.output, dict):
                return response.output.get('text', '')
            else:
                return str(response.output)
        except:
            return ''
    
    def assess_quality(self, transcription: Dict[str, Any], llm_model_override: Optional[str] = None, params_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        text = transcription.get('text', '')
        if not text:
            return {
                'quality_score': 0.0,
                'completeness': 'invalid',
                'coherence': 'invalid',
                'value_level': 'none',
                'issues': ['empty transcription'],
                'recommendation': 'unusable'
            }
        full_prompt = self.quality_prompt + f"\n\n{text}"
        try:
            ov_model = llm_model_override or self.llm_model
            ov_temp = (params_override or {}).get('temperature', 0.1)
            ov_top_p = (params_override or {}).get('top_p', 0.95)

            response = Generation.call(
                model=ov_model,
                prompt=full_prompt,
                temperature=ov_temp,
                top_p=ov_top_p
            )
            if response.status_code == 200:
                result_text = response.output.text.strip()
                quality_result = self._parse_quality_json(result_text)
                return quality_result
            else:
                return self._default_quality_assessment(text)
        except Exception:
            return self._default_quality_assessment(text)
    
    def _parse_quality_json(self, text: str) -> Dict[str, Any]:
        try:
            if '{' in text and '}' in text:
                json_start = text.index('{')
                json_end = text.rindex('}') + 1
                json_text = text[json_start:json_end]
                quality = json.loads(json_text)
                return quality
            else:
                return self._default_quality_assessment("")
        except json.JSONDecodeError:
            return self._default_quality_assessment("")
    
    def _default_quality_assessment(self, text: str) -> Dict[str, Any]:
        if not text:
            score = 0.0
            completeness = 'invalid'
        elif len(text) < 10:
            score = 0.3
            completeness = 'poor'
        elif len(text) < 50:
            score = 0.6
            completeness = 'fair'
        else:
            score = 0.8
            completeness = 'good'
        return {
            'quality_score': score,
            'completeness': completeness,
            'coherence': 'unassessed',
            'value_level': 'medium' if score >= 0.6 else 'low',
            'issues': ['rule-based fallback'],
            'recommendation': 'manual review' if score < 0.7 else 'usable'
        }
    
    def validate_whisper_params(self) -> Dict[str, Any]:
        expected = {
            'beam_size': 5,
            'language': 'zh',
            'temperature': 0.0,
            'initial_prompt': 'Output simplified Chinese with punctuation',
            'n_threads': 8
        }
        validation = {
            'expected': expected,
            'current': self.WHISPER_PARAMS,
            'matches': expected == self.WHISPER_PARAMS,
            'differences': []
        }
        for key, expected_value in expected.items():
            current_value = self.WHISPER_PARAMS.get(key)
            if current_value != expected_value:
                validation['differences'].append({
                    'parameter': key,
                    'expected': expected_value,
                    'current': current_value
                })
        return validation

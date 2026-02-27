import os
from typing import Dict, Any, Optional
from dashscope import MultiModalConversation
import dashscope

class ImageHandler:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get('dashscope', {}).get('api_key')
        dashscope.api_key = self.api_key
        self.model = config.get('models', {}).get('image_model', 'qwen-vl-plus')
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, str]:
        prompts = {}
        prompt_dir = os.path.join(os.path.dirname(__file__), '..', 'prompts')
        classification_path = os.path.join(prompt_dir, 'image_classification.txt')
        if os.path.exists(classification_path):
            with open(classification_path, 'r', encoding='utf-8') as f:
                prompts['classification'] = f.read()
        return prompts
    
    def process_image(self, image_path: str, qwen_client=None) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            return {'error': 'File not found', 'path': image_path}
        try:
            if qwen_client and hasattr(qwen_client, 'classify_with_voting'):
                classification_prompts = [
                    self.prompts.get('classification', 'Classify this image'),
                    "Is this image drug-related?",
                    "Determine if this image contains drug content"
                ]
                classification = qwen_client.classify_with_voting(image_path, classification_prompts)
            else:
                classification = self.classify_image(image_path)
            
            result = {
                'path': image_path,
                'classification': classification.get('category_final', 'normal'),
                'confidence': classification.get('confidence', 0.0)
            }
            
            if classification.get('category_final') == 'drug_related':
                details = self.extract_details(image_path)
                result['details'] = details
            
            return result
        except Exception as e:
            return {'error': str(e), 'path': image_path}
    
    def classify_image(self, image_path: str) -> Dict[str, Any]:
        prompt = self.prompts.get('classification', 'Classify this image as drug_related or normal')
        try:
            messages = [{'role': 'user', 'content': [
                {'image': f'file://{image_path}'},
                {'text': prompt}
            ]}]
            response = MultiModalConversation.call(model=self.model, messages=messages)
            if response.status_code == 200:
                text = response.output.choices[0].message.content[0]['text']
                category = 'drug_related' if 'drug' in text.lower() else 'normal'
                return {'category_final': category, 'confidence': 0.7}
            else:
                return {'category_final': 'normal', 'confidence': 0.0}
        except Exception:
            return {'category_final': 'normal', 'confidence': 0.0}
    
    def extract_details(self, image_path: str) -> Dict[str, Any]:
        try:
            ocr_text = self.extract_text(image_path)
            return {
                'ocr_text': ocr_text.get('text', ''),
                'entities': []
            }
        except Exception:
            return {'ocr_text': '', 'entities': []}
    
    def extract_text(self, image_path: str) -> Dict[str, Any]:
        try:
            messages = [{'role': 'user', 'content': [
                {'image': f'file://{image_path}'},
                {'text': 'Extract all text from this image'}
            ]}]
            response = MultiModalConversation.call(model=self.model, messages=messages)
            if response.status_code == 200:
                text = response.output.choices[0].message.content[0]['text']
                return {'success': True, 'text': text}
            else:
                return {'success': False, 'text': ''}
        except Exception:
            return {'success': False, 'text': ''}
    
    def _assess_quality(self, classification: Dict, ocr: Dict) -> Dict[str, Any]:
        w_cls = 0.5
        w_ocr = 0.3
        w_ent = 0.2
        
        cls_conf = classification.get('confidence', 0.0)
        ocr_score = min(len(ocr.get('text', '')) / 100.0, 1.0)
        ent_score = min(len(ocr.get('entities', [])) / 5.0, 1.0)
        
        combined_score = w_cls * cls_conf + w_ocr * ocr_score + w_ent * ent_score
        
        variance = sum([(cls_conf - combined_score)**2, (ocr_score - combined_score)**2, (ent_score - combined_score)**2]) / 3
        reliability = 1.0 - min(variance, 0.3)
        
        final_score = combined_score * reliability
        
        return {
            'quality_score': min(final_score, 1.0),
            'usable': final_score >= 0.6,
            'components': {
                'classification': cls_conf,
                'ocr': ocr_score,
                'entities': ent_score,
                'reliability': reliability
            }
        }

import time
from typing import Dict, Any, List
from collections import defaultdict

from ..base import BaseAgent, Task, TaskResult, ModalityType
import logging

logger = logging.getLogger(__name__)

class ImageAgent(BaseAgent):
    def __init__(self, name: str = "ImageAgent", config: Dict[str, Any] = None):
        super().__init__(name, config or {})
        self.modality = ModalityType.IMAGE
        self.llm_client = None
        self.voting_prompts = [
            "Please classify this image as drug_box|drug_list|other",
            "Classify: drug_box/drug_list/other. Only output class",
            "Identify: drug_box/drug_list/other. Only class, no explanation"
        ]
        self.output_format = {
            'id': None,
            'type': 'image',
            'category': None,
            'context': None,
            'sender': None,
            'timestamp': None,
            'path': None
        }

    async def process(self, task: Task) -> TaskResult:
        start_time = time.time()
        try:
            image_path = task.content
            metadata = task.metadata
            classification = await self._classify_with_voting(image_path)

            result_data = {
                'type': 'image',
                'category': classification['category_final'],
                'confidence': classification.get('confidence', 0.0),
                'path': image_path,
                'voting_details': classification
            }

            if classification['category_final'] in ['drug_box', 'drug_list']:
                details = await self._extract_details(image_path, classification['category_final'])
                result_data.update(details)

            image_entry = self._build_image_entry(result_data, metadata)

            processing_time = time.time() - start_time
            self._record_success(processing_time)

            return TaskResult(
                task_id=task.task_id,
                success=True,
                result=image_entry,
                confidence=classification.get('confidence', 0.0),
                processing_time=processing_time,
                metadata={'category': classification['category_final']}
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self._record_failure(processing_time)
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                processing_time=processing_time
            )

    async def _classify_with_voting(self, image_path: str) -> Dict[str, Any]:
        if not self.llm_client:
            return {
                'category_final': 'drug_box',
                'confidence': 0.8,
                'votes': {'drug_box': 2, 'drug_list': 0, 'other': 1},
                'raw_responses': ['mock1', 'mock2', 'mock3']
            }

        votes = defaultdict(int)
        raw_responses = []

        for prompt in self.voting_prompts:
            try:
                result = self.llm_client.classify_image(image_path, prompt)
                text = result.get('result', '')
                category = self._parse_category(text)
                votes[category] += 1
                raw_responses.append(text)
            except Exception as e:
                logger.warning(f"[{self.name}] Classification prompt failed: {e}")

        if votes:
            final_category = max(votes.items(), key=lambda x: x[1])[0]
            confidence = votes[final_category] / len(self.voting_prompts)
        else:
            final_category = 'other'
            confidence = 0.0

        return {
            'category_final': final_category,
            'confidence': confidence,
            'votes': dict(votes),
            'raw_responses': raw_responses
        }

    def _parse_category(self, text: str) -> str:
        text = (text or '').strip().lower()
        if 'drug_box' in text:
            return 'drug_box'
        if 'drug_list' in text:
            return 'drug_list'
        return 'other'

    async def _extract_details(self, image_path: str, category: str) -> Dict[str, Any]:
        if not self.llm_client:
            return {'ocr_text': '', 'entities': []}

        try:
            ocr_result = self.llm_client.extract_text(image_path, "Extract all text from this image")
            ocr_text = ocr_result.get('result', '')

            entities = []
            if category == 'drug_box':
                entity_result = self.llm_client.analyze_image(
                    image_path,
                    "Extract drug name, price, manufacturer from drug box image"
                )
                entities_text = entity_result.get('result', '')
                entities = self._parse_drug_entities(entities_text)

            return {
                'ocr_text': ocr_text,
                'entities': entities
            }
        except Exception as e:
            logger.warning(f"[{self.name}] Detail extraction failed: {e}")
            return {'ocr_text': '', 'entities': []}

    def _parse_drug_entities(self, text: str) -> List[Dict[str, str]]:
        entities = []
        return entities

    def _build_image_entry(self, result_data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'id': result_data.get('id', ''),
            'type': 'image',
            'category': result_data.get('category', 'unknown'),
            'context': result_data.get('ocr_text', ''),
            'sender': metadata.get('sender', ''),
            'timestamp': metadata.get('timestamp', ''),
            'path': result_data.get('path', ''),
            'entities': result_data.get('entities', []),
            'confidence': result_data.get('confidence', 0.0)
        }

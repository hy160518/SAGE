import re
import time
from typing import Dict, Any

from ..base import BaseAgent, Task, TaskResult, ModalityType
import logging

logger = logging.getLogger(__name__)

class TextAgent(BaseAgent):
    def __init__(self, name: str = "TextAgent", config: dict = None):
        super().__init__(name, config or {})
        self.modality = ModalityType.TEXT
        self.llm_client = None

    async def process(self, task: Task) -> TaskResult:
        start_time = time.time()
        try:
            content = task.content
            metadata = task.metadata

            message_extraction = self._extract_message(content, metadata)
            behavior_normalization = await self._normalize_behavior(content, metadata)
            entity_extraction = await self._extract_entities(content)
            text_entry = self._build_text_entry(message_extraction, behavior_normalization, entity_extraction)

            processing_time = time.time() - start_time
            self._record_success(processing_time)

            return TaskResult(
                task_id=task.task_id,
                success=True,
                result=text_entry,
                confidence=0.8,
                processing_time=processing_time,
                metadata={'behavior_type': behavior_normalization.get('type', 'message')}
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

    def _extract_message(self, content: str, metadata: Dict) -> Dict[str, Any]:
        return {
            'id': metadata.get('message_id', ''),
            'sender': metadata.get('sender', ''),
            'receiver': metadata.get('receiver', ''),
            'timestamp': metadata.get('timestamp', ''),
            'device': metadata.get('device_id', ''),
            'content': content,
            'message_type': metadata.get('message_type', 'text')
        }

    async def _normalize_behavior(self, content: str, metadata: Dict) -> Dict[str, Any]:
        if not self.llm_client:
            return self._rule_based_behavior_parse(content, metadata)

        behavior_prompt = f"""
        Analyze the following WeChat message and identify behavior type.

        Supported types:
        - transfer: transfer (e.g., transfer XX yuan)
        - call_voice: voice call
        - call_video: video call
        - red_envelope: red envelope
        - add_contact: add contact
        - location: location sharing
        - message: normal message

        Message: {content}

        Output JSON: {{"type": "behavior_type", "normalized": "normalized description", "entities": {{}}}}
        Extract drug information from:
        {content}

        Output JSON: {{"drugs": [{{"name": "drug_name", "quantity": "quantity"}}], "persons": ["person_name"]}}
        """

        try:
            result = self.llm_client.generate(drug_prompt)
            return self._parse_entity_result(result)
        except:
            return {'drugs': [], 'persons': []}

    def _parse_entity_result(self, result: str) -> Dict[str, Any]:
        try:
            import json
            if '{' in result and '}' in result:
                json_start = result.find('{')
                json_end = result.rindex('}') + 1
                return json.loads(result[json_start:json_end])
        except:
            pass
        return {'drugs': [], 'persons': []}

    def _rule_based_entity_extraction(self, content: str) -> Dict[str, Any]:
        drug_keywords = [
            'amoxicillin', 'ibuprofen', 'paracetamol', 'aspirin', 'panadol',
            'cephalexin', 'metformin', 'atorvastatin', 'amlodipine', 'omeprazole',
            'losartan', 'metoprolol', 'gabapentin', 'sertraline', 'escitalopram',
            'fluoxetine', 'prednisone', 'azithromycin', 'ciprofloxacin', 'doxycycline'
        ]

        drugs = []
        content_lower = content.lower()

        for drug in drug_keywords:
            if drug in content_lower:
                qty_match = re.search(r'(\d+)\s*(?:mg|boxes?|pieces?|tablets?)', content_lower)
                quantity = qty_match.group(0) if qty_match else 'unknown'
                drugs.append({'name': drug.title(), 'quantity': quantity})

        persons = []
        words = content.split()
        for word in words:
            if re.match(r'^[\u4e00-\u9fa5]{2,4}$', word) or (word and word[0].isupper() and len(word) > 1):
                if word not in ['CNY', 'USD', 'Transfer', 'Red', 'Envelope']:
                    persons.append({'name': word, 'role': 'unknown'})

        return {'drugs': drugs, 'persons': persons}

    def _build_text_entry(self, message: Dict, behavior: Dict, entities: Dict) -> Dict[str, Any]:
        return {
            'id': message.get('id', ''),
            'sender': message.get('sender', ''),
            'receiver': message.get('receiver', ''),
            'timestamp': message.get('timestamp', ''),
            'device': message.get('device', ''),
            'content': message.get('content', ''),
            'behavior': behavior,
            'entities': entities
        }

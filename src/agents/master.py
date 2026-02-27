import time
import asyncio
from typing import Dict, Any, List, Callable

from .base import BaseAgent, Task, TaskResult, ModalityType
import logging

logger = logging.getLogger(__name__)

class MasterAgent(BaseAgent):
    def __init__(self, name: str = "MasterAgent", config: Dict[str, Any] = None):
        super().__init__(name, config or {})
        self.workers: Dict[ModalityType, BaseAgent] = {}
        self.modality_classifier = None
        self.pending_tasks: Dict[str, Task] = {}
        self.task_results: Dict[str, TaskResult] = {}
        self.classification_threshold = config.get('classification_threshold', 0.7) if config else 0.7
        self.enable_async = config.get('enable_async', True) if config else True

    def register_worker(self, modality: ModalityType, worker: BaseAgent):
        self.workers[modality] = worker
        logger.info(f"[{self.name}] Registered worker: {worker.name} for modality: {modality.value}")

    def set_modality_classifier(self, classifier: Callable):
        self.modality_classifier = classifier

    async def process(self, task: Task) -> TaskResult:
        start_time = time.time()
        try:
            modality_result = await self._classify_modality(task)
            if modality_result['modality'] != ModalityType.UNKNOWN:
                task.modality = modality_result['modality']

            worker = self.workers.get(task.modality)
            if not worker:
                return TaskResult(
                    task_id=task.task_id,
                    success=False,
                    error=f"No worker registered for modality: {task.modality.value}",
                    processing_time=time.time() - start_time
                )

            worker_result = await worker.process(task)
            self.task_results[task.task_id] = worker_result

            processing_time = time.time() - start_time
            self._record_success(processing_time)

            return TaskResult(
                task_id=task.task_id,
                success=worker_result.success,
                result={
                    'modality_classification': modality_result,
                    'worker_result': worker_result.to_dict()
                },
                confidence=modality_result.get('confidence', 0.5) * worker_result.confidence,
                processing_time=processing_time,
                metadata={
                    'classified_modality': task.modality.value,
                    'worker': worker.name
                }
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self._record_failure(processing_time)
            logger.error(f"[{self.name}] Error processing task {task.task_id}: {str(e)}")
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                processing_time=processing_time
            )

    async def _classify_modality(self, task: Task) -> Dict[str, Any]:
        if self.modality_classifier:
            return await self.modality_classifier(task)
        else:
            return self._rule_based_classification(task)

    def _rule_based_classification(self, task: Task) -> Dict[str, Any]:
        content = str(task.content).lower()
        if any(ext in content for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']):
            modality = ModalityType.IMAGE
            confidence = 0.9
        elif any(ext in content for ext in ['.mp3', '.wav', '.m4a', '.amr', '.aac']):
            modality = ModalityType.VOICE
            confidence = 0.9
        elif task.metadata.get('message_type') == 'image':
            modality = ModalityType.IMAGE
            confidence = 0.8
        elif task.metadata.get('message_type') == 'voice':
            modality = ModalityType.VOICE
            confidence = 0.8
        else:
            modality = ModalityType.TEXT
            confidence = 0.7

        return {
            'modality': modality,
            'confidence': confidence,
            'probabilities': {
                'text': 1.0 - confidence if modality == ModalityType.TEXT else confidence * 0.2,
                'image': 1.0 - confidence if modality == ModalityType.IMAGE else confidence * 0.2,
                'voice': 1.0 - confidence if modality == ModalityType.VOICE else confidence * 0.2,
            },
            'method': 'rule_based_fallback'
        }

    async def dispatch_batch(self, tasks: List[Task]) -> List[TaskResult]:
        if self.enable_async:
            results = await asyncio.gather(*[self.process(task) for task in tasks], return_exceptions=True)
            return [r if isinstance(r, TaskResult) else TaskResult(task_id=tasks[i].task_id, success=False, error=str(r)) for i, r in enumerate(results)]
        else:
            return [await self.process(task) for task in tasks]

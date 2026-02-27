import time

from ..base import BaseAgent, Task, TaskResult, ModalityType
import logging

logger = logging.getLogger(__name__)

class VoiceAgent(BaseAgent):
    def __init__(self, name: str = "VoiceAgent", config: dict = None):
        super().__init__(name, config or {})
        self.modality = ModalityType.VOICE
        self.asr_client = None

    def set_asr_client(self, client):
        self.asr_client = client

    async def process(self, task: Task) -> TaskResult:
        start_time = time.time()
        try:
            voice_path = task.content
            metadata = task.metadata

            transcription = await self._transcribe(voice_path)
            quality = await self._assess_quality(transcription)
            voice_entry = self._build_voice_entry(transcription, quality, metadata, voice_path)

            processing_time = time.time() - start_time
            self._record_success(processing_time)

            return TaskResult(
                task_id=task.task_id,
                success=transcription.get('success', False),
                result=voice_entry,
                confidence=quality.get('quality_score', 0.0),
                processing_time=processing_time,
                metadata={'duration': transcription.get('duration', 0)}
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

    async def _transcribe(self, voice_path: str) -> dict:
        if not self.asr_client:
            return {
                'success': True,
                'text': 'Mock transcription result',
                'duration': 10.0
            }

        try:
            result = self.asr_client.transcribe(voice_path)
            return {
                'success': result.get('success', False),
                'text': result.get('text', ''),
                'duration': result.get('duration', 0)
            }
        except Exception as e:
            logger.warning(f"[{self.name}] Transcription failed: {e}")
            return {'success': False, 'text': '', 'error': str(e)}

    async def _assess_quality(self, transcription: dict) -> dict:
        text = transcription.get('text', '')
        if not text:
            return {'quality_score': 0.0, 'completeness': 'invalid'}

        if len(text) < 10:
            score = 0.3
        elif len(text) < 50:
            score = 0.6
        else:
            score = 0.8

        return {
            'quality_score': score,
            'completeness': 'good' if score > 0.6 else 'poor'
        }

    def _build_voice_entry(self, transcription: dict, quality: dict, metadata: dict, voice_path: str) -> dict:
        return {
            'id': '',
            'context': transcription.get('text', ''),
            'duration': transcription.get('duration', 0),
            'sender': metadata.get('sender', ''),
            'timestamp': metadata.get('timestamp', ''),
            'path': voice_path,
            'quality_score': quality.get('quality_score', 0.0)
        }

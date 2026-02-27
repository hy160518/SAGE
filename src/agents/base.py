import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ModalityType(Enum):
    TEXT = "text"
    IMAGE = "image"
    VOICE = "voice"
    UNKNOWN = "unknown"

@dataclass
class Task:
    task_id: str
    modality: ModalityType
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'modality': self.modality.value,
            'content': str(self.content)[:100],
            'metadata': self.metadata,
            'priority': self.priority,
            'created_at': self.created_at.isoformat()
        }

@dataclass
class TaskResult:
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    confidence: float = 0.0
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'success': self.success,
            'result': self.result,
            'error': self.error,
            'confidence': self.confidence,
            'processing_time': self.processing_time,
            'metadata': self.metadata
        }

@dataclass
class AgentMessage:
    sender: str
    receiver: str
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None

class BaseAgent(ABC):
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.message_queue: List[AgentMessage] = []
        self.stats = {
            'processed': 0,
            'success': 0,
            'failed': 0,
            'total_time': 0.0
        }

    @abstractmethod
    async def process(self, task: Task) -> TaskResult:
        pass

    def send_message(self, receiver: 'BaseAgent', message_type: str, content: Dict[str, Any]):
        msg = AgentMessage(
            sender=self.name,
            receiver=receiver.name,
            message_type=message_type,
            content=content
        )
        receiver.receive_message(msg)

    def receive_message(self, message: AgentMessage):
        self.message_queue.append(message)
        logger.info(f"[{self.name}] Received message from {message.sender}: {message.message_type}")

    def get_stats(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            **self.stats,
            'avg_time': self.stats['total_time'] / max(self.stats['processed'], 1)
        }

    def _record_success(self, processing_time: float):
        self.stats['processed'] += 1
        self.stats['success'] += 1
        self.stats['total_time'] += processing_time

    def _record_failure(self, processing_time: float):
        self.stats['processed'] += 1
        self.stats['failed'] += 1
        self.stats['total_time'] += processing_time

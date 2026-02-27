from .base import ModalityType, Task, TaskResult, AgentMessage, BaseAgent
from .master import MasterAgent
from .workers.image_agent import ImageAgent
from .workers.voice_agent import VoiceAgent
from .workers.text_agent import TextAgent
from .fusion.orchestrator import FusionOrchestrator

def create_sage_agents(config: dict = None, clients: dict = None):
    config = config or {}
    clients = clients or {}

    master_agent = MasterAgent("MasterAgent", config)

    image_agent = ImageAgent("ImageAgent", config)
    if clients.get('qwen_vl'):
        image_agent.llm_client = clients['qwen_vl']

    voice_agent = VoiceAgent("VoiceAgent", config)
    if clients.get('whisper'):
        voice_agent.asr_client = clients['whisper']

    text_agent = TextAgent("TextAgent", config)
    if clients.get('chatglm'):
        text_agent.llm_client = clients['chatglm']

    master_agent.register_worker(ModalityType.IMAGE, image_agent)
    master_agent.register_worker(ModalityType.VOICE, voice_agent)
    master_agent.register_worker(ModalityType.TEXT, text_agent)

    return {
        'master': master_agent,
        'image': image_agent,
        'voice': voice_agent,
        'text': text_agent
    }

__all__ = [
    'ModalityType',
    'Task',
    'TaskResult',
    'AgentMessage',
    'BaseAgent',
    'MasterAgent',
    'ImageAgent',
    'VoiceAgent',
    'TextAgent',
    'FusionOrchestrator',
    'create_sage_agents'
]

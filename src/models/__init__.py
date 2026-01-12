"""
Model clients for external APIs
Unified interface for Qwen-VL, ChatGLM, and Whisper
Includes ModelFactory for professional initialization from YAML configs.
"""

from .qwen_vl_client import QwenVLClient
from .chatglm_client import ChatGLMClient
from .whisper_client import WhisperClient
from .factory import ModelFactory

__all__ = [
    'QwenVLClient',
    'ChatGLMClient',
    'WhisperClient',
    'ModelFactory'
]

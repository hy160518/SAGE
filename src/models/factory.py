from pathlib import Path
from typing import Optional, Dict, Any
import yaml

from .qwen_vl_client import QwenVLClient
from .chatglm_client import ChatGLMClient
from .whisper_client import WhisperClient

def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

class ModelFactory:
    def __init__(self, repo_root: Optional[Path] = None, api_keys_path: Optional[Path] = None, model_cfg_path: Optional[Path] = None):
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
        self.api_keys_path = Path(api_keys_path) if api_keys_path else self.repo_root / 'configs' / 'api_keys.yaml'
        self.model_cfg_path = Path(model_cfg_path) if model_cfg_path else Path(__file__).resolve().parent / 'model_config.yaml'
        self.api_keys = load_yaml(self.api_keys_path)
        if not self.api_keys:
            example = self.repo_root / 'configs' / 'api_keys.yaml.example'
            self.api_keys = load_yaml(example)
        self.model_cfg = load_yaml(self.model_cfg_path)

    def build_qwen_vl(self) -> QwenVLClient:
        cfg = self.model_cfg.get('qwen_vl') or {}
        api_key = cfg.get('api_key') or (self.api_keys.get('api_keys') or {}).get('qwen_vl')
        return QwenVLClient(
            api_key=api_key,
            api_url=cfg.get('api_url')
        )

    def build_chatglm(self) -> ChatGLMClient:
        cfg = self.model_cfg.get('chatglm') or {}
        api_key = cfg.get('api_key') or (self.api_keys.get('api_keys') or {}).get('chatglm')
        return ChatGLMClient(
            api_key=api_key,
            api_url=cfg.get('api_url')
        )

    def build_whisper(self) -> WhisperClient:
        cfg = self.model_cfg.get('whisper') or {}
        api_key = cfg.get('api_key') or (self.api_keys.get('api_keys') or {}).get('whisper')
        return WhisperClient(
            api_key=api_key,
            api_url=cfg.get('api_url')
        )

    def build_all(self) -> Dict[str, Any]:
        return {
            'qwen_vl': self.build_qwen_vl(),
            'chatglm': self.build_chatglm(),
            'whisper': self.build_whisper()
        }

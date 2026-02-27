import requests
import time
from typing import Dict, Any, Optional
from pathlib import Path

class WhisperClient:    
    def __init__(self, api_key: str = None, api_url: str = None):

        self.api_key = api_key or ""
        self.api_url = api_url or "https://api.openai.com/v1/audio/transcriptions"
        self.model = "whisper-1"
        self.timeout = 60
        self.max_retries = 3
    
    def transcribe(self, audio_path: str, language: str = "zh") -> Dict[str, Any]:
        if self.api_key == "":
            return {
                "text": "[MOCK] Audio transcription not available",
                "language": language,
                "duration": 0.0,
                "error": "API key not configured"
            }
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            audio_path = Path(audio_path)
            if not audio_path.exists():
                return {"error": f"Audio file not found: {audio_path}", "text": ""}
            
            with open(audio_path, 'rb') as audio_file:
                files = {
                    'file': (audio_path.name, audio_file, self._get_mime_type(audio_path)),
                    'model': (None, self.model),
                    'language': (None, language)
                }
                
                for attempt in range(self.max_retries):
                    try:
                        response = requests.post(
                            self.api_url,
                            headers=headers,
                            files=files,
                            timeout=self.timeout
                        )
                        response.raise_for_status()
                        result = response.json()
                        
                        return {
                            "text": result.get("text", ""),
                            "language": result.get("language", language),
                            "duration": result.get("duration", 0.0),
                            "model": self.model
                        }
                    
                    except requests.exceptions.RequestException as e:
                        if attempt == self.max_retries - 1:
                            return {"error": str(e), "text": ""}
                        time.sleep(2 ** attempt)
        
        except Exception as e:
            return {"error": str(e), "text": ""}
    
    def transcribe_with_timestamps(self, audio_path: str, language: str = "zh") -> Dict[str, Any]:

        result = self.transcribe(audio_path, language)
        result["timestamps"] = []
        return result
    
    def _get_mime_type(self, audio_path: Path) -> str:
        ext = audio_path.suffix.lower()
        mime_types = {
            '.wav': 'audio/wav',
            '.mp3': 'audio/mpeg',
            '.m4a': 'audio/m4a',
            '.aac': 'audio/aac',
            '.flac': 'audio/flac',
            '.ogg': 'audio/ogg'
        }
        return mime_types.get(ext, 'audio/wav')
    
    def batch_transcribe(self, audio_paths: list, language: str = "zh") -> list:

        results = []
        for audio_path in audio_paths:
            result = self.transcribe(audio_path, language)
            results.append(result)
        return results

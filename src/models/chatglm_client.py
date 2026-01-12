"""
ChatGLM Model Client
For text understanding and generation
"""

import requests
import time
import json
from typing import Dict, Any, List, Optional


class ChatGLMClient:
    """Client for ChatGLM language model"""
    
    def __init__(self, api_key: str = None, api_url: str = None):
        """
        Initialize ChatGLM client
        
        Args:
            api_key: Zhipu AI API key
            api_url: API endpoint URL
        """
        self.api_key = api_key or "YOUR_CHATGLM_API_KEY"
        self.api_url = api_url or "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        self.model = "glm-4"
        self.timeout = 30
        self.max_retries = 3
    
    def extract_entities(self, text: str, prompt: str) -> Dict[str, Any]:
        """
        Extract named entities from text
        
        Args:
            text: Input text
            prompt: Extraction prompt/instruction
            
        Returns:
            dict: Extracted entities
        """
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ]
        return self._call_api(messages)
    
    def analyze_text(self, text: str, instruction: str) -> Dict[str, Any]:
        """
        Analyze text with custom instruction
        
        Args:
            text: Input text
            instruction: Analysis instruction
            
        Returns:
            dict: Analysis result
        """
        messages = [
            {"role": "user", "content": f"{instruction}\n\n{text}"}
        ]
        return self._call_api(messages)
    
    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        General chat interface
        
        Args:
            messages: Conversation history
            
        Returns:
            dict: Model response
        """
        return self._call_api(messages)
    
    def _call_api(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Internal API call method"""
        
        # Mock mode if no API key
        if self.api_key == "YOUR_CHATGLM_API_KEY":
            return {
                "result": "mock_response",
                "entities": [],
                "raw_response": "[MOCK] ChatGLM API not configured",
                "error": "API key not configured"
            }
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7
            }
            
            # Retry logic
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(
                        self.api_url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
                    # Try to parse entities if JSON format
                    entities = self._parse_entities(content)
                    
                    return {
                        "result": content,
                        "entities": entities,
                        "raw_response": content,
                        "model": self.model
                    }
                
                except requests.exceptions.RequestException as e:
                    if attempt == self.max_retries - 1:
                        return {"error": str(e), "result": "", "entities": []}
                    time.sleep(2 ** attempt)
        
        except Exception as e:
            return {"error": str(e), "result": "", "entities": []}
    
    def _parse_entities(self, text: str) -> list:
        """Try to parse entities from response"""
        try:
            # Try to find JSON in response
            if "{" in text and "}" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                data = json.loads(text[start:end])
                return data.get("entities", [])
        except:
            pass
        return []
    
    def batch_process(self, texts: list, prompt: str) -> list:
        """
        Process multiple texts
        
        Args:
            texts: List of input texts
            prompt: Common prompt for all texts
            
        Returns:
            list: List of results
        """
        results = []
        for text in texts:
            result = self.extract_entities(text, prompt)
            results.append(result)
        return results

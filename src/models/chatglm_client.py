import requests
import time
import json
from typing import Dict, Any, List, Optional

class ChatGLMClient:
    def __init__(self, api_key: str = None, api_url: str = None):
        self.api_key = api_key or "YOUR_CHATGLM_API_KEY"
        self.api_url = api_url or "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        self.model = "glm-4"
        self.timeout = 30
        self.max_retries = 3

    def extract_entities(self, text: str, prompt: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ]
        return self._call_api(messages)

    def analyze_text(self, text: str, instruction: str) -> Dict[str, Any]:
        messages = [
            {"role": "user", "content": f"{instruction}\n\n{text}"}
        ]
        return self._call_api(messages)

    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        return self._call_api(messages)

    def _call_api(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
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
        try:
            if "{" in text and "}" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                data = json.loads(text[start:end])
                return data.get("entities", [])
        except:
            pass
        return []

    def batch_process(self, texts: list, prompt: str) -> list:
        results = []
        for text in texts:
            result = self.extract_entities(text, prompt)
            results.append(result)
        return results

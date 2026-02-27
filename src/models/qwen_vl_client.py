import requests
import time
import json
import base64
from typing import Dict, Any, Optional
from pathlib import Path

class QwenVLClient:

    def __init__(self, api_key: str = None, api_url: str = None):

        self.api_key = api_key or ""
        self.api_url = api_url or "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        self.model = "qwen-vl-plus"
        self.timeout = 30
        self.max_retries = 3

    def classify_image(self, image_path: str, prompt: str) -> Dict[str, Any]:

        return self._call_api(image_path, prompt)

    def extract_text(self, image_path: str, prompt: str = None) -> Dict[str, Any]:

        if not prompt:
            prompt = "请识别并提取图片中的所有文字内容，保持原始格式。"
        return self._call_api(image_path, prompt)

    def analyze_image(self, image_path: str, prompt: str) -> Dict[str, Any]:

        return self._call_api(image_path, prompt)

    def classify_with_voting(self, image_path: str, prompts: Optional[list] = None) -> Dict[str, Any]:

        default_prompts = [
            "请判断图片是否为药品盒、手写药品清单或其他类别，并只输出类别：drug_box|drug_list|other",
            "分类该图片：drug_box（药品盒）、drug_list（手写清单）、other（其他）。只输出类别",
            "识别图像类别：drug_box/drug_list/other。只给类别，不要解释"
        ]
        prompts = prompts or default_prompts

        votes = {}
        raw = []
        for p in prompts:
            res = self._call_api(image_path, p)
            raw.append(res.get('raw_response') or res.get('result', ''))
            label = self._extract_category(res.get('result', ''))
            votes[label] = votes.get(label, 0) + 1

        final = max(votes.items(), key=lambda x: x[1])[0] if votes else 'other'
        return {
            'category_final': final,
            'votes': votes,
            'raw': raw
        }

    def _extract_category(self, text: str) -> str:
        t = (text or '').strip().lower()
        if 'drug_box' in t:
            return 'drug_box'
        if 'drug_list' in t:
            return 'drug_list'
        return 'other'

    def _call_api(self, image_path: str, prompt: str) -> Dict[str, Any]:

        if self.api_key == "":
            return {
                "result": "mock_response",
                "confidence": 0.9,
                "raw_response": "[MOCK] Qwen-VL API not configured",
                "error": "API key not configured"
            }

        try:
            with open(image_path, 'rb') as f:
                image_base64 = base64.b64encode(f.read()).decode('utf-8')

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model,
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"image": f"data:image/jpeg;base64,{image_base64}"},
                                {"text": prompt}
                            ]
                        }
                    ]
                }
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

                    text_output = result.get("output", {}).get("choices", [{}])[0].get("message", {}).get("content", "")

                    return {
                        "result": text_output,
                        "confidence": 0.9,
                        "raw_response": text_output,
                        "model": self.model
                    }

                except requests.exceptions.RequestException as e:
                    if attempt == self.max_retries - 1:
                        return {"error": str(e), "result": ""}
                    time.sleep(2 ** attempt)

        except Exception as e:
            return {"error": str(e), "result": ""}

    def batch_process(self, image_paths: list, prompt: str) -> list:

        results = []
        for image_path in image_paths:
            result = self._call_api(image_path, prompt)
            results.append(result)
        return results

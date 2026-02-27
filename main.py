import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, os.path.dirname(__file__))

from src.models.factory import ModelFactory
from src.processors.image_handler import ImageHandler
from src.processors.voice_handler import VoiceHandler
from src.processors.text_handler import TextHandler
from src.fusion.uidn_builder import UIDN
from src.data.intake import load_chat_messages


class SAGEPipeline:
    def __init__(self, config_root: str = None):
        self.config_root = Path(config_root) if config_root else Path(__file__).parent
        self.factory = ModelFactory(repo_root=self.config_root)
        self.uidn = UIDN()
        
    def process_images(self, image_paths: List[str]) -> Dict[str, Any]:
        qwen_client = self.factory.build_qwen_vl()
        handler = ImageHandler({"dashscope": {"api_key": qwen_client.api_key}})
        results = []
        for img_path in image_paths:
            if os.path.exists(img_path):
                result = handler.process_image(img_path, qwen_client=qwen_client)
                results.append(result)
        return {"images": results}
    
    def process_voices(self, voice_entries: List[Dict]) -> Dict[str, Any]:
        handler = VoiceHandler({"dashscope": {"api_key": self.factory.api_keys.get('api_keys', {}).get('whisper')}})
        return {"voices": handler.process_batch(voice_entries)}
    
    def process_texts(self, text_entries: List[Dict]) -> Dict[str, Any]:
        handler = TextHandler({"dashscope": {"api_key": self.factory.api_keys.get('api_keys', {}).get('chatglm')}})
        return {"texts": handler.process_batch(text_entries)}
    
    def run(self, data_path: str = None, output_dir: str = "output") -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        
        if data_path and os.path.exists(data_path):
            valids, errors = load_chat_messages(data_path)
            text_entries = valids[:100] if valids else []
        else:
            text_entries = []
        
        results = {
            "text": self.process_texts(text_entries) if text_entries else {},
            "images": {},
            "voices": {}
        }
        
        self.uidn.process_worker_results(results)
        network = self.uidn.export_results()
        
        output_path = os.path.join(output_dir, "sage_output.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"results": results, "network": network}, f, ensure_ascii=False, indent=2)
        
        return {"output": output_path, "entities": len(network.get("entities", [])), "relationships": len(network.get("relationships", []))}


def main():
    parser = argparse.ArgumentParser(description="SAGE: Smart Agent for Gathering Evidence")
    parser.add_argument("--data", type=str, help="Path to data file or directory")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    args = parser.parse_args()
    
    pipeline = SAGEPipeline()
    result = pipeline.run(data_path=args.data, output_dir=args.output)
    
    if result:
        print(f"Output: {result['output']}")
        print(f"Entities: {result['entities']}, Relationships: {result['relationships']}")


if __name__ == "__main__":
    main()


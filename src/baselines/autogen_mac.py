import time
from typing import Dict, List, Any, Optional

from ..processors.text_handler import TextHandler
from ..processors.voice_handler import VoiceHandler
from ..processors.image_handler import ImageHandler


class AutoGenMACBaseline:

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.text_handler = TextHandler(self.config)
        self.voice_handler = VoiceHandler(self.config)
        self.image_handler = ImageHandler(self.config)

        self._autogen_available = False
        try:
            from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager  # type: ignore
            self._autogen_available = True
            self._AssistantAgent = AssistantAgent
            self._UserProxyAgent = UserProxyAgent
            self._GroupChat = GroupChat
            self._GroupChatManager = GroupChatManager
        except Exception:
            self._autogen_available = False

    def run(self, data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:

        if self._autogen_available:
            return self._run_with_autogen(data)
        else:
            return self._run_fallback(data)

    def _run_with_autogen(self, data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:

        AssistantAgent = self._AssistantAgent
        UserProxyAgent = self._UserProxyAgent
        GroupChat = self._GroupChat
        GroupChatManager = self._GroupChatManager

        def text_tool(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return self.text_handler.process_batch(batch)

        text_agent = AssistantAgent("Text-Agent")
        def voice_tool(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return self.voice_handler.process_batch(batch)
        def image_tool(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            # ImageHandler exposes single-image API; wrap for batch
            results: List[Dict[str, Any]] = []
            for entry in batch:
                res = self.image_handler.process_image(entry.get('file_path', ''))
                res['uuid'] = entry.get('uuid')
                results.append(res)
            return results

        text_agent = AssistantAgent("Text-Agent")
        image_agent = AssistantAgent("Image-Agent")
        voice_agent = AssistantAgent("Voice-Agent")

        user_proxy = UserProxyAgent("User")

        def run_modality(entries: List[Dict], first_agent_name: str, tool_fn):
            if not entries:
                return []
            group = GroupChat(agents=[user_proxy, text_agent, image_agent, voice_agent], messages=[])
            manager = GroupChatManager(groupchat=group, speaker_selection_method="auto")

            user_proxy.initiate_chat(manager, message=f"Process {len(entries)} {first_agent_name.lower()} items with your tool.")
            return tool_fn(entries)

        text_results = run_modality(data.get('text_entries', []), "Text-Agent", text_tool)
        voice_results = run_modality(data.get('voice_entries', []), "Voice-Agent", voice_tool)
        image_results = run_modality(data.get('image_entries', []), "Image-Agent", image_tool)

        return {
            'text_results': text_results,
            'voice_results': voice_results,
            'image_results': image_results,
        }

    def _run_fallback(self, data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        start = time.time()
        text_results = self.text_handler.process_batch(data.get('text_entries', []))
        voice_results = self.voice_handler.process_batch(data.get('voice_entries', []))
        image_results: List[Dict[str, Any]] = []
        for entry in data.get('image_entries', []):
            res = self.image_handler.process_image(entry.get('file_path', ''))
            res['uuid'] = entry.get('uuid')
            image_results.append(res)

        _ = time.time() - start  # runtime available if needed
        return {
            'text_results': text_results,
            'voice_results': voice_results,
            'image_results': image_results,
        }

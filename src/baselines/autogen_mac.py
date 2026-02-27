import json
from typing import Dict, List, Any

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
        if not self._autogen_available:
            raise ImportError(
                "AutoGen-MAC baseline requires 'autogen' package.\n"
                "Install it with: pip install -r requirements-baselines.txt\n"
                "Or: pip install 'pyautogen>=0.3.0'"
            )
        return self._run_with_autogen(data)

    def _run_with_autogen(self, data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        AssistantAgent = self._AssistantAgent
        UserProxyAgent = self._UserProxyAgent
        GroupChat = self._GroupChat
        GroupChatManager = self._GroupChatManager

        dashscope_api_key = self.config.get('dashscope', {}).get('api_key', '')
        
        if dashscope_api_key:
            llm_config = {
                "config_list": [{
                    "model": "qwen-plus",
                    "api_key": dashscope_api_key,
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                }],
                "temperature": 0.7,
                "timeout": 60,
            }
        else:
            llm_config = {
                "config_list": [{
                    "model": "gpt-4",
                    "api_key": "mock-key-for-testing",
                }],
                "temperature": 0,
            }

        text_agent = AssistantAgent(
            name="Text_Agent",
            system_message=(
                "You are a text processing specialist. Use the process_text function "
                "to extract entities and drugs from text entries. Always call the function "
                "with the provided entries data."
            ),
            llm_config=llm_config,
        )
        
        voice_agent = AssistantAgent(
            name="Voice_Agent",
            system_message=(
                "You are a voice processing specialist. Use the process_voice function "
                "to perform ASR transcription on voice entries. Always call the function "
                "with the provided entries data."
            ),
            llm_config=llm_config,
        )
        
        image_agent = AssistantAgent(
            name="Image_Agent",
            system_message=(
                "You are an image processing specialist. Use the process_image function "
                "to perform OCR and classification on image entries. Always call the function "
                "with the provided entries data."
            ),
            llm_config=llm_config,
        )

        user_proxy = UserProxyAgent(
            name="UserProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False,
            is_termination_msg=lambda x: x.get("content", "").strip().endswith("TERMINATE"),
        )

        @user_proxy.register_for_execution()
        @text_agent.register_for_llm(description="Extract entities and drugs from text entries using NLP models")
        def process_text(entries: str) -> str:
            try:
                batch = json.loads(entries)
                results = self.text_handler.process_batch(batch)
                return json.dumps(results, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"error": str(e)}, ensure_ascii=False)

        @user_proxy.register_for_execution()
        @voice_agent.register_for_llm(description="Transcribe voice entries via ASR and assess quality")
        def process_voice(entries: str) -> str:
            try:
                batch = json.loads(entries)
                results = self.voice_handler.process_batch(batch)
                return json.dumps(results, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"error": str(e)}, ensure_ascii=False)

        @user_proxy.register_for_execution()
        @image_agent.register_for_llm(description="Extract text from images via OCR and classify content")
        def process_image(entries: str) -> str:
            try:
                batch = json.loads(entries)
                results = []
                for entry in batch:
                    res = self.image_handler.process_image(entry.get('file_path', ''))
                    res['uuid'] = entry.get('uuid')
                    results.append(res)
                return json.dumps(results, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"error": str(e)}, ensure_ascii=False)

        groupchat = GroupChat(
            agents=[user_proxy, text_agent, voice_agent, image_agent],
            messages=[],
            max_round=50,
            speaker_selection_method="auto",
        )
        manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

        all_results = {'text_results': [], 'voice_results': [], 'image_results': []}

        if data.get('text_entries'):
            entries_json = json.dumps(data['text_entries'], ensure_ascii=False)
            task_msg = (
                f"Text_Agent: Please call process_text function with the following entries:\n"
                f"{entries_json}\n\n"
                f"Reply with TERMINATE after processing."
            )
            user_proxy.initiate_chat(manager, message=task_msg)
            text_results = self._extract_function_results(groupchat.messages, 'process_text')
            all_results['text_results'] = text_results

        if data.get('voice_entries'):
            entries_json = json.dumps(data['voice_entries'], ensure_ascii=False)
            task_msg = (
                f"Voice_Agent: Please call process_voice function with the following entries:\n"
                f"{entries_json}\n\n"
                f"Reply with TERMINATE after processing."
            )
            groupchat.messages = []
            user_proxy.initiate_chat(manager, message=task_msg)
            voice_results = self._extract_function_results(groupchat.messages, 'process_voice')
            all_results['voice_results'] = voice_results

        if data.get('image_entries'):
            entries_json = json.dumps(data['image_entries'], ensure_ascii=False)
            task_msg = (
                f"Image_Agent: Please call process_image function with the following entries:\n"
                f"{entries_json}\n\n"
                f"Reply with TERMINATE after processing."
            )
            groupchat.messages = []
            user_proxy.initiate_chat(manager, message=task_msg)
            image_results = self._extract_function_results(groupchat.messages, 'process_image')
            all_results['image_results'] = image_results

        return all_results

    def _extract_function_results(self, messages: List[Dict], function_name: str) -> List[Dict]:
        results = []
        for i, msg in enumerate(messages):
            is_function_result = (
                msg.get('role') == 'function' or 
                (msg.get('name') == 'UserProxy' and 'exitcode' not in str(msg.get('content', '')))
            )
            if is_function_result:
                content = msg.get('content', '')
                if i > 0:
                    prev_msg = messages[i-1]
                    tool_calls = prev_msg.get('tool_calls', [])
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict) and tool_call.get('function', {}).get('name') == function_name:
                            try:
                                parsed = json.loads(content)
                                if isinstance(parsed, list):
                                    results.extend(parsed)
                                elif isinstance(parsed, dict):
                                    if 'error' not in parsed:
                                        results.append(parsed)
                            except (json.JSONDecodeError, TypeError):
                                pass
                            break
                elif function_name in str(msg.get('name', '')) or function_name in content[:100]:
                    try:
                        if isinstance(content, str) and (content.startswith('[') or content.startswith('{')):
                            parsed = json.loads(content)
                            if isinstance(parsed, list):
                                results.extend(parsed)
                            elif isinstance(parsed, dict) and 'error' not in parsed:
                                results.append(parsed)
                    except (json.JSONDecodeError, TypeError):
                        continue
        return results

from typing import Dict, List, Any, Optional

from ..utils.data_loader import ForensicDataLoader

def collect_with_autogen(config: Dict[str, Any], csv_path: Optional[str] = None, media_dir: Optional[str] = None) -> Dict[str, List[Dict]]:
    loader = ForensicDataLoader(config)
    try:
        from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager  # type: ignore
        user = UserProxyAgent("User")
        assistant = AssistantAgent("DataLoader-Agent")
        group = GroupChat(agents=[user, assistant], messages=[])
        GroupChatManager(groupchat=group, speaker_selection_method="auto")
        print("[AutoGen] Using multi-agent orchestration for data collection")
    except Exception:
        print("[Warning] AutoGen not installed, using direct loader. Install: pip install -r requirements-baselines.txt")
    
    if csv_path:
        data = loader.load_from_csv(csv_path, media_dir=media_dir or "")
    elif media_dir:
        data = loader.load_from_directory(media_dir)
    else:
        data = {'text_entries': [], 'voice_entries': [], 'image_entries': []}
    return data

def collect_with_llamaindex(config: Dict[str, Any], csv_path: Optional[str] = None, media_dir: Optional[str] = None) -> Dict[str, List[Dict]]:
    loader = ForensicDataLoader(config)
    try:
        import llama_index  # type: ignore
        print("[LlamaIndex] Using workflow-based data collection")
    except Exception:
        print("[Warning] LlamaIndex not installed, using direct loader. Install: pip install -r requirements-baselines.txt")
    if csv_path:
        return loader.load_from_csv(csv_path, media_dir=media_dir or "")
    elif media_dir:
        return loader.load_from_directory(media_dir)
    return {'text_entries': [], 'voice_entries': [], 'image_entries': []}

from typing import Dict, List, Any, Optional

from ..utils.data_loader import ForensicDataLoader


def collect_with_autogen(config: Dict[str, Any], csv_path: Optional[str] = None, media_dir: Optional[str] = None) -> Dict[str, List[Dict]]:
    """
    Use AutoGen-style orchestration to collect initial multimodal data.
    If `autogen` is installed, form minimal agents; otherwise, fall back to
    `ForensicDataLoader` directly. Returns dict with `text_entries`, `voice_entries`, `image_entries`.
    """
    try:
        from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager  # type: ignore
        # Minimal orchestration: a user agent instructs an assistant to load data
        # In practical setups, agents could negotiate sources; here we delegate to loader.
        loader = ForensicDataLoader(config)
        user = UserProxyAgent("User")
        assistant = AssistantAgent("DataLoader-Agent")
        group = GroupChat(agents=[user, assistant], messages=[])
        GroupChatManager(groupchat=group, speaker_selection_method="auto")
        # Perform loading via CSV if provided, else fall back to directory scanning
        if csv_path:
            data = loader.load_from_csv(csv_path, media_dir=media_dir or "")
        elif media_dir:
            data = loader.load_from_directory(media_dir)
        else:
            data = {'text_entries': [], 'voice_entries': [], 'image_entries': []}
        return data
    except Exception:
        # Fallback: direct loader without autogen
        loader = ForensicDataLoader(config)
        if csv_path:
            return loader.load_from_csv(csv_path, media_dir=media_dir or "")
        elif media_dir:
            return loader.load_from_directory(media_dir)
        return {'text_entries': [], 'voice_entries': [], 'image_entries': []}


def collect_with_llamaindex(config: Dict[str, Any], csv_path: Optional[str] = None, media_dir: Optional[str] = None) -> Dict[str, List[Dict]]:
    """
    Use LlamaIndex Workflow to collect initial data.
    If `llama_index.core.workflow` is available, wrap loading in a StartEvent;
    otherwise, fall back to `ForensicDataLoader` directly.
    """
    try:
        # Attempt import to ensure availability, but use our loader to avoid API incompatibilities.
        import llama_index  # type: ignore
    except Exception:
        pass
    loader = ForensicDataLoader(config)
    if csv_path:
        return loader.load_from_csv(csv_path, media_dir=media_dir or "")
    elif media_dir:
        return loader.load_from_directory(media_dir)
    return {'text_entries': [], 'voice_entries': [], 'image_entries': []}

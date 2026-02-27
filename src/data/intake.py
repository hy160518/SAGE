import json
from typing import List, Dict, Any, Tuple
import os

REQUIRED_KEYS = ["uuid", "content", "timestamp"]
# Also accept "id" as alternative to "uuid" for paper compatibility

def _validate_item(item: Dict[str, Any]) -> Tuple[bool, str]:
    # Support both "uuid" (paper format) and "id"
    id_field = "uuid" if "uuid" in item else "id"
    content_field = "content"

    # Check for content
    if content_field not in item:
        return False, f"missing key: {content_field}"
    if not isinstance(item.get(content_field), str) or not item[content_field].strip():
        return False, "content invalid"

    # Check for ID (uuid or id)
    if id_field not in item:
        return False, f"missing key: uuid or id"
    if not isinstance(item.get(id_field), str) or not item[id_field].strip():
        return False, "uuid invalid"

    # Check timestamp
    ts = item.get("timestamp")
    if not isinstance(ts, (int, float, str)):
        return False, "timestamp invalid type"

    return True, ""

def load_chat_messages(path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not os.path.exists(path):
        return [], [{"index": -1, "reason": "file not found"}]

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        return [], [{"index": -1, "reason": "root not array"}]

    valids: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for idx, item in enumerate(data):
        ok, reason = _validate_item(item)
        if ok:
            valids.append(item)
        else:
            errors.append({"index": idx, "reason": reason})
    return valids, errors

def save_quality_report(output_path: str, valid_count: int, error_list: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report = {
        "valid_count": valid_count,
        "error_count": len(error_list),
        "errors": error_list[:200]  # 限制输出长度
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

import os
import re
import json
from typing import List, Dict, Any, Tuple
from datetime import datetime
import csv

class ForensicDataLoader:
    MESSAGE_TYPES = {
        '1': 'text',
        '3': 'image',
        '34': 'voice',
        '43': 'video',
        '47': 'emoji',
        '49': 'link',
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.stats = {
            'total_records': 0,
            'text_count': 0,
            'voice_count': 0,
            'image_count': 0,
            'skipped': 0
        }

    def load_from_csv(self, csv_path: str, media_dir: str = None) -> Dict[str, List[Dict]]:
        print(f"Loading CSV: {csv_path}")

        data = {
            'text_entries': [],
            'voice_entries': [],
            'image_entries': []
        }

        if not os.path.exists(csv_path):
            print(f"Error: CSV not found - {csv_path}")
            return data

        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)

            for row in reader:
                self.stats['total_records'] += 1

                try:
                    entry = self._parse_csv_row(row, media_dir)

                    if entry:
                        if entry['type'] == 'text':
                            data['text_entries'].append(entry)
                            self.stats['text_count'] += 1
                        elif entry['type'] == 'voice':
                            data['voice_entries'].append(entry)
                            self.stats['voice_count'] += 1
                        elif entry['type'] == 'image':
                            data['image_entries'].append(entry)
                            self.stats['image_count'] += 1
                    else:
                        self.stats['skipped'] += 1

                except Exception as e:
                    print(f"Parse failed: {str(e)}")
                    self.stats['skipped'] += 1
                    continue

        print(f"\nCSV Loaded:")
        print(f"  Total: {self.stats['total_records']}")
        print(f"  Text: {self.stats['text_count']}")
        print(f"  Voice: {self.stats['voice_count']}")
        print(f"  Image: {self.stats['image_count']}")
        print(f"  Skipped: {self.stats['skipped']}")

        return data

    def _parse_csv_row(self, row: Dict[str, str], media_dir: str = None) -> Dict[str, Any]:
        uuid = row.get('UUID') or row.get('msgSvrId') or row.get('id')
        msg_type = row.get('type') or row.get('msgType') or row.get('Type')
        content = row.get('content') or row.get('Content') or ''
        sender = row.get('talker') or row.get('sender') or row.get('Sender') or ''
        receiver = row.get('des') or row.get('receiver') or row.get('Receiver') or ''
        timestamp = row.get('createTime') or row.get('timestamp') or row.get('Timestamp')

        if not uuid or not msg_type:
            return None

        create_time = self._parse_timestamp(timestamp)

        msg_type_str = self.MESSAGE_TYPES.get(str(msg_type), 'unknown')

        if msg_type_str == 'text':
            return {
                'type': 'text',
                'uuid': uuid,
                'content': content,
                'sender': sender,
                'receiver': receiver,
                'timestamp': create_time,
                'original_row': row
            }

        elif msg_type_str == 'voice':
            voice_file = self._find_media_file(uuid, media_dir, ['.mp3', '.wav', '.amr', '.m4a'])

            if voice_file:
                return {
                    'type': 'voice',
                    'uuid': uuid,
                    'file_path': voice_file,
                    'sender': sender,
                    'receiver': receiver,
                    'timestamp': create_time,
                    'duration': self._extract_duration(content),
                    'original_row': row
                }
            else:
                return None

        elif msg_type_str == 'image':
            image_file = self._find_media_file(uuid, media_dir, ['.jpg', '.jpeg', '.png', '.bmp'])

            if image_file:
                return {
                    'type': 'image',
                    'uuid': uuid,
                    'file_path': image_file,
                    'sender': sender,
                    'receiver': receiver,
                    'timestamp': create_time,
                    'original_row': row
                }
            else:
                return None

        else:
            return None

    def _parse_timestamp(self, timestamp_str: str) -> str:
        if not timestamp_str:
            return ''

        try:
            if timestamp_str.isdigit():
                timestamp = int(timestamp_str)
                if timestamp > 10000000000:
                    timestamp = timestamp / 1000
                dt = datetime.fromtimestamp(timestamp)
                return dt.strftime('%Y-%m-%d %H:%M:%S')

            else:
                dt = datetime.fromisoformat(timestamp_str)
                return dt.strftime('%Y-%m-%d %H:%M:%S')

        except Exception as e:
            return timestamp_str

    def _find_media_file(self, uuid: str, media_dir: str, extensions: List[str]) -> str:
        if not media_dir or not os.path.exists(media_dir):
            return None

        clean_uuid = re.sub(r'[^a-zA-Z0-9_-]', '', str(uuid))

        for root, dirs, files in os.walk(media_dir):
            for file in files:
                file_lower = file.lower()

                if clean_uuid in file and any(file_lower.endswith(ext) for ext in extensions):
                    return os.path.join(root, file)

        return None

    def _extract_duration(self, content: str) -> int:
        if not content:
            return 0

        try:
            if '{' in content and '}' in content:
                data = json.loads(content)
                return int(data.get('voiceLength', 0))

            match = re.search(r'voiceLength["\']?\s*:\s*["\']?(\d+)', content)
            if match:
                return int(match.group(1))

        except:
            pass

        return 0

    def load_from_directory(self, dir_path: str) -> Dict[str, List[Dict]]:
        print(f"Scanning directory: {dir_path}")

        data = {
            'text_entries': [],
            'voice_entries': [],
            'image_entries': []
        }

        if not os.path.isdir(dir_path):
            print(f"Error: Directory not found - {dir_path}")
            return data

        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_lower = file.lower()

                uuid = os.path.splitext(file)[0]

                if file_lower.endswith('.txt'):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        data['text_entries'].append({
                            'type': 'text',
                            'uuid': uuid,
                            'content': content,
                            'file_path': file_path,
                            'sender': 'unknown',
                            'receiver': 'unknown'
                        })
                        self.stats['text_count'] += 1
                    except Exception as e:
                        print(f"Read text failed {file}: {str(e)}")

                elif any(file_lower.endswith(ext) for ext in ['.mp3', '.wav', '.m4a', '.amr']):
                    data['voice_entries'].append({
                        'type': 'voice',
                        'uuid': uuid,
                        'file_path': file_path,
                        'sender': 'unknown',
                        'receiver': 'unknown'
                    })
                    self.stats['voice_count'] += 1

                elif any(file_lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
                    data['image_entries'].append({
                        'type': 'image',
                        'uuid': uuid,
                        'file_path': file_path,
                        'sender': 'unknown',
                        'receiver': 'unknown'
                    })
                    self.stats['image_count'] += 1

        print(f"\nDirectory scan complete:")
        print(f"  Text: {self.stats['text_count']}")
        print(f"  Voice: {self.stats['voice_count']}")
        print(f"  Image: {self.stats['image_count']}")

        return data

    def load_from_json(self, json_path: str) -> Dict[str, List[Dict]]:
        print(f"Loading JSON: {json_path}")

        if not os.path.exists(json_path):
            print(f"Error: JSON not found - {json_path}")
            return {
                'text_entries': [],
                'voice_entries': [],
                'image_entries': []
            }

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            print("Error: Invalid JSON format")
            return {
                'text_entries': [],
                'voice_entries': [],
                'image_entries': []
            }

        self.stats['text_count'] = len(data.get('text_entries', []))
        self.stats['voice_count'] = len(data.get('voice_entries', []))
        self.stats['image_count'] = len(data.get('image_entries', []))

        print(f"JSON Loaded:")
        print(f"  Text: {self.stats['text_count']}")
        print(f"  Voice: {self.stats['voice_count']}")
        print(f"  Image: {self.stats['image_count']}")

        return data

    def get_statistics(self) -> Dict[str, int]:
        return self.stats.copy()

"""
真实取证数据加载器
解析微信取证数据格式（2023-石电-64-J1格式）
参考DataCollector的实际数据结构
"""
import os
import re
import json
from typing import List, Dict, Any, Tuple
from datetime import datetime
import csv


class ForensicDataLoader:
    """
    取证数据加载器
    
    支持的数据格式：
    1. 微信聊天记录CSV（标准取证格式）
    2. 文件目录扫描（语音、图像文件）
    3. JSON格式的结构化数据
    
    参考DataCollector中的实际数据结构：
    - 聊天记录包含：UUID, 发送人, 接收人, 消息类型, 内容, 时间戳等
    - 媒体文件以UUID命名，关联到聊天记录
    """
    
    # 微信消息类型映射
    MESSAGE_TYPES = {
        '1': 'text',        # 文本消息
        '3': 'image',       # 图片消息
        '34': 'voice',      # 语音消息
        '43': 'video',      # 视频消息
        '47': 'emoji',      # 表情包
        '49': 'link',       # 链接/文件
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据加载器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.stats = {
            'total_records': 0,
            'text_count': 0,
            'voice_count': 0,
            'image_count': 0,
            'skipped': 0
        }
    
    def load_from_csv(self, csv_path: str, media_dir: str = None) -> Dict[str, List[Dict]]:
        """
        从CSV文件加载微信聊天记录
        CSV格式参考DataCollector的数据结构：
        UUID, 发送人, 接收人, 消息类型, 消息内容, 时间, 媒体文件路径
        
        Args:
            csv_path: CSV文件路径
            media_dir: 媒体文件目录（包含语音、图像文件）
        
        Returns:
            分类后的数据字典
        """
        print(f"正在加载CSV数据: {csv_path}")
        
        data = {
            'text_entries': [],
            'voice_entries': [],
            'image_entries': []
        }
        
        if not os.path.exists(csv_path):
            print(f"错误: CSV文件不存在 - {csv_path}")
            return data
        
        # 读取CSV
        with open(csv_path, 'r', encoding='utf-8-sig') as f:  # utf-8-sig处理BOM
            reader = csv.DictReader(f)
            
            for row in reader:
                self.stats['total_records'] += 1
                
                try:
                    # 解析一行记录
                    entry = self._parse_csv_row(row, media_dir)
                    
                    if entry:
                        # 根据类型分类
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
                    print(f"解析行失败: {str(e)}")
                    self.stats['skipped'] += 1
                    continue
        
        # 打印统计
        print(f"\n✓ CSV加载完成:")
        print(f"  总记录数: {self.stats['total_records']}")
        print(f"  文本消息: {self.stats['text_count']}")
        print(f"  语音消息: {self.stats['voice_count']}")
        print(f"  图像消息: {self.stats['image_count']}")
        print(f"  跳过: {self.stats['skipped']}")
        
        return data
    
    def _parse_csv_row(self, row: Dict[str, str], media_dir: str = None) -> Dict[str, Any]:
        """
        解析CSV的一行数据
        
        CSV字段（参考实际取证数据）：
        - UUID / msgSvrId: 消息ID
        - talker: 发送人
        - content: 消息内容
        - type: 消息类型（1=文本, 3=图片, 34=语音）
        - createTime: 创建时间（时间戳）
        - des: 接收人
        
        Args:
            row: CSV行数据
            media_dir: 媒体文件目录
        
        Returns:
            解析后的条目，如果无法解析则返回None
        """
        # 兼容不同的字段名
        uuid = row.get('UUID') or row.get('msgSvrId') or row.get('id')
        msg_type = row.get('type') or row.get('msgType') or row.get('Type')
        content = row.get('content') or row.get('Content') or ''
        sender = row.get('talker') or row.get('sender') or row.get('Sender') or ''
        receiver = row.get('des') or row.get('receiver') or row.get('Receiver') or ''
        timestamp = row.get('createTime') or row.get('timestamp') or row.get('Timestamp')
        
        if not uuid or not msg_type:
            return None
        
        # 解析时间
        create_time = self._parse_timestamp(timestamp)
        
        # 根据消息类型处理
        msg_type_str = self.MESSAGE_TYPES.get(str(msg_type), 'unknown')
        
        if msg_type_str == 'text':
            # 文本消息
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
            # 语音消息 - 需要找到对应的语音文件
            voice_file = self._find_media_file(uuid, media_dir, ['.mp3', '.wav', '.amr', '.m4a'])
            
            if voice_file:
                return {
                    'type': 'voice',
                    'uuid': uuid,
                    'file_path': voice_file,
                    'sender': sender,
                    'receiver': receiver,
                    'timestamp': create_time,
                    'duration': self._extract_duration(content),  # 从content中提取时长
                    'original_row': row
                }
            else:
                # 语音文件不存在，跳过
                return None
        
        elif msg_type_str == 'image':
            # 图像消息 - 找到对应的图像文件
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
            # 其他类型暂不处理
            return None
    
    def _parse_timestamp(self, timestamp_str: str) -> str:
        """
        解析时间戳
        支持多种格式：Unix时间戳、ISO格式等
        
        Args:
            timestamp_str: 时间戳字符串
        
        Returns:
            格式化的时间字符串
        """
        if not timestamp_str:
            return ''
        
        try:
            # 尝试作为Unix时间戳解析
            if timestamp_str.isdigit():
                timestamp = int(timestamp_str)
                # 处理毫秒时间戳
                if timestamp > 10000000000:
                    timestamp = timestamp / 1000
                dt = datetime.fromtimestamp(timestamp)
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # 尝试作为ISO格式解析
            else:
                dt = datetime.fromisoformat(timestamp_str)
                return dt.strftime('%Y-%m-%d %H:%M:%S')
        
        except Exception as e:
            # 解析失败，返回原始字符串
            return timestamp_str
    
    def _find_media_file(self, uuid: str, media_dir: str, extensions: List[str]) -> str:
        """
        查找媒体文件
        根据UUID在媒体目录中查找对应的文件
        
        Args:
            uuid: 消息UUID
            media_dir: 媒体文件目录
            extensions: 文件扩展名列表
        
        Returns:
            文件路径，如果未找到返回None
        """
        if not media_dir or not os.path.exists(media_dir):
            return None
        
        # 清理UUID（去除特殊字符）
        clean_uuid = re.sub(r'[^a-zA-Z0-9_-]', '', str(uuid))
        
        # 在媒体目录中搜索
        for root, dirs, files in os.walk(media_dir):
            for file in files:
                file_lower = file.lower()
                
                # 检查文件名是否包含UUID且扩展名匹配
                if clean_uuid in file and any(file_lower.endswith(ext) for ext in extensions):
                    return os.path.join(root, file)
        
        return None
    
    def _extract_duration(self, content: str) -> int:
        """
        从消息内容中提取语音时长
        微信语音消息的content可能包含 "voiceLength":"32" 这样的信息
        
        Args:
            content: 消息内容
        
        Returns:
            时长（秒），无法提取返回0
        """
        if not content:
            return 0
        
        try:
            # 尝试从JSON格式提取
            if '{' in content and '}' in content:
                data = json.loads(content)
                return int(data.get('voiceLength', 0))
            
            # 尝试正则匹配
            match = re.search(r'voiceLength["\']?\s*:\s*["\']?(\d+)', content)
            if match:
                return int(match.group(1))
        
        except:
            pass
        
        return 0
    
    def load_from_directory(self, dir_path: str) -> Dict[str, List[Dict]]:
        """
        从目录加载数据（简单扫描）
        用于没有CSV的情况，直接扫描媒体文件
        
        Args:
            dir_path: 目录路径
        
        Returns:
            数据字典
        """
        print(f"正在扫描目录: {dir_path}")
        
        data = {
            'text_entries': [],
            'voice_entries': [],
            'image_entries': []
        }
        
        if not os.path.isdir(dir_path):
            print(f"错误: 目录不存在 - {dir_path}")
            return data
        
        # 扫描文件
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_lower = file.lower()
                
                # 提取UUID（假设文件名就是UUID）
                uuid = os.path.splitext(file)[0]
                
                if file_lower.endswith('.txt'):
                    # 文本文件
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
                        print(f"读取文本文件失败 {file}: {str(e)}")
                
                elif any(file_lower.endswith(ext) for ext in ['.mp3', '.wav', '.m4a', '.amr']):
                    # 语音文件
                    data['voice_entries'].append({
                        'type': 'voice',
                        'uuid': uuid,
                        'file_path': file_path,
                        'sender': 'unknown',
                        'receiver': 'unknown'
                    })
                    self.stats['voice_count'] += 1
                
                elif any(file_lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
                    # 图像文件
                    data['image_entries'].append({
                        'type': 'image',
                        'uuid': uuid,
                        'file_path': file_path,
                        'sender': 'unknown',
                        'receiver': 'unknown'
                    })
                    self.stats['image_count'] += 1
        
        print(f"\n✓ 目录扫描完成:")
        print(f"  文本: {self.stats['text_count']}")
        print(f"  语音: {self.stats['voice_count']}")
        print(f"  图像: {self.stats['image_count']}")
        
        return data
    
    def load_from_json(self, json_path: str) -> Dict[str, List[Dict]]:
        """
        从JSON文件加载数据
        用于预处理好的结构化数据
        
        Args:
            json_path: JSON文件路径
        
        Returns:
            数据字典
        """
        print(f"正在加载JSON数据: {json_path}")
        
        if not os.path.exists(json_path):
            print(f"错误: JSON文件不存在 - {json_path}")
            return {
                'text_entries': [],
                'voice_entries': [],
                'image_entries': []
            }
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 验证数据格式
        if not isinstance(data, dict):
            print("错误: JSON格式不正确，应该是字典类型")
            return {
                'text_entries': [],
                'voice_entries': [],
                'image_entries': []
            }
        
        # 统计
        self.stats['text_count'] = len(data.get('text_entries', []))
        self.stats['voice_count'] = len(data.get('voice_entries', []))
        self.stats['image_count'] = len(data.get('image_entries', []))
        
        print(f"✓ JSON加载完成:")
        print(f"  文本: {self.stats['text_count']}")
        print(f"  语音: {self.stats['voice_count']}")
        print(f"  图像: {self.stats['image_count']}")
        
        return data
    
    def get_statistics(self) -> Dict[str, int]:
        """获取加载统计信息"""
        return self.stats.copy()

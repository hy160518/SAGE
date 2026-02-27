import hashlib
import hmac
import re
import json
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from datetime import datetime

class PseudonymizationStats:
    def __init__(self):
        self.total_processed = 0
        self.entities_found = defaultdict(int)
        self.context_patterns = defaultdict(int)
        self.nested_cases = 0
        self.ambiguous_cases = []
        
    def to_dict(self) -> Dict:
        return {
            'total_processed': self.total_processed,
            'entities_found': dict(self.entities_found),
            'context_patterns': dict(self.context_patterns),
            'nested_cases': self.nested_cases,
            'ambiguous_cases': self.ambiguous_cases[:50]  
        }

class Pseudonymizer:

    COMMON_SURNAMES = {
        '王', '李', '张', '刘', '陈', '杨', '黄', '赵', '周', '吴',
        '徐', '孙', '马', '朱', '胡', '郭', '何', '高', '林', '罗',
        '郑', '梁', '谢', '宋', '唐', '许', '韩', '冯', '邓', '曹',
        '彭', '曾', '肖', '田', '董', '袁', '潘', '于', '蒋', '蔡',
        '余', '杜', '叶', '程', '苏', '魏', '吕', '丁', '任', '沈',
        '姚', '卢', '姜', '崔', '钟', '谭', '陆', '汪', '范', '金',
        '石', '廖', '贾', '夏', '韦', '付', '方', '白', '邹', '孟',
        '熊', '秦', '邱', '江', '尹', '薛', '闫', '段', '雷', '侯',
        '龙', '史', '陶', '黎', '贺', '顾', '毛', '郝', '龚', '邵',
        '万', '钱', '严', '覃', '武', '戴', '莫', '孔', '向', '汤'
    }
    
    ADDRESS_KEYWORDS = {
        '省', '市', '区', '县', '镇', '乡', '村', '街道', '路', '号',
        '小区', '栋', '单元', '室', '楼', '层', '弄', '巷', '里', '庄'
    }

    def __init__(self, salt: str, reversible: bool = False, enable_context: bool = True):

        self.salt = salt.encode("utf-8")
        self.cache: Dict[str, str] = {}
        self.reverse_cache: Optional[Dict[str, str]] = {} if reversible else None
        self.reversible = reversible
        self.enable_context = enable_context
        self.stats = PseudonymizationStats()
        
        self._compile_patterns()
    
    def _compile_patterns(self):
        self.patterns = {
            'phone': re.compile(r'\b1[3-9]\d{9}\b'),
            'id_card': re.compile(r'\b\d{15}(\d{2}[0-9Xx])?\b'),
            'wechat': re.compile(r'\b[a-zA-Z][a-zA-Z0-9_\-]{3,20}\b'),
            'long_number': re.compile(r'\b\d{10,}\b'),
            'email': re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'),
            'url': re.compile(r'https?://[^\s]+'),
            'chinese_name': re.compile(r'[\u4e00-\u9fa5]{2,4}'),
            'address': re.compile(r'[\u4e00-\u9fa5]{2,}(?:省|市|区|县|镇|乡|村|街道|路|号|小区|栋|单元|室|楼|层)[^\s，。！？;；]*')
        }

    def _tag(self, kind: str, raw: str) -> str:
        if raw in self.cache:
            return self.cache[raw]
        
        digest = hmac.new(self.salt, raw.encode("utf-8"), hashlib.sha256).hexdigest()[:10]
        tag = f"{kind}_{digest}"
        
        self.cache[raw] = tag
        if self.reversible and self.reverse_cache is not None:
            self.reverse_cache[tag] = raw
        
        self.stats.entities_found[kind] += 1
        return tag

    def _is_likely_chinese_name(self, text: str) -> bool:
        if len(text) < 2 or len(text) > 4:
            return False
        
        if text[0] not in self.COMMON_SURNAMES:
            return False
        
        for keyword in self.ADDRESS_KEYWORDS:
            if keyword in text:
                return False
        
        return True

    def _check_context(self, text: str, match_obj, match_type: str) -> bool:
        if not self.enable_context:
            return True
        
        start = match_obj.start()
        end = match_obj.end()
        
        context_before = text[max(0, start-20):start]
        context_after = text[end:min(len(text), end+20)]
        
        if match_type == 'long_number':
            if any(keyword in context_before + context_after 
                   for keyword in ['订单', '编号', '流水', '单号', 'ID', '序号']):
                return False
        
        if match_type == 'wechat':
            if 'http' in context_before or '://' in context_before:
                return False
        
        if match_type == 'chinese_name':
            matched_text = match_obj.group(0)
            if any(keyword in context_after[:5] 
                   for keyword in ['片', '胶囊', '颗粒', '丸', '注射液', '口服液']):
                return False
        
        return True

    def _extract_nested_entities(self, text: str) -> List[Tuple[int, int, str, str]]:
        entities = []
        
        priority_order = [
            ('phone', self.patterns['phone']),
            ('id_card', self.patterns['id_card']),
            ('email', self.patterns['email']),
            ('url', self.patterns['url']),
            ('address', self.patterns['address']),
            ('chinese_name', self.patterns['chinese_name']),
            ('wechat', self.patterns['wechat']),
            ('long_number', self.patterns['long_number'])
        ]
        
        for entity_type, pattern in priority_order:
            for match in pattern.finditer(text):
                if not self._check_context(text, match, entity_type):
                    continue
                
                if entity_type == 'chinese_name':
                    if not self._is_likely_chinese_name(match.group(0)):
                        continue
                
                entities.append((
                    match.start(),
                    match.end(),
                    entity_type,
                    match.group(0)
                ))
        
        entities = self._resolve_overlaps(entities)
        
        return entities

    def _resolve_overlaps(self, entities: List[Tuple[int, int, str, str]]) -> List[Tuple[int, int, str, str]]:
        if not entities:
            return []
        
        entities.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        
        resolved = []
        last_end = -1
        
        for entity in entities:
            start, end, entity_type, text = entity
            
            if start >= last_end:
                resolved.append(entity)
                last_end = end
            else:
                self.stats.nested_cases += 1
        
        return resolved

    def transform(self, text: str) -> str:
        if not text:
            return text
        
        self.stats.total_processed += 1
        entities = self._extract_nested_entities(text)
        
        if not entities:
            return text
        
        entities.sort(key=lambda x: x[0], reverse=True)
        
        result = text
        for start, end, entity_type, matched_text in entities:
            tag = self._tag(entity_type.upper(), matched_text)
            result = result[:start] + tag + result[end:]
        
        return result

    def reverse_transform(self, text: str) -> Optional[str]:
        if not self.reversible or not self.reverse_cache:
            return None
        
        result = text
        pattern = re.compile(r'\b(PHONE|ID_CARD|WECHAT|EMAIL|URL|ADDRESS|CHINESE_NAME|LONG_NUMBER)_[a-f0-9]{10}\b')
        
        for match in pattern.finditer(text):
            tag = match.group(0)
            if tag in self.reverse_cache:
                original = self.reverse_cache[tag]
                result = result.replace(tag, original)
        
        return result

    def get_statistics(self) -> Dict:
        return self.stats.to_dict()
    
    def reset_statistics(self):
        self.stats = PseudonymizationStats()
    
    def export_mapping(self, output_path: str):
        mapping = {
            'timestamp': datetime.now().isoformat(),
            'total_entities': len(self.cache),
            'mappings': [
                {'original': original, 'pseudonym': pseudo}
                for original, pseudo in self.cache.items()
            ],
            'statistics': self.get_statistics()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
    
    def batch_transform(self, texts: List[str]) -> List[str]:
        return [self.transform(text) for text in texts]

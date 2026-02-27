import re
from typing import Dict, List, Optional, Any, Tuple
from difflib import SequenceMatcher
from datetime import datetime

class EntityConflictResolver:

    def __init__(self):
        self.attr_priority = {
            'name': 10,
            'phone': 10,
            'wechat': 10,
            'idcard': 9,
            'account': 8,
            'gender': 5,
            'age': 3,
            'occupation': 3
        }
    
    def resolve_conflict(self, existing_value: Any, new_value: Any, 
                        confidence_existing: float, confidence_new: float,
                        attr_name: str) -> Tuple[Any, float, str]:
        if existing_value == new_value:
            avg_conf = (confidence_existing + confidence_new) / 2
            return existing_value, avg_conf, 'identical'

        if not existing_value:
            return new_value, confidence_new, 'new_value'

        if not new_value:
            return existing_value, confidence_existing, 'existing_value'

        if confidence_new > confidence_existing:
            return new_value, confidence_new, 'confidence_based'
        else:
            return existing_value, confidence_existing, 'confidence_based'

class EntityRegistry:

    DETERMINISTIC_THRESHOLD = 1.0
    SEMANTIC_THRESHOLD = 0.85
    CROSS_MODAL_THRESHOLD = 0.75

    def __init__(self):
        self.entities = {}
        self.phone_index = {}
        self.name_index = {}
        self.wechat_index = {}
        self.idcard_index = {}
        self.account_index = {}
        self.next_id = 1
        self.conflict_resolver = EntityConflictResolver()
        self.match_history = []
    
    def register_entity(self, entity_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        match_info = {
            'match_level': None,
            'matched_entity_id': None,
            'confidence': 0.0,
            'matching_field': None,
            'strategy': None
        }
        
        phone = entity_data.get('phone') or entity_data.get('phone_number')
        name = entity_data.get('name')
        wechat = entity_data.get('wechat') or entity_data.get('wechat_id') or entity_data.get('handle')
        idcard = entity_data.get('id_card') or entity_data.get('id_card_number')
        account = entity_data.get('account') or entity_data.get('bank_account') or entity_data.get('payment_account')
        
        if phone:
            norm_phone = self._normalize_phone(phone)
            if norm_phone in self.phone_index:
                existing_id = self.phone_index[norm_phone]
                match_info['match_level'] = 1
                match_info['matched_entity_id'] = existing_id
                match_info['confidence'] = 1.0
                match_info['matching_field'] = 'phone'
                match_info['strategy'] = 'deterministic'
                
                self._merge_entity(existing_id, entity_data, match_info)
                return existing_id, match_info
        
        if wechat:
            norm_wechat = self._normalize_wechat(wechat)
            if norm_wechat in self.wechat_index:
                existing_id = self.wechat_index[norm_wechat]
                match_info['match_level'] = 1
                match_info['matched_entity_id'] = existing_id
                match_info['confidence'] = 1.0
                match_info['matching_field'] = 'wechat'
                match_info['strategy'] = 'deterministic'
                
                self._merge_entity(existing_id, entity_data, match_info)
                return existing_id, match_info
        
        if idcard:
            norm_idcard = self._normalize_idcard(idcard)
            if norm_idcard in self.idcard_index:
                existing_id = self.idcard_index[norm_idcard]
                match_info['match_level'] = 1
                match_info['matched_entity_id'] = existing_id
                match_info['confidence'] = 0.95
                match_info['matching_field'] = 'idcard'
                match_info['strategy'] = 'deterministic'
                
                self._merge_entity(existing_id, entity_data, match_info)
                return existing_id, match_info
        
        if account:
            norm_account = self._normalize_account(account)
            if norm_account in self.account_index:
                existing_id = self.account_index[norm_account]
                match_info['match_level'] = 1
                match_info['matched_entity_id'] = existing_id
                match_info['confidence'] = 0.9
                match_info['matching_field'] = 'account'
                match_info['strategy'] = 'deterministic'
                
                self._merge_entity(existing_id, entity_data, match_info)
                return existing_id, match_info
        
        if name:
            norm_name = self._normalize_name(name)
            best_match_id, best_score = self._find_best_name_match(norm_name)
            
            if best_match_id and best_score >= self.SEMANTIC_THRESHOLD:
                match_info['match_level'] = 2
                match_info['matched_entity_id'] = best_match_id
                match_info['confidence'] = best_score
                match_info['matching_field'] = 'name'
                match_info['strategy'] = 'semantic'
                
                self._merge_entity(best_match_id, entity_data, match_info)
                return best_match_id, match_info
        
        cross_modal_match = self._find_cross_modal_match(entity_data)
        if cross_modal_match:
            matched_id, confidence = cross_modal_match
            if confidence >= self.CROSS_MODAL_THRESHOLD:
                match_info['match_level'] = 3
                match_info['matched_entity_id'] = matched_id
                match_info['confidence'] = confidence
                match_info['matching_field'] = 'cross_modal'
                match_info['strategy'] = 'cross_modal'
                
                self._merge_entity(matched_id, entity_data, match_info)
                return matched_id, match_info
        
        entity_id = f"ENTITY_{self.next_id:06d}"
        self.next_id += 1
        
        self.entities[entity_id] = {
            **entity_data,
            'entity_id': entity_id,
            'sources': [entity_data.get('source', 'unknown')],
            'merged_count': 1,
            'confidence': entity_data.get('confidence', 1.0),
            'match_history': [match_info],
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
        
        if phone:
            self.phone_index[self._normalize_phone(phone)] = entity_id
        if name:
            self.name_index[self._normalize_name(name)] = entity_id
        if wechat:
            self.wechat_index[self._normalize_wechat(wechat)] = entity_id
        if idcard:
            self.idcard_index[self._normalize_idcard(idcard)] = entity_id
        if account:
            self.account_index[self._normalize_account(account)] = entity_id
        
        match_info['match_level'] = 0
        match_info['strategy'] = 'new_entity'
        match_info['confidence'] = 1.0
        
        return entity_id, match_info
    
    def _normalize_phone(self, phone: str) -> str:
        digits = re.sub(r'\D', '', str(phone))
        
        if len(digits) > 11:
            digits = digits[-11:]  
        
        return digits
    
    def _normalize_name(self, name: str) -> str:
        return str(name).strip().lower()

    def _normalize_wechat(self, handle: str) -> str:
        return str(handle).strip().lower()

    def _normalize_idcard(self, idcard: str) -> str:
        s = re.sub(r"\s+", "", str(idcard))
        return s.upper()

    def _normalize_account(self, account: str) -> str:
        return re.sub(r"\s+", "", str(account))
    
    def _find_best_name_match(self, norm_name: str) -> Tuple[Optional[str], float]:
        if not norm_name:
            return None, 0.0
        
        best_match_id = None
        best_score = 0.0
        
        for existing_norm_name, entity_id in self.name_index.items():
            seq_ratio = SequenceMatcher(None, norm_name, existing_norm_name).ratio()

            lev_dist = self._levenshtein_distance(norm_name, existing_norm_name)
            max_len = max(len(norm_name), len(existing_norm_name))
            lev_ratio = 1.0 - (lev_dist / max_len) if max_len > 0 else 0.0
            
            common_chars = set(norm_name) & set(existing_norm_name)
            union_chars = set(norm_name) | set(existing_norm_name)
            jaccard = len(common_chars) / len(union_chars) if union_chars else 0.0
            
            combined_score = 0.5 * seq_ratio + 0.3 * lev_ratio + 0.2 * jaccard
            
            adaptive_threshold = self.SEMANTIC_THRESHOLD - (0.1 if len(norm_name) <= 3 else 0.0)
            
            if combined_score > best_score and combined_score >= adaptive_threshold:
                best_score = combined_score
                best_match_id = entity_id
        
        return best_match_id, best_score
    
    def _find_cross_modal_match(self, entity_data: Dict[str, Any]) -> Optional[Tuple[str, float]]:
        name = entity_data.get('name')
        if not name:
            return None
        
        norm_name = self._normalize_name(name)
        best_match = None
        best_conf = 0.0
        for entity_id, entity in self.entities.items():
            existing_name = entity.get('name')
            if not existing_name:
                continue
            
            norm_existing = self._normalize_name(existing_name)
            name_sim = SequenceMatcher(None, norm_name, norm_existing).ratio()
            
            if name_sim < 0.75:
                continue
            

            matching_count = 0
            total_count = 0
            

            if entity_data.get('phone') and entity.get('phone'):
                total_count += 1
                if self._normalize_phone(entity_data.get('phone')) == self._normalize_phone(entity.get('phone')):
                    matching_count += 1
            
            if entity_data.get('account') and entity.get('account'):
                total_count += 1
                if self._normalize_account(entity_data.get('account')) == self._normalize_account(entity.get('account')):
                    matching_count += 1
            
            if total_count > 0:
                secondary_conf = matching_count / total_count
                combined_conf = 0.6 * name_sim + 0.4 * secondary_conf
                
                if combined_conf > best_conf:
                    best_conf = combined_conf
                    best_match = entity_id
        
        return (best_match, best_conf) if best_match and best_conf >= self.CROSS_MODAL_THRESHOLD else None
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        
        return prev_row[-1]
    
    def _merge_entity(self, entity_id: str, new_data: Dict[str, Any], match_info: Dict[str, Any]):
        entity = self.entities[entity_id]
        
        for key, value in new_data.items():
            if key in entity:
                existing = entity[key]
                if existing and existing != value:
                    existing_conf = entity.get('confidence', 1.0)
                    new_conf = new_data.get('confidence', 1.0)
                    
                    resolved, resolved_conf, strategy = self.conflict_resolver.resolve_conflict(
                        existing, value, existing_conf, new_conf, key
                    )
                    
                    entity[key] = resolved
                    entity[f'{key}_conflict_history'] = entity.get(f'{key}_conflict_history', [])
                    entity[f'{key}_conflict_history'].append({
                        'existing': existing,
                        'new': value,
                        'resolved': resolved,
                        'strategy': strategy
                    })
                elif not existing and value:
                    entity[key] = value
            else:
                entity[key] = value
        
        new_source = new_data.get('source', 'unknown')
        if new_source not in entity['sources']:
            entity['sources'].append(new_source)
        
        entity['merged_count'] += 1
        entity['match_history'].append(match_info)
        entity['last_updated'] = datetime.now().isoformat()
        
        alpha = 0.3  
        new_conf = new_data.get('confidence', 1.0)
        entity['confidence'] = alpha * new_conf + (1 - alpha) * entity['confidence']
    
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        return self.entities.get(entity_id)
    
    def get_all_entities(self) -> List[Dict[str, Any]]:
        return list(self.entities.values())
    
    def get_entity_by_phone(self, phone: str) -> Optional[Dict[str, Any]]:
        norm_phone = self._normalize_phone(phone)
        entity_id = self.phone_index.get(norm_phone)
        return self.entities.get(entity_id) if entity_id else None
    
    def get_entity_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        norm_name = self._normalize_name(name)
        entity_id = self.name_index.get(norm_name)
        return self.entities.get(entity_id) if entity_id else None
    
    def get_statistics(self) -> Dict[str, int]:
        return {
            'total_entities': len(self.entities),
            'phone_indexed': len(self.phone_index),
            'name_indexed': len(self.name_index),
            'wechat_indexed': len(self.wechat_index),
            'idcard_indexed': len(self.idcard_index),
            'account_indexed': len(self.account_index),
            'avg_merged_count': sum(e['merged_count'] for e in self.entities.values()) / len(self.entities) if self.entities else 0
        }

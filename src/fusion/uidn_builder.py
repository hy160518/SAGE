from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from .entity_matcher import EntityRegistry
from .graph_builder import RelationshipGraph

class UIDNStatistics:    
    def __init__(self):
        self.match_counts = {
            'level_1_deterministic': 0,
            'level_2_semantic': 0,
            'level_3_cross_modal': 0,
            'new_entities': 0
        }
        self.modality_counts = {
            'image': 0,
            'voice': 0,
            'text': 0,
            'external': 0
        }
        self.edge_counts = {
            'co-occurrence': 0,
            'temporal': 0,
            'semantic': 0,
            'multi-modal': 0,
            'external': 0
        }
        self.conflicts = []  
        self.temporal_coverage = None  
    def to_dict(self) -> Dict[str, Any]:
        return {
            'match_counts': self.match_counts,
            'modality_counts': self.modality_counts,
            'edge_counts': self.edge_counts,
            'conflict_count': len(self.conflicts),
            'temporal_coverage': self.temporal_coverage
        }

class UIDN:
    def __init__(self):
        self.registry = EntityRegistry()
        self.graph = RelationshipGraph()
        self.timeline = []
        self.stats = UIDNStatistics()
        self.entity_mappings = {}  
    
    def process_worker_results(self, results: Dict[str, List[Dict]]) -> Dict[str, Any]:

        print("\n[UIDN] Starting multi-modal entity fusion...")

        for modality, key in [('image', 'image_results'), 
                              ('voice', 'voice_results'), 
                              ('text', 'text_results')]:
            modality_results = results.get(key, [])
            
            for result in modality_results:
                entities = result.get('entities', [])
                
                for entity in entities:
                    entity['source'] = f"{modality}:{result.get('id', result.get('uuid', 'unknown'))}"
                    entity['modality'] = modality
                    entity['timestamp'] = result.get('timestamp')
                    
                    entity_id, match_info = self.registry.register_entity(entity)
                    
                    match_level = match_info.get('match_level', 0)
                    if match_level == 1:
                        self.stats.match_counts['level_1_deterministic'] += 1
                    elif match_level == 2:
                        self.stats.match_counts['level_2_semantic'] += 1
                    elif match_level == 3:
                        self.stats.match_counts['level_3_cross_modal'] += 1
                    else:
                        self.stats.match_counts['new_entities'] += 1
                    
                    self.stats.modality_counts[modality] += 1
                    
                    print(f"  [{modality}] Entity: {entity_id} (match_level: {match_level}, confidence: {match_info.get('confidence', 0):.2f})")
        
        print(f"[UIDN] Entity fusion completed: {len(self.registry.get_all_entities())} unique entities")
        
        return self.stats.to_dict()
    
    def build_relationship_graph(self) -> Dict[str, Any]:
        print("\n[UIDN] Building relationship graph...")
        
        entities = self.registry.get_all_entities()
        
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                sources1 = set(entity1['sources'])
                sources2 = set(entity2['sources'])
                common_sources = sources1 & sources2
                
                if common_sources:
                    edge = self.graph.add_edge(
                        entity1['entity_id'],
                        entity2['entity_id'],
                        relation_type=RelationshipGraph.COOCCURRENCE,
                        weight=len(common_sources),
                        metadata={'common_sources': list(common_sources)}
                    )
                    self.stats.edge_counts['co-occurrence'] += 1
                
                ts1 = entity1.get('timestamp')
                ts2 = entity2.get('timestamp')
                if ts1 and ts2:
                    try:
                        dt1 = datetime.fromisoformat(ts1) if isinstance(ts1, str) else ts1
                        dt2 = datetime.fromisoformat(ts2) if isinstance(ts2, str) else ts2
                        time_diff = abs((dt1 - dt2).total_seconds())
                        
                        if 0 < time_diff < 86400:
                            self.graph.add_edge(
                                entity1['entity_id'],
                                entity2['entity_id'],
                                relation_type=RelationshipGraph.TEMPORAL,
                                weight=1.0 / (1 + time_diff / 3600),
                                timestamp=str(min(dt1, dt2))
                            )
                            self.stats.edge_counts['temporal'] += 1
                    except (ValueError, KeyError, TypeError) as e:
                        import logging
                        logging.getLogger(__name__).debug(f"Temporal edge skip: {e}")
                
                mod1 = entity1.get('modality')
                mod2 = entity2.get('modality')
                if mod1 and mod2 and mod1 != mod2:
                    if entity1.get('match_history') and entity2.get('match_history'):
                        self.graph.add_edge(
                            entity1['entity_id'],
                            entity2['entity_id'],
                            relation_type=RelationshipGraph.MULTI_MODAL,
                            weight=0.8
                        )
                        self.stats.edge_counts['multi-modal'] += 1
        
        stats = self.graph.get_statistics()
        print(f"[UIDN] Graph built: {stats['node_count']} nodes, {stats['edge_count']} edges")
        
        return stats
    
    def detect_conflicts(self) -> List[Dict[str, Any]]:
        conflicts = []
        
        for entity in self.registry.get_all_entities():
            for key in entity.keys():
                if key.endswith('_conflict_history'):
                    conflict_history = entity[key]
                    for conflict in conflict_history:
                        conflicts.append({
                            'entity_id': entity['entity_id'],
                            'type': 'attribute_conflict',
                            'attribute': key.replace('_conflict_history', ''),
                            'details': conflict
                        })
            
            if self.graph.get_degree(entity['entity_id']) == 0 and entity['merged_count'] == 1:
                conflicts.append({
                    'entity_id': entity['entity_id'],
                    'type': 'isolated_entity',
                    'details': f"Entity {entity['entity_id']} has no relationships"
                })
        
        self.stats.conflicts = conflicts
        return conflicts
    
    def generate_timeline(self, min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        print("\n[UIDN] Generating timeline...")
        
        self.timeline = []
        
        entities = self.registry.get_all_entities()
        for entity in entities:
            if entity.get('confidence', 1.0) >= min_confidence:
                timestamp = entity.get('timestamp')
                if timestamp:
                    self.timeline.append({
                        'timestamp': timestamp,
                        'entity_id': entity['entity_id'],
                        'entity_name': entity.get('name', 'unknown'),
                        'event_type': 'entity_activity',
                        'modality': entity.get('modality'),
                        'source': entity.get('source'),
                        'confidence': entity.get('confidence', 1.0),
                        'details': entity
                    })
        
        self.timeline.sort(key=lambda x: x['timestamp'] or '')
        
        if self.timeline:
            self.stats.temporal_coverage = {
                'start': self.timeline[0]['timestamp'],
                'end': self.timeline[-1]['timestamp'],
                'events': len(self.timeline)
            }
        
        print(f"[UIDN] Timeline generated: {len(self.timeline)} events")
        
        return self.timeline
    
    def integrate_external_data(self, external_records: List[Dict[str, Any]]) -> Dict[str, int]:
        print(f"\n[UIDN] Integrating {len(external_records)} external records...")
        
        stats = {
            'processed': 0,
            'matched_to_existing': 0,
            'created_new': 0
        }
        
        for rec in external_records:
            entity_attrs = {
                'name': rec.get('name') or rec.get('user_name'),
                'phone': rec.get('phone') or rec.get('phone_number'),
                'wechat': rec.get('wechat') or rec.get('wechat_id'),
                'idcard': rec.get('id_card') or rec.get('id_card_number'),
                'account': rec.get('account') or rec.get('bank_account'),
                'timestamp': rec.get('timestamp'),
                'confidence': rec.get('confidence', 0.9),
                'source': f"external:{rec.get('source', 'unknown')}"
            }
            
            before_count = len(self.registry.entities)
            entity_id, match_info = self.registry.register_entity(entity_attrs)
            after_count = len(self.registry.entities)
            
            stats['processed'] += 1
            if before_count == after_count:
                stats['matched_to_existing'] += 1
            else:
                stats['created_new'] += 1
            
            self.stats.modality_counts['external'] += 1
            
            counterparty_phone = rec.get('counterparty_phone')
            if counterparty_phone:
                other = self.registry.get_entity_by_phone(counterparty_phone)
                if other:
                    edge = self.graph.add_edge(
                        entity_id, 
                        other['entity_id'],
                        relation_type=RelationshipGraph.EXTERNAL,
                        weight=rec.get('weight', 1.0),
                        timestamp=rec.get('timestamp'),
                        metadata={'record_type': rec.get('type', 'transaction')}
                    )
                    self.stats.edge_counts['external'] += 1
            
            if entity_attrs['timestamp']:
                self.timeline.append({
                    'timestamp': entity_attrs['timestamp'],
                    'entity_id': entity_id,
                    'event_type': rec.get('type', 'external_activity'),
                    'modality': 'external',
                    'source': entity_attrs['source'],
                    'details': rec
                })
        
        self.timeline.sort(key=lambda x: x['timestamp'] or '')
        
        print(f"[UIDN] External integration completed")
        
        return stats
    
    def export_results(self) -> Dict[str, Any]:
        return {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_entities': len(self.registry.get_all_entities()),
                'total_edges': self.graph.get_edge_count(),
                'timeline_events': len(self.timeline),
                'conflicts_detected': len(self.stats.conflicts)
            },
            'entities': self.registry.get_all_entities(),
            'entity_statistics': self.registry.get_statistics(),
            'relationship_graph': self.graph.export_to_dict(),
            'timeline': self.timeline,
            'conflicts': self.stats.conflicts,
            'processing_statistics': self.stats.to_dict()
        }
    
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        return self.registry.get_entity(entity_id)
    
    def get_entity_by_phone(self, phone: str) -> Optional[Dict[str, Any]]:
        return self.registry.get_entity_by_phone(phone)
    
    def get_entity_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        return self.registry.get_entity_by_name(name)
    
    def get_related_entities(self, entity_id: str) -> List[Tuple[str, float]]:
        return self.graph.get_neighbors(entity_id)
    
    def get_ego_network(self, entity_id: str, depth: int = 2) -> Dict[str, Any]:
        return self.graph.get_ego_network(entity_id, depth)
    
    def get_temporal_context(self, entity_id: str, time_window: int = 86400) -> List[Dict[str, Any]]:
        entity = self.get_entity(entity_id)
        if not entity or not entity.get('timestamp'):
            return []

        try:
            target_time = datetime.fromisoformat(entity['timestamp']) if isinstance(entity['timestamp'], str) else entity['timestamp']
        except (ValueError, TypeError) as e:
            import logging
            logging.getLogger(__name__).debug(f"Failed to parse timestamp: {e}")
            return []

        related = []
        for related_id, weight in self.get_related_entities(entity_id):
            related_entity = self.get_entity(related_id)
            if related_entity and related_entity.get('timestamp'):
                try:
                    related_time = datetime.fromisoformat(related_entity['timestamp']) if isinstance(related_entity['timestamp'], str) else related_entity['timestamp']
                    time_diff = abs((target_time - related_time).total_seconds())
                    if time_diff <= time_window:
                        related.append({
                            'entity_id': related_id,
                            'entity_name': related_entity.get('name'),
                            'time_diff_hours': time_diff / 3600,
                            'relationship_weight': weight
                        })
                except (ValueError, TypeError):
                    continue
        
        return sorted(related, key=lambda x: x['time_diff_hours'])


import networkx as nx
from community import community_louvain
from typing import List, Dict, Any, Tuple, Set
import json


class UIDNBuilder:

    ALPHA = 0.4  
    BETA = 0.3  
    GAMMA = 0.3  
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.graph = nx.Graph()
        self.entities = {}  # {entity_id: entity_data}
        self.entity_counter = 0
        
        weights = config.get('uidn', {}).get('weights', {})
        self.ALPHA = weights.get('alpha', self.ALPHA)
        self.BETA = weights.get('beta', self.BETA)
        self.GAMMA = weights.get('gamma', self.GAMMA)
        
    def build_graph(self, multimodal_results: Dict[str, List[Dict]]) -> nx.Graph:

        print("\nStart building...")
        
        all_entities = self._extract_all_entities(multimodal_results)
        print(f"Extracted  {len(all_entities)} entities")
        
        aligned_entities = self._align_entities(all_entities)
        print(f"Aligned to {len(aligned_entities)} unique entities")
        
        self._build_nodes(aligned_entities)
        
        self._build_edges(aligned_entities, multimodal_results)
        
        print(f"UIDN graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def detect_communities(self) -> Dict[str, List[str]]:

        print("\nExecuting community detection...")
        
        partition = community_louvain.best_partition(self.graph)
        
        communities = {}
        for node, community_id in partition.items():
            community_key = f"community_{community_id}"
            if community_key not in communities:
                communities[community_key] = []
            communities[community_key].append(node)
        
        print(f"Detected {len(communities)} communities")
        for comm_id, members in communities.items():
            print(f"  {comm_id}: {len(members)} 个成员")
        
        return communities
    
    def _extract_all_entities(self, multimodal_results: Dict) -> List[Dict]:
        all_entities = []
        
        for result in multimodal_results.get('text_results', []):
            entities_data = result.get('entities', {})
            
            for person in entities_data.get('persons', []):
                all_entities.append({
                    'type': 'person',
                    'name': person.get('name'),
                    'role': person.get('role'),
                    'source': 'text',
                    'uuid': result.get('uuid'),
                    'confidence': result.get('confidence', 0.5)
                })
            
            for drug in entities_data.get('drugs', []):
                all_entities.append({
                    'type': 'drug',
                    'name': drug.get('name'),
                    'quantity': drug.get('quantity'),
                    'source': 'text',
                    'uuid': result.get('uuid'),
                    'confidence': result.get('confidence', 0.5)
                })
        
        for result in multimodal_results.get('image_results', []):
            classification = result.get('classification', {})
            if classification.get('category') == 'person_image':
                all_entities.append({
                    'type': 'person',
                    'name': f"person_image_{result.get('uuid')}",
                    'source': 'image',
                    'uuid': result.get('uuid'),
                    'confidence': result.get('confidence', 0.5)
                })
        
        return all_entities
    
    def _align_entities(self, entities: List[Dict]) -> List[Dict]:

        aligned = []
        processed = set()
        
        for i, entity in enumerate(entities):
            if i in processed:
                continue
            
            similar_group = [entity]
            processed.add(i)
            
            for j, other_entity in enumerate(entities[i+1:], start=i+1):
                if j in processed:
                    continue
                
                similarity = self._calculate_entity_similarity(entity, other_entity)
                
                if similarity > 0.7:
                    similar_group.append(other_entity)
                    processed.add(j)
            
            fused_entity = self._fuse_entity_group(similar_group)
            aligned.append(fused_entity)
        
        return aligned
    
    def _calculate_entity_similarity(self, entity1: Dict, entity2: Dict) -> float:

        if entity1.get('type') != entity2.get('type'):
            return 0.0

        name1 = entity1.get('name', '').lower()
        name2 = entity2.get('name', '').lower()
        
        if name1 == name2:
            name_sim = 1.0
        elif name1 in name2 or name2 in name1:
            name_sim = 0.8
        else:
            name_sim = 0.0
        
        conf1 = entity1.get('confidence', 0.5)
        conf2 = entity2.get('confidence', 0.5)
        conf_factor = (conf1 + conf2) / 2
        
        similarity = name_sim * conf_factor
        
        return similarity
    
    def _fuse_entity_group(self, entity_group: List[Dict]) -> Dict:

        sorted_group = sorted(entity_group, key=lambda x: x.get('confidence', 0), reverse=True)

        fused = sorted_group[0].copy()
        
        fused['sources'] = [e.get('source') for e in entity_group]
        fused['uuids'] = [e.get('uuid') for e in entity_group]
        fused['mention_count'] = len(entity_group)
        
        self.entity_counter += 1
        fused['entity_id'] = f"entity_{self.entity_counter}"
        
        return fused
    
    def _build_nodes(self, entities: List[Dict]):

        for entity in entities:
            entity_id = entity['entity_id']
            self.entities[entity_id] = entity
            
            self.graph.add_node(
                entity_id,
                type=entity.get('type'),
                name=entity.get('name'),
                confidence=entity.get('confidence'),
                mention_count=entity.get('mention_count', 1)
            )
    
    def _build_edges(self, entities: List[Dict], multimodal_results: Dict):

        uuid_entities = {}
        for entity in entities:
            for uuid in entity.get('uuids', []):
                if uuid not in uuid_entities:
                    uuid_entities[uuid] = []
                uuid_entities[uuid].append(entity['entity_id'])
        
        for uuid, entity_ids in uuid_entities.items():
            for i, id1 in enumerate(entity_ids):
                for id2 in entity_ids[i+1:]:
                    weight = self._calculate_edge_weight(
                        self.entities[id1],
                        self.entities[id2]
                    )
                    
                    self.graph.add_edge(id1, id2, weight=weight, relation='co-occurrence')
        
        for result in multimodal_results.get('text_results', []):
            relations = result.get('entities', {}).get('relations', [])
            for rel in relations:

                pass
    
    def _calculate_edge_weight(self, entity1: Dict, entity2: Dict) -> float:

        common_uuids = set(entity1.get('uuids', [])) & set(entity2.get('uuids', []))
        co_occurrence = len(common_uuids)
        
        conf1 = entity1.get('confidence', 0.5)
        conf2 = entity2.get('confidence', 0.5)
        avg_confidence = (conf1 + conf2) / 2
        
        weight = co_occurrence * avg_confidence
        
        return weight
    
    def export_graph(self, output_path: str):
        graph_data = {
            'nodes': [
                {
                    'id': node,
                    **self.graph.nodes[node]
                }
                for node in self.graph.nodes()
            ],
            'edges': [
                {
                    'source': u,
                    'target': v,
                    **self.graph.edges[u, v]
                }
                for u, v in self.graph.edges()
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ The UIDN graph has been exported to: {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'connected_components': nx.number_connected_components(self.graph),
            'average_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0
        }

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
        self.entities = {}
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
            item_id = result.get('uuid') or result.get('id', '')
            confidence = result.get('confidence', 0.8)

            for person in entities_data.get('persons', []):
                name = person.get('name')
                if name:
                    all_entities.append({
                        'type': 'person',
                        'name': name,
                        'role': person.get('role', 'unknown'),
                        'source': 'text',
                        'uuid': item_id,
                        'confidence': confidence
                    })

            for drug in entities_data.get('drugs', []):
                name = drug.get('name')
                if name:
                    all_entities.append({
                        'type': 'drug',
                        'name': name,
                        'quantity': drug.get('quantity', ''),
                        'source': 'text',
                        'uuid': item_id,
                        'confidence': confidence
                    })

        for result in multimodal_results.get('image_results', []):
            classification = result.get('classification', {})
            if classification.get('category') == 'person_image':
                item_id = result.get('uuid') or result.get('id', '')
                all_entities.append({
                    'type': 'person',
                    'name': f"person_image_{item_id}",
                    'source': 'image',
                    'uuid': item_id,
                    'confidence': result.get('confidence', 0.5)
                })

        return all_entities

    def _align_entities(self, entities: List[Dict]) -> List[Dict]:
        aligned = []
        processed = set()

        name_groups = {}
        for i, entity in enumerate(entities):
            if i in processed:
                continue
            name = entity.get('name', '').lower().strip()
            if name:
                if name not in name_groups:
                    name_groups[name] = []
                name_groups[name].append((i, entity))

        for name, group in name_groups.items():
            group_entities = [e for _, e in group]
            fused = self._fuse_entity_group(group_entities, match_level='deterministic')
            aligned.append(fused)
            for idx, _ in group:
                processed.add(idx)

        remaining = [(i, e) for i, e in enumerate(entities) if i not in processed]
        for i, entity in remaining:
            if i in processed:
                continue

            similar_group = [entity]
            processed.add(i)

            name1 = entity.get('name', '').lower()

            for j, other_entity in remaining[i + 1:]:
                if j in processed:
                    continue

                match_result = self._calculate_entity_similarity(entity, other_entity)

                if match_result['level'] == 'deterministic' and match_result['score'] > 0.95:
                    similar_group.append(other_entity)
                    processed.add(j)
                elif match_result['level'] in ('semantic', 'cross-modal') and match_result['score'] > 0.6:
                    similar_group.append(other_entity)
                    processed.add(j)

            if len(similar_group) > 1:
                fused = self._fuse_entity_group(similar_group, match_result['level'])
            else:
                fused = self._fuse_entity_group(similar_group, 'single')
            aligned.append(fused)

        return aligned

    def _calculate_entity_similarity(self, entity1: Dict, entity2: Dict) -> Dict:
        if entity1.get('type') != entity2.get('type'):
            return {'score': 0.0, 'level': 'none'}

        name1 = entity1.get('name', '').lower().strip()
        name2 = entity2.get('name', '').lower().strip()

        if not name1 or not name2:
            return {'score': 0.0, 'level': 'none'}

        if name1 == name2:
            conf1 = entity1.get('confidence', 0.5)
            conf2 = entity2.get('confidence', 0.5)
            avg_conf = (conf1 + conf2) / 2
            return {'score': avg_conf, 'level': 'deterministic'}

        if name1 in name2 or name2 in name1:
            conf1 = entity1.get('confidence', 0.5)
            conf2 = entity2.get('confidence', 0.5)
            avg_conf = (conf1 + conf2) / 2
            return {'score': 0.8 * avg_conf, 'level': 'semantic'}

        source1 = entity1.get('source', 'text')
        source2 = entity2.get('source', 'text')

        if source1 != source2:
            edit_dist = self._levenshtein_distance(name1, name2)
            max_len = max(len(name1), len(name2))
            if max_len > 0:
                str_sim = 1.0 - (edit_dist / max_len)
                if str_sim > 0.6:
                    conf1 = entity1.get('confidence', 0.5)
                    conf2 = entity2.get('confidence', 0.5)
                    avg_conf = (conf1 + conf2) / 2
                    return {'score': str_sim * avg_conf * 0.7, 'level': 'cross-modal'}

        return {'score': 0.0, 'level': 'none'}

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _fuse_entity_group(self, entity_group: List[Dict], match_level: str = 'semantic') -> Dict:
        if not entity_group:
            return {}

        sorted_group = sorted(entity_group, key=lambda x: x.get('confidence', 0), reverse=True)

        fused = sorted_group[0].copy()

        fused['sources'] = list(set([e.get('source') for e in entity_group]))
        fused['uuids'] = [e.get('uuid') for e in entity_group if e.get('uuid')]
        fused['mention_count'] = len(entity_group)
        fused['match_level'] = match_level

        confidences = [e.get('confidence', 0.5) for e in entity_group]
        if match_level == 'deterministic':
            fused['confidence'] = min(1.0, sum(confidences) / len(confidences) * 1.2)
        elif match_level == 'semantic':
            fused['confidence'] = sum(confidences) / len(confidences)
        else:
            fused['confidence'] = sum(confidences) / len(confidences) * 0.8

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

        entity_name_to_id = {}
        for entity in entities:
            name = entity.get('name', '').lower()
            if name and entity['entity_id'] not in entity_name_to_id:
                entity_name_to_id[name] = entity['entity_id']

        for result in multimodal_results.get('text_results', []):
            relations = result.get('entities', {}).get('relations', [])
            for rel in relations:
                source_name = rel.get('source', '').lower()
                target_name = rel.get('target', '').lower()
                rel_type = rel.get('type', 'associated')

                source_id = entity_name_to_id.get(source_name)
                target_id = entity_name_to_id.get(target_name)

                if source_id and target_id and source_id != target_id:
                    if self.graph.has_edge(source_id, target_id):
                        current_weight = self.graph[source_id][target_id].get('weight', 0)
                        new_weight = min(1.0, current_weight + 0.3)
                        self.graph[source_id][target_id]['weight'] = new_weight
                    else:
                        self.graph.add_edge(
                            source_id, target_id,
                            weight=0.3,
                            relation=rel_type
                        )

        for result in multimodal_results.get('image_results', []):
            image_persons = result.get('persons', [])
            for person in image_persons:
                person_name = person.get('name', '').lower()
                text_entity_id = entity_name_to_id.get(person_name)
                if text_entity_id:
                    image_entity_id = result.get('uuid', '')
                    if image_entity_id:
                        entity_id = f"image_{image_entity_id}"
                        if entity_id in self.entities:
                            if self.graph.has_edge(text_entity_id, entity_id):
                                self.graph[text_entity_id][entity_id]['weight'] = min(1.0,
                                    self.graph[text_entity_id][entity_id]['weight'] + 0.5)
                            else:
                                self.graph.add_edge(
                                    text_entity_id, entity_id,
                                    weight=0.5,
                                    relation='cross-modal'
                                )
    
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

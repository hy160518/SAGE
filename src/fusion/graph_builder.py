from typing import Dict, List, Any, Tuple, Set
from datetime import datetime
from collections import defaultdict

class RelationshipEdge:

    COOCCURRENCE = 'co-occurrence'     
    TEMPORAL = 'temporal'              
    SEMANTIC = 'semantic'              
    MULTI_MODAL = 'multi-modal'        
    EXTERNAL = 'external'              
    
    def __init__(self, source: str, target: str, relation_type: str = COOCCURRENCE,
                 weight: float = 1.0, timestamp: str = None, metadata: Dict = None):
        self.source = source
        self.target = target
        self.relation_type = relation_type
        self.weight = weight
        self.timestamp = timestamp or datetime.now().isoformat()
        self.metadata = metadata or {}
        self.frequency = 1  
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source,
            'target': self.target,
            'relation_type': self.relation_type,
            'weight': self.weight,
            'timestamp': self.timestamp,
            'frequency': self.frequency,
            'metadata': self.metadata
        }

class RelationshipGraph:
    def __init__(self):
        self.nodes = set()
        self.edges = []  
        self.adjacency = defaultdict(list)  
        self.edge_map = {}  
        
        self.temporal_index = defaultdict(list)  
        
        self.relation_type_index = defaultdict(list)  
    
    def add_node(self, node_id: str, node_type: str = 'entity', attributes: Dict = None):
        self.nodes.add(node_id)
        if node_id not in self.adjacency:
            self.adjacency[node_id] = []
    
    def add_edge(self, source: str, target: str, relation_type: str = 'co-occurrence',
                 weight: float = 1.0, timestamp: str = None, metadata: Dict = None) -> RelationshipEdge:
        self.add_node(source)
        self.add_node(target)
        
        edge_key = (source, target) if source <= target else (target, source)
        
        if edge_key in self.edge_map:
            edge = self.edge_map[edge_key]
            edge.weight += weight
            edge.frequency += 1
            
            if timestamp and timestamp < edge.timestamp:
                edge.timestamp = timestamp
        else:
            edge = RelationshipEdge(source, target, relation_type, weight, timestamp, metadata)
            self.edges.append(edge)
            self.edge_map[edge_key] = edge
            
            self.adjacency[source].append((target, edge))
            if source != target:
                self.adjacency[target].append((source, edge))
        
        self.relation_type_index[relation_type].append(edge)
        if timestamp:
            self.temporal_index[timestamp].append(edge)
        
        return edge
    
    def get_neighbors(self, node_id: str) -> List[Tuple[str, float]]:
        neighbors = []
        for neighbor_id, edge in self.adjacency.get(node_id, []):
            neighbors.append((neighbor_id, edge.weight))
        return neighbors
    
    def get_edges(self, relation_type: str = None) -> List[Dict[str, Any]]:
        if relation_type:
            return [edge.to_dict() for edge in self.relation_type_index.get(relation_type, [])]
        return [edge.to_dict() for edge in self.edges]
    
    def get_edge_count(self) -> int:
        return len(self.edges)
    
    def get_node_count(self) -> int:
        return len(self.nodes)
    
    def get_degree(self, node_id: str) -> int:
        return len(self.adjacency.get(node_id, []))
    
    def get_weighted_degree(self, node_id: str) -> float:
        return sum(weight for _, weight in self.get_neighbors(node_id))
    
    def get_statistics(self) -> Dict[str, Any]:
        if not self.nodes:
            return {
                'node_count': 0,
                'edge_count': 0,
                'avg_degree': 0,
                'max_degree': 0,
                'min_degree': 0,
                'density': 0,
                'relation_type_distribution': {}
            }
        
        degrees = [self.get_degree(node) for node in self.nodes]
        total_weight = sum(edge.weight for edge in self.edges)
        
        rel_dist = defaultdict(int)
        for edge in self.edges:
            rel_dist[edge.relation_type] += 1
        
        possible_edges = len(self.nodes) * (len(self.nodes) - 1) / 2
        density = len(self.edges) / possible_edges if possible_edges > 0 else 0
        
        return {
            'node_count': self.get_node_count(),
            'edge_count': self.get_edge_count(),
            'avg_degree': sum(degrees) / len(degrees) if degrees else 0,
            'max_degree': max(degrees) if degrees else 0,
            'min_degree': min(degrees) if degrees else 0,
            'avg_edge_weight': total_weight / len(self.edges) if self.edges else 0,
            'total_weight': total_weight,
            'density': density,
            'relation_type_distribution': dict(rel_dist),
            'connected_components': len(self.find_connected_components())
        }
    
    def find_connected_components(self) -> List[Set[str]]:
        visited = set()
        components = []
        
        def dfs(node, component):
            visited.add(node)
            component.add(node)
            for neighbor, _ in self.get_neighbors(node):
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        for node in self.nodes:
            if node not in visited:
                component = set()
                dfs(node, component)
                components.append(component)
        
        return components
    
    def get_shortest_path(self, source: str, target: str) -> List[str]:
        if source not in self.nodes or target not in self.nodes:
            return []
        
        if source == target:
            return [source]
        
        queue = [(source, [source])]
        visited = {source}
        
        while queue:
            node, path = queue.pop(0)
            
            for neighbor, _ in self.get_neighbors(node):
                if neighbor == target:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return []
    
    def get_temporal_edges(self, start_time: str = None, end_time: str = None) -> List[Dict[str, Any]]:
        filtered_edges = []
        for edge in self.edges:
            if edge.timestamp:
                if start_time and edge.timestamp < start_time:
                    continue
                if end_time and edge.timestamp > end_time:
                    continue
            filtered_edges.append(edge.to_dict())
        return filtered_edges
    
    def get_ego_network(self, node_id: str, depth: int = 1) -> Dict[str, Any]:
        ego = {node_id}
        current_layer = {node_id}
        
        for _ in range(depth):
            next_layer = set()
            for node in current_layer:
                for neighbor, _ in self.get_neighbors(node):
                    if neighbor not in ego:
                        next_layer.add(neighbor)
                        ego.add(neighbor)
            current_layer = next_layer
        
        ego_edges = [e.to_dict() for e in self.edges if e.source in ego and e.target in ego]
        
        return {
            'ego_node': node_id,
            'depth': depth,
            'nodes': list(ego),
            'node_count': len(ego),
            'edge_count': len(ego_edges),
            'edges': ego_edges
        }
    
    def export_to_dict(self) -> Dict[str, Any]:
        return {
            'nodes': list(self.nodes),
            'edges': [e.to_dict() for e in self.edges],
            'statistics': self.get_statistics()
        }

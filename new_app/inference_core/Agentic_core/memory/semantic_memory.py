"""
Semantic Memory with Knowledge Graph Storage.

Adheres to:
- Algorithmic Complexity: O(1) entity lookup, O(E) neighborhood traversal (E=edges).
- Memory Layout: Adjacency list representation for cache-friendly iteration.
- Zero-Copy: Direct dict access without serialization overhead.
- Failure Domain: Result types for all operations.
"""
import asyncio
import logging
import json
from typing import Dict, Set, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
from ..core.result import Result, Ok, Err
from ..core.config import get_config

logger = logging.getLogger(__name__)
CONFIG = get_config()

# ============================================================================
# KNOWLEDGE GRAPH ARCHITECTURE
# ============================================================================
# Representation: Directed Property Graph
# - Nodes: Entities (concepts, objects, people)
# - Edges: Relations (typed, directional)
# - Properties: Attributes on both nodes and edges
#
# Storage Format:
# {
#   "entities": {
#     "entity_id": {"type": "Person", "name": "Alice", ...},
#     ...
#   },
#   "relations": {
#     "entity_id": [
#       ("relation_type", "target_entity_id", {"since": "2024", ...}),
#       ...
#     ]
#   }
# }
#
# Complexity Analysis:
# - Add entity: O(1)
# - Add relation: O(1)  
# - Get neighbors: O(deg(v)) where deg=out-degree
# - Path finding (BFS): O(V + E)
# ============================================================================

@dataclass
class Entity:
    """
    Knowledge graph entity (node).
    
    Field ordering (descending size):
    - properties: Dict (8 bytes pointer)
    - entity_id: str (8 bytes pointer)
    - entity_type: str (8 bytes pointer)
    """
    entity_id: str
    entity_type: str
    properties: Dict[str, Any]


@dataclass
class Relation:
    """
    Knowledge graph relation (edge).
    
    Field ordering (descending size):
    - properties: Dict (8 bytes pointer)
    - source_id: str (8 bytes pointer)
    - target_id: str (8 bytes pointer)
    - relation_type: str (8 bytes pointer)
    """
    source_id: str
    relation_type: str
    target_id: str
    properties: Dict[str, Any]


class SemanticMemory:
    """
    Knowledge graph for semantic relationships.
    
    Performance Characteristics:
    - Entity lookup: O(1) via hash map
    - Relation traversal: O(deg(v)) per entity
    - Path finding: O(V + E) via BFS
    - Storage: ~100KB per 1000 entities (in-memory)
    """
    
    def __init__(self, persistence_path: Optional[str] = None):
        """
        Initialize semantic memory.
        
        Args:
            persistence_path: Path to save/load knowledge graph (JSON)
        """
        # Adjacency list: entity_id -> [(relation_type, target_id, properties)]
        self._graph: Dict[str, List[Tuple[str, str, Dict]]] = defaultdict(list)
        
        # Entity metadata: entity_id -> properties
        self._entities: Dict[str, Dict[str, Any]] = {}
        
        # Reverse index for efficient backward traversal
        self._reverse_graph: Dict[str, List[Tuple[str, str, Dict]]] = defaultdict(list)
        
        self.persistence_path = persistence_path or f"{CONFIG.storage_path.replace('.db', '_kg.json')}"
        self._dirty = False  # Track if graph needs saving
    
    async def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> Result[bool, Exception]:
        """
        Add or update entity.
        
        Complexity: O(1)
        
        Args:
            entity_id: Unique entity identifier
            entity_type: Entity type (Person, Concept, Object, etc.)
            properties: Optional key-value properties
            
        Returns:
            Ok(True) on success
        """
        try:
            if not entity_id or not entity_type:
                return Err(ValueError("entity_id and entity_type required"))
            
            self._entities[entity_id] = {
                "type": entity_type,
                **(properties or {})
            }
            
            self._dirty = True
            logger.debug(f"Added entity: {entity_id} ({entity_type})")
            return Ok(True)
            
        except Exception as e:
            return Err(e)
    
    async def add_relation(
        self,
        source_id: str,
        relation_type: str,
        target_id: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> Result[bool, Exception]:
        """
        Add directed relation between entities.
        
        Complexity: O(1)
        
        Args:
            source_id: Source entity ID
            relation_type: Relation type (WORKS_AT, KNOWS, PART_OF, etc.)
            target_id: Target entity ID
            properties: Optional relation properties
            
        Returns:
            Ok(True) on success
        """
        try:
            # Validate entities exist
            if source_id not in self._entities:
                return Err(ValueError(f"Source entity {source_id} not found"))
            if target_id not in self._entities:
                return Err(ValueError(f"Target entity {target_id} not found"))
            
            # Add to forward adjacency list
            edge_data = (relation_type, target_id, properties or {})
            self._graph[source_id].append(edge_data)
            
            # Add to reverse index
            reverse_edge = (relation_type, source_id, properties or {})
            self._reverse_graph[target_id].append(reverse_edge)
            
            self._dirty = True
            logger.debug(f"Added relation: {source_id} --[{relation_type}]--> {target_id}")
            return Ok(True)
            
        except Exception as e:
            return Err(e)
    
    async def get_entity(self, entity_id: str) -> Result[Optional[Entity], Exception]:
        """
        Retrieve entity by ID.
        
        Complexity: O(1)
        """
        try:
            if entity_id not in self._entities:
                return Ok(None)
            
            props = self._entities[entity_id]
            entity = Entity(
                entity_id=entity_id,
                entity_type=props["type"],
                properties={k: v for k, v in props.items() if k != "type"}
            )
            
            return Ok(entity)
            
        except Exception as e:
            return Err(e)
    
    async def get_neighbors(
        self,
        entity_id: str,
        relation_type: Optional[str] = None,
        direction: str = "outgoing"
    ) -> Result[List[Relation], Exception]:
        """
        Get neighboring entities.
        
        Complexity: O(deg(v)) where deg=out-degree or in-degree
        
        Args:
            entity_id: Entity to query
            relation_type: Optional filter by relation type
            direction: "outgoing", "incoming", or "both"
            
        Returns:
            Ok(List[Relation]) of matching relations
        """
        try:
            if entity_id not in self._entities:
                return Err(ValueError(f"Entity {entity_id} not found"))
            
            relations = []
            
            # Outgoing relations
            if direction in ["outgoing", "both"]:
                for rel_type, target_id, props in self._graph.get(entity_id, []):
                    if relation_type is None or rel_type == relation_type:
                        relations.append(Relation(
                            source_id=entity_id,
                            relation_type=rel_type,
                            target_id=target_id,
                            properties=props
                        ))
            
            # Incoming relations
            if direction in ["incoming", "both"]:
                for rel_type, source_id, props in self._reverse_graph.get(entity_id, []):
                    if relation_type is None or rel_type == relation_type:
                        relations.append(Relation(
                            source_id=source_id,
                            relation_type=rel_type,
                            target_id=entity_id,
                            properties=props
                        ))
            
            return Ok(relations)
            
        except Exception as e:
            return Err(e)
    
    async def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 3
    ) -> Result[Optional[List[str]], Exception]:
        """
        Find shortest path between entities using BFS.
        
        Complexity: O(V + E) worst case
        
        Args:
            start_id: Starting entity
            end_id: Target entity
            max_depth: Maximum path length
            
        Returns:
            Ok(List[entity_ids]) representing path, or Ok(None) if no path
        """
        try:
            if start_id not in self._entities or end_id not in self._entities:
                return Err(ValueError("Start or end entity not found"))
            
            # BFS implementation
            from collections import deque
            
            queue = deque([(start_id, [start_id])])
            visited = {start_id}
            
            while queue:
                current_id, path = queue.popleft()
                
                # Check depth limit
                if len(path) > max_depth:
                    continue
                
                # Found target
                if current_id == end_id:
                    return Ok(path)
                
                # Explore neighbors
                for rel_type, neighbor_id, props in self._graph.get(current_id, []):
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append((neighbor_id, path + [neighbor_id]))
            
            # No path found
            return Ok(None)
            
        except Exception as e:
            return Err(e)
    
    async def query_by_property(
        self,
        entity_type: Optional[str] = None,
        **property_filters
    ) -> Result[List[Entity], Exception]:
        """
        Query entities by type and/or properties.
        
        Complexity: O(n) where n = total entities
        
        Args:
            entity_type: Filter by entity type
            **property_filters: Key-value filters
            
        Returns:
            Ok(List[Entity]) matching filters
        """
        try:
            matches = []
            
            for entity_id, props in self._entities.items():
                # Type filter
                if entity_type and props.get("type") != entity_type:
                    continue
                
                # Property filters
                match = True
                for key, value in property_filters.items():
                    if props.get(key) != value:
                        match = False
                        break
                
                if match:
                    matches.append(Entity(
                        entity_id=entity_id,
                        entity_type=props["type"],
                        properties={k: v for k, v in props.items() if k != "type"}
                    ))
            
            return Ok(matches)
            
        except Exception as e:
            return Err(e)
    
    async def save(self) -> Result[bool, Exception]:
        """
        Persist knowledge graph to disk.
        
        Format: JSON with entities and adjacency list.
        """
        try:
            if not self._dirty:
                return Ok(True)  # No changes
            
            data = {
                "entities": self._entities,
                "relations": {
                    entity_id: [
                        {
                            "type": rel_type,
                            "target": target_id,
                            "properties": props
                        }
                        for rel_type, target_id, props in edges
                    ]
                    for entity_id, edges in self._graph.items()
                }
            }
            
            import os
            os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
            
            with open(self.persistence_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self._dirty = False
            logger.info(f"Saved knowledge graph to {self.persistence_path}")
            return Ok(True)
            
        except Exception as e:
            return Err(e)
    
    async def load(self) -> Result[bool, Exception]:
        """
        Load knowledge graph from disk.
        """
        try:
            import os
            if not os.path.exists(self.persistence_path):
                logger.info("No existing knowledge graph found, starting fresh")
                return Ok(True)
            
            with open(self.persistence_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._entities = data.get("entities", {})
            
            # Rebuild adjacency lists
            self._graph.clear()
            self._reverse_graph.clear()
            
            relations = data.get("relations", {})
            for source_id, edges in relations.items():
                for edge in edges:
                    rel_type = edge["type"]
                    target_id = edge["target"]
                    props = edge.get("properties", {})
                    
                    self._graph[source_id].append((rel_type, target_id, props))
                    self._reverse_graph[target_id].append((rel_type, source_id, props))
            
            logger.info(
                f"Loaded knowledge graph: {len(self._entities)} entities, "
                f"{sum(len(edges) for edges in self._graph.values())} relations"
            )
            self._dirty = False
            return Ok(True)
            
        except Exception as e:
            return Err(e)
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get graph statistics.
        """
        total_entities = len(self._entities)
        total_relations = sum(len(edges) for edges in self._graph.values())
        
        entity_types = defaultdict(int)
        for props in self._entities.values():
            entity_types[props.get("type", "unknown")] += 1
        
        relation_types = defaultdict(int)
        for edges in self._graph.values():
            for rel_type, _, _ in edges:
                relation_types[rel_type] += 1
        
        return {
            "total_entities": total_entities,
            "total_relations": total_relations,
            "entity_types": dict(entity_types),
            "relation_types": dict(relation_types),
            "dirty": self._dirty
        }

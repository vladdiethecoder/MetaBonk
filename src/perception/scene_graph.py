"""Scene Graph for Spatial Reasoning.

Relational scene representation for game understanding:
- Entity relationship extraction (NEAR, FAR, BEHIND, etc.)
- Spatial reasoning primitives (collision, path-finding)
- Threat/resource classification and prioritization

References:
- Johnson et al., "Image Retrieval using Scene Graphs"
- SIMA 2: Perception layer scene understanding

Usage:
    from grounded_sam import GroundedSAMPerception
    from scene_graph import SceneGraph
    
    perception = GroundedSAMPerception()
    entities = perception.detect(frame)
    graph = SceneGraph.from_entities(entities)
    
    # Query: find enemies near player
    enemies = graph.query("enemy NEAR player", max_distance=100)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from enum import Enum, auto
import math

import numpy as np

# Import from sibling module
try:
    from .grounded_sam import Entity, EntityCategory, BoundingBox
except ImportError:
    # Fallback for standalone testing
    from grounded_sam import Entity, EntityCategory, BoundingBox


class SpatialRelation(Enum):
    """Spatial relationships between entities."""
    NEAR = auto()           # Within close distance
    FAR = auto()            # Beyond far distance
    LEFT_OF = auto()        # To the left
    RIGHT_OF = auto()       # To the right
    ABOVE = auto()          # Above vertically
    BELOW = auto()          # Below vertically
    INSIDE = auto()         # Bounding boxes overlap significantly
    TOUCHING = auto()       # Boxes touch but don't overlap much
    FACING = auto()         # Entity velocity points toward target
    APPROACHING = auto()    # Distance decreasing
    RETREATING = auto()     # Distance increasing


@dataclass
class SceneGraphEdge:
    """Edge in the scene graph representing a relationship."""
    source_id: str
    target_id: str
    relation: SpatialRelation
    distance: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneGraphConfig:
    """Configuration for scene graph construction."""
    
    # Distance thresholds (in pixels)
    near_threshold: float = 100.0
    far_threshold: float = 300.0
    touch_threshold: float = 10.0
    
    # Relation computation
    compute_directional: bool = True
    compute_temporal: bool = True  # Velocity-based relations
    
    # Frame dimensions for normalization
    frame_width: int = 1920
    frame_height: int = 1080


class SceneGraph:
    """Graph-based scene representation for spatial reasoning.
    
    Nodes: Entities detected in the scene
    Edges: Spatial relationships between entities
    
    Supports:
    - Relational queries ("enemy NEAR player")
    - Path finding (can entity A reach entity B?)
    - Threat assessment (which enemies are approaching?)
    - Resource prioritization (nearest pickup)
    """
    
    def __init__(self, cfg: Optional[SceneGraphConfig] = None):
        self.cfg = cfg or SceneGraphConfig()
        self.nodes: Dict[str, Entity] = {}
        self.edges: List[SceneGraphEdge] = []
        self.timestamp: float = 0.0
        
        # Index for fast queries
        self._edge_index: Dict[str, List[SceneGraphEdge]] = {}
        
    @classmethod
    def from_entities(
        cls,
        entities: List[Entity],
        cfg: Optional[SceneGraphConfig] = None,
        prev_graph: Optional["SceneGraph"] = None,
    ) -> "SceneGraph":
        """Construct scene graph from detected entities."""
        graph = cls(cfg)
        
        # Add nodes
        for entity in entities:
            graph.nodes[entity.entity_id] = entity
        
        # Compute all pairwise relations
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                edges = graph._compute_relations(e1, e2, prev_graph)
                graph.edges.extend(edges)
        
        # Build index
        graph._build_edge_index()
        
        return graph
    
    def _compute_relations(
        self,
        e1: Entity,
        e2: Entity,
        prev_graph: Optional["SceneGraph"],
    ) -> List[SceneGraphEdge]:
        """Compute all relations between two entities."""
        edges = []
        
        # Distance
        distance = e1.distance_to(e2)
        
        # Near/Far
        if distance <= self.cfg.near_threshold:
            edges.append(SceneGraphEdge(
                e1.entity_id, e2.entity_id, SpatialRelation.NEAR, distance
            ))
            edges.append(SceneGraphEdge(
                e2.entity_id, e1.entity_id, SpatialRelation.NEAR, distance
            ))
        elif distance >= self.cfg.far_threshold:
            edges.append(SceneGraphEdge(
                e1.entity_id, e2.entity_id, SpatialRelation.FAR, distance
            ))
            edges.append(SceneGraphEdge(
                e2.entity_id, e1.entity_id, SpatialRelation.FAR, distance
            ))
        
        # Directional relations
        if self.cfg.compute_directional:
            dx = e2.position[0] - e1.position[0]
            dy = e2.position[1] - e1.position[1]
            
            # Horizontal
            if abs(dx) > abs(dy):
                if dx > 0:
                    edges.append(SceneGraphEdge(
                        e2.entity_id, e1.entity_id, SpatialRelation.LEFT_OF, distance
                    ))
                    edges.append(SceneGraphEdge(
                        e1.entity_id, e2.entity_id, SpatialRelation.RIGHT_OF, distance
                    ))
                else:
                    edges.append(SceneGraphEdge(
                        e1.entity_id, e2.entity_id, SpatialRelation.LEFT_OF, distance
                    ))
                    edges.append(SceneGraphEdge(
                        e2.entity_id, e1.entity_id, SpatialRelation.RIGHT_OF, distance
                    ))
            else:
                # Vertical
                if dy > 0:
                    edges.append(SceneGraphEdge(
                        e1.entity_id, e2.entity_id, SpatialRelation.ABOVE, distance
                    ))
                    edges.append(SceneGraphEdge(
                        e2.entity_id, e1.entity_id, SpatialRelation.BELOW, distance
                    ))
                else:
                    edges.append(SceneGraphEdge(
                        e2.entity_id, e1.entity_id, SpatialRelation.ABOVE, distance
                    ))
                    edges.append(SceneGraphEdge(
                        e1.entity_id, e2.entity_id, SpatialRelation.BELOW, distance
                    ))
        
        # Overlap/Touch
        iou = e1.box.iou(e2.box)
        if iou > 0.3:
            edges.append(SceneGraphEdge(
                e1.entity_id, e2.entity_id, SpatialRelation.INSIDE, distance,
                {"iou": iou}
            ))
        elif iou > 0 or distance < self.cfg.touch_threshold:
            edges.append(SceneGraphEdge(
                e1.entity_id, e2.entity_id, SpatialRelation.TOUCHING, distance
            ))
        
        # Temporal relations (velocity-based)
        if self.cfg.compute_temporal and prev_graph:
            edges.extend(self._compute_temporal_relations(e1, e2, prev_graph))
        
        return edges
    
    def _compute_temporal_relations(
        self,
        e1: Entity,
        e2: Entity,
        prev_graph: "SceneGraph",
    ) -> List[SceneGraphEdge]:
        """Compute velocity-based temporal relations."""
        edges = []
        
        prev_e1 = prev_graph.nodes.get(e1.entity_id)
        prev_e2 = prev_graph.nodes.get(e2.entity_id)
        
        if prev_e1 and prev_e2:
            prev_dist = prev_e1.distance_to(prev_e2)
            curr_dist = e1.distance_to(e2)
            
            if curr_dist < prev_dist - 5:  # Threshold for noise
                edges.append(SceneGraphEdge(
                    e1.entity_id, e2.entity_id, SpatialRelation.APPROACHING, curr_dist,
                    {"delta": prev_dist - curr_dist}
                ))
            elif curr_dist > prev_dist + 5:
                edges.append(SceneGraphEdge(
                    e1.entity_id, e2.entity_id, SpatialRelation.RETREATING, curr_dist,
                    {"delta": curr_dist - prev_dist}
                ))
        
        # Check if e1 is facing e2 (velocity direction)
        if e1.velocity != (0.0, 0.0):
            vx, vy = e1.velocity
            dx = e2.position[0] - e1.position[0]
            dy = e2.position[1] - e1.position[1]
            
            # Dot product normalized
            v_mag = math.sqrt(vx**2 + vy**2)
            d_mag = math.sqrt(dx**2 + dy**2)
            
            if v_mag > 0 and d_mag > 0:
                dot = (vx * dx + vy * dy) / (v_mag * d_mag)
                if dot > 0.7:  # ~45 degree cone
                    edges.append(SceneGraphEdge(
                        e1.entity_id, e2.entity_id, SpatialRelation.FACING,
                        e1.distance_to(e2), {"angle_similarity": dot}
                    ))
        
        return edges
    
    def _build_edge_index(self):
        """Build index for fast edge lookup."""
        self._edge_index.clear()
        for edge in self.edges:
            if edge.source_id not in self._edge_index:
                self._edge_index[edge.source_id] = []
            self._edge_index[edge.source_id].append(edge)
    
    def get_relations(
        self,
        entity_id: str,
        relation: Optional[SpatialRelation] = None,
    ) -> List[SceneGraphEdge]:
        """Get all relations for an entity, optionally filtered by type."""
        edges = self._edge_index.get(entity_id, [])
        if relation:
            return [e for e in edges if e.relation == relation]
        return edges
    
    def query(
        self,
        pattern: str,
        max_distance: Optional[float] = None,
    ) -> List[Entity]:
        """Query entities matching a relational pattern.
        
        Pattern format: "<label> <RELATION> <label>"
        Examples:
            "enemy NEAR player"
            "pickup LEFT_OF player"
            "projectile APPROACHING player"
        """
        parts = pattern.upper().split()
        if len(parts) != 3:
            raise ValueError(f"Invalid pattern: {pattern}. Expected '<label> <RELATION> <label>'")
        
        source_label, relation_name, target_label = parts
        
        # Parse relation
        try:
            relation = SpatialRelation[relation_name]
        except KeyError:
            raise ValueError(f"Unknown relation: {relation_name}")
        
        # Find matching entities
        results = []
        
        # Find all target entities
        targets = [e for e in self.nodes.values() 
                   if target_label.lower() in e.label.lower() or 
                   target_label.lower() in e.category.name.lower()]
        
        for entity in self.nodes.values():
            if source_label.lower() not in entity.label.lower() and \
               source_label.lower() not in entity.category.name.lower():
                continue
            
            # Check if entity has the required relation to any target
            for edge in self.get_relations(entity.entity_id, relation):
                target = self.nodes.get(edge.target_id)
                if target and target in targets:
                    if max_distance is None or edge.distance <= max_distance:
                        results.append(entity)
                        break
        
        return results
    
    def get_threats(self, player_id: Optional[str] = None) -> List[Tuple[Entity, float]]:
        """Get threats sorted by danger (distance * category weight).
        
        Returns list of (entity, threat_score) tuples.
        """
        if player_id is None:
            # Find player entity
            players = [e for e in self.nodes.values() 
                       if e.category == EntityCategory.PLAYER]
            if not players:
                return []
            player = players[0]
        else:
            player = self.nodes.get(player_id)
            if not player:
                return []
        
        # Weight by category
        category_weights = {
            EntityCategory.BOSS: 2.0,
            EntityCategory.ENEMY: 1.0,
            EntityCategory.PROJECTILE: 1.5,
        }
        
        threats = []
        for entity in self.nodes.values():
            weight = category_weights.get(entity.category)
            if weight is None:
                continue
            
            distance = entity.distance_to(player)
            # Inverse distance: closer = higher threat
            threat_score = weight * (1000.0 / max(distance, 1.0))
            
            # Bonus for approaching
            for edge in self.get_relations(entity.entity_id, SpatialRelation.APPROACHING):
                if edge.target_id == player.entity_id:
                    threat_score *= 1.5
                    break
            
            threats.append((entity, threat_score))
        
        # Sort by threat (highest first)
        return sorted(threats, key=lambda x: x[1], reverse=True)
    
    def get_pickups(
        self,
        player_id: Optional[str] = None,
        max_distance: Optional[float] = None,
    ) -> List[Tuple[Entity, float]]:
        """Get pickups sorted by priority (value / distance)."""
        if player_id is None:
            players = [e for e in self.nodes.values() 
                       if e.category == EntityCategory.PLAYER]
            if not players:
                return []
            player = players[0]
        else:
            player = self.nodes.get(player_id)
            if not player:
                return []
        
        pickups = []
        for entity in self.nodes.values():
            if entity.category != EntityCategory.PICKUP:
                continue
            
            distance = entity.distance_to(player)
            if max_distance and distance > max_distance:
                continue
            
            # Priority based on type
            value = 1.0
            label_lower = entity.label.lower()
            if "health" in label_lower:
                value = 3.0
            elif "gem" in label_lower or "xp" in label_lower:
                value = 1.0
            elif "chest" in label_lower:
                value = 2.0
            
            priority = value / max(distance, 1.0)
            pickups.append((entity, priority))
        
        return sorted(pickups, key=lambda x: x[1], reverse=True)
    
    def to_observation(
        self,
        player_id: Optional[str] = None,
        max_entities: int = 32,
    ) -> np.ndarray:
        """Convert scene graph to fixed-size observation vector.
        
        Output: [max_entities * 6] vector where each entity has:
            [rel_x, rel_y, vx, vy, category_onehot(2)]
        """
        if player_id is None:
            players = [e for e in self.nodes.values() 
                       if e.category == EntityCategory.PLAYER]
            player = players[0] if players else None
        else:
            player = self.nodes.get(player_id)
        
        obs = np.zeros(max_entities * 6)
        
        # Get entities sorted by distance to player
        entities = []
        for entity in self.nodes.values():
            if entity.entity_id == (player.entity_id if player else None):
                continue
            if player:
                distance = entity.distance_to(player)
            else:
                distance = 0
            entities.append((entity, distance))
        
        entities = sorted(entities, key=lambda x: x[1])[:max_entities]
        
        for i, (entity, _) in enumerate(entities):
            base = i * 6
            
            # Relative position (normalized)
            if player:
                rel_x = (entity.position[0] - player.position[0]) / self.cfg.frame_width
                rel_y = (entity.position[1] - player.position[1]) / self.cfg.frame_height
            else:
                rel_x = entity.position[0] / self.cfg.frame_width
                rel_y = entity.position[1] / self.cfg.frame_height
            
            obs[base] = rel_x
            obs[base + 1] = rel_y
            obs[base + 2] = entity.velocity[0] / 100.0  # Normalize velocity
            obs[base + 3] = entity.velocity[1] / 100.0
            
            # Category encoding (threat=1, pickup=-1, else=0)
            if entity.category in [EntityCategory.ENEMY, EntityCategory.BOSS, EntityCategory.PROJECTILE]:
                obs[base + 4] = 1.0  # Threat
            elif entity.category == EntityCategory.PICKUP:
                obs[base + 5] = 1.0  # Pickup
        
        return obs
    
    def visualize(self) -> str:
        """Return ASCII visualization of the scene graph."""
        lines = ["Scene Graph:"]
        lines.append(f"  Nodes: {len(self.nodes)}")
        lines.append(f"  Edges: {len(self.edges)}")
        lines.append("")
        
        for entity_id, entity in list(self.nodes.items())[:10]:
            lines.append(f"  [{entity.category.name}] {entity.label} @ {entity.position}")
            for edge in self.get_relations(entity_id)[:3]:
                target = self.nodes.get(edge.target_id)
                if target:
                    lines.append(f"    --{edge.relation.name}--> {target.label}")
        
        return "\n".join(lines)


__all__ = [
    "SceneGraph",
    "SceneGraphConfig",
    "SceneGraphEdge",
    "SpatialRelation",
]

"""Grounded-SAM Perception System.

Zero-shot semantic scene understanding for game environments:
- Grounding DINO: Text-prompted object detection (prompt → boxes)
- SAM: Segment Anything Model for precise masks (boxes → masks)
- Real-time entity tracking and classification

References:
- Liu et al., "Grounding DINO: Marrying DINO with Grounded Pre-Training"
- Kirillov et al., "Segment Anything"
- SIMA 2: Perception layer for semantic scene graphs

Capabilities:
- Zero-shot: Detect any entity via text prompts ("enemy", "health potion")
- Precise segmentation: Pixel-level masks for occlusion handling
- Temporal tracking: Consistent entity IDs across frames
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from enum import Enum, auto
import hashlib

import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


class EntityCategory(Enum):
    """High-level entity categories for Megabonk."""
    PLAYER = auto()
    ENEMY = auto()
    BOSS = auto()
    PROJECTILE = auto()
    PICKUP = auto()
    OBSTACLE = auto()
    UI_ELEMENT = auto()
    UNKNOWN = auto()


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def iou(self, other: "BoundingBox") -> float:
        """Intersection over Union."""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def to_xyxy(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]
    
    def to_xywh(self) -> List[float]:
        return [self.x1, self.y1, self.width, self.height]


@dataclass
class Entity:
    """Detected entity in the game world."""
    entity_id: str
    label: str
    category: EntityCategory
    box: BoundingBox
    confidence: float
    mask: Optional[np.ndarray] = None  # [H, W] binary mask
    
    # Temporal tracking
    track_id: Optional[int] = None
    velocity: Tuple[float, float] = (0.0, 0.0)
    last_seen: float = 0.0
    
    # Game-specific attributes
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def position(self) -> Tuple[float, float]:
        return self.box.center
    
    def distance_to(self, other: "Entity") -> float:
        """Euclidean distance to another entity."""
        dx = self.position[0] - other.position[0]
        dy = self.position[1] - other.position[1]
        return np.sqrt(dx**2 + dy**2)


@dataclass
class GroundedSAMConfig:
    """Configuration for Grounded-SAM perception."""
    
    # Model configs
    grounding_dino_model: str = "IDEA-Research/grounding-dino-tiny"
    sam_model: str = "facebook/sam-vit-base"
    device: str = "cuda"
    
    # Detection thresholds
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    
    # Default prompts for Megabonk
    default_prompts: List[str] = field(default_factory=lambda: [
        "player character",
        "enemy monster",
        "boss",
        "health pickup",
        "experience gem",
        "treasure chest",
        "projectile bullet",
        "damage number",
        "menu button",
        "upgrade card",
    ])
    
    # Category mappings
    category_map: Dict[str, EntityCategory] = field(default_factory=lambda: {
        "player": EntityCategory.PLAYER,
        "player character": EntityCategory.PLAYER,
        "enemy": EntityCategory.ENEMY,
        "enemy monster": EntityCategory.ENEMY,
        "monster": EntityCategory.ENEMY,
        "boss": EntityCategory.BOSS,
        "health": EntityCategory.PICKUP,
        "health pickup": EntityCategory.PICKUP,
        "gem": EntityCategory.PICKUP,
        "experience gem": EntityCategory.PICKUP,
        "chest": EntityCategory.PICKUP,
        "treasure chest": EntityCategory.PICKUP,
        "projectile": EntityCategory.PROJECTILE,
        "bullet": EntityCategory.PROJECTILE,
        "projectile bullet": EntityCategory.PROJECTILE,
        "button": EntityCategory.UI_ELEMENT,
        "menu button": EntityCategory.UI_ELEMENT,
        "card": EntityCategory.UI_ELEMENT,
        "upgrade card": EntityCategory.UI_ELEMENT,
    })
    
    # Tracking
    iou_threshold: float = 0.3
    max_age: float = 0.5  # seconds before track is lost
    
    # Performance
    resize_for_detection: Optional[Tuple[int, int]] = (640, 480)
    use_fp16: bool = True


class EntityTracker:
    """Simple IoU-based entity tracking across frames."""
    
    def __init__(self, iou_threshold: float = 0.3, max_age: float = 0.5):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: Dict[int, Entity] = {}
        self.next_id = 0
        
    def update(
        self,
        entities: List[Entity],
        timestamp: float,
    ) -> List[Entity]:
        """Update tracks with new detections."""
        if not self.tracks:
            # First frame: assign new IDs
            for entity in entities:
                entity.track_id = self.next_id
                entity.last_seen = timestamp
                self.tracks[self.next_id] = entity
                self.next_id += 1
            return entities
        
        # Match detections to existing tracks via Hungarian algorithm (simplified)
        unmatched_detections = list(range(len(entities)))
        matched: Set[int] = set()
        
        for track_id, track in list(self.tracks.items()):
            best_iou = 0.0
            best_idx = -1
            
            for i in unmatched_detections:
                if entities[i].label != track.label:
                    continue
                iou = entities[i].box.iou(track.box)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_idx = i
            
            if best_idx >= 0:
                # Match found: update track
                entity = entities[best_idx]
                entity.track_id = track_id
                entity.velocity = (
                    entity.position[0] - track.position[0],
                    entity.position[1] - track.position[1],
                )
                entity.last_seen = timestamp
                self.tracks[track_id] = entity
                unmatched_detections.remove(best_idx)
                matched.add(track_id)
        
        # Prune old tracks
        to_remove = []
        for track_id, track in self.tracks.items():
            if track_id not in matched and timestamp - track.last_seen > self.max_age:
                to_remove.append(track_id)
        for track_id in to_remove:
            del self.tracks[track_id]
        
        # New tracks for unmatched detections
        for i in unmatched_detections:
            entity = entities[i]
            entity.track_id = self.next_id
            entity.last_seen = timestamp
            self.tracks[self.next_id] = entity
            self.next_id += 1
        
        return entities


class GroundedSAMPerception:
    """Zero-shot perception using Grounding DINO + SAM.
    
    Usage:
        perception = GroundedSAMPerception()
        entities = perception.detect(frame, ["enemy", "health pickup"])
    """
    
    def __init__(self, cfg: Optional[GroundedSAMConfig] = None):
        self.cfg = cfg or GroundedSAMConfig()
        self.tracker = EntityTracker(
            iou_threshold=self.cfg.iou_threshold,
            max_age=self.cfg.max_age,
        )
        
        # Lazy model loading
        self._grounding_dino = None
        self._sam = None
        self._sam_processor = None
        self._grounding_processor = None
        
    def _load_models(self):
        """Lazy load models on first use."""
        if self._grounding_dino is not None:
            return
        
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            from transformers import SamModel, SamProcessor
            
            device = self.cfg.device
            
            # Load Grounding DINO
            self._grounding_processor = AutoProcessor.from_pretrained(
                self.cfg.grounding_dino_model
            )
            self._grounding_dino = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.cfg.grounding_dino_model
            ).to(device)
            
            if self.cfg.use_fp16 and HAS_TORCH:
                self._grounding_dino = self._grounding_dino.half()
            
            # Load SAM
            self._sam_processor = SamProcessor.from_pretrained(self.cfg.sam_model)
            self._sam = SamModel.from_pretrained(self.cfg.sam_model).to(device)
            
            if self.cfg.use_fp16 and HAS_TORCH:
                self._sam = self._sam.half()
                
        except Exception as e:
            raise RuntimeError(f"[GroundedSAM] Failed to load models: {type(e).__name__}: {e}") from e
    
    def _categorize(self, label: str) -> EntityCategory:
        """Map label to category."""
        label_lower = label.lower().strip()
        return self.cfg.category_map.get(label_lower, EntityCategory.UNKNOWN)
    
    def _generate_entity_id(self, label: str, box: BoundingBox) -> str:
        """Generate deterministic entity ID."""
        data = f"{label}:{box.x1:.0f}:{box.y1:.0f}:{box.x2:.0f}:{box.y2:.0f}"
        return hashlib.md5(data.encode()).hexdigest()[:8]
    
    def detect(
        self,
        frame: np.ndarray,
        prompts: Optional[List[str]] = None,
        with_masks: bool = True,
        track: bool = True,
    ) -> List[Entity]:
        """Detect entities in frame using text prompts.
        
        Args:
            frame: RGB image [H, W, 3]
            prompts: Text prompts for detection (uses defaults if None)
            with_masks: Whether to compute SAM masks
            track: Whether to track entities across frames
            
        Returns:
            List of detected entities
        """
        self._load_models()
        prompts = prompts or self.cfg.default_prompts

        if not HAS_PIL:
            raise RuntimeError("[GroundedSAM] PIL is required for perception.")
        if not HAS_TORCH:
            raise RuntimeError("[GroundedSAM] torch is required for perception.")
        if self._grounding_dino is None or self._grounding_processor is None:
            raise RuntimeError("[GroundedSAM] GroundingDINO model is not initialized.")
        
        timestamp = time.time()
        
        # Convert to PIL
        image = Image.fromarray(frame)
        
        # Resize for efficiency
        if self.cfg.resize_for_detection:
            orig_size = image.size
            image = image.resize(self.cfg.resize_for_detection)
            scale_x = orig_size[0] / self.cfg.resize_for_detection[0]
            scale_y = orig_size[1] / self.cfg.resize_for_detection[1]
        else:
            scale_x = scale_y = 1.0
        
        # Combine prompts into text query
        text_query = ". ".join(prompts)
        
        entities = []
        
        try:
            # Grounding DINO detection
            inputs = self._grounding_processor(
                images=image,
                text=text_query,
                return_tensors="pt"
            ).to(self.cfg.device)
            
            with torch.no_grad():
                outputs = self._grounding_dino(**inputs)
            
            # Post-process detections
            results = self._grounding_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=self.cfg.box_threshold,
                text_threshold=self.cfg.text_threshold,
                target_sizes=[image.size[::-1]],  # (H, W)
            )[0]
            
            boxes = results["boxes"].cpu().numpy()
            scores = results["scores"].cpu().numpy()
            labels = results["labels"]
            
            # Create entities
            for i in range(len(boxes)):
                box = BoundingBox(
                    x1=boxes[i][0] * scale_x,
                    y1=boxes[i][1] * scale_y,
                    x2=boxes[i][2] * scale_x,
                    y2=boxes[i][3] * scale_y,
                )
                
                label = labels[i]
                entity = Entity(
                    entity_id=self._generate_entity_id(label, box),
                    label=label,
                    category=self._categorize(label),
                    box=box,
                    confidence=float(scores[i]),
                )
                entities.append(entity)
            
            # SAM segmentation
            if with_masks and entities and self._sam is not None:
                entities = self._add_masks(image, entities)
                
        except Exception as e:
            raise RuntimeError(f"[GroundedSAM] Detection error: {type(e).__name__}: {e}") from e
        
        # Tracking
        if track:
            entities = self.tracker.update(entities, timestamp)
        
        return entities
    
    def _add_masks(
        self,
        image: "Image.Image",
        entities: List[Entity],
    ) -> List[Entity]:
        """Add SAM masks to entities."""
        if not HAS_TORCH or self._sam is None:
            return entities
        
        try:
            # Prepare boxes for SAM
            boxes = [[e.box.x1, e.box.y1, e.box.x2, e.box.y2] for e in entities]
            
            inputs = self._sam_processor(
                image,
                input_boxes=[boxes],
                return_tensors="pt"
            ).to(self.cfg.device)
            
            with torch.no_grad():
                outputs = self._sam(**inputs)
            
            masks = self._sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )[0]
            
            # Assign masks to entities
            for i, entity in enumerate(entities):
                if i < len(masks):
                    # Take first mask (highest score)
                    entity.mask = masks[i, 0].numpy().astype(np.uint8) * 255
                    
        except Exception as e:
            print(f"[GroundedSAM] SAM error: {e}")
        
        return entities
    
    def get_entities_by_category(
        self,
        entities: List[Entity],
        category: EntityCategory,
    ) -> List[Entity]:
        """Filter entities by category."""
        return [e for e in entities if e.category == category]
    
    def get_nearest_entity(
        self,
        entities: List[Entity],
        position: Tuple[float, float],
        category: Optional[EntityCategory] = None,
    ) -> Optional[Entity]:
        """Get nearest entity to a position."""
        filtered = entities
        if category:
            filtered = self.get_entities_by_category(entities, category)
        
        if not filtered:
            return None
        
        # Create reference entity for distance calculation
        ref = Entity(
            entity_id="ref",
            label="ref",
            category=EntityCategory.UNKNOWN,
            box=BoundingBox(position[0], position[1], position[0], position[1]),
            confidence=1.0,
        )
        
        return min(filtered, key=lambda e: e.distance_to(ref))
    
    def get_threat_level(self, entities: List[Entity]) -> float:
        """Estimate threat level from detected entities (0-1)."""
        enemies = self.get_entities_by_category(entities, EntityCategory.ENEMY)
        bosses = self.get_entities_by_category(entities, EntityCategory.BOSS)
        projectiles = self.get_entities_by_category(entities, EntityCategory.PROJECTILE)
        
        # Weighted threat calculation
        threat = (
            len(enemies) * 0.1 +
            len(bosses) * 0.5 +
            len(projectiles) * 0.15
        )
        
        return min(1.0, threat)


__all__ = [
    "GroundedSAMPerception",
    "GroundedSAMConfig",
    "Entity",
    "EntityCategory",
    "BoundingBox",
    "EntityTracker",
]

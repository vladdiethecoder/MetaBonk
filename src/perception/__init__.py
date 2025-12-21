"""Perception package initialization."""

from .capture import (
    DXGICapture,
    FallbackCapture,
    CaptureConfig,
    CapturedFrame,
    RingBuffer,
    create_capture,
)

from .slot_attention import (
    SlotAttentionConfig,
    SlotAttentionAutoEncoder,
    SlotAttentionModule,
    DualFrequencyPerception,
    ConvEncoder,
    SlotDecoder,
)

from .grounded_sam import (
    GroundedSAMPerception,
    GroundedSAMConfig,
    Entity,
    EntityCategory,
    BoundingBox,
    EntityTracker,
)

from .scene_graph import (
    SceneGraph,
    SceneGraphConfig,
    SceneGraphEdge,
    SpatialRelation,
)

__all__ = [
    # Capture
    "DXGICapture",
    "FallbackCapture",
    "CaptureConfig",
    "CapturedFrame",
    "RingBuffer",
    "create_capture",
    # Slot Attention
    "SlotAttentionConfig",
    "SlotAttentionAutoEncoder",
    "SlotAttentionModule",
    "DualFrequencyPerception",
    "ConvEncoder",
    "SlotDecoder",
    # Grounded-SAM
    "GroundedSAMPerception",
    "GroundedSAMConfig",
    "Entity",
    "EntityCategory",
    "BoundingBox",
    "EntityTracker",
    # Scene Graph
    "SceneGraph",
    "SceneGraphConfig",
    "SceneGraphEdge",
    "SpatialRelation",
]


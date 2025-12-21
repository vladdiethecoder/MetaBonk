"""Neural Telemetry Visualization with Rerun.io.

Implements multimodal visualization for the "mind" of the AI:
- 3D spatial attention vectors
- Value estimate graphs
- Policy entropy heatmaps
- Time-travel debugging

References:
- Rerun.io SDK
- Neuromorphic Broadcast Engine specification
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import rerun as rr
    import rerun.blueprint as rrb
    HAS_RERUN = True
except ImportError:
    HAS_RERUN = False


@dataclass
class NeuralState:
    """Neural state of an agent at a point in time."""
    
    timestamp: float
    agent_id: int
    
    # Position
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    
    # Value estimate
    value_estimate: float = 0.0
    
    # Policy
    policy_entropy: float = 0.0
    action_probs: Optional[np.ndarray] = None
    
    # Attention
    attention_targets: List[Tuple[np.ndarray, float]] = field(default_factory=list)
    # List of (target_position, attention_weight)
    
    # Emotions (derived)
    fear: float = 0.0
    confidence: float = 0.0
    aggression: float = 0.0


@dataclass
class VisualizationConfig:
    """Configuration for neural visualization."""
    
    # Recording
    recording_name: str = "metabonk_neural"
    
    # Colors
    agent_color: Tuple[int, int, int] = (0, 255, 128)
    enemy_color: Tuple[int, int, int] = (255, 64, 64)
    attention_high_color: Tuple[int, int, int] = (255, 0, 0)
    attention_low_color: Tuple[int, int, int] = (64, 64, 255)
    
    # Sizes
    agent_size: float = 0.5
    enemy_size: float = 0.3
    attention_line_width: float = 2.0
    
    # History
    history_seconds: float = 5.0
    position_trail_length: int = 60


class NeuralVisualizer:
    """Visualizes agent neural states using Rerun.io.
    
    Features:
    - 3D attention vectors
    - Value/entropy graphs
    - Time-travel debugging
    """
    
    def __init__(self, cfg: Optional[VisualizationConfig] = None):
        self.cfg = cfg or VisualizationConfig()
        self.initialized = False
        
        # History for trails and time-travel
        self.position_history: Dict[int, List[np.ndarray]] = {}
        self.value_history: Dict[int, List[Tuple[float, float]]] = {}
        self.entropy_history: Dict[int, List[Tuple[float, float]]] = {}
        
        # Current frame
        self.current_time: float = 0.0
    
    def initialize(self):
        """Initialize Rerun recording."""
        if not HAS_RERUN:
            print("Rerun not installed - visualization disabled")
            return
        
        rr.init(self.cfg.recording_name, spawn=True)
        
        # Set up blueprint (layout)
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(name="World", origin="/world"),
                rrb.Vertical(
                    rrb.TimeSeriesView(name="Value Estimate", origin="/metrics/value"),
                    rrb.TimeSeriesView(name="Entropy", origin="/metrics/entropy"),
                ),
            )
        )
        
        rr.send_blueprint(blueprint)
        
        self.initialized = True
        print("Rerun visualization initialized")
    
    def log_agent_state(
        self,
        state: NeuralState,
        enemies: Optional[List[np.ndarray]] = None,
    ):
        """Log agent neural state to Rerun."""
        if not self.initialized or not HAS_RERUN:
            return
        
        self.current_time = state.timestamp
        rr.set_time_seconds("game_time", state.timestamp)
        
        agent_id = state.agent_id
        
        # Log agent position
        rr.log(
            f"/world/agent_{agent_id}/position",
            rr.Points3D(
                [state.position],
                colors=[self.cfg.agent_color],
                radii=[self.cfg.agent_size],
            ),
        )
        
        # Position trail
        if agent_id not in self.position_history:
            self.position_history[agent_id] = []
        
        self.position_history[agent_id].append(state.position.copy())
        
        # Trim trail
        if len(self.position_history[agent_id]) > self.cfg.position_trail_length:
            self.position_history[agent_id].pop(0)
        
        if len(self.position_history[agent_id]) >= 2:
            trail = np.array(self.position_history[agent_id])
            rr.log(
                f"/world/agent_{agent_id}/trail",
                rr.LineStrips3D(
                    [trail],
                    colors=[(0, 255, 128, 128)],  # Semi-transparent
                ),
            )
        
        # Log enemies
        if enemies:
            rr.log(
                f"/world/enemies",
                rr.Points3D(
                    enemies,
                    colors=[self.cfg.enemy_color] * len(enemies),
                    radii=[self.cfg.enemy_size] * len(enemies),
                ),
            )
        
        # Log attention vectors
        if state.attention_targets:
            attention_lines = []
            attention_colors = []
            
            for target_pos, weight in state.attention_targets:
                attention_lines.append([state.position, target_pos])
                
                # Interpolate color based on weight
                t = min(weight, 1.0)
                color = (
                    int(self.cfg.attention_low_color[0] * (1-t) + self.cfg.attention_high_color[0] * t),
                    int(self.cfg.attention_low_color[1] * (1-t) + self.cfg.attention_high_color[1] * t),
                    int(self.cfg.attention_low_color[2] * (1-t) + self.cfg.attention_high_color[2] * t),
                )
                attention_colors.append(color)
            
            rr.log(
                f"/world/agent_{agent_id}/attention",
                rr.LineStrips3D(
                    attention_lines,
                    colors=attention_colors,
                    radii=[self.cfg.attention_line_width * w for _, w in state.attention_targets],
                ),
            )
        
        # Log metrics
        rr.log(
            f"/metrics/value/agent_{agent_id}",
            rr.Scalar(state.value_estimate),
        )
        
        rr.log(
            f"/metrics/entropy/agent_{agent_id}",
            rr.Scalar(state.policy_entropy),
        )
        
        # Store history
        if agent_id not in self.value_history:
            self.value_history[agent_id] = []
            self.entropy_history[agent_id] = []
        
        self.value_history[agent_id].append((state.timestamp, state.value_estimate))
        self.entropy_history[agent_id].append((state.timestamp, state.policy_entropy))
        
        # Trim history
        cutoff = state.timestamp - self.cfg.history_seconds
        self.value_history[agent_id] = [
            (t, v) for t, v in self.value_history[agent_id] if t > cutoff
        ]
        self.entropy_history[agent_id] = [
            (t, v) for t, v in self.entropy_history[agent_id] if t > cutoff
        ]
    
    def log_emotions(self, agent_id: int, fear: float, confidence: float, aggression: float):
        """Log derived emotional states."""
        if not self.initialized or not HAS_RERUN:
            return
        
        rr.log(f"/metrics/emotions/fear", rr.Scalar(fear))
        rr.log(f"/metrics/emotions/confidence", rr.Scalar(confidence))
        rr.log(f"/metrics/emotions/aggression", rr.Scalar(aggression))
    
    def log_decision(
        self,
        agent_id: int,
        options: List[str],
        probabilities: List[float],
        chosen: int,
    ):
        """Log decision visualization."""
        if not self.initialized or not HAS_RERUN:
            return
        
        # Bar chart of action probabilities
        rr.log(
            f"/decisions/agent_{agent_id}",
            rr.BarChart(probabilities),
        )
        
        # Text annotation
        text = f"Chose: {options[chosen]} ({probabilities[chosen]:.1%})"
        rr.log(
            f"/decisions/agent_{agent_id}/text",
            rr.TextLog(text),
        )
    
    def set_time(self, timestamp: float):
        """Set the current visualization time (for time-travel)."""
        if HAS_RERUN:
            rr.set_time_seconds("game_time", timestamp)
            self.current_time = timestamp


class NeuralHUD:
    """Generates HUD overlay data for broadcast."""
    
    def __init__(self):
        self.active_agent: Optional[int] = None
        self.last_state: Optional[NeuralState] = None
    
    def update(self, state: NeuralState):
        """Update HUD with new state."""
        self.active_agent = state.agent_id
        self.last_state = state
    
    def get_hud_data(self) -> Dict[str, Any]:
        """Get data for rendering HUD overlay."""
        if self.last_state is None:
            return {}
        
        state = self.last_state
        
        # Compute "emotions" from neural metrics
        # Fear: high when value estimate drops sharply
        # Confidence: low entropy = high confidence
        # Aggression: derived from action distribution
        
        fear = max(0, -state.value_estimate) / 10.0  # Normalize
        confidence = 1.0 - min(state.policy_entropy, 1.0)
        aggression = state.aggression
        
        return {
            "agent_id": state.agent_id,
            "position": state.position.tolist(),
            "value_estimate": state.value_estimate,
            "entropy": state.policy_entropy,
            "fear": fear,
            "confidence": confidence,
            "aggression": aggression,
            "attention_count": len(state.attention_targets),
            "top_attention": state.attention_targets[:3] if state.attention_targets else [],
        }


class TelemetryBridge:
    """Bridge between game telemetry and visualization.
    
    Receives data from BepInEx SideChannel and feeds visualizers.
    """
    
    def __init__(self, visualizer: NeuralVisualizer, hud: NeuralHUD):
        self.visualizer = visualizer
        self.hud = hud
        
        # Buffer for batch processing
        self.state_buffer: List[NeuralState] = []
        self.buffer_size = 10
    
    def process_telemetry(
        self,
        raw_data: bytes,
        timestamp: Optional[float] = None,
    ):
        """Process raw telemetry from game."""
        import struct
        
        ts = timestamp or time.time()
        
        # Parse binary format (simplified)
        # Format: agent_id(4) + pos(12) + vel(12) + value(4) + entropy(4) + attention_count(4) + ...
        
        try:
            if len(raw_data) < 40:
                return
            
            offset = 0
            agent_id = struct.unpack("<I", raw_data[offset:offset+4])[0]
            offset += 4
            
            position = np.array(struct.unpack("<3f", raw_data[offset:offset+12]))
            offset += 12
            
            velocity = np.array(struct.unpack("<3f", raw_data[offset:offset+12]))
            offset += 12
            
            value = struct.unpack("<f", raw_data[offset:offset+4])[0]
            offset += 4
            
            entropy = struct.unpack("<f", raw_data[offset:offset+4])[0]
            offset += 4
            
            # Create state
            state = NeuralState(
                timestamp=ts,
                agent_id=agent_id,
                position=position,
                velocity=velocity,
                value_estimate=value,
                policy_entropy=entropy,
            )
            
            # Parse attention targets if present
            if len(raw_data) > offset + 4:
                attention_count = struct.unpack("<I", raw_data[offset:offset+4])[0]
                offset += 4
                
                for _ in range(min(attention_count, 10)):  # Cap at 10
                    if len(raw_data) < offset + 16:
                        break
                    
                    target = np.array(struct.unpack("<3f", raw_data[offset:offset+12]))
                    weight = struct.unpack("<f", raw_data[offset+12:offset+16])[0]
                    state.attention_targets.append((target, weight))
                    offset += 16
            
            # Buffer state
            self.state_buffer.append(state)
            
            # Flush buffer
            if len(self.state_buffer) >= self.buffer_size:
                self._flush_buffer()
        
        except Exception as e:
            print(f"Telemetry parse error: {e}")
    
    def _flush_buffer(self):
        """Flush buffered states to visualizer."""
        for state in self.state_buffer:
            self.visualizer.log_agent_state(state)
            self.hud.update(state)
        
        self.state_buffer.clear()

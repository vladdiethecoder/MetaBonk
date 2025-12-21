"""Automated Director: Algorithmic Cinematography for AI Broadcasts.

Implements the "Interestingness Score" I(t) to automatically switch
cameras between agents in the simulation swarm.

Features:
- Risk/Skill/Novelty/CrowdAffinity metrics
- Hysteresis to prevent jarring cuts  
- OBS WebSocket integration for scene switching
- Dynamic layouts (Solo, Split, Grid, PiP)

References:
- Neuromorphic Broadcast Engine specification
- OBS WebSocket Protocol 5.0
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False


class BroadcastLayout(Enum):
    """Available broadcast layouts."""
    
    SOLO = "solo"           # Single agent focus
    SPLIT_2X2 = "split_2x2" # 4-way split
    SPLIT_2X1 = "split_2x1" # 2-way horizontal
    PIP = "pip"             # Picture-in-Picture (neural viz focus)
    GRID = "grid"           # N-way grid


@dataclass
class AgentMetrics:
    """Real-time metrics for an agent."""
    
    agent_id: int
    health: float = 100.0
    max_health: float = 100.0
    score: float = 0.0
    kill_count: int = 0
    survival_time: float = 0.0
    enemy_density: float = 0.0
    velocity: float = 0.0
    
    # Neural metrics
    value_estimate: float = 0.0
    policy_entropy: float = 0.0
    
    # Events
    boss_active: bool = False
    legendary_found: bool = False
    near_death: bool = False
    
    # Crowd
    crowd_affinity: float = 0.0
    
    @property
    def risk(self) -> float:
        """Risk score - danger level."""
        if self.health <= 0:
            return 0.0
        return self.enemy_density / max(self.health / self.max_health, 0.01)
    
    @property
    def skill(self) -> float:
        """Skill score - rate of reward accumulation."""
        return self.score / max(self.survival_time, 1.0)


@dataclass
class DirectorConfig:
    """Configuration for the Automated Director."""
    
    # Scoring weights
    w_risk: float = 0.3
    w_skill: float = 0.3
    w_novelty: float = 0.2
    w_crowd: float = 0.2
    
    # Hysteresis
    switch_threshold: float = 0.1  # Minimum score difference to switch
    switch_delay: float = 3.0      # Seconds to hold before switching
    
    # Novelty decay
    novelty_decay: float = 0.95
    
    # Update rate
    sample_rate_hz: float = 1.0
    
    # OBS connection
    obs_host: str = "localhost"
    obs_port: int = 4455
    obs_password: str = ""


class InterestingnessScorer:
    """Computes the Interestingness Score I(t) for each agent."""
    
    def __init__(self, cfg: DirectorConfig):
        self.cfg = cfg
        self.novelty_scores: Dict[int, float] = {}
    
    def compute_score(
        self,
        metrics: AgentMetrics,
    ) -> float:
        """Compute I(t) for an agent.
        
        I(t) = w1*Risk + w2*Skill + w3*Novelty + w4*CrowdAffinity
        """
        # Risk component
        risk = min(metrics.risk, 10.0) / 10.0  # Normalize to [0, 1]
        
        # Skill component
        skill = min(metrics.skill, 100.0) / 100.0
        
        # Novelty component (decays over time)
        novelty = self.novelty_scores.get(metrics.agent_id, 0.0)
        
        # Add novelty impulses for events
        if metrics.boss_active:
            novelty = max(novelty, 1.0)
        if metrics.legendary_found:
            novelty = max(novelty, 0.8)
        if metrics.near_death:
            novelty = max(novelty, 0.6)
        
        # Store and decay
        self.novelty_scores[metrics.agent_id] = novelty * self.cfg.novelty_decay
        
        # Crowd affinity
        crowd = min(metrics.crowd_affinity, 1.0)
        
        # Weighted sum
        score = (
            self.cfg.w_risk * risk +
            self.cfg.w_skill * skill +
            self.cfg.w_novelty * novelty +
            self.cfg.w_crowd * crowd
        )
        
        return score
    
    def trigger_novelty(self, agent_id: int, magnitude: float = 1.0):
        """Trigger a novelty event for an agent."""
        current = self.novelty_scores.get(agent_id, 0.0)
        self.novelty_scores[agent_id] = max(current, magnitude)


class OBSController:
    """Controls OBS Studio via WebSocket protocol.
    
    Handles scene switching, layout changes, and source transforms.
    """
    
    def __init__(self, cfg: DirectorConfig):
        self.cfg = cfg
        self.ws = None
        self.connected = False
        self.request_id = 0
        
        # Scene mappings
        self.scenes = {
            BroadcastLayout.SOLO: "Agent_Solo",
            BroadcastLayout.SPLIT_2X2: "Grid_2x2",
            BroadcastLayout.SPLIT_2X1: "Split_Horizontal",
            BroadcastLayout.PIP: "Neural_PiP",
            BroadcastLayout.GRID: "Agent_Grid",
        }
    
    async def connect(self) -> bool:
        """Connect to OBS WebSocket."""
        if not HAS_WEBSOCKETS:
            print("websockets not installed - OBS control disabled")
            return False
        
        try:
            url = f"ws://{self.cfg.obs_host}:{self.cfg.obs_port}"
            self.ws = await websockets.connect(url)

            # OBS WebSocket 5 handshake: Hello (op=0) -> Identify (op=1) -> Identified (op=2)
            hello_raw = await self.ws.recv()
            hello = json.loads(hello_raw)
            if hello.get("op") != 0:
                raise RuntimeError(f"unexpected OBS hello: {hello}")

            d = hello.get("d") or {}
            rpc_version = int(d.get("rpcVersion") or 1)
            auth = None
            auth_info = d.get("authentication")
            if auth_info and self.cfg.obs_password:
                try:
                    salt = str(auth_info.get("salt") or "")
                    challenge = str(auth_info.get("challenge") or "")
                    secret = base64.b64encode(
                        hashlib.sha256((self.cfg.obs_password + salt).encode("utf-8")).digest()
                    ).decode("utf-8")
                    auth = base64.b64encode(
                        hashlib.sha256((secret + challenge).encode("utf-8")).digest()
                    ).decode("utf-8")
                except Exception:
                    auth = None

            identify: Dict[str, Any] = {"op": 1, "d": {"rpcVersion": rpc_version}}
            if auth:
                identify["d"]["authentication"] = auth
            await self.ws.send(json.dumps(identify))

            # Await Identified.
            while True:
                identified_raw = await self.ws.recv()
                identified = json.loads(identified_raw)
                if identified.get("op") == 2:
                    break
                # Ignore other frames (events, etc.) during startup.

            self.connected = True
            print(f"Connected to OBS at {url}")
            return True
            
        except Exception as e:
            print(f"OBS connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from OBS."""
        if self.ws:
            await self.ws.close()
            self.connected = False
    
    async def send_request(
        self,
        request_type: str,
        request_data: Optional[Dict] = None,
    ) -> Dict:
        """Send request to OBS and await response."""
        if not self.connected or not self.ws:
            return {"error": "Not connected"}
        
        self.request_id += 1
        req_id = str(self.request_id)
        
        message = {
            "op": 6,  # Request
            "d": {
                "requestType": request_type,
                "requestId": req_id,
            }
        }
        
        if request_data:
            message["d"]["requestData"] = request_data
        
        await self.ws.send(json.dumps(message))

        # Wait for the matching response (op=7). Ignore events (op=5).
        while True:
            raw = await self.ws.recv()
            msg = json.loads(raw)
            op = msg.get("op")
            if op == 7:
                d = msg.get("d") or {}
                if str(d.get("requestId")) != req_id:
                    continue
                status = d.get("requestStatus") or {}
                if status.get("result", False):
                    return d.get("responseData") or {}
                return {
                    "error": status.get("comment") or "request_failed",
                    "code": status.get("code"),
                    "requestType": d.get("requestType"),
                }
            # Other messages: events/hello/etc. Ignore for now.
    
    async def switch_scene(self, layout: BroadcastLayout) -> bool:
        """Switch to a broadcast layout scene."""
        scene_name = self.scenes.get(layout, "Agent_Solo")
        
        result = await self.send_request(
            "SetCurrentProgramScene",
            {"sceneName": scene_name}
        )
        
        return "error" not in result
    
    async def set_source_transform(
        self,
        scene_name: str,
        source_id: int,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> bool:
        """Set transform for a source in a scene."""
        result = await self.send_request(
            "SetSceneItemTransform",
            {
                "sceneName": scene_name,
                "sceneItemId": source_id,
                "sceneItemTransform": {
                    "positionX": x,
                    "positionY": y,
                    "boundsWidth": width,
                    "boundsHeight": height,
                    "boundsType": "OBS_BOUNDS_SCALE_INNER",
                }
            }
        )
        
        return "error" not in result
    
    async def create_grid_layout(
        self,
        agent_source_ids: List[int],
        canvas_width: int = 1920,
        canvas_height: int = 1080,
    ):
        """Dynamically create a grid layout for N agents."""
        n = len(agent_source_ids)
        if n == 0:
            return
        
        # Calculate grid dimensions
        cols = int(n ** 0.5)
        if cols * cols < n:
            cols += 1
        rows = (n + cols - 1) // cols
        
        cell_width = canvas_width / cols
        cell_height = canvas_height / rows
        
        for i, source_id in enumerate(agent_source_ids):
            row = i // cols
            col = i % cols
            
            x = col * cell_width
            y = row * cell_height
            
            await self.set_source_transform(
                "Agent_Grid",
                source_id,
                x, y,
                cell_width, cell_height,
            )


class AutomatedDirector:
    """The brain of the broadcast - decides what to show and when.
    
    Monitors all agents, computes Interestingness scores, and
    orchestrates camera switches with proper hysteresis.
    """
    
    def __init__(self, cfg: Optional[DirectorConfig] = None):
        self.cfg = cfg or DirectorConfig()
        
        self.scorer = InterestingnessScorer(self.cfg)
        self.obs = OBSController(self.cfg)
        
        # Current state
        self.active_agent_id: Optional[int] = None
        self.active_layout = BroadcastLayout.SOLO
        self.switch_pending_since: Optional[float] = None
        self.switch_pending_to: Optional[int] = None
        
        # Agent tracking
        self.agent_metrics: Dict[int, AgentMetrics] = {}
        self.agent_scores: Dict[int, float] = {}
        
        # Stats
        self.total_switches = 0
        self.running = False
    
    def update_metrics(self, agent_id: int, metrics: AgentMetrics):
        """Update metrics for an agent."""
        self.agent_metrics[agent_id] = metrics
        self.agent_scores[agent_id] = self.scorer.compute_score(metrics)
    
    def get_top_agents(self, n: int = 4) -> List[int]:
        """Get top N agents by Interestingness."""
        sorted_agents = sorted(
            self.agent_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [agent_id for agent_id, _ in sorted_agents[:n]]
    
    def should_switch(self) -> Optional[int]:
        """Determine if camera should switch to a new agent.
        
        Implements hysteresis to prevent rapid switching.
        """
        if not self.agent_scores:
            return None
        
        # Get current and best agents
        best_agent = max(self.agent_scores, key=self.agent_scores.get)
        best_score = self.agent_scores[best_agent]
        
        current_score = 0.0
        if self.active_agent_id is not None:
            current_score = self.agent_scores.get(self.active_agent_id, 0.0)
        
        # Check threshold
        if best_score > current_score + self.cfg.switch_threshold:
            if self.switch_pending_to != best_agent:
                # New pending switch
                self.switch_pending_to = best_agent
                self.switch_pending_since = time.time()
                return None
            
            # Check if delay has passed
            if time.time() - self.switch_pending_since >= self.cfg.switch_delay:
                return best_agent
        else:
            # Clear pending
            self.switch_pending_to = None
            self.switch_pending_since = None
        
        return None
    
    def determine_layout(self) -> BroadcastLayout:
        """Determine optimal layout based on agent scores."""
        top_agents = self.get_top_agents(4)
        
        if len(top_agents) < 2:
            return BroadcastLayout.SOLO
        
        # Check score variance
        scores = [self.agent_scores[a] for a in top_agents]
        score_range = max(scores) - min(scores)
        
        # If scores are close, show split view
        if score_range < 0.1 and len(top_agents) >= 4:
            return BroadcastLayout.SPLIT_2X2
        elif score_range < 0.15 and len(top_agents) >= 2:
            return BroadcastLayout.SPLIT_2X1
        
        # Check for high entropy (neural viz focus)
        if self.active_agent_id is not None:
            agent = self.agent_metrics.get(self.active_agent_id)
            if agent and agent.policy_entropy > 0.8:
                return BroadcastLayout.PIP
        
        return BroadcastLayout.SOLO
    
    async def director_loop(self):
        """Main director loop - runs at sample_rate_hz."""
        self.running = True
        interval = 1.0 / self.cfg.sample_rate_hz
        
        # Connect to OBS
        if HAS_WEBSOCKETS:
            await self.obs.connect()
        
        while self.running:
            loop_start = time.time()
            
            # Check for camera switch
            new_agent = self.should_switch()
            if new_agent is not None and new_agent != self.active_agent_id:
                self.active_agent_id = new_agent
                self.total_switches += 1
                print(f"Director: Switching to Agent {new_agent} (Score: {self.agent_scores[new_agent]:.3f})")
                
                # OBS switch would happen here
                # await self.obs.switch_scene(self.active_layout)
            
            # Check for layout change
            new_layout = self.determine_layout()
            if new_layout != self.active_layout:
                self.active_layout = new_layout
                print(f"Director: Layout changed to {new_layout.value}")
                
                if self.obs.connected:
                    await self.obs.switch_scene(new_layout)
            
            # Sleep for remaining interval
            elapsed = time.time() - loop_start
            if elapsed < interval:
                await asyncio.sleep(interval - elapsed)
    
    def stop(self):
        """Stop the director loop."""
        self.running = False
    
    def get_broadcast_state(self) -> Dict[str, Any]:
        """Get current broadcast state for display."""
        return {
            "active_agent": self.active_agent_id,
            "active_layout": self.active_layout.value,
            "top_agents": self.get_top_agents(4),
            "agent_scores": dict(self.agent_scores),
            "total_switches": self.total_switches,
            "pending_switch": self.switch_pending_to,
        }


class BroadcastOverlay:
    """Generates overlay data for Dear PyGui integration."""
    
    def __init__(self, director: AutomatedDirector):
        self.director = director
        
        # History for graphs
        self.score_history: Dict[int, List[float]] = {}
        self.history_length = 100
    
    def update(self):
        """Update overlay data."""
        for agent_id, score in self.director.agent_scores.items():
            if agent_id not in self.score_history:
                self.score_history[agent_id] = []
            
            self.score_history[agent_id].append(score)
            
            # Trim history
            if len(self.score_history[agent_id]) > self.history_length:
                self.score_history[agent_id].pop(0)
    
    def get_overlay_data(self) -> Dict[str, Any]:
        """Get data for overlay rendering."""
        state = self.director.get_broadcast_state()
        
        # Add active agent details
        if state["active_agent"] is not None:
            metrics = self.director.agent_metrics.get(state["active_agent"])
            if metrics:
                state["active_agent_metrics"] = {
                    "health": metrics.health,
                    "score": metrics.score,
                    "kill_count": metrics.kill_count,
                    "survival_time": metrics.survival_time,
                    "value_estimate": metrics.value_estimate,
                    "entropy": metrics.policy_entropy,
                }
        
        # Add score history for graphs
        if state["active_agent"] in self.score_history:
            state["score_history"] = self.score_history[state["active_agent"]]
        
        return state

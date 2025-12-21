#!/usr/bin/env python3
"""Apex Agent - Complete integration of all NSAIG components.

Combines world model, liquid dynamics, skill tokens, and Active Inference
into a single deployable agent that can:
- Navigate menus via VLM
- Play gameplay via learned skills
- Adapt via Task Arithmetic

Usage:
    python scripts/run_agent.py --mode autonomous
    python scripts/run_agent.py --mode menu_only
"""

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class AgentConfig:
    """Configuration for Apex Agent."""
    
    # Checkpoints
    world_model_ckpt: str = "./checkpoints/world_model.pt"
    skill_vqvae_ckpt: str = "./checkpoints/skill_vqvae.pt"
    liquid_dynamics_ckpt: str = "./checkpoints/liquid_dynamics.pt"
    decision_transformer_ckpt: str = "./checkpoints/decision_transformer.pt"
    
    # Runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    planning_horizon: int = 15
    
    # Connection
    bridge_host: str = "127.0.0.1"
    bridge_port: int = 5555
    
    # VLM
    vlm_enabled: bool = True
    vlm_model: str = "llava"


class ApexAgent:
    """Complete Apex Agent integrating all NSAIG components."""
    
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # Components (lazy loaded)
        self._world_model = None
        self._skill_vqvae = None
        self._liquid_dynamics = None
        self._decision_transformer = None
        self._vlm_navigator = None
        self._bridge = None
        
        # State
        self.current_skill_vector = None
        self.game_state = "menu"  # "menu", "playing", "paused"
        self.episode_step = 0
    
    def load_models(self):
        """Load all model checkpoints."""
        print("Loading Apex models...")
        
        # World Model
        if Path(self.cfg.world_model_ckpt).exists():
            from src.learner.world_model import WorldModel
            ckpt = torch.load(self.cfg.world_model_ckpt, map_location=self.device, weights_only=True)
            self._world_model = WorldModel(ckpt["config"]).to(self.device)
            self._world_model.load_state_dict(ckpt["model_state_dict"])
            self._world_model.eval()
            print("  ✓ World Model loaded")
        
        # Skill VQ-VAE
        if Path(self.cfg.skill_vqvae_ckpt).exists():
            from src.learner.skill_tokens import SkillVQVAE
            ckpt = torch.load(self.cfg.skill_vqvae_ckpt, map_location=self.device, weights_only=True)
            self._skill_vqvae = SkillVQVAE(ckpt["config"]).to(self.device)
            self._skill_vqvae.load_state_dict(ckpt["model_state_dict"])
            self._skill_vqvae.eval()
            print("  ✓ Skill VQ-VAE loaded")
        
        # Liquid Dynamics
        if Path(self.cfg.liquid_dynamics_ckpt).exists():
            from src.learner.liquid_networks import LiquidWorldModel
            ckpt = torch.load(self.cfg.liquid_dynamics_ckpt, map_location=self.device, weights_only=True)
            self._liquid_dynamics = LiquidWorldModel().to(self.device)
            self._liquid_dynamics.load_state_dict(ckpt["model_state_dict"])
            self._liquid_dynamics.eval()
            print("  ✓ Liquid Dynamics loaded")
        
        # Decision Transformer
        if Path(self.cfg.decision_transformer_ckpt).exists():
            from src.learner.decision_transformer import DecisionTransformer
            ckpt = torch.load(self.cfg.decision_transformer_ckpt, map_location=self.device, weights_only=True)
            self._decision_transformer = DecisionTransformer(ckpt["config"]).to(self.device)
            self._decision_transformer.load_state_dict(ckpt["model_state_dict"])
            self._decision_transformer.eval()
            print("  ✓ Decision Transformer loaded")
        
        print("Models loaded!")
    
    async def connect_bridge(self):
        """Connect to Unity bridge."""
        from src.bridge.unity_bridge import UnityBridge, BridgeConfig
        
        bridge_cfg = BridgeConfig()
        self._bridge = UnityBridge(bridge_cfg)
        
        connected = await self._bridge.connect()
        if connected:
            print("  ✓ Connected to game bridge")
        else:
            print("  ✗ Bridge connection failed (game may not be running)")
        
        return connected
    
    async def navigate_menu(self, goal: str):
        """Use VLM to navigate menus."""
        if not self.cfg.vlm_enabled:
            print("VLM navigation disabled")
            return False
        
        try:
            from scripts.menu_navigator import ActiveInferenceNavigator, NavigatorConfig
            
            nav_cfg = NavigatorConfig(vlm_model=self.cfg.vlm_model)
            navigator = ActiveInferenceNavigator(nav_cfg)
            
            success = await navigator.navigate_to_goal(goal, max_steps=15)
            return success
        except Exception as e:
            print(f"Menu navigation error: {e}")
            return False
    
    @torch.no_grad()
    def get_action(self, observation: torch.Tensor) -> torch.Tensor:
        """Get action from observation using best available model."""
        obs = observation.to(self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        # Try Decision Transformer first (most general)
        if self._decision_transformer is not None:
            # Simplified: just use observation as state
            action = self._decision_transformer.get_action(
                states=obs.unsqueeze(1),
                actions=torch.zeros(1, 1, 6, device=self.device),
                returns_to_go=torch.ones(1, 1, 1, device=self.device),
                timesteps=torch.zeros(1, 1, dtype=torch.long, device=self.device),
            )
            return action.squeeze()
        
        raise RuntimeError(
            "No policy is loaded (DecisionTransformer missing). Refusing to emit random actions; "
            "provide real checkpoints or disable gameplay execution."
        )
    
    @torch.no_grad()
    def imagine_future(
        self,
        observation: torch.Tensor,
        action_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """Imagine future states using world model."""
        if self._world_model is None:
            return None
        obs0 = observation.to(self.device)
        if obs0.dim() == 1:
            obs0 = obs0.unsqueeze(0)

        actions = action_sequence.to(self.device)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        try:
            out = self._world_model.imagine_rollout(
                obs0,
                actions,
                horizon=self.cfg.planning_horizon,
                deterministic=False,
            )
            return out["pred_obs"].detach()
        except Exception as e:
            print(f"[ApexAgent] imagination error: {e}")
            return None
    
    async def run_episode(self, max_steps: int = 1000):
        """Run one episode of gameplay."""
        if self._bridge is None:
            print("Bridge not connected!")
            return
        
        self.episode_step = 0
        total_reward = 0.0
        
        print(f"Starting episode (max {max_steps} steps)...")
        
        for step in range(max_steps):
            # Get frame from game
            frame = await self._bridge.read_frame()
            if frame is None:
                await asyncio.sleep(0.016)  # ~60 FPS
                continue
            
            # Extract observation from game state
            obs = self._state_to_observation(frame.state)
            
            # Get action
            action = self.get_action(obs)
            
            # Send to game
            await self._send_action(action)
            
            # Track
            self.episode_step += 1
            
            # Check for episode end
            if frame.state.get("done", False):
                break
        
        print(f"Episode complete: {self.episode_step} steps")
    
    def _state_to_observation(self, state: Dict) -> torch.Tensor:
        """Convert game state dict to observation tensor."""
        obs = torch.zeros(204)
        
        # Extract what we can
        if "playerPosition" in state:
            pos = state["playerPosition"]
            obs[0] = pos.get("x", 0) / 100.0
            obs[1] = pos.get("y", 0) / 100.0
            obs[2] = pos.get("z", 0) / 100.0
        
        if "playerHealth" in state:
            obs[10] = state["playerHealth"] / 100.0
        
        return obs
    
    async def _send_action(self, action: torch.Tensor):
        """Send action to game via bridge."""
        if self._bridge is None:
            return
        # Intentionally avoid hard-coded control mappings (W/A/S/D, aim, etc.).
        # Provide explicit action primitives (keys/mouse) through a higher-level
        # learned mapper or by passing a structured action dict.
        if isinstance(action, dict):
            keys = action.get("keys") or {}
            for k, v in keys.items():
                try:
                    await self._bridge.send_key(str(k), bool(v))
                except Exception:
                    pass
            click = action.get("click")
            if isinstance(click, dict) and "x" in click and "y" in click:
                try:
                    await self._bridge.send_mouse_click(int(click["x"]), int(click["y"]), int(click.get("button", 0)))
                except Exception:
                    pass
        return
    
    async def run_autonomous(self):
        """Full autonomous mode: menu navigation + gameplay."""
        print("\nStarting Apex Agent in Autonomous Mode")
        print("="*50)
        
        # Load models
        self.load_models()
        
        # Try to connect
        connected = await self.connect_bridge()
        
        if not connected:
            print("\nGame bridge not available. Running in VLM-only mode.")
            
            # Just try menu navigation
            if self.cfg.vlm_enabled:
                await self.navigate_menu("Start Game")
            return
        
        # Navigate to gameplay
        self.game_state = "menu"
        await self.navigate_menu("Start Game")
        
        # Play
        self.game_state = "playing"
        await self.run_episode()


async def main():
    parser = argparse.ArgumentParser(description="Apex Agent Runner")
    parser.add_argument("--mode", default="autonomous", choices=["autonomous", "menu_only", "play_only"])
    parser.add_argument("--vlm", default="llava", help="VLM model")
    parser.add_argument("--no-vlm", action="store_true", help="Disable VLM")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    cfg = AgentConfig(
        device=args.device,
        vlm_enabled=not args.no_vlm,
        vlm_model=args.vlm,
    )
    
    agent = ApexAgent(cfg)
    
    if args.mode == "menu_only":
        agent.load_models()
        await agent.navigate_menu("Start Game")
    elif args.mode == "play_only":
        agent.load_models()
        await agent.connect_bridge()
        await agent.run_episode()
    else:  # autonomous
        await agent.run_autonomous()


if __name__ == "__main__":
    asyncio.run(main())

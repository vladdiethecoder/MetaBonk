#!/usr/bin/env python3
"""Rollout collector for world model training.

Connects to running MetaBonk instances and collects (obs, action, reward)
trajectories for training the DreamerV3 world model.

Usage:
    python -m scripts.collect_rollouts --instances 4 --episodes 100

Or run alongside training:
    python -m scripts.collect_rollouts --continuous --output-dir ./rollouts
"""

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

try:
    import httpx
except ImportError:
    httpx = None
    print("Warning: httpx not installed. Run: pip install httpx")

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class CollectorConfig:
    """Configuration for rollout collection."""
    
    # Connection
    orchestrator_url: str = "http://localhost:8040"
    worker_base_port: int = 5000
    num_instances: int = 4
    
    # Collection
    episodes_per_instance: int = 25
    max_steps_per_episode: int = 2000
    
    # Output
    output_dir: str = "./rollouts"
    
    # Mode
    continuous: bool = False
    collection_interval: float = 5.0  # seconds between checks


class RolloutCollector:
    """Collects rollouts from running game instances."""
    
    def __init__(self, cfg: CollectorConfig):
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.running = True
        self.episodes_collected = 0
    
    async def collect_from_worker(
        self,
        worker_url: str,
        instance_id: str,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Collect a single episode from a worker."""
        if not httpx:
            return None
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Get worker status
                resp = await client.get(f"{worker_url}/status")
                if resp.status_code != 200:
                    return None
                
                status = resp.json()
                
                # Check if worker has rollout data
                if "rollout" not in status:
                    # Request rollout collection
                    resp = await client.post(
                        f"{worker_url}/collect_rollout",
                        json={"max_steps": self.cfg.max_steps_per_episode},
                    )
                    if resp.status_code != 200:
                        return None
                    
                    rollout_data = resp.json()
                else:
                    rollout_data = status["rollout"]
                
                if not rollout_data:
                    return None
                
                # Convert to tensors
                episode = {
                    "observations": torch.tensor(rollout_data.get("observations", [])),
                    "actions": torch.tensor(rollout_data.get("actions", [])),
                    "rewards": torch.tensor(rollout_data.get("rewards", [])),
                    "dones": torch.tensor(rollout_data.get("dones", [])),
                    "instance_id": instance_id,
                    "timestamp": time.time(),
                }
                
                return episode if len(episode["observations"]) > 0 else None
                
            except Exception as e:
                print(f"Error collecting from {worker_url}: {e}")
                return None
    
    async def collect_all_workers(self) -> List[Dict[str, torch.Tensor]]:
        """Collect from all available workers."""
        tasks = []
        
        for i in range(self.cfg.num_instances):
            port = self.cfg.worker_base_port + i
            url = f"http://localhost:{port}"
            instance_id = f"inst-{i}"
            tasks.append(self.collect_from_worker(url, instance_id))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        episodes = []
        for r in results:
            if isinstance(r, dict):
                episodes.append(r)
        
        return episodes
    
    def save_episode(self, episode: Dict[str, torch.Tensor]) -> Path:
        """Save episode to disk."""
        timestamp = int(time.time() * 1000)
        instance_id = episode.get("instance_id", "unknown")
        filename = f"episode_{instance_id}_{timestamp}.pt"
        path = self.output_dir / filename
        torch.save(episode, path)
        return path
    
    async def run_continuous(self):
        """Continuously collect rollouts."""
        print(f"Starting continuous collection to {self.output_dir}")
        print("Press Ctrl+C to stop")
        
        while self.running:
            episodes = await self.collect_all_workers()
            
            for ep in episodes:
                path = self.save_episode(ep)
                self.episodes_collected += 1
                print(f"Collected episode {self.episodes_collected}: {path.name}")
            
            if not episodes:
                print("No episodes collected, waiting...")
            
            await asyncio.sleep(self.cfg.collection_interval)
    
    async def run_batch(self, total_episodes: int):
        """Collect a fixed number of episodes."""
        print(f"Collecting {total_episodes} episodes to {self.output_dir}")
        
        while self.episodes_collected < total_episodes and self.running:
            episodes = await self.collect_all_workers()
            
            for ep in episodes:
                if self.episodes_collected >= total_episodes:
                    break
                path = self.save_episode(ep)
                self.episodes_collected += 1
                print(f"[{self.episodes_collected}/{total_episodes}] Saved: {path.name}")
            
            if not episodes:
                print("Waiting for workers...")
                await asyncio.sleep(2.0)
        
        print(f"\nCollection complete! {self.episodes_collected} episodes saved.")


def generate_synthetic_rollouts(
    output_dir: str,
    num_episodes: int = 100,
    obs_dim: int = 204,
    action_dim: int = 6,
):
    raise RuntimeError(
        "Synthetic rollout generation has been removed. "
        "Collect rollouts from real workers via the orchestrator/learner stack."
    )


async def main():
    parser = argparse.ArgumentParser(description="Collect rollouts for world model training")
    parser.add_argument("--instances", type=int, default=4, help="Number of game instances")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes to collect")
    parser.add_argument("--output-dir", default="./rollouts", help="Output directory")
    parser.add_argument("--continuous", action="store_true", help="Continuous collection mode")
    parser.add_argument("--orchestrator", default="http://localhost:8040")
    args = parser.parse_args()
    
    cfg = CollectorConfig(
        orchestrator_url=args.orchestrator,
        num_instances=args.instances,
        output_dir=args.output_dir,
        continuous=args.continuous,
    )
    
    collector = RolloutCollector(cfg)
    
    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\nStopping collection...")
        collector.running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    
    if cfg.continuous:
        await collector.run_continuous()
    else:
        await collector.run_batch(args.episodes)


if __name__ == "__main__":
    asyncio.run(main())

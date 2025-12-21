#!/usr/bin/env python3
"""World Model training script for MetaBonk Apex.

Trains the DreamerV3-style RSSM world model on collected gameplay data.
The trained model enables "imagination" - planning by simulating futures
without interacting with the actual environment.

Usage:
    python -m scripts.train_world_model --data-dir ./rollouts --epochs 100

Training phases:
1. Movement only (empty arena) - learn basic physics
2. Simple enemies - learn prediction of entity dynamics
3. Full gameplay - learn complete world dynamics
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.learner.world_model import WorldModel, WorldModelConfig, LatentState


@dataclass
class TrainingConfig:
    """Configuration for world model training."""
    
    # Data
    data_dir: str = "./rollouts"
    sequence_length: int = 64
    batch_size: int = 32
    
    # Model
    obs_dim: int = 204
    action_dim: int = 6
    
    # Training
    epochs: int = 100
    learning_rate: float = 3e-4
    grad_clip: float = 100.0
    
    # Checkpointing
    save_dir: str = "./checkpoints/world_model"
    save_every: int = 10
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class RolloutDataset(Dataset):
    """Dataset of collected rollouts for training."""
    
    def __init__(
        self,
        data_dir: str,
        sequence_length: int,
        obs_dim: int,
        action_dim: int,
    ):
        self.sequence_length = sequence_length
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Load all rollout files
        self.episodes: List[Dict[str, torch.Tensor]] = []
        data_path = Path(data_dir)
        
        if data_path.exists():
            for f in data_path.glob("*.pt"):
                try:
                    episode = torch.load(f, weights_only=True)
                    if self._validate_episode(episode):
                        self.episodes.append(episode)
                except Exception as e:
                    print(f"Skipping {f}: {e}")
        
        if not self.episodes:
            raise RuntimeError(
                f"No valid .pt episodes found in {data_dir}. "
                "Synthetic data generation is intentionally disabled; provide real rollouts."
            )
    
    def _validate_episode(self, episode: Dict) -> bool:
        """Check if episode has required fields."""
        required = ["observations", "actions", "rewards", "dones"]
        return all(k in episode for k in required)
    
    def __len__(self) -> int:
        return len(self.episodes) * 10  # Sample multiple sequences per episode
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        # Pick random episode
        episode_idx = idx % len(self.episodes)
        episode = self.episodes[episode_idx]
        
        obs = episode["observations"]
        actions = episode["actions"]
        rewards = episode["rewards"]
        dones = episode["dones"]
        
        T = len(obs)
        
        # Random start position
        if T <= self.sequence_length:
            start = 0
            end = T
        else:
            start = random.randint(0, T - self.sequence_length)
            end = start + self.sequence_length
        
        # Slice and pad if needed
        obs_seq = obs[start:end]
        act_seq = actions[start:end]
        rew_seq = rewards[start:end]
        done_seq = dones[start:end]
        
        # Pad if necessary
        if len(obs_seq) < self.sequence_length:
            pad_len = self.sequence_length - len(obs_seq)
            obs_seq = torch.cat([obs_seq, torch.zeros(pad_len, self.obs_dim)])
            act_seq = torch.cat([act_seq, torch.zeros(pad_len, self.action_dim)])
            rew_seq = torch.cat([rew_seq, torch.zeros(pad_len)])
            done_seq = torch.cat([done_seq, torch.ones(pad_len)])  # Pad with terminal
        
        return obs_seq, act_seq, rew_seq, done_seq


def train_epoch(
    model: WorldModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    cfg: TrainingConfig,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    device = torch.device(cfg.device)
    
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_reward = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for obs, actions, rewards, dones in pbar:
        obs = obs.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)
        
        optimizer.zero_grad()
        
        # Compute losses
        losses = model.compute_loss(obs, actions, rewards, dones)
        loss = losses["total_loss"]
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        
        optimizer.step()
        
        # Accumulate
        total_loss += losses["total_loss"].item()
        total_recon += losses["recon_loss"].item()
        total_kl += losses["kl_loss"].item()
        total_reward += losses["reward_loss"].item()
        num_batches += 1
        
        pbar.set_postfix({
            "loss": f"{total_loss/num_batches:.4f}",
            "recon": f"{total_recon/num_batches:.4f}",
        })
    
    return {
        "total_loss": total_loss / max(num_batches, 1),
        "recon_loss": total_recon / max(num_batches, 1),
        "kl_loss": total_kl / max(num_batches, 1),
        "reward_loss": total_reward / max(num_batches, 1),
    }


def evaluate_imagination(
    model: WorldModel,
    dataloader: DataLoader,
    cfg: TrainingConfig,
    num_samples: int = 5,
) -> Dict[str, float]:
    """Evaluate imagination quality by comparing predicted vs actual."""
    model.eval()
    device = torch.device(cfg.device)
    
    total_pred_error = 0.0
    total_reward_error = 0.0
    num_tests = 0
    
    with torch.no_grad():
        for obs, actions, rewards, dones in dataloader:
            if num_tests >= num_samples:
                break
            
            obs = obs.to(device)
            actions = actions.to(device)
            
            B, T, _ = obs.shape
            
            # Get initial state from first few observations
            init_obs = obs[:, :5]
            init_act = actions[:, :5]
            
            states, _ = model.observe_sequence(init_obs, init_act)
            start_state = states[-1]
            
            # Create a simple policy that just uses the actual actions
            class ReplayPolicy(nn.Module):
                def __init__(self, actions):
                    super().__init__()
                    self.actions = actions
                    self.t = 0
                
                def forward(self, state):
                    act = self.actions[:, min(self.t, self.actions.shape[1]-1)]
                    self.t += 1
                    return act
            
            # Imagine future
            horizon = min(10, T - 5)
            policy = ReplayPolicy(actions[:, 5:])
            imagined_states, pred_rewards, _ = model.imagine_trajectory(
                start_state, policy, horizon
            )
            
            # Compare predictions
            actual_rewards = rewards[:, 5:5+horizon]
            reward_error = (pred_rewards - actual_rewards).abs().mean()
            
            total_reward_error += reward_error.item()
            num_tests += 1
    
    return {
        "imagination_reward_error": total_reward_error / max(num_tests, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train DreamerV3 World Model")
    parser.add_argument("--data-dir", default="./rollouts", help="Rollout data directory")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--save-dir", default="./checkpoints/world_model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    cfg = TrainingConfig(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_dir=args.save_dir,
        device=args.device,
    )
    
    print(f"Training World Model on {cfg.device}")
    print(f"Data directory: {cfg.data_dir}")
    
    # Create directories
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    dataset = RolloutDataset(
        cfg.data_dir,
        cfg.sequence_length,
        cfg.obs_dim,
        cfg.action_dim,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    
    print(f"Dataset size: {len(dataset)} samples from {len(dataset.episodes)} episodes")
    
    # Create model
    model_cfg = WorldModelConfig(
        obs_dim=cfg.obs_dim,
        action_dim=cfg.action_dim,
    )
    model = WorldModel(model_cfg).to(cfg.device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    
    # Training loop
    best_loss = float("inf")
    
    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        
        # Train
        train_metrics = train_epoch(model, dataloader, optimizer, cfg)
        
        print(f"  Loss: {train_metrics['total_loss']:.4f} | "
              f"Recon: {train_metrics['recon_loss']:.4f} | "
              f"KL: {train_metrics['kl_loss']:.4f} | "
              f"Reward: {train_metrics['reward_loss']:.4f}")
        
        # Evaluate imagination
        if epoch % 5 == 0:
            eval_metrics = evaluate_imagination(model, dataloader, cfg)
            print(f"  Imagination Reward Error: {eval_metrics['imagination_reward_error']:.4f}")
        
        # Save checkpoint
        if epoch % cfg.save_every == 0 or train_metrics["total_loss"] < best_loss:
            if train_metrics["total_loss"] < best_loss:
                best_loss = train_metrics["total_loss"]
                path = Path(cfg.save_dir) / "best_model.pt"
            else:
                path = Path(cfg.save_dir) / f"model_epoch_{epoch}.pt"
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": train_metrics["total_loss"],
                "config": model_cfg,
            }, path)
            print(f"  Saved checkpoint: {path}")
    
    print("\nTraining complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {cfg.save_dir}")


if __name__ == "__main__":
    main()

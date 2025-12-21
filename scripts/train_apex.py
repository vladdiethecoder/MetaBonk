#!/usr/bin/env python3
"""Apex Training Pipeline - Complete end-to-end training orchestration.

Coordinates training of all NSAIG components:
1. World Model (RSSM/LNN)
2. Skill Tokenizer (VQ-VAE)
3. Decision Transformer (behavioral cloning)
4. Free Energy objective (Active Inference)

Usage:
    python scripts/train_apex.py --phase 1  # World model only
    python scripts/train_apex.py --phase all  # Full pipeline
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ApexTrainingConfig:
    """Configuration for Apex training pipeline."""
    
    # Paths
    data_dir: str = "./rollouts"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Data
    obs_dim: int = 204
    action_dim: int = 6
    sequence_length: int = 64
    batch_size: int = 32
    
    # Training
    epochs: int = 50
    learning_rate: float = 3e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Components to train
    train_world_model: bool = True
    train_skill_tokens: bool = True
    train_decision_transformer: bool = True
    train_liquid_dynamics: bool = True


def generate_synthetic_data(config: ApexTrainingConfig, num_episodes: int = 100):
    raise RuntimeError(
        "Synthetic/demo data generation has been removed. "
        "Provide real .pt rollouts in --data-dir (see scripts/video_pretrain.py for video-derived rollouts)."
    )


def train_world_model(config: ApexTrainingConfig) -> Dict[str, float]:
    """Train DreamerV3-style world model."""
    from src.learner.world_model import WorldModel, WorldModelConfig
    
    print("\n" + "="*60)
    print("Phase 1: Training World Model (RSSM)")
    print("="*60)
    
    # Load data
    data_path = Path(config.data_dir)
    episodes = []
    for f in sorted(data_path.glob("*.pt"))[:50]:  # Limit for speed
        episodes.append(torch.load(f, weights_only=True))
    
    if not episodes:
        print("No training data found!")
        return {}
    
    print(f"Loaded {len(episodes)} episodes")
    
    # Create model
    model_cfg = WorldModelConfig(
        obs_dim=config.obs_dim,
        action_dim=config.action_dim,
    )
    model = WorldModel(model_cfg).to(config.device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    losses = []
    for epoch in range(min(config.epochs, 20)):  # Quick training
        epoch_loss = 0.0
        n_batches = 0
        
        for ep in episodes:
            obs = ep["observations"].to(config.device)
            actions = ep["actions"].to(config.device)
            rewards = ep["rewards"].to(config.device)
            dones = ep["dones"].to(config.device)
            
            # Add batch dimension
            obs = obs.unsqueeze(0)
            actions = actions.unsqueeze(0)
            rewards = rewards.unsqueeze(0)
            dones = dones.unsqueeze(0)
            
            # Forward
            loss_dict = model.compute_loss(obs, actions, rewards, dones)
            loss = loss_dict["total_loss"]
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
    
    # Save checkpoint
    ckpt_path = Path(config.checkpoint_dir) / "world_model.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model_cfg,
    }, ckpt_path)
    print(f"Saved: {ckpt_path}")
    
    return {"final_loss": losses[-1] if losses else 0.0}


def train_skill_tokens(config: ApexTrainingConfig) -> Dict[str, float]:
    """Train skill VQ-VAE for action tokenization."""
    from src.learner.skill_tokens import SkillVQVAE, SkillVQConfig
    
    print("\n" + "="*60)
    print("Phase 2: Training Skill Tokens (VQ-VAE)")
    print("="*60)
    
    # Load data
    data_path = Path(config.data_dir)
    all_action_seqs = []
    
    for f in sorted(data_path.glob("*.pt"))[:30]:
        ep = torch.load(f, weights_only=True)
        actions = ep["actions"]
        
        # Split into chunks
        chunk_size = 16
        for i in range(0, len(actions) - chunk_size, chunk_size // 2):
            chunk = actions[i:i + chunk_size]
            if len(chunk) == chunk_size:
                all_action_seqs.append(chunk)
    
    if not all_action_seqs:
        print("No action sequences found!")
        return {}
    
    print(f"Extracted {len(all_action_seqs)} action sequences")
    
    # Create model
    vq_cfg = SkillVQConfig(
        action_dim=config.action_dim,
        sequence_length=16,
        num_codes=256,
    )
    model = SkillVQVAE(vq_cfg).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training
    action_tensor = torch.stack(all_action_seqs).to(config.device)
    
    losses = []
    for epoch in range(min(config.epochs, 30)):
        # Shuffle
        perm = torch.randperm(len(action_tensor))
        shuffled = action_tensor[perm]
        
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, len(shuffled), config.batch_size):
            batch = shuffled[i:i + config.batch_size]
            
            _, _, loss, metrics = model(batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            usage = metrics.get("codebook_utilization", 0)
            print(f"  Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Codebook Usage = {usage:.1%}")
    
    # Save
    ckpt_path = Path(config.checkpoint_dir) / "skill_vqvae.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": vq_cfg,
    }, ckpt_path)
    print(f"Saved: {ckpt_path}")
    
    return {"final_loss": losses[-1] if losses else 0.0}


def train_liquid_dynamics(config: ApexTrainingConfig) -> Dict[str, float]:
    """Train Liquid Neural Network for continuous dynamics."""
    from src.learner.liquid_networks import LiquidWorldModel
    
    print("\n" + "="*60)
    print("Phase 3: Training Liquid Dynamics (CfC)")
    print("="*60)
    
    # Load data
    data_path = Path(config.data_dir)
    episodes = []
    for f in sorted(data_path.glob("*.pt"))[:30]:
        episodes.append(torch.load(f, weights_only=True))
    
    if not episodes:
        print("No data!")
        return {}
    
    # Create model
    model = LiquidWorldModel(
        obs_dim=config.obs_dim,
        action_dim=config.action_dim,
        latent_dim=128,
    ).to(config.device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    losses = []
    for epoch in range(min(config.epochs, 20)):
        epoch_loss = 0.0
        n_samples = 0
        
        for ep in episodes:
            obs = ep["observations"].to(config.device)
            actions = ep["actions"].to(config.device)
            
            # Predict next observation from current
            for t in range(len(obs) - 1):
                pred_obs, pred_reward = model(
                    obs[t:t+1],
                    actions[t:t+1],
                )
                
                target = obs[t+1:t+2]
                loss = torch.nn.functional.mse_loss(pred_obs, target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_samples += 1
        
        avg_loss = epoch_loss / max(n_samples, 1)
        losses.append(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}: Loss = {avg_loss:.6f}")
    
    # Save
    ckpt_path = Path(config.checkpoint_dir) / "liquid_dynamics.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
    }, ckpt_path)
    print(f"Saved: {ckpt_path}")
    
    return {"final_loss": losses[-1] if losses else 0.0}


def train_decision_transformer(config: ApexTrainingConfig) -> Dict[str, float]:
    """Train Decision Transformer for behavioral cloning."""
    from src.learner.decision_transformer import (
        DecisionTransformer, DTConfig, TrajectoryDataset
    )
    
    print("\n" + "="*60)
    print("Phase 4: Training Decision Transformer")
    print("="*60)
    
    # Load data
    data_path = Path(config.data_dir)
    trajectories = []
    for f in sorted(data_path.glob("*.pt"))[:30]:
        trajectories.append(torch.load(f, weights_only=True))
    
    if not trajectories:
        print("No data!")
        return {}
    
    # Create dataset
    dataset = TrajectoryDataset(
        trajectories,
        context_length=20,
    )
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Create model
    dt_cfg = DTConfig(
        state_dim=config.obs_dim,
        action_dim=config.action_dim,
        n_layer=3,
        n_head=4,
        n_embd=128,
    )
    model = DecisionTransformer(dt_cfg).to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    losses = []
    for epoch in range(min(config.epochs, 20)):
        epoch_loss = 0.0
        n_batches = 0
        
        for states, actions, rtg, timesteps in dataloader:
            states = states.to(config.device)
            actions = actions.to(config.device)
            rtg = rtg.to(config.device)
            timesteps = timesteps.to(config.device)
            
            # Forward
            action_preds = model(states, actions, rtg, timesteps)
            
            # Loss
            loss = torch.nn.functional.mse_loss(
                action_preds[:, :-1],
                actions[:, 1:],
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
    
    # Save
    ckpt_path = Path(config.checkpoint_dir) / "decision_transformer.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": dt_cfg,
    }, ckpt_path)
    print(f"Saved: {ckpt_path}")
    
    return {"final_loss": losses[-1] if losses else 0.0}


def main():
    parser = argparse.ArgumentParser(description="Apex Training Pipeline")
    parser.add_argument("--phase", default="all", help="Phase to train: 1,2,3,4 or 'all'")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-dir", default="./rollouts")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    config = ApexTrainingConfig(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    
    print("="*60)
    print("MetaBonk Apex Training Pipeline")
    print("="*60)
    print(f"Device: {config.device}")
    print(f"Data: {config.data_dir}")
    
    if not list(Path(config.data_dir).glob("*.pt")):
        raise RuntimeError(
            f"No .pt episodes found in {config.data_dir}. "
            "Provide real rollouts (e.g., scripts/video_pretrain.py --phase export_pt)."
        )
    
    results = {}
    
    # Run training phases
    phases = args.phase.lower()
    
    if phases in ["1", "all"]:
        results["world_model"] = train_world_model(config)
    
    if phases in ["2", "all"]:
        results["skill_tokens"] = train_skill_tokens(config)
    
    if phases in ["3", "all"]:
        results["liquid_dynamics"] = train_liquid_dynamics(config)
    
    if phases in ["4", "all"]:
        results["decision_transformer"] = train_decision_transformer(config)
    
    # Summary
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    for name, metrics in results.items():
        print(f"  {name}: {metrics}")
    
    print(f"\nCheckpoints saved to: {config.checkpoint_dir}")


if __name__ == "__main__":
    main()

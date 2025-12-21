#!/usr/bin/env python3
"""High-Throughput RL Training Script.

Complete training pipeline for RTX 5090:
- 10 parallel game instances
- FP8 Transformer policy
- Sample Factory APPO
- Population Based Training

Usage:
    python scripts/train_high_throughput.py --num-workers 10 --use-fp8 --use-pbt
"""

import argparse
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class HighThroughputConfig:
    """Configuration for high-throughput training."""
    
    # Workers
    num_workers: int = 10
    num_envs_per_worker: int = 1
    
    # Batching
    batch_size: int = 16384  # Massive for GPU saturation
    rollout_length: int = 32
    
    # Learning
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    
    # Policy
    use_fp8: bool = True
    context_length: int = 512  # Enabled by FP8 efficiency
    num_layers: int = 4
    embed_dim: int = 256
    
    # PBT
    use_pbt: bool = True
    pbt_interval: int = 1_000_000
    population_size: int = 10
    
    # Hardware
    device: str = "cuda"
    mixed_precision: bool = True
    
    # Logging
    log_interval: int = 1000
    save_interval: int = 100_000
    experiment_name: str = "megabonk_high_throughput"
    
    # Paths
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")


def create_policy(cfg: HighThroughputConfig, hparams: Optional[Dict] = None):
    """Create FP8 Transformer policy."""
    from src.learner.fp8_transformer import TransformerXLPolicy, TransformerPolicyConfig
    
    # Override with PBT hyperparameters if provided
    lr = hparams.get("learning_rate", cfg.learning_rate) if hparams else cfg.learning_rate
    
    policy_cfg = TransformerPolicyConfig(
        embed_dim=cfg.embed_dim,
        num_layers=cfg.num_layers,
        context_length=cfg.context_length,
        use_fp8=cfg.use_fp8,
    )
    
    return TransformerXLPolicy(policy_cfg)


def create_env(instance_id: int):
    """Create environment for a worker."""
    from src.env.megabonk_gym import MegabonkEnv, MegabonkEnvConfig
    
    env_cfg = MegabonkEnvConfig(
        use_dense_reward=True,
    )
    
    return MegabonkEnv(env_cfg)


def train_with_sample_factory(cfg: HighThroughputConfig):
    """Train using Sample Factory."""
    from src.learner.sample_factory import (
        SampleFactoryRunner,
        SampleFactoryConfig,
    )
    
    sf_cfg = SampleFactoryConfig(
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        rollout_length=cfg.rollout_length,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        entropy_coef=cfg.entropy_coef,
        device=cfg.device,
    )
    
    runner = SampleFactoryRunner(
        policy_factory=lambda: create_policy(cfg),
        env_factory=create_env,
        cfg=sf_cfg,
    )
    
    print("\n" + "=" * 60)
    print("Starting Sample Factory Training")
    print("=" * 60)
    print(f"Workers: {cfg.num_workers}")
    print(f"Batch Size: {cfg.batch_size:,}")
    print(f"FP8 Enabled: {cfg.use_fp8}")
    print(f"Context Length: {cfg.context_length}")
    
    runner.train(total_steps=100_000_000)


def train_with_pbt(cfg: HighThroughputConfig):
    """Train using Population Based Training."""
    from src.learner.pbt import PBTTrainer, PBTConfig
    
    pbt_cfg = PBTConfig(
        population_size=cfg.population_size,
        exploit_interval=cfg.pbt_interval,
        checkpoint_dir=cfg.checkpoint_dir / "pbt",
    )
    
    trainer = PBTTrainer(
        policy_factory=lambda hparams: create_policy(cfg, hparams),
        env_factory=create_env,
        cfg=pbt_cfg,
    )
    
    print("\n" + "=" * 60)
    print("Starting Population Based Training")
    print("=" * 60)
    print(f"Population Size: {cfg.population_size}")
    print(f"Evolution Interval: {cfg.pbt_interval:,} steps")
    print(f"FP8 Enabled: {cfg.use_fp8}")
    
    trainer.train(total_steps=100_000_000)


def main():
    parser = argparse.ArgumentParser(description="High-Throughput RL Training")
    
    # Workers
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16384)
    
    # Policy
    parser.add_argument("--use-fp8", action="store_true")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    
    # Training
    parser.add_argument("--use-pbt", action="store_true")
    parser.add_argument("--pbt-interval", type=int, default=1_000_000)
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda")
    
    # Paths
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--experiment", type=str, default="megabonk_high_throughput")
    
    args = parser.parse_args()
    
    # Build config
    cfg = HighThroughputConfig(
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        use_fp8=args.use_fp8,
        context_length=args.context_length,
        num_layers=args.num_layers,
        use_pbt=args.use_pbt,
        pbt_interval=args.pbt_interval,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=args.experiment,
    )
    
    # Print system info
    print("\n" + "=" * 60)
    print("METABONK HIGH-THROUGHPUT RL INFRASTRUCTURE")
    print("=" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        
        # Check for Transformer Engine (FP8)
        try:
            import transformer_engine
            print(f"Transformer Engine: Available (FP8 enabled)")
        except ImportError:
            print(f"Transformer Engine: Not available")
            cfg.use_fp8 = False
    except ImportError:
        print("PyTorch: Not available")
    
    print(f"Workers: {cfg.num_workers}")
    print(f"Batch Size: {cfg.batch_size:,}")
    print(f"FP8: {cfg.use_fp8}")
    print(f"PBT: {cfg.use_pbt}")
    print(f"Context Length: {cfg.context_length}")
    
    # Create directories
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    if cfg.use_pbt:
        train_with_pbt(cfg)
    else:
        train_with_sample_factory(cfg)


if __name__ == "__main__":
    main()

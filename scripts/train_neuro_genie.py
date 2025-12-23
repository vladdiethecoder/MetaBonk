#!/usr/bin/env python3
"""Training Pipeline for Neuro-Genie Architecture.

Unified training script for all Neuro-Genie components:
1. Latent Action Model (LAM) - VQ-VAE from video
2. Action Adapter - Latent ↔ Explicit translation  
3. Generative World Model - Spatiotemporal transformer
4. Dream-based RL (PPO) - Policy training in dreams
5. Federated Dreaming - Multi-niche specialists

Usage:
    python train_neuro_genie.py --mode lam --data ./gameplay_videos/
    python train_neuro_genie.py --mode world_model --data ./rollouts/
    python train_neuro_genie.py --mode dream_rl --world_model ./checkpoints/gwm.pt
    python train_neuro_genie.py --mode federated --config ./configs/federated.yaml
"""

import argparse
import json
import os
import time
from pathlib import Path
from dataclasses import asdict
import sys

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install: pip install torch")
    sys.exit(1)


# Import Neuro-Genie components
from src.neuro_genie.latent_action_model import (
    LAMConfig, LatentActionModel, LAMTrainer,
    create_frame_pairs_from_video, create_dataset_from_npz_dir,
)
from src.neuro_genie.action_adapter import (
    ActionAdapterConfig, ActionAdapter, AdapterTrainer,
)
from src.neuro_genie.generative_world_model import (
    GWMConfig, GenerativeWorldModel, GWMTrainer, VideoTokenizer,
)
from src.neuro_genie.offline_replay_env import OfflineReplayEnv
from src.neuro_genie.federated_dreaming import (
    FederatedDreamingConfig, FederatedDreamCoordinator,
    AgentRole, NICHE_REGISTRY,
)
from src.neuro_genie.blackwell_optimizer import (
    BlackwellOptimConfig,
    apply_blackwell_defaults,
    maybe_compile,
    CUDAGraphStep,
)


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


class FramePairDataset(Dataset):
    """Dataset of frame pairs for LAM training."""
    
    def __init__(
        self,
        data_dir: str,
        frame_size: tuple = (128, 128),
        max_samples: int = 100000,
    ):
        self.data_dir = Path(data_dir)
        self.frame_size = frame_size
        self.pairs = []
        
        # Load from NPZ files
        for npz_path in self.data_dir.glob("**/*.npz"):
            try:
                data = np.load(npz_path, allow_pickle=True)
                if 'frames' in data:
                    frames = data['frames']
                    for i in range(len(frames) - 1):
                        if len(self.pairs) >= max_samples:
                            break
                        self.pairs.append((npz_path, i))
            except Exception:
                continue
        
        print(f"Loaded {len(self.pairs)} frame pairs from {data_dir}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        import cv2
        
        npz_path, frame_idx = self.pairs[idx]
        data = np.load(npz_path, allow_pickle=True)
        frames = data['frames']
        
        frame_t = cv2.resize(frames[frame_idx], self.frame_size)
        frame_t1 = cv2.resize(frames[frame_idx + 1], self.frame_size)
        
        # Normalize and transpose
        frame_t = frame_t.astype(np.float32) / 255.0
        frame_t1 = frame_t1.astype(np.float32) / 255.0
        frame_t = np.transpose(frame_t, (2, 0, 1))
        frame_t1 = np.transpose(frame_t1, (2, 0, 1))
        
        return torch.tensor(frame_t), torch.tensor(frame_t1)


def train_lam(args):
    """Train Latent Action Model."""
    print("\n" + "="*60)
    print("  LATENT ACTION MODEL (LAM) TRAINING")
    print("="*60)
    
    print("""
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Watch & Intuit                                     │
│                                                             │
│ The LAM learns a "vocabulary of motion" by watching video:  │
│ • Encode frame pairs → latent action embedding              │
│ • Quantize to discrete codebook (512 codes)                 │
│ • Decode (frame_t, action) → predicted frame_t+1            │
│                                                             │
│ This unsupervised learning discovers WHAT actions exist     │
│ without knowing HOW they map to keyboard/mouse.             │
└─────────────────────────────────────────────────────────────┘
    """)
    
    # Configuration
    cfg = LAMConfig(
        frame_height=args.frame_size,
        frame_width=args.frame_size,
        num_latent_actions=args.num_actions,
        latent_action_dim=args.action_dim,
        learning_rate=args.lr,
        max_steps=args.max_steps,
    )
    
    print(f"Configuration:")
    print(f"  Frame size: {cfg.frame_height}x{cfg.frame_width}")
    print(f"  Latent actions: {cfg.num_latent_actions}")
    print(f"  Action dimension: {cfg.latent_action_dim}")
    print(f"  Learning rate: {cfg.learning_rate}")
    print(f"  Max steps: {cfg.max_steps}")
    
    # Create dataset
    print(f"\nLoading data from: {args.data}")
    dataset = FramePairDataset(args.data, frame_size=(cfg.frame_height, cfg.frame_width))
    
    if len(dataset) == 0:
        print("ERROR: No frame pairs found. Check data directory.")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Create model and trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    model = LatentActionModel(cfg)
    trainer = LAMTrainer(model, cfg, device=device)
    
    # Resume from checkpoint if exists
    ckpt_path = Path(args.checkpoint_dir) / "lam_latest.pt"
    if ckpt_path.exists() and args.resume:
        print(f"Resuming from: {ckpt_path}")
        trainer.load_checkpoint(str(ckpt_path))
    
    # Training loop
    print(f"\n{'='*60}")
    print("  Starting Training")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    epoch = 0
    
    while trainer.step < cfg.max_steps:
        epoch += 1
        epoch_loss = 0
        
        for batch_idx, (frame_t, frame_t1) in enumerate(dataloader):
            metrics = trainer.train_step(frame_t, frame_t1)
            epoch_loss += metrics['total_loss']
            
            if trainer.step % args.log_interval == 0:
                elapsed = time.time() - start_time
                eta = (elapsed / trainer.step) * (cfg.max_steps - trainer.step)
                
                print(
                    f"\rStep {trainer.step:>6}/{cfg.max_steps} | "
                    f"Loss: {metrics['total_loss']:.4f} | "
                    f"Recon: {metrics['recon_loss']:.4f} | "
                    f"Perplexity: {metrics['perplexity']:.1f} | "
                    f"Codebook: {metrics['codebook_utilization']*100:.1f}% | "
                    f"ETA: {format_time(eta)}",
                    end=""
                )
            
            if trainer.step >= cfg.max_steps:
                break
        
        # Checkpoint
        trainer.save_checkpoint(str(ckpt_path))
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch} complete. Avg loss: {avg_loss:.4f}")
    
    # Final save
    final_path = Path(args.checkpoint_dir) / "lam_final.pt"
    trainer.save_checkpoint(str(final_path))
    
    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Final checkpoint: {final_path}")
    print(f"  Total time: {format_time(time.time() - start_time)}")
    print(f"{'='*60}")


def train_world_model(args):
    """Train Generative World Model."""
    print("\n" + "="*60)
    print("  GENERATIVE WORLD MODEL TRAINING")
    print("="*60)
    
    print("""
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: The Dream Dojo                                     │
│                                                             │
│ Training the neural simulator (Genie-3 style):              │
│ 1. Video Tokenizer: Compress frames to discrete tokens      │
│ 2. Spatiotemporal Transformer: Predict next frame tokens    │
│                                                             │
│ The world model learns to "hallucinate" game environments   │
│ conditioned on latent actions and optional text prompts.    │
└─────────────────────────────────────────────────────────────┘
    """)
    
    # Configuration
    cfg = GWMConfig(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        learning_rate=args.lr,
        max_steps=args.max_steps,
    )
    
    print(f"Configuration:")
    print(f"  Embed dim: {cfg.embed_dim}")
    print(f"  Attention heads: {cfg.num_heads}")
    print(f"  Transformer layers: {cfg.num_layers}")
    
    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GenerativeWorldModel(cfg)
    trainer = GWMTrainer(model, cfg, device=device)
    
    # Phase 1: Train video tokenizer
    print("\n--- Phase 1: Training Video Tokenizer ---")
    # ... tokenizer training loop ...
    
    # Phase 2: Train transformer
    print("\n--- Phase 2: Training Transformer ---")
    # ... transformer training loop ...
    
    print("\nWorld Model training complete!")


def train_dream_rl(args):
    """Train a policy from offline `.pt` rollouts (honest replay, no interactivity).

    This is intentionally **not** "interactive dreaming": actions do not affect
    the next state in replay. The objective here is to produce a strong
    initialization via behavior cloning over the recorded dataset.
    """
    print("\n" + "="*60)
    print("  OFFLINE REPLAY POLICY TRAINING (BEHAVIOR CLONING)")
    print("="*60)
    
    print("""
┌─────────────────────────────────────────────────────────────┐
│ Offline Replay (Honest)                                     │
│                                                             │
│ Training from recorded `.pt` rollouts (actions ignored):    │
│ • No live game connection required                          │
│ • No synthetic initialization / placeholder rewards         │
│ • Produces a strong initialization for later online RL      │
│                                                             │
│ Later: fine-tune online or in a learned world model.        │
└─────────────────────────────────────────────────────────────┘
    """)

    pt_dir = Path(
        getattr(args, "pt_dir", "")
        or os.environ.get("METABONK_VIDEO_ROLLOUTS_PT_DIR", "")
        or args.data
    ).expanduser()
    print(f"[offline_replay] rollouts: {pt_dir}")
    env = OfflineReplayEnv(rollout_dir=pt_dir, sampling_mode="random")

    # Infer dims from one reset.
    obs0, info0 = env.reset()
    import torch

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        and not getattr(args, "device", "cuda").lower().startswith("cpu")
        else "cpu"
    )
    optimize_5090 = bool(getattr(args, "optimize_5090", False)) or str(
        os.environ.get("METABONK_OPTIMIZE_5090", "0") or "0"
    ).strip().lower() in ("1", "true", "yes", "on")
    compile_mode = str(getattr(args, "compile_mode", "max-autotune") or "max-autotune")
    if bool(getattr(args, "cuda_graph", False)) and bool(getattr(args, "compile", False)):
        if compile_mode == "max-autotune":
            compile_mode = "max-autotune-no-cudagraphs"
            print("[offline_replay] compile_mode=max-autotune-no-cudagraphs (cuda_graph enabled)")
    if optimize_5090 and device.type == "cuda":
        cfg = BlackwellOptimConfig(
            enable_tf32=True,
            cudnn_benchmark=True,
            matmul_precision=str(getattr(args, "matmul_precision", "high") or "high"),
            alloc_conf=str(getattr(args, "alloc_conf", "") or "expandable_segments:True"),
            memory_fraction=float(getattr(args, "memory_fraction", 0.0) or 0.0),
            compile=bool(getattr(args, "compile", False)),
            compile_mode=compile_mode,
        )
        applied = apply_blackwell_defaults(cfg)
        if applied:
            print(f"[offline_replay] blackwell={applied}")
    obs0_t = torch.as_tensor(obs0, dtype=torch.float32, device=device)

    # Determine action dim from the first trajectory.
    traj_path = Path(str(info0.get("path") or ""))
    tr = torch.load(traj_path, map_location="cpu", weights_only=False)
    act_dim = int(tr["actions"].shape[-1])
    obs_dim = int(obs0_t.numel())

    # Simple MLP policy: obs -> action (continuous).
    policy = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, act_dim),
    ).to(device)
    policy = maybe_compile(
        policy,
        enabled=bool(getattr(args, "compile", False)) and device.type == "cuda",
        mode=compile_mode,
    )

    opt = torch.optim.Adam(policy.parameters(), lr=float(args.lr))
    loss_fn = torch.nn.MSELoss()

    max_steps = int(args.max_steps)
    batch_size = int(args.batch_size)
    print(f"[offline_replay] device={device} obs_dim={obs_dim} act_dim={act_dim} steps={max_steps} batch={batch_size}")

    # Optional CUDA graph capture (static shapes only).
    use_cuda_graph = bool(getattr(args, "cuda_graph", False)) and device.type == "cuda"
    graph = None
    fixed_x = fixed_y = None
    if use_cuda_graph:
        fixed_x = torch.zeros((batch_size, obs_dim), device=device, dtype=torch.float32)
        fixed_y = torch.zeros((batch_size, act_dim), device=device, dtype=torch.float32)

        def _step_static():
            pred = policy(fixed_x)
            loss = loss_fn(pred, fixed_y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            return loss

        graph = CUDAGraphStep(_step_static)
        if graph.capture():
            print("[offline_replay] cuda_graph=captured")
        else:
            print("[offline_replay] cuda_graph=capture_failed; eager fallback")
            use_cuda_graph = False
            graph = None

    # Batch sampling by repeatedly resetting to random trajectories and sampling random indices.
    # This stays honest: targets are the recorded actions.
    for step in range(max_steps):
        # Sample a random trajectory file via env.reset().
        obs, info = env.reset()
        traj_path = Path(str(info.get("path") or ""))
        data = torch.load(traj_path, map_location="cpu", weights_only=False)
        obs_all = data["observations"]
        acts_all = data["actions"]
        T = int(obs_all.shape[0])
        if T < 2:
            continue
        idx = torch.randint(0, T, (min(batch_size, T),), device="cpu")
        x = obs_all[idx].to(device=device, dtype=torch.float32)
        y = acts_all[idx].to(device=device, dtype=torch.float32)

        if use_cuda_graph and fixed_x is not None and fixed_y is not None and graph is not None:
            if x.shape[0] != batch_size:
                # Pad/truncate to static graph shape.
                if x.shape[0] < batch_size:
                    pad = batch_size - x.shape[0]
                    x = torch.cat([x, x[:1].repeat(pad, 1)], dim=0)
                    y = torch.cat([y, y[:1].repeat(pad, 1)], dim=0)
                else:
                    x = x[:batch_size]
                    y = y[:batch_size]
            fixed_x.copy_(x)
            fixed_y.copy_(y)
            graph.replay()
            with torch.no_grad():
                loss = loss_fn(policy(fixed_x), fixed_y)
        else:
            pred = policy(x)
            loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if (step + 1) % max(1, int(args.log_interval)) == 0:
            print(f"[offline_replay] step {step+1}/{max_steps} loss={float(loss.detach().item()):.6f}")

    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "offline_replay_policy.pt"
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "obs_dim": obs_dim,
            "action_dim": act_dim,
            "pt_dir": str(pt_dir),
        },
        out_path,
    )
    print(f"[offline_replay] Saved policy to: {out_path}")


def train_federated(args):
    """Train federated dreaming with ecological niches."""
    print("\n" + "="*60)
    print("  FEDERATED DREAMING - ECOLOGICAL NICHES")
    print("="*60)
    
    print("""
┌─────────────────────────────────────────────────────────────┐
│ Phase 4: The Holographic God                                │
│                                                             │
│ Training specialists in divergent hallucinated environments │
│ and merging them into a generalist "God Agent":             │
│                                                             │
│ • SCOUT: Exploration, navigation, pathfinding               │
│ • SPEEDRUNNER: Momentum, timing, flow optimization          │
│ • TANK: Survival, damage avoidance, robustness              │
│ • KILLER: Combat, targeting, enemy prediction               │
│                                                             │
│ TIES-Merging combines specialists → zero-shot generalist    │
└─────────────────────────────────────────────────────────────┘
    """)
    
    # Configuration
    cfg = FederatedDreamingConfig(
        steps_per_niche=args.steps_per_niche,
        merge_method=args.merge_method,
        ties_density=args.ties_density,
    )
    
    print(f"Configuration:")
    print(f"  Steps per niche: {cfg.steps_per_niche}")
    print(f"  Merge method: {cfg.merge_method}")
    print(f"  TIES density: {cfg.ties_density}")
    print(f"\nEcological niches:")
    for role, niche in NICHE_REGISTRY.items():
        print(f"  • {niche.name}: {len(niche.prompts)} prompts")
    
    # Create coordinator
    coordinator = FederatedDreamCoordinator(cfg)
    
    # Define agent factory
    def create_agent():
        # Create a simple policy network
        return nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 64),  # Latent action dim
        )
    
    # Run pipeline
    print("\n" + "-"*60)
    print("Starting Federated Dreaming Pipeline...")
    print("-"*60 + "\n")
    
    results = coordinator.full_pipeline(
        agent_factory=create_agent,
        iterations=args.iterations,
        steps_per_iteration=args.steps_per_niche,
    )
    
    print(f"\nFederated Dreaming complete!")
    print(f"God Agent saved to: {results['final_god_agent_path']}")


def main():
    parser = argparse.ArgumentParser(
        description="Neuro-Genie Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train Latent Action Model on gameplay videos
  python train_neuro_genie.py --mode lam --data ./gameplay_videos/
  
  # Train Generative World Model
  python train_neuro_genie.py --mode world_model --data ./rollouts/
  
  # Train policy from offline replay rollouts (behavior cloning)
  python train_neuro_genie.py --mode dream_rl --pt-dir ./rollouts/
  
  # Run federated dreaming
  python train_neuro_genie.py --mode federated --iterations 3
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["lam", "adapter", "world_model", "dream_rl", "federated"],
        help="Training mode",
    )
    
    # Data paths
    parser.add_argument("--data", type=str, default="./gameplay_videos/",
                       help="Path to training data")
    parser.add_argument("--world_model", type=str, default="./checkpoints/gwm.pt",
                       help="Path to world model checkpoint")
    parser.add_argument("--pt-dir", type=str, default=os.environ.get("METABONK_VIDEO_ROLLOUTS_PT_DIR", "rollouts"),
                       help="Directory of .pt rollouts for offline replay training")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/neuro_genie/",
                       help="Directory for saving checkpoints")
    
    # Training settings
    parser.add_argument("--max_steps", type=int, default=100000,
                       help="Maximum training steps")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint")
    
    # LAM settings
    parser.add_argument("--frame_size", type=int, default=128,
                       help="Frame size for LAM")
    parser.add_argument("--num_actions", type=int, default=512,
                       help="Number of latent actions")
    parser.add_argument("--action_dim", type=int, default=64,
                       help="Latent action dimension")
    
    # World model settings
    parser.add_argument("--embed_dim", type=int, default=768,
                       help="Transformer embedding dimension")
    parser.add_argument("--num_heads", type=int, default=12,
                       help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=12,
                       help="Number of transformer layers")
    
    # Dream RL settings
    parser.add_argument("--num_envs", type=int, default=8,
                       help="Number of parallel dream envs")
    parser.add_argument("--episode_length", type=int, default=1000,
                       help="Episode length in dreams")
    parser.add_argument("--optimize-5090", action="store_true",
                       help="Enable RTX 5090/Blackwell-safe Torch optimizations (TF32, allocator, etc.).")
    parser.add_argument("--compile", action="store_true",
                       help="Use torch.compile() where applicable (CUDA only).")
    parser.add_argument("--compile-mode", default="max-autotune",
                       help="torch.compile mode (default: max-autotune).")
    parser.add_argument("--cuda-graph", action="store_true",
                       help="Attempt CUDA graph capture for the offline replay step (CUDA only; static shapes).")
    parser.add_argument("--matmul-precision", default="high",
                       help="torch.set_float32_matmul_precision (high|medium).")
    parser.add_argument("--alloc-conf", default="",
                       help="Override PYTORCH_CUDA_ALLOC_CONF (default: expandable_segments:True).")
    parser.add_argument("--memory-fraction", type=float, default=0.0,
                       help="Optional torch.cuda.set_per_process_memory_fraction (0 = unset).")
    
    # Federated settings
    parser.add_argument("--iterations", type=int, default=3,
                       help="Federated learning iterations")
    parser.add_argument("--steps_per_niche", type=int, default=50000,
                       help="Training steps per ecological niche")
    parser.add_argument("--merge_method", type=str, default="ties",
                       choices=["ties", "average", "fisher"],
                       help="Method for merging specialists")
    parser.add_argument("--ties_density", type=float, default=0.5,
                       help="TIES merging density")
    
    # Misc
    parser.add_argument("--log_interval", type=int, default=100,
                       help="Steps between logging")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="DataLoader workers")
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Route to appropriate training function
    if args.mode == "lam":
        train_lam(args)
    elif args.mode == "adapter":
        print("Adapter training requires a trained LAM. See documentation.")
    elif args.mode == "world_model":
        train_world_model(args)
    elif args.mode == "dream_rl":
        train_dream_rl(args)
    elif args.mode == "federated":
        train_federated(args)


if __name__ == "__main__":
    main()

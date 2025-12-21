"""SIMA 2 Training Pipeline.

End-to-end training orchestration for the SIMA 2 architecture:
- Phase 1: Diffusion policy pretraining (behavioral cloning)
- Phase 2: Consistency distillation (60Hz motor)
- Phase 3: Eureka reward evolution
- Phase 4: End-to-end RL with full SIMA 2 stack

Usage:
    python scripts/train_sima2.py --config configs/sima2.yaml
    python scripts/train_sima2.py --test-mode --max-steps 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class SIMA2TrainingConfig:
    """Configuration for SIMA 2 training pipeline.
    
    Optimized for RTX 5090 (31.4GB VRAM) with parallel processing.
    """
    
    # General
    experiment_name: str = "sima2_default"
    seed: int = 42
    device: str = "cuda"
    log_dir: str = "checkpoints/sima2"
    
    # Parallel optimization (RTX 5090)
    num_workers: int = 0             # Use 0 to avoid multiprocessing overhead with large datasets
    prefetch_factor: int = 4       # Batches to prefetch per worker
    use_compile: bool = True       # torch.compile() for faster kernels (PyTorch 2.0+)
    use_amp: bool = True           # Automatic mixed precision
    pin_memory: bool = True        # Faster CPU-GPU transfer
    gradient_accumulation: int = 4 # Effective batch size multiplier

    
    # Phase 1: Diffusion pretraining (larger batches for 31GB VRAM)
    diffusion_epochs: int = 100
    diffusion_batch_size: int = 128   # Increased from 64
    diffusion_lr: float = 1e-4
    rollout_dir: str = "rollouts/video_demos"
    video_dir: str = "gameplay_videos"
    
    # Phase 2: Consistency distillation
    consistency_epochs: int = 50
    consistency_batch_size: int = 64  # Increased from 32
    
    # Phase 3: Eureka reward (parallel candidate evaluation)
    eureka_generations: int = 10
    eureka_population: int = 16
    eureka_steps_per_candidate: int = 10000
    eureka_parallel_eval: int = 4     # Evaluate 4 candidates in parallel
    
    # Phase 4: End-to-end RL (high throughput)
    rl_total_steps: int = 1_000_000
    rl_batch_size: int = 512          # Increased from 256
    rl_lr: float = 3e-4
    eval_freq: int = 10000
    save_freq: int = 50000
    
    # Environment (parallel)
    num_envs: int = 8                 # Increased from 4
    frame_stack: int = 4
    
    # Test mode
    test_mode: bool = False
    max_test_steps: int = 1000





def load_trajectories(rollout_dir: str) -> List[Dict]:
    """Load trajectory data from rollout files.
    
    Handles corrupted .npz files gracefully by skipping them.
    """
    trajectories = []
    rollout_path = Path(rollout_dir)
    
    if not rollout_path.exists():
        print(f"[SIMA2] Warning: Rollout directory not found: {rollout_dir}")
        return trajectories
    
    npz_files = list(rollout_path.glob("*.npz"))
    print(f"[SIMA2] Found {len(npz_files)} trajectory files in {rollout_dir}")
    
    loaded = 0
    skipped = 0
    total_frames = 0
    
    for npz_file in npz_files:
        try:
            import numpy as np
            
            # Check file size first
            file_size = npz_file.stat().st_size
            if file_size < 1000:  # Skip tiny/corrupted files
                print(f"[SIMA2] Skipping small file: {npz_file.name} ({file_size} bytes)")
                skipped += 1
                continue
            
            data = np.load(npz_file, allow_pickle=True)
            traj = {
                "states": data.get("observations", data.get("states")),
                "actions": data.get("actions"),
                "rewards": data.get("rewards"),
                "dones": data.get("dones", data.get("terminals")),
            }
            
            if traj["states"] is not None and traj["actions"] is not None:
                num_frames = len(traj["states"])
                total_frames += num_frames
                trajectories.append(traj)
                loaded += 1
                
        except Exception as e:
            print(f"[SIMA2] Error loading {npz_file.name}: {type(e).__name__}: {e}")
            skipped += 1
    
    print(f"[SIMA2] Loaded {loaded} trajectories ({total_frames:,} frames), skipped {skipped}")
    return trajectories


# Module-level Dataset class for pickling with multiprocessing DataLoader
class TrajectoryDataset:
    """GPU-optimized trajectory dataset with prefetching.
    
    Defined at module level to enable pickling for DataLoader workers.
    """
    def __init__(self, trajectories, obs_horizon, horizon):
        self.samples = []
        self.obs_horizon = obs_horizon
        self.horizon = horizon
        
        for traj in trajectories:
            states = traj["states"]
            actions = traj["actions"]
            T = len(states)
            
            if T < horizon + obs_horizon:
                continue
            
            # Pre-generate all valid sample indices
            for start in range(T - horizon - obs_horizon):
                self.samples.append((states, actions, start))
        
        print(f"[SIMA2] Dataset: {len(self.samples):,} samples from {len(trajectories)} trajectories")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        import torch
        import numpy as np
        
        states, actions, start = self.samples[idx]
        
        # Extract frames and actions
        obs_frames = states[start:start + self.obs_horizon]
        act_seq = actions[start:start + self.horizon]
        
        # Convert to tensors with float32 for pinned memory
        obs_tensor = torch.from_numpy(np.asarray(obs_frames)).float()
        act_tensor = torch.from_numpy(np.asarray(act_seq)).float()
        
        return obs_tensor, act_tensor


def phase1_diffusion_pretraining(cfg: SIMA2TrainingConfig) -> Optional[Any]:

    """Phase 1: Pretrain diffusion policy via behavioral cloning.
    
    Handles image observations (224x224x3) by encoding to latent vectors.
    Uses GPU acceleration for RTX 5090.
    """
    print("\n" + "="*60)
    print("PHASE 1: Diffusion Policy Pretraining")
    print("="*60)
    
    try:
        import torch
        import torch.nn as nn
        from src.learner.diffusion_policy import DiffusionPolicy, DiffusionPolicyConfig
    except ImportError as e:
        print(f"[SIMA2] Skipping diffusion pretraining: {e}")
        return None
    
    # Check GPU availability
    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"[SIMA2] Using device: {device}")
    if device.type == "cuda":
        print(f"[SIMA2] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[SIMA2] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load trajectories
    trajectories = load_trajectories(cfg.rollout_dir)
    if not trajectories:
        print("[SIMA2] No trajectories found, skipping pretraining")
        return None
    
    # Check observation shape from first trajectory
    sample_obs = trajectories[0]["states"][0]
    if hasattr(sample_obs, 'shape'):
        print(f"[SIMA2] Observation shape: {sample_obs.shape}")
    
    # CNN Encoder for image observations
    class ImageEncoder(nn.Module):
        """Encode 224x224x3 images to 256-dim latent vectors."""
        def __init__(self, latent_dim: int = 256):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 8, stride=4, padding=2),  # 224 -> 56
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 56 -> 28
                nn.ReLU(),
                nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 28 -> 14
                nn.ReLU(),
                nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 14 -> 7
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),  # 7 -> 1
            )
            self.fc = nn.Linear(256, latent_dim)
        
        def forward(self, x):
            # Ensure device-consistent forward (avoid CPU/CUDA mismatches).
            try:
                dev = next(self.parameters()).device
                x = x.to(dev)
            except Exception:
                pass
            # x: [B, H, W, C] or [B, C, H, W]
            if x.dim() == 4 and x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2)  # BHWC -> BCHW
            elif x.dim() == 3 and x.shape[-1] == 3:
                x = x.permute(2, 0, 1).unsqueeze(0)  # HWC -> 1CHW
            x = x.float() / 255.0  # Normalize to [0, 1]
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    # Initialize models with channels-last for better GPU performance on convs
    latent_dim = 256
    encoder = ImageEncoder(latent_dim).to(device)
    if device.type == 'cuda':
        encoder = encoder.to(memory_format=torch.channels_last)
    
    policy_cfg = DiffusionPolicyConfig(
        obs_dim=latent_dim,
        action_dim=6,
        horizon=16,
    )
    policy = DiffusionPolicy(policy_cfg).to(device)
    
    # Apply torch.compile for faster kernels (PyTorch 2.0+)
    if cfg.use_compile and hasattr(torch, 'compile'):
        try:
            encoder = torch.compile(encoder, mode="reduce-overhead")
            print("[SIMA2] Applied torch.compile() to encoder")
        except Exception as e:
            print(f"[SIMA2] torch.compile failed: {e}")
    
    # Fused AdamW optimizer (faster CUDA kernels)
    try:
        encoder_optimizer = torch.optim.AdamW(
            encoder.parameters(), 
            lr=cfg.diffusion_lr,
            fused=True  # CUDA fused optimizer
        )
        print("[SIMA2] Using fused AdamW optimizer")
    except TypeError:
        encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=cfg.diffusion_lr)
    
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda' and cfg.use_amp))
    
    # Training config - push batch size higher for 31GB VRAM
    if cfg.test_mode:
        num_epochs = 2
    else:
        num_epochs = cfg.diffusion_epochs
    
    batch_size = cfg.diffusion_batch_size
    grad_accum_steps = cfg.gradient_accumulation
    
    print(f"[SIMA2] Training for {num_epochs} epochs on {len(trajectories)} trajectories")
    print(f"[SIMA2] Batch size: {batch_size}, Gradient accumulation: {grad_accum_steps}")
    print(f"[SIMA2] Effective batch size: {batch_size * grad_accum_steps}")
    
    # PRE-LOAD DATA TO GPU for maximum throughput
    import numpy as np
    print("[SIMA2] Pre-loading data to GPU memory...")
    all_obs = []

    all_act = []
    for traj in trajectories:
        states = traj["states"]
        actions = traj["actions"]
        T = len(states)
        if T < policy_cfg.horizon + policy_cfg.obs_horizon:
            continue
        # Sample random windows from each trajectory
        num_samples = min(2000, T - policy_cfg.horizon - policy_cfg.obs_horizon)
        for _ in range(num_samples):
            start = np.random.randint(0, T - policy_cfg.horizon - policy_cfg.obs_horizon)
            obs_frames = states[start:start + policy_cfg.obs_horizon]
            act_seq = actions[start:start + policy_cfg.horizon]
            all_obs.append(obs_frames)
            all_act.append(act_seq)
    
    # Convert to GPU tensors (this fits in 31GB)
    all_obs_tensor = torch.tensor(np.array(all_obs), dtype=torch.float32, device=device)
    all_act_tensor = torch.tensor(np.array(all_act), dtype=torch.float32, device=device)
    
    # Note: channels_last applied during forward pass (after reshape to rank 4)
    
    num_samples = len(all_obs_tensor)
    print(f"[SIMA2] GPU-resident: {num_samples:,} samples ({all_obs_tensor.element_size() * all_obs_tensor.nelement() / 1024**3:.2f} GB)")

    
    encoder.train()
    policy.train()
    
    batches_per_epoch = 512  # More batches now that data is on GPU
    print(f"[SIMA2] Fast GPU training: batch_size={batch_size}, batches/epoch={batches_per_epoch}")
    
    # Training loop with GPU-resident data (ZERO CPU-GPU transfer!)
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        # Shuffle indices on GPU
        perm = torch.randperm(num_samples, device=device)
        
        for batch_idx in range(batches_per_epoch):
            # Direct GPU indexing (no DataLoader overhead)
            start_idx = (batch_idx * batch_size) % num_samples
            indices = perm[start_idx:start_idx + batch_size]
            
            if len(indices) < batch_size:
                # Wrap around
                indices = torch.cat([indices, perm[:batch_size - len(indices)]])
            
            obs_batch = all_obs_tensor[indices]  # Already on GPU!
            act_batch = all_act_tensor[indices]  # Already on GPU!
            
            # Mixed precision forward pass
            with torch.amp.autocast(device_type='cuda', enabled=device.type == 'cuda' and cfg.use_amp):
                # Encode observations [B, obs_horizon, H, W, C] -> [B, obs_horizon, latent_dim]
                B, obs_horizon, H, W, C = obs_batch.shape
                obs_flat = obs_batch.view(B * obs_horizon, H, W, C)
                obs_encoded = encoder(obs_flat)  # [B*obs_horizon, latent_dim]
                obs_encoded = obs_encoded.view(B, obs_horizon, -1).float()
            
            # Update diffusion policy
            encoder_optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            losses = policy.update(act_batch, obs_encoded)
            
            # Backprop with gradient scaling
            if "loss" in losses and isinstance(losses["loss"], torch.Tensor):
                scaler.scale(losses["loss"]).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % grad_accum_steps == 0:
                    scaler.unscale_(encoder_optimizer)
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                    scaler.step(encoder_optimizer)
                    scaler.update()
            
            loss_val = losses.get("loss", 0)

            if isinstance(loss_val, torch.Tensor):
                loss_val = loss_val.item()
            epoch_loss += loss_val
        
        # Print epoch stats
        avg_loss = epoch_loss / batches_per_epoch
        print(f"  Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f} ({batches_per_epoch} batches)")
        
        # Clear CUDA cache periodically
        if device.type == "cuda" and (epoch + 1) % 10 == 0:
            torch.cuda.empty_cache()
    

    # Save checkpoint
    save_path = Path(cfg.log_dir) / f"{cfg.experiment_name}_diffusion.pt"


    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "policy": policy.state_dict(),
        "encoder": encoder.state_dict(),
        "config": {
            "latent_dim": latent_dim,
            "action_dim": 6,
            "horizon": 16,
        }
    }
    torch.save(checkpoint, save_path)
    print(f"[SIMA2] Saved diffusion policy + encoder to {save_path}")
    
    return {"policy": policy, "encoder": encoder}




def phase2_consistency_distillation(
    cfg: SIMA2TrainingConfig,
    teacher: Optional[Any] = None,
) -> Optional[Any]:
    """Phase 2: Distill diffusion policy to consistency model.
    
    Uses GPU acceleration and loads encoder from Phase 1.
    """
    print("\n" + "="*60)
    print("PHASE 2: Consistency Model Distillation")
    print("="*60)
    
    try:
        import torch
        import torch.nn as nn
        from src.learner.consistency_policy import ConsistencyPolicy, CPQEConfig
    except ImportError as e:
        print(f"[SIMA2] Skipping consistency distillation: {e}")
        return None
    
    # Setup device
    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"[SIMA2] Using device: {device}")
    if device.type == "cuda":
        print(f"[SIMA2] GPU: {torch.cuda.get_device_name(0)}")
    
    # Try to load encoder from phase 1
    encoder = None
    phase1_checkpoint = Path(cfg.log_dir) / f"{cfg.experiment_name}_diffusion.pt"
    if phase1_checkpoint.exists():
        print(f"[SIMA2] Loading encoder from {phase1_checkpoint}")
        checkpoint = torch.load(phase1_checkpoint, map_location=device, weights_only=False)
        
        # Recreate encoder architecture
        class ImageEncoder(nn.Module):
            def __init__(self, latent_dim: int = 256):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(3, 32, 8, stride=4, padding=2),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                )
                self.fc = nn.Linear(256, latent_dim)
            
            def forward(self, x):
                # Ensure device-consistent forward (avoid CPU/CUDA mismatches).
                try:
                    dev = next(self.parameters()).device
                    x = x.to(dev)
                except Exception:
                    pass
                if x.dim() == 4 and x.shape[-1] == 3:
                    x = x.permute(0, 3, 1, 2)
                elif x.dim() == 3 and x.shape[-1] == 3:
                    x = x.permute(2, 0, 1).unsqueeze(0)
                x = x.float()
                # Support both uint8-like [0..255] and normalized [0..1] inputs.
                try:
                    if float(x.max().detach().cpu().item()) > 1.5:
                        x = x / 255.0
                except Exception:
                    x = x / 255.0
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)
        
        encoder = ImageEncoder(256).to(device)
        if "encoder" in checkpoint:
            encoder.load_state_dict(checkpoint["encoder"])
            encoder.eval()  # Freeze encoder
            print("[SIMA2] Loaded encoder weights")
    
    # Initialize student
    student_cfg = CPQEConfig(
        obs_dim=256,
        action_dim=6,
        horizon=16,
    )
    student = ConsistencyPolicy(student_cfg).to(device)
    
    # Load trajectories
    trajectories = load_trajectories(cfg.rollout_dir)
    
    if cfg.test_mode:
        num_epochs = 2
    else:
        num_epochs = cfg.consistency_epochs
    
    print(f"[SIMA2] Training for {num_epochs} epochs")
    student.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for traj in trajectories:
            try:
                import numpy as np
                states = traj["states"]
                actions = traj["actions"]
                
                if len(actions) < student_cfg.horizon:
                    continue
                
                batch_size = min(16, len(states) // student_cfg.horizon)
                
                for _ in range(batch_size):
                    start = np.random.randint(0, len(actions) - student_cfg.horizon)
                    
                    # Process observations
                    if encoder is not None:
                        # Use image encoder
                        obs_frames = states[start:start+2]
                        obs_tensor = torch.tensor(obs_frames, dtype=torch.float32, device=device)
                        with torch.no_grad():
                            obs = encoder(obs_tensor).unsqueeze(0)  # [1, 2, 256]
                    else:
                        # Flatten if no encoder
                        obs_frames = states[start:start+2]
                        obs = torch.tensor(obs_frames, dtype=torch.float32, device=device)
                        obs = obs.view(1, 2, -1)[:, :, :256]  # Truncate to 256
                    
                    # Get actions on GPU
                    act = torch.tensor(
                        actions[start:start + student_cfg.horizon],
                        dtype=torch.float32,
                        device=device
                    ).unsqueeze(0)
                    
                    losses = student.update_policy(act, obs)
                    epoch_loss += losses.get("consistency_loss", 0)
                    num_batches += 1
                    
            except Exception as e:
                if epoch == 0:
                    print(f"[SIMA2] Batch error: {e}")
                continue
        
        if num_batches > 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: loss={epoch_loss/num_batches:.4f} ({num_batches} batches)")
        
        # Clear CUDA cache
        if device.type == "cuda" and (epoch + 1) % 10 == 0:
            torch.cuda.empty_cache()
    
    # Benchmark inference speed on GPU
    print("\n[SIMA2] Benchmarking inference speed...")
    if device.type == "cuda":
        benchmark = student.benchmark_inference_speed(target_hz=60.0)
        print(f"  Latency: {benchmark['latency_ms']:.2f}ms")
        print(f"  Achieved: {benchmark['achieved_hz']:.1f}Hz")
        print(f"  Target: {benchmark['target_hz']:.1f}Hz {'✓' if benchmark['meets_target'] else '✗'}")
    
    # Save
    save_path = Path(cfg.log_dir) / f"{cfg.experiment_name}_consistency.pt"
    torch.save({
        "policy": student.state_dict(),
        "config": {"obs_dim": 256, "action_dim": 6, "horizon": 16},
    }, save_path)
    print(f"[SIMA2] Saved consistency policy to {save_path}")
    
    return student




def phase3_eureka_reward(cfg: SIMA2TrainingConfig) -> Optional[str]:
    """Phase 3: Evolve reward function via Eureka.
    
    GPU-accelerated fitness evaluation for reward function candidates.
    """
    print("\n" + "="*60)
    print("PHASE 3: Eureka Reward Evolution")
    print("="*60)
    
    try:
        import torch
        from src.learner.eureka_reward import EurekaRewardEvolver, EurekaConfig
    except ImportError as e:
        print(f"[SIMA2] Skipping Eureka: {e}")
        return None
    
    # Setup device
    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"[SIMA2] Using device: {device}")
    if device.type == "cuda":
        print(f"[SIMA2] GPU: {torch.cuda.get_device_name(0)}")
    
    eureka_cfg = EurekaConfig(
        population_size=cfg.eureka_population if not cfg.test_mode else 4,
        generations=cfg.eureka_generations if not cfg.test_mode else 2,
        training_steps=cfg.eureka_steps_per_candidate if not cfg.test_mode else 100,
    )

    
    eureka = EurekaRewardEvolver(cfg=eureka_cfg)
    
    task_description = """
    Megabonk survival game:
    - Survive as long as possible against waves of enemies
    - Collect XP gems to level up and unlock upgrades
    - Avoid taking damage, manage health
    - Kill enemies for rewards
    - Explore to find chests and power-ups
    """
    
    best_reward_code = eureka.evolve(task_description)
    
    # Save results
    save_path = Path(cfg.log_dir) / f"{cfg.experiment_name}_eureka_results.json"
    eureka.save_results(str(save_path))
    print(f"[SIMA2] Saved Eureka results to {save_path}")
    
    # Save best reward function
    reward_path = Path(cfg.log_dir) / f"{cfg.experiment_name}_reward.py"
    with open(reward_path, "w") as f:
        f.write("# Best Eureka-evolved reward function\n\n")
        f.write(best_reward_code)
    print(f"[SIMA2] Saved best reward function to {reward_path}")
    
    return best_reward_code


def phase4_end_to_end_rl(
    cfg: SIMA2TrainingConfig,
    motor_policy: Optional[Any] = None,
    reward_code: Optional[str] = None,
) -> None:
    """Phase 4: End-to-end RL with full SIMA 2 stack.
    
    Offline "dream" training from real rollouts (no synthetic frames).

    Note: This recovery repo intentionally avoids synthetic/mock gameplay loops.
    For live end-to-end RL, run real workers feeding rollouts into the learner
    service. For offline pretraining from video, use scripts/video_pretrain.py
    to export `.pt` rollouts and then run this phase.
    """
    print("\n" + "="*60)
    print("PHASE 4: End-to-End RL Training")
    print("="*60)
    
    try:
        import torch
    except ImportError as e:
        print(f"[SIMA2] Cannot run end-to-end: {e}")
        return
    
    # Setup device
    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"[SIMA2] Using device: {device}")
    if device.type == "cuda":
        print(f"[SIMA2] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[SIMA2] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Offline rollouts (vector observations) exported from labeled video demos.
    pt_dir = Path(os.environ.get("METABONK_VIDEO_ROLLOUTS_PT_DIR", "rollouts/video_rollouts"))
    if not pt_dir.exists():
        raise RuntimeError(
            f"[SIMA2] Missing PT rollouts dir: {pt_dir}. "
            "Run scripts/video_pretrain.py (export_pt) after labeling actions+rewards."
        )

    pt_files = sorted(pt_dir.glob("*.pt"))
    if not pt_files:
        raise RuntimeError(
            f"[SIMA2] No .pt rollouts found in {pt_dir}. "
            "Run scripts/video_pretrain.py --phase export_pt (requires labeled actions+rewards)."
        )

    from src.learner.world_model import WorldModel, WorldModelConfig
    from src.learner.ppo import PolicyLearner, PPOConfig

    # Load episodes into memory (small subset in test mode).
    max_eps = 4 if cfg.test_mode else int(os.environ.get("METABONK_PHASE4_MAX_EPISODES", "0") or 0)
    episodes = []
    for f in pt_files[: max_eps or None]:
        episodes.append(torch.load(f, map_location=device, weights_only=False))

    obs0 = episodes[0]["observations"]
    act0 = episodes[0]["actions"]
    obs_dim = int(obs0.shape[-1])
    action_dim = int(act0.shape[-1])

    wm_cfg = WorldModelConfig(obs_dim=obs_dim, action_dim=action_dim)
    wm = WorldModel(wm_cfg).to(device)
    wm.train()

    wm_epochs = 1 if cfg.test_mode else int(os.environ.get("METABONK_PHASE4_WM_EPOCHS", "10"))
    print(f"[SIMA2] Training world model for {wm_epochs} epochs on {len(episodes)} episodes")
    for ep_i in range(wm_epochs):
        losses = []
        for ep in episodes:
            obs = ep["observations"].to(device)
            actions = ep["actions"].to(device)
            rewards = ep.get("rewards")
            if rewards is None:
                raise RuntimeError(
                    "[SIMA2] Episode missing `rewards`. "
                    "Export rollouts from video with rewards via `python scripts/video_pretrain.py --phase reward_label export_pt`."
                )
            rewards = rewards.to(device) if isinstance(rewards, torch.Tensor) else torch.as_tensor(
                rewards, device=device, dtype=torch.float32
            )
            dones = ep.get("dones")
            if dones is not None:
                dones = dones.to(device)
                if dones.dtype != torch.bool:
                    dones = dones.to(dtype=torch.bool)
            ld = wm.update_from_rollout(obs, actions, rewards, dones=dones)
            losses.append(float(ld.get("wm_recon", 0.0)))
        if losses:
            print(f"[SIMA2] WM epoch {ep_i+1}/{wm_epochs} wm_recon={float(sum(losses)/len(losses)):.4f}")

    # Dream (imagination) policy update in latent space.
    ppo_cfg = PPOConfig(continuous_dim=action_dim, discrete_branches=())
    dreamer = PolicyLearner(obs_dim=obs_dim, cfg=ppo_cfg, device=str(device))

    dream_steps = cfg.max_test_steps if cfg.test_mode else int(os.environ.get("METABONK_PHASE4_DREAM_STEPS", "2000"))
    dream_batch = int(os.environ.get("METABONK_PHASE4_DREAM_BATCH", "256"))
    dream_horizon = int(os.environ.get("METABONK_DREAM_HORIZON", "5"))
    dream_starts = int(os.environ.get("METABONK_DREAM_STARTS", "8"))

    print(f"[SIMA2] Dream training for {dream_steps} steps (batch={dream_batch}, horizon={dream_horizon})")
    wm.eval()
    torch.manual_seed(int(cfg.seed))
    if device.type == "cuda":
        try:
            torch.cuda.manual_seed_all(int(cfg.seed))
        except Exception:
            pass
    for step in range(int(dream_steps)):
        ep = episodes[int(torch.randint(0, len(episodes), (1,), device=device).item())]
        obs = ep["observations"].to(device)
        if obs.shape[0] < 2:
            continue
        B = min(dream_batch, int(obs.shape[0]))
        idx = torch.randint(0, obs.shape[0], (B,), device=device)
        obs_batch = obs[idx]
        dl = dreamer.dream_update(wm, obs_batch, horizon=dream_horizon, num_starts=dream_starts)
        if (step + 1) % 200 == 0:
            print(
                f"[SIMA2] dream step {step+1}/{dream_steps} "
                f"dream_loss={float(dl.get('dream_loss', 0.0)):.4f} "
                f"dream_return={float(dl.get('dream_return', 0.0)):.4f}"
            )

    # Save artifacts.
    out_dir = Path(cfg.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wm_path = out_dir / f"{cfg.experiment_name}_world_model.pt"
    torch.save({"model_state_dict": wm.state_dict(), "config": wm_cfg}, wm_path)
    print(f"[SIMA2] Saved world model to {wm_path}")

    dream_path = out_dir / f"{cfg.experiment_name}_dream_policy.pt"
    torch.save(
        {
            "policy_state_dict": dreamer.net.state_dict(),
            "config": ppo_cfg,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
        },
        dream_path,
    )
    print(f"[SIMA2] Saved dream policy to {dream_path}")





def main():
    parser = argparse.ArgumentParser(description="SIMA 2 Training Pipeline")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--experiment", type=str, default="sima2_run", help="Experiment name")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode (quick)")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps for test mode")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4], help="Run specific phase only")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Load config
    cfg = SIMA2TrainingConfig(
        experiment_name=args.experiment,
        test_mode=args.test_mode,
        max_test_steps=args.max_steps,
        device=args.device,
    )
    
    if args.config:
        try:
            import yaml
            with open(args.config) as f:
                config_dict = yaml.safe_load(f)
            for k, v in config_dict.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
    
    print("="*60)
    print("SIMA 2 Training Pipeline")
    print("="*60)
    print(f"Experiment: {cfg.experiment_name}")
    print(f"Device: {cfg.device}")
    print(f"Test mode: {cfg.test_mode}")
    print()
    
    # Run phases
    diffusion_policy = None
    consistency_policy = None
    reward_code = None
    
    if args.phase is None or args.phase == 1:
        diffusion_policy = phase1_diffusion_pretraining(cfg)
    
    if args.phase is None or args.phase == 2:
        consistency_policy = phase2_consistency_distillation(cfg, diffusion_policy)
    
    if args.phase is None or args.phase == 3:
        reward_code = phase3_eureka_reward(cfg)
    
    if args.phase is None or args.phase == 4:
        phase4_end_to_end_rl(cfg, consistency_policy, reward_code)
    
    print("\n" + "="*60)
    print("SIMA 2 Training Pipeline Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

"""Dream Bridge: Deprecated pixel-level dreaming environments.

Historically this module wrapped the pixel Generative World Model as a Gym
environment and then trained RL "inside dreams". That path relied on synthetic
frame initialization and placeholder reward logic, which destroys
troubleshooting signal.

MetaBonk now trains world-model updates and dreaming *offline* from real
rollouts exported from either:
  - video demos (`scripts/video_pretrain.py --phase export_pt|world_model|dream`)
  - live workers (saved `.pt` rollouts)

The `DreamBridgeEnv`/`BatchedDreamEnv` symbols remain for API compatibility, but
they intentionally raise with instructions to use the real-data pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


try:  # Optional dependency
    import gymnasium as gym  # type: ignore
except Exception:  # pragma: no cover
    try:
        import gym  # type: ignore
    except Exception:  # pragma: no cover
        gym = None  # type: ignore


@dataclass
class DreamBridgeConfig:
    """Configuration for Dream Bridge (kept for compatibility)."""

    # Frame dimensions
    frame_height: int = 128
    frame_width: int = 128
    frame_channels: int = 3

    # World model settings
    world_model_checkpoint: str = "checkpoints/world_model.pt"

    # Episode settings
    max_episode_steps: int = 1000
    context_frames: int = 4

    # Action space
    num_latent_actions: int = 512
    use_continuous_actions: bool = False


_DISABLED_MSG = (
    "DreamBridgeEnv (pixel-level gym wrapper) is intentionally disabled.\n"
    "Use offline dreaming from real `.pt` rollouts instead:\n"
    "  - `python scripts/video_pretrain.py --phase dream`\n"
    "  - or `POST /offline/dream/train` on the learner service.\n"
    "This avoids synthetic initialization and placeholder rewards."
)


class DreamBridgeEnv(gym.Env if gym is not None else object):  # type: ignore[misc]
    """Compatibility shim (raises).

    The dream environment is no longer the supported training path in MetaBonk.
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        cfg: Optional[DreamBridgeConfig] = None,
        world_model=None,
        render_mode: Optional[str] = None,
    ):
        raise RuntimeError(_DISABLED_MSG)


class BatchedDreamEnv:
    """Compatibility shim (raises)."""

    def __init__(
        self,
        num_envs: int,
        cfg: Optional[DreamBridgeConfig] = None,
        world_model=None,
    ):
        raise RuntimeError(_DISABLED_MSG)


def make_dream_env(
    cfg: Optional[DreamBridgeConfig] = None,
    world_model=None,
    render_mode: Optional[str] = None,
) -> DreamBridgeEnv:
    """Create a dream environment (disabled)."""

    return DreamBridgeEnv(cfg=cfg, world_model=world_model, render_mode=render_mode)


def make_batched_dream_env(
    num_envs: int,
    cfg: Optional[DreamBridgeConfig] = None,
    world_model=None,
) -> BatchedDreamEnv:
    """Create a batched dream environment (disabled)."""

    return BatchedDreamEnv(num_envs=num_envs, cfg=cfg, world_model=world_model)

